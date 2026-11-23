// lyricsAlignment.ts — the alignment pipeline behind Retranscribe and Re-align.
//
// The shape is the one the scope document argues for: Whisper is the push (it
// covers any language and produces a monotone map from audio to lyric index),
// the in-repo CTC forced aligner is the precision (it is scored against the
// lyrics the song was actually given, so a word can never be invented), and a
// statistics pass polices the result.
//
// Two constraints decide the rest:
//
//   * MMS_FA layer-normalises over its WHOLE input, so a full track is one
//     forward. That is why a full run is not windowed, and why a re-align of a
//     selection is a deliberate crop: the model then normalises over the
//     selection's own audio, which is a different (and here better) input.
//   * The aligner's CTC alphabet is 28 romanised symbols. Lyrics outside it are
//     not force-fitted into spans that would be wrong; the run falls back to
//     Whisper and says so in `diagnostics.fallback`. Inside it, the text is
//     spelled the way it is sung rather than the way it is written: `alignerLyrics`
//     expands acronyms and reads digits as words, because the engine keeps only
//     `[a-z']` from a token and would otherwise align `AGI` as "agi" and drop
//     `2024` entirely (see `lyricsPronunciation.ts`).
//
// The corridor from the scope document is drawn in two places. On a full run it
// is the anchor set: Whisper's reconciled words are the loose map the CTC pass
// is audited against after the fact (`anchorDrift`), which measures and moves
// nothing — the exact lyrics already make the lattice monotone. On a selection it is the crop itself: only the selected
// lines' text is sent, against only the audio under them, so the lattice cannot
// wander into a repeated chorus elsewhere in the track.

import fs from 'fs';
import { HOP_SEC, fileEnvelope } from './audioActivity.js';
import { removeSlice, sliceToWav } from './audioSlice.js';
import { vocalStems } from './vocalStemCache.js';
import { normaliseWord, pairByText } from './textPairing.js';
import { alignerLyrics } from './lyricsPronunciation.js';
import { resolveYue2AlignerModel } from './training/yue2Align.js';
import { yue2Align } from './backends/yue2/client.js';
import { yue2LyricsJson } from './backends/yue2/align.js';
import {
  reconcileLyrics,
  type AnchorDrift,
  type LyricsDiagnostics,
  type LyricsJson,
  type LyricsLine,
  type LyricsWord,
  type PolishRepair,
} from './lyricsReconcile.js';
import { transcribeTrack, type WhisperOptions, type WhisperResult } from './whisperTranscribe.js';

/** Shortest word span this pipeline will write. Same floor the editor uses. */
const MIN_WORD_SEC = 0.05;
/** How far an energy onset may be from a flagged word's onset and still move it. */
const SNAP_TOLERANCE_SEC = 0.12;
/** A cohort needs this many words before its median is trusted. */
const MIN_COHORT_WORDS = 5;
/** Fallback cohort: this many lines either side when a section is too small. */
const SECTION_FALLBACK_LINES = 4;
/** How far to look for same-layer words when a section run is too small: a
 *  harmony line is often two words, and a mixed window would hand it the lead's
 *  pace (or the lead the harmony's). */
const SAME_LAYER_FALLBACK_LINES = 8;
/** Robust sigma: 1.4826 x MAD assumes a normal core. */
const MAD_TO_SIGMA = 1.4826;
/** A floor for the spread, so a run of equal notes cannot narrow the band to zero. */
const MIN_SIGMA_SEC = 0.05;
/** How many sigma a duration may sit from its section median before it is flagged. */
const DURATION_SIGMAS = 3.5;
/** Below this CTC confidence a word is eligible for onset re-timing. */
const SNAP_CONFIDENCE = 0.35;
/** Snapping must improve an onset by this much, or it is not worth the risk. */
const SNAP_HYSTERESIS_SEC = 0.03;
/** A snapped onset must stay voiced this long, so a click is not a syllable. */
const SNAP_VOICED_SEC = 0.04;
/** How far above its local level a frame must sit to count as voice. */
const VOICED_RATIO = 0.45;
/** Absolute gate, so a silent file's noise floor is never voice. */
const VOICED_PEAK_RATIO = 0.02;
/** Audio kept either side of a selected range, so a clipped first/last word is
 *  still fully inside the crop. */
const RANGE_PAD_SEC = 2;
/** Above this share of non-Latin letters the CTC aligner is skipped. */
const MAX_NON_ROMANISED = 0.15;
/** How far a CTC onset may sit from its anchor before it is counted as drift. */
const ANCHOR_OUTLIER_SEC = 0.25;
/** How far a repeated line's word may sit from the earlier occurrence's offset
 *  before the repeat is treated as evidence and the word is re-placed. */
const REPEAT_TOLERANCE_SEC = 0.2;
/** A repeat's pace may differ from the first occurrence's by this much, and no
 *  more — beyond it the two are not the same delivery and nothing is reused. */
const REPEAT_SCALE_MIN = 0.85;
const REPEAT_SCALE_MAX = 1.15;
/** A word shorter than this did not take a sung syllable's time, so a span made
 *  of such words measures the aligner's error, not the repeat's pace. */
const REPEAT_COLLAPSED_SEC = 0.12;
/** A source word the aligner itself doubted is not a prediction: copying it
 *  would carry the same bad onset into every later occurrence. Same line the
 *  statistics pass draws between a measurement and a guess. */
const REPEAT_SOURCE_CONFIDENCE = SNAP_CONFIDENCE;
/** A near repeat shares at least this many words ... */
const REPEAT_MIN_SHARED_WORDS = 3;
/** ... and at least this share of the longer line ... */
const REPEAT_MIN_SHARED_PCT = 60;
/** ... differing by no more than this many words on either side. */
const REPEAT_MAX_VARIATION_WORDS = 2;
/** How many disagreements an audit names, worst first. */
const ANCHOR_WORST = 8;

export interface AlignRangeRequest {
  /** Inclusive indices into the existing `.lyrics.json` lines. */
  fromLine: number;
  toLine: number;
  /** Derived from those lines by the caller; kept for the diagnostics block. */
  startSec: number;
  endSec: number;
}

export interface AlignmentRequest {
  audioPath: string;
  /** The render's own lyric text, for a full run. Ignored for a range. */
  lyrics: string;
  /** The sidecar being updated. Required for a range, optional metadata for a full run. */
  existing?: LyricsJson | null;
  range?: AlignRangeRequest;
  model?: string;
  language?: string;
  beamSize?: number;
  dtw?: boolean;
  isolateVocals?: boolean;
  useCtc?: boolean;
  signal?: AbortSignal;
  onProgress?: (stage: string) => void;
}

export interface AlignmentOutcome {
  lines: LyricsLine[];
  diagnostics: LyricsDiagnostics;
  vocalsIsolated: boolean;
}

// ── Pure helpers ────────────────────────────────────────────────────

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Nearest-rank percentile of an ascending array. */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

interface FlatTiming {
  word: string;
  start: number;
  line: number;
  wordIndex: number;
}

function flattenTimings(lines: LyricsLine[]): FlatTiming[] {
  const out: FlatTiming[] = [];
  lines.forEach((line, li) => line.words.forEach((word, wi) => {
    out.push({ word: word.word, start: word.start, line: li, wordIndex: wi });
  }));
  return out;
}

/**
 * Measure the CTC timings against Whisper's anchors. Nothing is moved.
 *
 * This is the audit the scope document's corridor argument actually needs: the
 * exact lyrics already make the CTC path monotone, so the question is not
 * whether it can wander off, but how far inside a line it drifts from the words
 * that were heard. A large number here is the evidence for building an
 * engine-side corridor; a small one says the corridor would be solving a problem
 * this pipeline does not have.
 */
export function anchorDrift(
  ctc: LyricsLine[], anchors: LyricsLine[], toleranceSec = ANCHOR_OUTLIER_SEC,
): AnchorDrift {
  const pairs = pairByText(flattenTimings(ctc), flattenTimings(anchors));
  const errors = pairs
    .map(([a, b]) => Math.abs(a.start - b.start) * 1000)
    .sort((x, y) => x - y);

  const sectionAt = (line: number) => (ctc[line]?.section || '').trim();
  const worst = pairs
    .map(([a, b]) => {
      const section = sectionAt(a.line);
      return {
        line: a.line,
        word: a.wordIndex,
        driftMs: Math.round(Math.abs(a.start - b.start) * 1000),
        anchorSec: Math.round(b.start * 1000) / 1000,
        ctcSec: Math.round(a.start * 1000) / 1000,
        ...(section ? { section } : {}),
      };
    })
    .sort((x, y) => y.driftMs - x.driftMs)
    .slice(0, ANCHOR_WORST);

  // Per section, because a whole-track median hides the case the corridor is
  // for: a chorus drifting 300 ms late and a verse 40 ms early average into one
  // reassuring number. The runs are the polish pass's own reading of a sidecar,
  // so a section's second appearance is a run rather than a search for its
  // label — which is exactly the repeat a corridor would have to constrain.
  const runs = sectionRuns(ctc);
  const sections = runs.map((run, index) => {
    const label = sectionLabel(ctc[run[0]]);
    const occurrence = runs.slice(0, index + 1)
      .filter(other => sectionLabel(ctc[other[0]]) === label).length;
    const within = pairs
      .filter(([a]) => run.includes(a.line))
      .map(([a, b]) => Math.abs(a.start - b.start) * 1000)
      .sort((x, y) => x - y);
    return {
      section: label === '—' ? '' : label,
      occurrence,
      words: within.length,
      medianMs: Math.round(percentile(within, 50)),
      p90Ms: Math.round(percentile(within, 90)),
      outliers: within.filter(ms => ms > toleranceSec * 1000).length,
    };
  }).filter(entry => entry.words > 0);

  return {
    anchors: anchors.reduce((n, line) => n + line.words.length, 0),
    words: ctc.reduce((n, line) => n + line.words.length, 0),
    medianMs: Math.round(percentile(errors, 50)),
    p90Ms: Math.round(percentile(errors, 90)),
    outliers: errors.filter(ms => ms > toleranceSec * 1000).length,
    worst,
    sections,
  };
}

/** One track's audit, as `npm run lyrics:drift` reads it back from a sidecar. */
export interface DriftRecord {
  /** The track, for the report only. */
  track: string;
  drift: AnchorDrift;
}

export interface DriftSummary {
  tracks: number;
  /** Anchor words behind the pooled audits: the denominator for `outliers`. */
  words: number;
  outliers: number;
  /** The median of the per-track medians: one track, one vote, so a long track
   *  with many words does not decide the number on its own. */
  medianMs: number;
  /** The largest P90 any single track had — the tail, not an average of it. */
  worstP90Ms: number;
  /** Per section label and occurrence, pooled across tracks. */
  sections: Array<{
    section: string;
    occurrence: number;
    tracks: number;
    words: number;
    medianMs: number;
    outliers: number;
  }>;
}

/**
 * Pool several tracks' anchor audits into the numbers D2-C is decided on.
 *
 * One track's median says the CTC path is usually close; it cannot say whether
 * the misses sit in one place. So the question is put to the sections: if a
 * chorus's second occurrence drifts like its first, a corridor has nothing to
 * constrain, and if it drifts far more, that is the case for one. This is not an
 * accuracy claim either way — the reference is Whisper, which is also machine
 * output, so it measures disagreement between two aligners.
 */
export function summariseDrift(records: DriftRecord[]): DriftSummary {
  const pooled = new Map<string, {
    section: string; occurrence: number; tracks: number; words: number;
    medians: number[]; outliers: number;
  }>();

  for (const { drift } of records) {
    // A sidecar written before the section breakdown existed has no rows: it is
    // still a track with a median, it just cannot speak to a repeat.
    for (const section of drift.sections ?? []) {
      const key = `${section.section}\u0000${section.occurrence}`;
      const entry = pooled.get(key) ?? {
        section: section.section, occurrence: section.occurrence,
        tracks: 0, words: 0, medians: [], outliers: 0,
      };
      entry.tracks++;
      entry.words += section.words;
      entry.medians.push(section.medianMs);
      entry.outliers += section.outliers;
      pooled.set(key, entry);
    }
  }

  return {
    tracks: records.length,
    words: records.reduce((n, record) => n + record.drift.anchors, 0),
    outliers: records.reduce((n, record) => n + record.drift.outliers, 0),
    medianMs: Math.round(percentile(records.map(r => r.drift.medianMs).sort((a, b) => a - b), 50)),
    worstP90Ms: records.reduce((n, record) => Math.max(n, record.drift.p90Ms), 0),
    sections: [...pooled.values()]
      .map(entry => ({
        section: entry.section,
        occurrence: entry.occurrence,
        tracks: entry.tracks,
        words: entry.words,
        medianMs: Math.round(percentile(entry.medians.sort((a, b) => a - b), 50)),
        outliers: entry.outliers,
      }))
      // Most outliers first: that is the row the eye has to explain.
      .sort((a, b) => b.outliers - a.outliers || b.medianMs - a.medianMs),
  };
}

/**
 * Share of a lyric's letters that are not Latin script.
 *
 * This is the honest version of "can the CTC aligner read this": MMS_FA's
 * labels are romanised, so a CJK or Cyrillic line has no symbol to map onto and
 * the aligner would drop it. Accented Latin is still Latin, so `café` passes.
 *
 * Digits are neither Latin nor foreign — they have no script — so they are not
 * counted at all. Counting them made `2024` a non-Latin word and could push an
 * otherwise romanised lyric over the threshold into a Whisper-only run.
 */
export function nonRomanisedRatio(lyrics: string): number {
  const letters = Array.from(lyrics.replace(/\[[^\]\n]*\]/g, ' '))
    .filter(ch => /\p{L}/u.test(ch));
  if (letters.length === 0) return 0;
  const foreign = letters.filter(ch => !/\p{Script=Latin}/u.test(ch)).length;
  return foreign / letters.length;
}

/** A line's text: its stored text, else its words. Section markers are not text. */
function lineText(line: LyricsLine): string {
  const direct = (line.text || '').trim();
  if (direct) return direct;
  return (line.words || []).map(w => w.word).join(' ').trim();
}

/** Comparison form for a line: case and punctuation are not identity. */
function normaliseLine(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

export interface SectionCarry {
  lines: LyricsLine[];
  /** How each label was placed, so a fallback is visible rather than silent. */
  matched: { text: number; index: number; time: number };
}

/**
 * Put the old sections back on a range run's new lines.
 *
 * A range run sends only the selected lines' words, so section labels — which
 * are not lyrics — never come back with it. Text is the stable thing here: the
 * words are the same, only their times were recomputed, so the label follows
 * the line it belongs to even when the block moved. Index is the fallback for a
 * line whose words were edited, and it only means anything when the run
 * returned the same number of lines. Nearest time is the last resort, and it is
 * exactly the case where the block moved, so it is reported as such.
 */
export function carrySections(oldLines: LyricsLine[], newLines: LyricsLine[]): SectionCarry {
  const byText = new Map<string, number[]>();
  oldLines.forEach((line, index) => {
    const key = normaliseLine(lineText(line));
    if (!key) return;
    const list = byText.get(key);
    if (list) list.push(index);
    else byText.set(key, [index]);
  });

  const used = new Set<number>();
  const matched = { text: 0, index: 0, time: 0 };
  const withSection = (line: LyricsLine, section: string | undefined): LyricsLine =>
    (section ? { ...line, section } : line);

  const lines = newLines.map((line, at) => {
    const candidates = byText.get(normaliseLine(lineText(line)));
    const fromText = candidates?.find(index => !used.has(index));
    if (fromText != null) {
      used.add(fromText);
      matched.text++;
      return withSection(line, oldLines[fromText].section);
    }

    if (newLines.length === oldLines.length && !used.has(at)) {
      used.add(at);
      matched.index++;
      return withSection(line, oldLines[at].section);
    }

    let nearest = -1;
    let best = Infinity;
    oldLines.forEach((candidate, index) => {
      if (used.has(index)) return;
      const distance = Math.abs(candidate.start - line.start);
      if (distance < best) { best = distance; nearest = index; }
    });
    if (nearest < 0) return line;
    used.add(nearest);
    matched.time++;
    return withSection(line, oldLines[nearest].section);
  });

  return { lines, matched };
}

/** A hand-checked line is only put back when the rerun agrees closely enough. */
const CHECKED_MOVE_SEC = 5;

export interface CheckedCarry {
  lines: LyricsLine[];
  /** Hand-placed lines restored, and matches rejected as too different. */
  kept: number;
  skipped: number;
}

/**
 * Put hand-placed lines back over a fresh run's timings.
 *
 * A re-align rebuilds every word from the audio, so the lines a person placed
 * by ear would otherwise be replaced by the same machine guess the placement
 * was correcting. Matching by text and restoring the words is what gives those
 * lines their weight: the statistics pass runs afterwards and repairs the rest
 * around them.
 *
 * A repeated line matches on text like any other, so the occurrence nearest in
 * time is the one it pairs with, not the first one in the file. `offsetSec` is
 * the crop a selection run worked on: the rerun's times are relative to it, and
 * the distance test has to compare like with like.
 *
 * The match is deliberately conservative. A line whose word count changed, or
 * that the rerun put more than a few seconds away, is not the line the person
 * heard — carrying it would stamp old times onto a different part of the song,
 * so it is counted and skipped instead.
 */
export function carryChecked(
  oldLines: LyricsLine[], newLines: LyricsLine[], offsetSec = 0,
): CheckedCarry {
  const byText = new Map<string, number[]>();
  oldLines.forEach((line, index) => {
    if (line.checked !== true) return;
    const key = normaliseLine(lineText(line));
    if (!key) return;
    const list = byText.get(key);
    if (list) list.push(index);
    else byText.set(key, [index]);
  });
  if (byText.size === 0) return { lines: newLines, kept: 0, skipped: 0 };

  let kept = 0;
  let skipped = 0;
  const lines = newLines.map(line => {
    const candidates = byText.get(normaliseLine(lineText(line)));
    if (!candidates?.length) return line;
    // The nearest occurrence, so a chorus matched to the repeat it actually is.
    let from = -1;
    let distance = Infinity;
    const at = line.start + offsetSec;
    for (const index of candidates) {
      const candidate = Math.abs(oldLines[index].start - at);
      if (candidate < distance) { distance = candidate; from = index; }
    }
    if (from < 0) return line;
    candidates.splice(candidates.indexOf(from), 1);
    const old = oldLines[from];
    if (old.words.length !== line.words.length || distance > CHECKED_MOVE_SEC) {
      skipped++;
      return line;
    }
    kept++;
    // Restored in the same clock as `newLines`. A selection's words are relative
    // to their crop and the caller shifts the block back afterwards, so keeping
    // the sidecar's absolute times here put a carried line one crop start late —
    // an anchor in the wrong place is worse than no anchor at all.
    const words = old.words.map(word => ({
      ...word,
      start: word.start - offsetSec,
      end: word.end - offsetSec,
      checked: true,
    }));
    return {
      ...line,
      start: words[0].start,
      end: words[words.length - 1].end,
      text: (line.text || '').trim() || words.map(w => w.word).join(' '),
      checked: true,
      words,
    };
  });

  return { lines, kept, skipped };
}

/** How the statistics pass reads the audio around one instant. */
interface EnergyProfile {
  env: number[];
  hopSec: number;
  peak: number;
}

export function energyProfile(env: number[], hopSec = HOP_SEC): EnergyProfile {
  return { env, hopSec, peak: env.reduce((m, v) => Math.max(m, v), 0) };
}

/** Median of a one-second window, so a quiet verse is judged by its own level. */
function localLevel(profile: EnergyProfile, frame: number): number {
  const radius = Math.max(4, Math.round(1 / profile.hopSec));
  const from = Math.max(0, frame - radius);
  const to = Math.min(profile.env.length - 1, frame + radius);
  const slice = profile.env.slice(from, to + 1);
  return slice.length ? median(slice) : 0;
}

function thresholdAt(profile: EnergyProfile, frame: number): number {
  return Math.max(
    profile.peak * VOICED_PEAK_RATIO,
    localLevel(profile, frame) * VOICED_RATIO,
    1e-5,
  );
}

function voicedAt(profile: EnergyProfile, frame: number): boolean {
  if (frame < 0 || frame >= profile.env.length) return false;
  return profile.env[frame] >= thresholdAt(profile, frame);
}

/** True when any frame in [startSec, endSec) is voiced. */
function voicedBetween(profile: EnergyProfile, startSec: number, endSec: number): boolean {
  const from = Math.max(0, Math.floor(startSec / profile.hopSec));
  const to = Math.min(profile.env.length - 1, Math.ceil(endSec / profile.hopSec));
  for (let i = from; i <= to; i++) if (voicedAt(profile, i)) return true;
  return false;
}

/** Nearest rising edge to `t`, inside the cue's legal window. */
function nearestOnset(
  profile: EnergyProfile, t: number, minStart: number, maxStart: number,
): number | null {
  const { env, hopSec } = profile;
  const from = Math.max(1, Math.floor((t - SNAP_TOLERANCE_SEC) / hopSec));
  const to = Math.min(env.length - 1, Math.ceil((t + SNAP_TOLERANCE_SEC) / hopSec));
  const voicedFrames = Math.max(1, Math.ceil(SNAP_VOICED_SEC / hopSec));
  let best: number | null = null;
  let bestDistance = Infinity;
  for (let i = from; i <= to; i++) {
    const threshold = thresholdAt(profile, i);
    if (env[i] < threshold || env[i - 1] >= threshold) continue;
    const candidate = i * hopSec;
    if (candidate < minStart + 0.01 || candidate + MIN_WORD_SEC > maxStart) continue;
    // A rise that does not stay voiced is a click, not a syllable.
    let voiced = 0;
    for (let k = i; k < env.length && voiced < voicedFrames; k++) {
      if (!voicedAt(profile, k)) break;
      voiced++;
    }
    if (voiced < voicedFrames) continue;
    const distance = Math.abs(candidate - t);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  return best;
}

export interface SectionBaseline {
  section: string;
  words: number;
  medianSec: number;
  madSec: number;
  lowSec: number;
  highSec: number;
}

export interface PolishStats {
  words: number;
  wordsFlagged: number;
  cutBack: number;
  snapped: number;
  /** Words on a hand-checked line: anchors the rest is repaired around. */
  anchored: number;
  /** Lines that had an earlier occurrence to reuse, and words it re-placed. */
  repeatedLines: number;
  repeatWordsMoved: number;
  monotonicRepairs: number;
  untouched: number;
  /** Timing changes actually applied: repeats, cut-backs plus snaps. */
  wordsMoved: number;
  medianSec: number;
  madSec: number;
  lowSec: number;
  highSec: number;
  sections: SectionBaseline[];
}

export interface PolishResult {
  lines: LyricsLine[];
  stats: PolishStats;
  repairs: PolishRepair[];
  repairsTruncated: boolean;
}

interface CohortWord {
  duration: number;
  syllables: number;
}

interface Baseline {
  median: number;
  mad: number;
  low: number;
  high: number;
  /** Median duration of one syllable, for the expected-duration model. */
  syllableSec: number;
}

/** Count vowel groups: a cheap syllable estimate, so "extraordinary" is allowed
 *  to last longer than "go" without asking for a pronunciation lexicon. */
function syllableCount(word: string): number {
  const groups = word.toLowerCase().replace(/[^a-zà-ÿ'’]/g, ' ').match(/[aeiouyà-ÿ]+/g);
  return Math.max(1, groups?.length ?? 1);
}

function toCohortWord(word: LyricsWord): CohortWord {
  return {
    duration: Math.max(MIN_WORD_SEC, word.end - word.start),
    syllables: syllableCount(word.word),
  };
}

function baselineOf(cohort: CohortWord[]): Baseline {
  const durations = cohort.map(c => c.duration);
  const medianSec = median(durations) || 0.25;
  const mad = median(durations.map(d => Math.abs(d - medianSec))) || 0;
  const sigma = Math.max(MIN_SIGMA_SEC, mad * MAD_TO_SIGMA);
  const syllableSec = median(cohort.map(c => c.duration / Math.max(1, c.syllables))) || medianSec;
  return {
    median: medianSec,
    mad,
    low: Math.max(MIN_WORD_SEC, medianSec - DURATION_SIGMAS * sigma),
    high: medianSec + DURATION_SIGMAS * sigma,
    syllableSec,
  };
}

/** A section is a run of consecutive lines sharing a label, so a repeated
 *  chorus is its own cohort even when another section sits between repeats. */
function sectionRuns(lines: LyricsLine[]): number[][] {
  const runs: number[][] = [];
  let previousKey: string | null = null;
  lines.forEach((line, index) => {
    const key = (line.section || '').trim() || '—';
    if (previousKey === key && runs.length) runs[runs.length - 1].push(index);
    else runs.push([index]);
    previousKey = key;
  });
  return runs;
}

function sectionLabel(line: LyricsLine): string {
  return (line.section || '').trim() || '—';
}

/** A word's own layer wins over its line's; anything unrecognised is lead. */
function wordLayer(line: LyricsLine, word: LyricsWord): string {
  const tag = word.layer ?? line.layer;
  return tag === 'backing' || tag === 'adlib' ? tag : 'lead';
}

/** The stream a line is judged in: its own tag, else its first tagged word. */
function lineLayer(line: LyricsLine): string {
  if (line.layer === 'backing' || line.layer === 'adlib') return line.layer;
  for (const word of line.words) {
    const tag = wordLayer(line, word);
    if (tag !== 'lead') return tag;
  }
  return 'lead';
}

/** The cohort a line's words are judged against.
 *
 *  Same layer first: a harmony's short syllables must not become the pace an
 *  ordinary lead word is measured against. A backing-majority section would
 *  otherwise pull the pooled median down and flag every lead word as tooLong.
 *  A harmony line is often only two words, so the window widens within the
 *  layer before giving up. Below `MIN_COHORT_WORDS` in the layer, the
 *  layer-blind windows are used: a band built from two words is worse than a
 *  mixed one. */
function cohortFor(
  lines: LyricsLine[], lineIndex: number, runOfLine: number[], runs: number[][],
): CohortWord[] {
  const layer = lineLayer(lines[lineIndex]);
  const layerWords = (li: number) => lines[li].words
    .filter(word => wordLayer(lines[li], word) === layer)
    .map(toCohortWord);

  const run = runOfLine[lineIndex] >= 0 ? runs[runOfLine[lineIndex]] : [];
  const inRun = run.flatMap(layerWords);
  if (inRun.length >= MIN_COHORT_WORDS) return inRun;

  const wideFrom = Math.max(0, lineIndex - SAME_LAYER_FALLBACK_LINES);
  const wideTo = Math.min(lines.length - 1, lineIndex + SAME_LAYER_FALLBACK_LINES);
  const layerWindow: CohortWord[] = [];
  for (let i = wideFrom; i <= wideTo; i++) layerWindow.push(...layerWords(i));
  if (layerWindow.length >= MIN_COHORT_WORDS) return layerWindow;

  const from = Math.max(0, lineIndex - SECTION_FALLBACK_LINES);
  const to = Math.min(lines.length - 1, lineIndex + SECTION_FALLBACK_LINES);
  const window: CohortWord[] = [];
  for (let i = from; i <= to; i++) window.push(...lines[i].words.map(toCohortWord));
  if (window.length >= MIN_COHORT_WORDS) return window;
  return lines.flatMap(line => line.words.map(toCohortWord));
}

const MAX_REPAIRS = 500;

/** Two lines are the same lyric when their text and word count match exactly. */
function repeatKey(line: LyricsLine): string {
  if (line.words.length < 2) return '';
  const text = normaliseLine(lineText(line));
  return text ? `${text}|${line.words.length}` : '';
}

/** Comparison form of a line's words, one entry per word so indices still line up. */
function normalisedWords(line: LyricsLine): string[] {
  return line.words.map(word => normaliseWord(word.word || ''));
}

/** A punctuation-only token has no comparison form, so it matches nothing and
 *  counts as part of a variation rather than as a shared word. */
function sameWord(a: string, b: string): boolean {
  return a !== '' && a === b;
}

interface RepeatSpan {
  /** Words shared as a leading run, then as a trailing run. */
  prefix: number;
  suffix: number;
  shared: number;
}

interface RepeatMatch {
  /** Index of the line being copied from. */
  index: number;
  span: RepeatSpan;
}

interface RepeatCandidate {
  index: number;
  words: string[];
  /** A line placed by ear is the copy source a repeat should prefer. */
  checked: boolean;
  /** The section this line is sung in, carried forward from its label. A line
   *  that merely shares a run of words in another section is a different
   *  lyric, not the same passage coming back. */
  section: string;
}

/**
 * How much of one line another line repeats, or null when they are not the same
 * lyric line.
 *
 * A chorus that comes back often comes back with one word changed — a pronoun,
 * a name, "I used to" against "I still" — while the words either side of that
 * change are sung the same way. So the comparison is the leading run plus the
 * trailing run: what is shared is reusable, and what differs is the line's own
 * variation, left to the aligner.
 *
 * Three guards keep this from pairing two lines that merely resemble each
 * other: the shared run needs `REPEAT_MIN_SHARED_WORDS` words and
 * `REPEAT_MIN_SHARED_PCT` of the longer line, and neither line may differ by
 * more than `REPEAT_MAX_VARIATION_WORDS`. A three-word line therefore stays
 * exact-only: two shared words out of three is not enough evidence.
 */
function sharedSpan(a: string[], b: string[]): RepeatSpan | null {
  if (a.length < REPEAT_MIN_SHARED_WORDS || b.length < REPEAT_MIN_SHARED_WORDS) return null;
  if (Math.abs(a.length - b.length) > REPEAT_MAX_VARIATION_WORDS * 2) return null;

  const limit = Math.min(a.length, b.length);
  let prefix = 0;
  while (prefix < limit && sameWord(a[prefix], b[prefix])) prefix++;
  let suffix = 0;
  while (suffix < limit - prefix && sameWord(a[a.length - 1 - suffix], b[b.length - 1 - suffix])) suffix++;

  const shared = prefix + suffix;
  const longer = Math.max(a.length, b.length);
  if (shared < REPEAT_MIN_SHARED_WORDS) return null;
  if (shared * 100 < longer * REPEAT_MIN_SHARED_PCT) return null;
  if (a.length - shared > REPEAT_MAX_VARIATION_WORDS) return null;
  if (b.length - shared > REPEAT_MAX_VARIATION_WORDS) return null;
  return { prefix, suffix, shared };
}

/**
 * The earlier line a repeat should copy from: a checked occurrence first,
 * because a person's timing outranks the machine's, then the line sharing the
 * most words, then the earliest, so the choice is stable across runs.
 *
 * Both lines must be in the same section. Without that, a verse that borrows a
 * chorus's opening words is matched on its prefix and suffix alone, with the
 * unrelated middle ignored — the one case where "the same words" and "the same
 * delivery" come apart. An exact repeat is found by its text before this runs,
 * so a chorus that returns in the outro still copies across the label change.
 */
function bestRepeat(target: RepeatCandidate, earlier: RepeatCandidate[]): RepeatMatch | null {
  let best: RepeatMatch | null = null;
  let bestChecked = false;
  for (const candidate of earlier) {
    if (target.section && candidate.section && target.section !== candidate.section) continue;
    const span = sharedSpan(target.words, candidate.words);
    if (!span) continue;
    const better = !best
      || (candidate.checked && !bestChecked)
      || (candidate.checked === bestChecked && span.shared > best.span.shared);
    if (better) {
      best = { index: candidate.index, span };
      bestChecked = candidate.checked;
    }
  }
  return best;
}

export interface RepeatReuse {
  /** Later occurrences that had an earlier one to copy from. */
  repeatedLines: number;
  /** Words re-placed because the earlier occurrence disagreed with this one. */
  wordsMoved: number;
}

/**
 * Re-place the words of a line that repeats earlier in the track.
 *
 * A chorus comes back with the same syllables in the same order, and usually in
 * the same rhythm: only the line's start has to be found again, and the aligner
 * has already done that. So the earlier occurrence supplies each word's offset
 * from its line start, and a word is touched only when this run put it further
 * than `REPEAT_TOLERANCE_SEC` from where that offset predicts — at that point
 * one of the two occurrences is wrong, and the earlier one has already been
 * listened to more often than the later one has.
 *
 * The prediction is then landed on the nearest energy rise inside the window,
 * so the reused offset is confirmed against the sound rather than trusted as
 * arithmetic. It is evidence, not authority: a hand-checked line is never
 * written over, and an earlier checked occurrence is preferred as the source.
 * Monotonic repair and the cut-back test run afterwards, so a reuse that
 * collides with its neighbour is cleaned up by the same rules as everything
 * else.
 */
export function reuseRepeats(
  lines: LyricsLine[],
  profile: EnergyProfile | null,
  record?: (at: { line: number; word: number }, before: { start: number; end: number }) => void,
): RepeatReuse {
  const stats: RepeatReuse = { repeatedLines: 0, wordsMoved: 0 };
  const first = new Map<string, number>();
  const firstChecked = new Map<string, number>();
  /** Earlier lines in comparison form, so a near repeat can be found by its
   *  shared prefix and suffix. Every line joins it: a line that was copied into
   *  is still an occurrence a later repeat may copy from. */
  const seen: RepeatCandidate[] = [];
  /** The section in effect, carried down from the last labelled line. */
  let currentSection = '';

  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    // Tracked before the short-line skip below: a one-word line can still open
    // a section, and the lines under it belong to that section.
    const label = (line.section || '').trim();
    if (label) currentSection = label;
    const key = repeatKey(line);
    if (!key) continue;
    const words = normalisedWords(line);

    // An exact repeat is the common case and needs no comparison: its words
    // line up one for one. Anything else is matched on what it shares.
    const exact = firstChecked.get(key) ?? first.get(key);
    let match: RepeatMatch | null = exact === undefined
      ? null
      : { index: exact, span: { prefix: line.words.length, suffix: 0, shared: line.words.length } };
    if (!match) {
      match = bestRepeat({ index: li, words, checked: line.checked === true, section: currentSection }, seen);
    }

    // Registered before the copy, so a later occurrence finds the closest
    // earlier one however this line itself was matched.
    if (!first.has(key)) first.set(key, li);
    if (line.checked === true && !firstChecked.has(key)) firstChecked.set(key, li);
    seen.push({ index: li, words, checked: line.checked === true, section: currentSection });

    if (!match) continue;

    const src = lines[match.index];
    const { prefix, suffix } = match.span;
    const count = line.words.length;
    const srcCount = src.words.length;
    stats.repeatedLines++;

    // Only the words both occurrences share are copied. A word one of them
    // does not have is the variation that tells the two lines apart, and it
    // keeps whatever the aligner gave it.
    const shared: Array<[number, number]> = [];
    for (let k = 0; k < prefix; k++) shared.push([k, k]);
    for (let k = 0; k < suffix; k++) shared.push([count - suffix + k, srcCount - suffix + k]);

    // Pace is measured across the shared words only, so a line with an extra
    // word in the middle is not read as a faster delivery of the whole line.
    const [tgtFirst, srcFirst] = shared[0];
    const [tgtLast, srcLast] = shared[shared.length - 1];
    const srcSpan = src.words[srcLast].end - src.words[srcFirst].start;
    const tgtSpan = line.words[tgtLast].end - line.words[tgtFirst].start;
    // A span is only a tempo when its words are singable lengths. A line the
    // aligner swallowed is short because its words are short, so scaling by it
    // sizes the repair to the mistake: one live run moved a word 1.1 s onto a
    // 15%-compressed copy of a 2 s line. Collapsed spans are therefore not a
    // measurement — the earlier occurrence's own pace is used unchanged.
    const tgtDurations = shared
      .map(([wi]) => Math.max(MIN_WORD_SEC, line.words[wi].end - line.words[wi].start))
      .sort((a, b) => a - b);
    const collapsed = tgtDurations.length > 0 && median(tgtDurations) <= REPEAT_COLLAPSED_SEC;
    // The source gets the same test. A swallowed earlier occurrence is not a
    // pace either, and the consequence here is worse: its offsets are only a
    // few hundredths apart, so every later occurrence would be rebuilt bunched
    // at its own first word — the whole line lands early against the sound.
    // Skipped rather than rescaled, because a collapsed span holds no rhythm to
    // rescale; the later occurrence keeps the aligner's own placement.
    const srcDurations = shared
      .map(([, srcWi]) => Math.max(MIN_WORD_SEC, src.words[srcWi].end - src.words[srcWi].start))
      .sort((a, b) => a - b);
    if (srcDurations.length > 0 && median(srcDurations) <= REPEAT_COLLAPSED_SEC) continue;
    // A compressed span is not a pace: the aligner did not give its words
    // syllable-length time, so the ratio is the size of the mistake. Scaling by
    // it at the 0.85 floor still moves the word most of the way to the wrong
    // place — one live run moved it 1.1 s — so the earlier occurrence's pace is
    // used whole instead. A span that runs long is different: that is a slower
    // delivery, and the 1.15 ceiling the decision asked for bounds it. So the
    // clamp stays for the slow side, and only the swallowed side is treated as
    // a measurement that never happened. The repeat is reused either way.
    const ratio = srcSpan > 0 ? tgtSpan / srcSpan : 0;
    const usable = !collapsed && ratio >= REPEAT_SCALE_MIN;
    const scale = usable ? Math.min(REPEAT_SCALE_MAX, ratio) : 1;
    const anchor = line.words[tgtFirst].start;
    const sourceAnchor = src.words[srcFirst].start;

    for (const [wi, srcWi] of shared) {
      const word = line.words[wi];
      // Placed by ear: a checked line is copied from, never written over.
      if (line.checked === true || word.checked === true) continue;

      const sourceWord = src.words[srcWi];
      // A word the aligner was unsure of is not evidence. Copying it would
      // spread one doubtful onset to every later occurrence of the line, so
      // this word is left where this run put it and the statistics passes
      // judge it like any other measurement.
      if (typeof sourceWord.confidence === 'number' && sourceWord.confidence < REPEAT_SOURCE_CONFIDENCE) continue;

      const predicted = anchor + (sourceWord.start - sourceAnchor) * scale;
      if (Math.abs(word.start - predicted) <= REPEAT_TOLERANCE_SEC) continue;

      const duration = Math.max(MIN_WORD_SEC, (sourceWord.end - sourceWord.start) * scale);
      const nextStart = wi + 1 < count ? line.words[wi + 1].start : Infinity;
      const previousEnd = wi > 0 ? line.words[wi - 1].end : 0;
      const onset = profile ? nearestOnset(profile, predicted, previousEnd, nextStart) : null;
      const start = Math.max(previousEnd, onset ?? predicted);
      const end = Math.min(nextStart, Math.max(start + MIN_WORD_SEC, start + duration));
      if (start >= nextStart || Math.abs(start - word.start) < MIN_WORD_SEC) continue;

      // Written before the record: `record` reads the word back for its `to`
      // side, so calling it first reported every repeat as a no-op even though
      // the word had moved.
      const before = { start: word.start, end: word.end };
      word.start = start;
      word.end = end;
      record?.({ line: li, word: wi }, before);
      stats.wordsMoved++;
    }
  }

  return stats;
}

/**
 * Repair an alignment without rewriting it wholesale.
 *
 * A word is only ever changed for one of four reasons:
 *
 *   repeat      a line repeats earlier in the track and this word sits further
 *               than the reuse tolerance from its earlier occurrence's offset;
 *   monotonic   it started before the previous word ended;
 *   cut-back    it is far longer than its section's words AND its tail either
 *               runs into the next onset or is silent. A held note that keeps
 *               sounding is a melisma, not an error, and is left alone;
 *   snap        it is flagged (low confidence, or long after the cut test) and a
 *               voiced onset lies within 120 ms, between its neighbours.
 *
 * A line someone ticked in the editor is a different kind of fact: those words
 * were placed by ear, so they are anchors and are never moved by any of the
 * four. The words around an anchor are what gets pushed instead.
 *
 * Baselines are per section and per vocal layer, widening to the same layer
 * nearby and only then to the layer-blind windows (see `cohortFor`): a
 * harmony's syllables are not the pace the lead is measured against. They use
 * median and MAD rather than a fixed multiplier, so a quiet verse and a loud
 * chorus are judged by their own singing pace.
 */
export interface PolishBounds {
  /** Earliest an unchecked word may start: the end of the line before a
   *  selection, in the same clock as `lines`. */
  startSec?: number;
  /** Latest an unchecked word may end: the start of the line after a
   *  selection, in the same clock as `lines`. */
  endSec?: number;
}

/** Which repairs a statistics pass may make. All default on. */
export interface PolishPasses {
  /** Reuse a line's earlier occurrence for its later repeats. */
  repeats?: boolean;
  /** Shorten a tail that runs into the next onset or over silence. */
  cutBack?: boolean;
  /** Move a flagged onset onto a nearby voiced rise. */
  snap?: boolean;
  /** Push a word that starts before the previous one ends. */
  monotonic?: boolean;
}

export function polishLines(
  lines: LyricsLine[], env: number[] = [], hopSec = HOP_SEC,
  opts: { dryRun?: boolean; bounds?: PolishBounds; passes?: PolishPasses } = {},
): PolishResult {
  const original = lines.map(line => ({ ...line, words: line.words.map(w => ({ ...w })) }));
  const out = lines.map(line => ({ ...line, words: line.words.map(w => ({ ...w })) }));

  const repairs: PolishRepair[] = [];
  let repairsTruncated = false;
  const record = (
    at: { line: number; word: number }, before: { start: number; end: number }, reason: PolishRepair['reason'],
  ) => {
    if (repairs.length >= MAX_REPAIRS) {
      repairsTruncated = true;
      return;
    }
    const after = out[at.line].words[at.word];
    repairs.push({
      line: at.line,
      word: at.word,
      from: before,
      to: { start: after.start, end: after.end },
      reason,
    });
  };

  const profile = energyProfile(env, hopSec);
  const hasEnv = env.length > 0;
  // Walls from outside this run. A selection's neighbours are not in `lines`,
  // so without them the pass can push or snap a word into a line it cannot
  // see. A full track has neither, and stays unbounded.
  const lowerBound = opts.bounds?.startSec ?? 0;
  const upperBound = opts.bounds?.endSec ?? Infinity;
  // A partial object from the editor's panel; every pass defaults on, which is
  // the behaviour the button had before it had settings.
  const passes = {
    repeats: opts.passes?.repeats !== false,
    cutBack: opts.passes?.cutBack !== false,
    snap: opts.passes?.snap !== false,
    monotonic: opts.passes?.monotonic !== false,
  };

  // A line someone ticked is placed by ear, so its words are anchors rather
  // than measurements: nothing below moves them, and the words around them are
  // what gets pushed. That is the weight a hand placement is owed.
  const anchored = new Set<string>();
  out.forEach((line, li) => line.words.forEach((word, wi) => {
    if (line.checked === true || word.checked === true) anchored.add(`${li}:${wi}`);
  }));
  const keyOf = (at: { line: number; word: number }) => `${at.line}:${at.word}`;

  // Repeats first: a chorus that came back is the strongest evidence there is
  // about where its words are, so the band-driven passes clean up after it
  // rather than the other way round.
  const repeats = passes.repeats
    ? reuseRepeats(out, hasEnv ? profile : null, (at, before) => record(at, before, 'repeat'))
    : { repeatedLines: 0, wordsMoved: 0 };

  const order: Array<{ line: number; word: number }> = [];
  // One stream per vocal layer. A harmony can sing over the lead, so "later in
  // the song" is only an ordering inside a layer: each gets its own monotonic
  // cursor rather than one cursor across the flattened track. Untagged words
  // are the lead, so every sidecar written before layers existed behaves
  // exactly as it did.
  const streams = new Map<string, Array<{ line: number; word: number }>>();
  out.forEach((line, li) => line.words.forEach((word, wi) => {
    const at = { line: li, word: wi };
    order.push(at);
    const layer = word.layer ?? line.layer ?? 'lead';
    const stream = streams.get(layer);
    if (stream) stream.push(at);
    else streams.set(layer, [at]);
  }));
  const byStart = (a: { line: number; word: number }, b: { line: number; word: number }) =>
    out[a.line].words[a.word].start - out[b.line].words[b.word].start;
  order.sort(byStart);
  for (const stream of streams.values()) stream.sort(byStart);

  // Baselines first: they must not see the repairs they gate.
  const runs = sectionRuns(out);
  const runOfLine: number[] = new Array(out.length).fill(-1);
  runs.forEach((run, index) => run.forEach(li => { runOfLine[li] = index; }));
  const bandForLine = new Map<number, Baseline>();
  const bandAt = (lineIndex: number): Baseline => {
    const cached = bandForLine.get(lineIndex);
    if (cached) return cached;
    const band = baselineOf(cohortFor(out, lineIndex, runOfLine, runs));
    bandForLine.set(lineIndex, band);
    return band;
  };

  let wordsFlagged = 0;
  let cutBack = 0;
  let snapped = 0;
  const eligible = new Set<string>();
  const cut = new Set<string>();

  // ── Cut-backs, against the aligner's own order ──────────────────
  //
  // This runs BEFORE the monotonic repair. A swallowed tail is exactly the
  // case where the next word's onset is earlier than the current word's end;
  // clamping the next word forward first would hide the collision and leave
  // the bad tail in place.
  for (const stream of streams.values()) {
    for (let i = 0; i < stream.length; i++) {
      const at = stream[i];
      // Placed by ear: not flagged, not cut, not measured.
      if (anchored.has(keyOf(at))) continue;
      const word = out[at.line].words[at.word];
      const nextAt = i + 1 < stream.length ? stream[i + 1] : null;
      const next = nextAt ? out[nextAt.line].words[nextAt.word] : null;
      // The next onset in this layer only: a harmony starting under a held lead
      // note is not a boundary the lead has to end at.
      const limit = next ? next.start : upperBound;

      // A hand placement is a wall, not only a protected word: a tail may not
      // reach into it even when the tail is too short to be flagged. The line
      // after a selection is the same kind of wall. So the tail is removed
      // whatever its length — ending at the wall is the one repair that leaves
      // a checked timestamp untouched, and a short word is visible in the bar
      // while an overlap is not. Where the room exists the word slides back
      // instead of shrinking, so a repair never invents a 10 ms syllable.
      const wall = nextAt ? anchored.has(keyOf(nextAt)) : Number.isFinite(upperBound);
      if (passes.cutBack && wall && Number.isFinite(limit) && word.end > limit) {
        const beforeTail = { start: word.start, end: word.end };
        const duration = Math.max(MIN_WORD_SEC, word.end - word.start);
        const previousEnd = i > 0 ? out[stream[i - 1].line].words[stream[i - 1].word].end : lowerBound;
        if (limit - duration >= previousEnd) {
          // The whole word fits before the wall: keep its length and land it
          // hard against the anchor.
          word.start = limit - duration;
          word.end = limit;
          cutBack++;
          cut.add(keyOf(at));
          record(at, beforeTail, 'cut-back');
        } else if (limit > previousEnd + MIN_WORD_SEC) {
          // Only part of it fits: fill the room that exists and stop at the wall.
          word.start = Math.min(Math.max(previousEnd, word.start), limit - MIN_WORD_SEC);
          word.end = limit;
          cutBack++;
          cut.add(keyOf(at));
          record(at, beforeTail, 'cut-back');
        }
        // Otherwise the word before this one already ends at the wall. Nothing
        // here can be cut without moving the checked timestamp, so the overlap
        // stays — unrepaired and uncounted rather than reported as a fix.
      }

      const band = bandAt(at.line);
      const expected = Math.max(MIN_WORD_SEC, syllableCount(word.word) * band.syllableSec);
      const duration = word.end - word.start;
      const tooLong = duration > Math.max(band.high, expected * 2);
      const lowConfidence = typeof word.confidence === 'number' && word.confidence < SNAP_CONFIDENCE;
      if (!tooLong && !lowConfidence) continue;
      wordsFlagged++;
      eligible.add(keyOf(at));
      if (!tooLong) continue;

      // A tail that collides with the next onset, or that sits over silence, is
      // a swallowed gap. A tail that keeps sounding is a held note: keep it.
      const before = { start: word.start, end: word.end };
      const collision = Number.isFinite(limit) && word.end >= limit - 0.05;
      const expectedEnd = Math.min(limit, word.start + Math.max(expected, band.median));
      const tailVoiced = hasEnv && voicedBetween(profile, Math.max(word.start, expectedEnd), word.end);
      if (!collision && !(hasEnv && !tailVoiced)) continue;
      const target = collision ? limit : expectedEnd;
      if (passes.cutBack && target > word.start + MIN_WORD_SEC && target < word.end) {
        word.end = target;
        cutBack++;
        cut.add(keyOf(at));
        record(at, before, 'cut-back');
      }
    }
  }

  // ── Monotonic repair ────────────────────────────────────────────
  let monotonicRepairs = 0;
  for (const stream of streams.values()) {
    if (!passes.monotonic) continue;
    let cursor = lowerBound;
    for (let i = 0; i < stream.length; i++) {
      const at = stream[i];
      const word = out[at.line].words[at.word];
      // An anchor does not move; the cursor moves past it, so an overlap is
      // resolved by pushing the next word rather than by trimming the one a
      // person placed.
      if (anchored.has(keyOf(at))) {
        cursor = Math.max(cursor, word.end);
        continue;
      }
      const before = { start: word.start, end: word.end };
      const start = Number.isFinite(word.start) && word.start >= 0 ? word.start : cursor;
      if (start < cursor) monotonicRepairs++;
      word.start = Math.max(start, cursor);
      if (Number.isFinite(before.end) && before.end > before.start && word.start > before.start) {
        // Pushed, the word keeps the duration it was aligned with and the whole
        // span moves. The next onset in this layer (or the line after the run)
        // is a ceiling, so the push cannot trade one overlap for another; a word
        // with no room left behind its neighbour still lands on the floor.
        const next = i + 1 < stream.length ? out[stream[i + 1].line].words[stream[i + 1].word] : null;
        const ceiling = next ? next.start : upperBound;
        word.end = Math.max(
          word.start + MIN_WORD_SEC,
          Math.min(ceiling, word.start + (before.end - before.start)),
        );
      } else if (!Number.isFinite(word.end) || word.end < word.start + MIN_WORD_SEC) {
        word.end = word.start + MIN_WORD_SEC;
      }
      if (word.start !== before.start || word.end !== before.end) record(at, before, 'monotonic');
      cursor = word.end;
    }
  }

  // ── Onset snapping ─────────────────────────────────────────────
  // Only a flagged word that was not just cut, and only onto a rise that keeps
  // a plausible duration. A correct onset stays where the aligner put it.
  for (const stream of streams.values()) {
    if (!passes.snap) continue;
    for (let i = 0; i < stream.length; i++) {
      const at = stream[i];
      const key = keyOf(at);
      if (!hasEnv || cut.has(key) || anchored.has(key) || !eligible.has(key)) continue;
      const word = out[at.line].words[at.word];
      const band = bandAt(at.line);
      const next = i + 1 < stream.length ? out[stream[i + 1].line].words[stream[i + 1].word] : null;
      const previousEnd = i > 0 ? out[stream[i - 1].line].words[stream[i - 1].word].end : lowerBound;
      const limit = next ? next.start : upperBound;
      const current = { start: word.start, end: word.end };
      const onset = nearestOnset(profile, word.start, previousEnd, limit);
      if (onset == null || Math.abs(onset - word.start) <= SNAP_HYSTERESIS_SEC) continue;
      const span = Math.max(MIN_WORD_SEC, Math.min(word.end - word.start, band.high));
      const end = Math.max(onset + MIN_WORD_SEC, Math.min(limit, onset + Math.max(span, band.low * 0.5)));
      const newDuration = end - onset;
      if (newDuration < Math.min(band.low * 0.5, 0.1) || newDuration > band.high * 1.5) continue;
      word.start = onset;
      word.end = end;
      snapped++;
      record(at, current, 'snap');
    }
  }

  // ── Report ──────────────────────────────────────────────────────
  let untouched = 0;
  out.forEach((line, li) => line.words.forEach((word, wi) => {
    const before = original[li]?.words?.[wi];
    if (before && word.start === before.start && word.end === before.end) untouched++;
  }));

  const global = baselineOf(order.map(at => toCohortWord(out[at.line].words[at.word])));
  const sections = runs.map(run => {
    const cohort = run.flatMap(li => out[li].words.map(toCohortWord));
    const band = baselineOf(cohort);
    return {
      section: sectionLabel(out[run[0]]),
      words: cohort.length,
      medianSec: band.median,
      madSec: band.mad,
      lowSec: band.low,
      highSec: band.high,
    };
  });

  // The sort below reorders lines, which is the only place a line's index
  // changes. `repairs` was recorded against those indices, so it is remapped
  // with them: a diagnostics file that names the wrong line is worse than one
  // that names none. Ties keep their original order so the map is stable.
  const kept = out
    .map((line, index) => ({ line, index }))
    .filter(({ line }) => line.words.length > 0 || lineText(line))
    .sort((a, b) => a.line.start - b.line.start || a.index - b.index);
  const bounded = kept.map(({ line }) => (line.words.length === 0 ? line : {
    ...line,
    start: line.words[0].start,
    end: line.words[line.words.length - 1].end,
    text: (line.text || '').trim() || line.words.map(w => w.word).join(' '),
  }));
  const positionOf = new Map(kept.map(({ index }, at) => [index, at]));
  for (const repair of repairs) {
    const at = positionOf.get(repair.line);
    if (at !== undefined) repair.line = at;
  }

  return {
    lines: opts.dryRun ? original : bounded,
    stats: {
      words: order.length,
      wordsFlagged,
      cutBack,
      snapped,
      anchored: anchored.size,
      repeatedLines: repeats.repeatedLines,
      repeatWordsMoved: repeats.wordsMoved,
      monotonicRepairs,
      untouched,
      wordsMoved: repeats.wordsMoved + cutBack + snapped,
      medianSec: global.median,
      madSec: global.mad,
      lowSec: global.low,
      highSec: global.high,
      sections,
    },
    repairs,
    repairsTruncated,
  };
}

export interface SeamClamp {
  lines: LyricsLine[];
  /** Unchecked words pushed clear of the line before the selection. */
  pushed: number;
  /** Unchecked words trimmed to stop inside the line after it. */
  trimmed: number;
}

/**
 * Hold a re-aligned selection to the gap the lines either side of it leave.
 *
 * A range run splices its block back between two lines that are kept exactly as
 * the editor left them, so the block is the only thing that can be wrong at a
 * seam — and it was aligned against a crop that never included those
 * neighbours. `polishLines` is told where the walls are, but repeats and the
 * final sort can still carry a word across one, so the splice gets its own
 * geometric check.
 *
 * A push keeps the word's duration and moves the span, and a hand-checked word
 * is not moved by a seam either: the cursor only moves past it.
 */
export function clampToNeighbours(
  block: LyricsLine[], before: LyricsLine | null, after: LyricsLine | null,
): SeamClamp {
  const lines = block.map(line => ({ ...line, words: line.words.map(word => ({ ...word })) }));
  const ceiling = after ? after.start : Infinity;
  let cursor = before ? before.end : 0;
  let pushed = 0;
  let trimmed = 0;

  for (const line of lines) {
    if (line.checked === true) {
      cursor = Math.max(cursor, line.end);
      continue;
    }
    for (const word of line.words) {
      if (word.checked === true) {
        cursor = Math.max(cursor, word.end);
        continue;
      }
      if (word.start < cursor) {
        const duration = Number.isFinite(word.end) && word.end > word.start
          ? word.end - word.start
          : MIN_WORD_SEC;
        word.start = cursor;
        word.end = Math.max(word.start + MIN_WORD_SEC, Math.min(ceiling, word.start + duration));
        pushed++;
      }
      cursor = Math.max(cursor, word.end);
    }
  }

  // The tail: the block may not run into the line after it, and that line is
  // not ours to move. Only the last word can cross the seam, and only when a
  // trim leaves room for the floor above.
  const lastLine = lines[lines.length - 1];
  const lastWord = lastLine && lastLine.checked !== true
    ? lastLine.words[lastLine.words.length - 1]
    : undefined;
  if (lastWord && lastWord.checked !== true && lastWord.end > ceiling
    && ceiling > lastWord.start + MIN_WORD_SEC) {
    lastWord.end = ceiling;
    trimmed++;
  }

  const bounded = lines.map(line => (line.words.length === 0 ? line : {
    ...line,
    start: line.words[0].start,
    end: line.words[line.words.length - 1].end,
  }));

  return { lines: bounded.sort((a, b) => a.start - b.start), pushed, trimmed };
}

/** Shift every word and line by `delta`, dropping anything before the track. */
function shiftLines(lines: LyricsLine[], delta: number): LyricsLine[] {
  if (delta === 0) return lines;
  const move = (t: number) => Math.max(0, t + delta);
  return lines.map(line => ({
    ...line,
    start: move(line.start),
    end: move(line.end),
    words: line.words.map(word => ({ ...word, start: move(word.start), end: move(word.end) })),
  }));
}

// ── The pipeline ────────────────────────────────────────────────────

async function whisperPass(
  audioPath: string, lyrics: string, req: AlignmentRequest,
): Promise<WhisperResult | null> {
  const options: WhisperOptions = {
    model: req.model,
    language: req.language,
    beamSize: req.beamSize,
    dtw: req.dtw !== false,
    ...(req.signal ? { signal: req.signal } : {}),
  };
  return transcribeTrack(audioPath, lyrics, options);
}

/**
 * Run the pipeline over one render, or over one selection of it.
 *
 * A full run returns every line from the aligner. A range run returns the whole
 * spliced line array: the selected block replaced, everything outside it kept
 * exactly as the editor last left it.
 */
export async function runSongAlignment(req: AlignmentRequest): Promise<AlignmentOutcome> {
  const startedAt = Date.now();
  const stages: string[] = [];
  const fallbacks: string[] = [];
  const progress = (stage: string) => {
    if (stages[stages.length - 1] !== stage) stages.push(stage);
    req.onProgress?.(stage);
  };

  const range = req.range;
  let audioPath = req.audioPath;
  let lyrics = req.lyrics;
  let cropPath: string | null = null;
  let cropStartSec = 0;
  let cropEndSec = 0;
  let rangeLines: LyricsLine[] = [];
  let rangeBefore: LyricsLine | null = null;
  let rangeAfter: LyricsLine | null = null;
  let sectionsMatched: SectionCarry['matched'] | null = null;
  let seams: { pushed: number; trimmed: number } | null = null;

  try {
    if (range) {
      if (!req.existing) throw new Error('A selection alignment needs the track’s current timings to update.');
      rangeLines = req.existing.lines.slice(range.fromLine, range.toLine + 1);
      if (rangeLines.length === 0) throw new Error('The selected lines are not in the lyric sidecar anymore.');
      // The lines the crop stops short of: they are kept as they are, so their
      // geometry is the wall the block has to stay inside.
      rangeBefore = req.existing.lines[range.fromLine - 1] ?? null;
      rangeAfter = req.existing.lines[range.toLine + 1] ?? null;
      lyrics = rangeLines.map(lineText).filter(Boolean).join('\n');
      if (!lyrics.trim()) throw new Error('The selected lines have no words to align.');
      progress('cut');
      cropStartSec = Math.max(0, range.startSec - RANGE_PAD_SEC);
      cropEndSec = range.endSec + RANGE_PAD_SEC;
      cropPath = await sliceToWav(req.audioPath, cropStartSec, cropEndSec, { signal: req.signal });
      audioPath = cropPath;
    }

    // A vocal stem improves the answer if one already exists, and separation is
    // a GPU job worth minutes, so the default is to reuse rather than produce:
    // `isolateVocals: true` asks for a fresh separation, `false` forbids a stem.
    // A crop of a few lines is aligned on the mix either way — the stem is a
    // whole-track file and the crop's timings are its own.
    let vocalsIsolated = false;
    if (req.isolateVocals !== false && !range) {
      const cached = vocalStems.find(req.audioPath);
      if (cached) {
        audioPath = cached;
        vocalsIsolated = true;
      } else if (req.isolateVocals === true) {
        progress('stem');
        try {
          audioPath = await vocalStems.vocalStem(req.audioPath);
          vocalsIsolated = true;
        } catch (err: any) {
          fallbacks.push(`vocals: ${err?.message || 'separation failed'}`);
        }
      }
    }

    let whisper: WhisperResult | null = null;
    try {
      progress('whisper');
      whisper = await whisperPass(audioPath, lyrics, req);
    } catch (err: any) {
      fallbacks.push(`whisper: ${err?.message || 'failed'}`);
    }
    if (req.signal?.aborted) throw new Error('Cancelled');
    if (!whisper) fallbacks.push('whisper: no transcription returned');

    // ── CTC forced alignment ────────────────────────────────────────
    const ratio = nonRomanisedRatio(lyrics);
    const alignerModel = resolveYue2AlignerModel();
    let ctcLines: LyricsLine[] | null = null;
    let ctcModel = '';
    if (req.useCtc === false) {
      fallbacks.push('ctc: turned off for this run');
    } else if (!alignerModel) {
      fallbacks.push('ctc: the MMS_FA aligner is not installed');
    } else if (ratio > MAX_NON_ROMANISED) {
      // Not an error: Whisper covers every language, so the run is timed by it
      // and says why, rather than force-fitting words onto symbols that do not
      // exist and returning spans that look confident and are wrong.
      fallbacks.push(
        `ctc: ${Math.round(ratio * 100)}% of the lyric letters are outside the aligner’s romanised alphabet, ` +
        `so Whisper timed this ${range ? 'selection' : 'track'}`,
      );
    } else {
      progress('align');
      try {
        // Spelled the way it is sung, not the way it is written; the map that
        // folds the spans back onto these words travels with it.
        const sung = alignerLyrics(lyrics);
        const aligned = await yue2Align(fs.readFileSync(audioPath), sung.text);
        ctcModel = aligned.model;
        const json = yue2LyricsJson(lyrics, aligned.words, sung);
        ctcLines = json.lines;
        if (ctcLines.length === 0) {
          ctcLines = null;
          fallbacks.push('ctc: aligned no words');
        }
      } catch (err: any) {
        fallbacks.push(`ctc: ${err?.message || 'alignment failed'}`);
      }
    }

    // ── Choose the timing source ────────────────────────────────────
    let lines: LyricsLine[];
    let method: LyricsDiagnostics['method'];
    /** Whisper's reconciled words, kept when CTC wins so the two can be compared. */
    let anchorLines: LyricsLine[] | null = null;
    if (ctcLines) {
      // The exact lyric sequence already bounds the CTC lattice to one
      // monotone pass. Whisper's map is still computed, not as the answer but
      // as the audit of the answer: `anchorDrift` measures where the CTC path
      // disagrees with the words that were heard, and moves nothing.
      lines = ctcLines;
      method = 'mms-fa';
      if (whisper) {
        try {
          progress('reconcile');
          anchorLines = reconcileLyrics(whisper, lyrics, whisper.meta?.model ?? '', vocalsIsolated).lines;
        } catch (err: any) {
          fallbacks.push(`anchors: ${err?.message || 'could not reconcile'}`);
        }
      }
    } else if (whisper) {
      progress('reconcile');
      lines = reconcileLyrics(whisper, lyrics, whisper.meta?.model ?? '', vocalsIsolated).lines;
      method = 'whisper';
    } else {
      throw new Error(
        `No timing source produced a result. ${fallbacks.join('; ') || 'Whisper and the aligner both failed.'}`,
      );
    }

    // Hand-placed lines are put back before anything is repaired, so the pass
    // below treats them as anchors instead of overwriting them.
    const checked = req.existing?.lines?.length
      ? carryChecked(req.existing.lines, lines, cropStartSec)
      : { lines, kept: 0, skipped: 0 };
    lines = checked.lines;

    // The audit runs on the aligner's own output, before the statistics pass
    // moves anything, so the number describes the aligner rather than this file.
    // Only a full track records one: a crop's line indices are its own and its
    // alignment answers a different question, so pooling the two in one report
    // would average two measurements into a number that describes neither.
    const drift = anchorLines && !range ? anchorDrift(lines, anchorLines) : null;

    // ── Statistics ──────────────────────────────────────────────────
    progress('polish');
    const env = fileEnvelope(audioPath);
    // A selection's neighbours sit outside the crop, so the pass is told where
    // they end and begin: a word snapped or cut into a line it cannot see would
    // only be pulled back at the splice.
    const polished = polishLines(lines, env, HOP_SEC, {
      bounds: {
        ...(rangeBefore ? { startSec: Math.max(0, rangeBefore.end - cropStartSec) } : {}),
        ...(rangeAfter ? { endSec: rangeAfter.start - cropStartSec } : {}),
      },
    });
    lines = polished.lines;

    if (range) {
      lines = shiftLines(lines, cropStartSec);
      // Sections are labels, not lyrics, so they are not in the text that was
      // sent; `carrySections` matches them back by text first.
      const carried = carrySections(rangeLines, lines);
      lines = carried.lines;
      sectionsMatched = carried.matched;
      if (req.existing) {
        // The splice is the last place a seam can be crossed, so the block is
        // held to the gap the two kept lines leave for it.
        const seam = clampToNeighbours(lines, rangeBefore, rangeAfter);
        lines = seam.lines;
        if (seam.pushed || seam.trimmed) seams = { pushed: seam.pushed, trimmed: seam.trimmed };
        lines = [
          ...req.existing.lines.slice(0, range.fromLine),
          ...lines,
          ...req.existing.lines.slice(range.toLine + 1),
        ].sort((a, b) => a.start - b.start);
      }
    }

    return {
      lines,
      vocalsIsolated,
      diagnostics: {
        method,
        stages,
        model: ctcModel,
        whisperModel: whisper?.meta?.model ?? '',
        windows: whisper?.meta?.windows ?? 0,
        dtw: whisper?.meta?.dtw ?? false,
        vocalsIsolated,
        words: polished.stats.words,
        untouched: polished.stats.untouched,
        wordsMoved: polished.stats.wordsMoved,
        wordsFlagged: polished.stats.wordsFlagged,
        cutBack: polished.stats.cutBack,
        snapped: polished.stats.snapped,
        anchored: polished.stats.anchored,
        repeatedLines: polished.stats.repeatedLines,
        repeatWordsMoved: polished.stats.repeatWordsMoved,
        checkedKept: checked.kept,
        monotonicRepairs: polished.stats.monotonicRepairs,
        baseline: {
          medianSec: polished.stats.medianSec,
          madSec: polished.stats.madSec,
          lowSec: polished.stats.lowSec,
          highSec: polished.stats.highSec,
        },
        sections: polished.stats.sections,
        repairs: polished.repairs,
        // A crop has no anchors of its own, so it records no audit — but it
        // must not take the track's last whole-track one with it: a selection
        // re-align is an ordinary edit, and the report pools by track.
        ...(drift
          ? { anchorDrift: drift }
          : range && req.existing?.diagnostics?.anchorDrift
            ? { anchorDrift: req.existing.diagnostics.anchorDrift }
            : {}),
        ...(polished.repairsTruncated ? { repairsTruncated: true } : {}),
        fallback: fallbacks.length ? fallbacks.join('; ') : null,
        elapsedMs: Date.now() - startedAt,
        ...(range ? {
          range: {
            fromLine: range.fromLine,
            toLine: range.toLine,
            startSec: range.startSec,
            endSec: range.endSec,
            cropStartSec,
            cropEndSec,
            ...(sectionsMatched ? { sectionsMatched } : {}),
            ...(seams ? { seamFixes: seams } : {}),
          },
        } : {}),
      },
    };
  } finally {
    if (cropPath) removeSlice(cropPath);
  }
}

/** Words are only useful to callers that keep the pipeline's view of them. */
export type { LyricsWord };
