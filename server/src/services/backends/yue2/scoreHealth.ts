// scoreHealth.ts — is a YuE2 lead sheet a song, or a planner that ran away?
//
// A render that hits the frame cap has two very different causes, and only
// one is a fault (becausereasons' CNZN model card, September 2026):
//
//   long     a score that reads as a song but ran to the cap. Since
//            2026-09-24 (f9223e1d) the app treats any cap as broken and
//            redraws it like a runaway: a ten-section lyric at 96 BPM that
//            wants six and a half minutes is the reporter's problem to
//            shorten, not a plan the renderer can finish.
//   runaway  an over-committed planner writing "intro > verse" for hundreds
//            of bars: minutes of groove and no singing. That IS the fault,
//            and it is visible in the ABC before a note is rendered.
//
// The score is the SheetSage2 dialect the planner trains on: `% section`
// comment lines, `V: Vocal` / `V: Ins` voice switches, bars split on `|`,
// chord symbols in double quotes, `Z<n>` for n whole-bar rests. Only the
// Vocal voice is measured — the instrumental line mirrors its bar count and
// says nothing about whether anyone sings.

import fs from 'node:fs';
import path from 'node:path';

/** Every check the judge makes, by name. A style can fail some of them by
 *  nature: a rap album's own lead sheets carry almost no vocal line
 *  (SheetSage transcribes melody, and rap has little), so a faithful adapter
 *  plans long vocal-silent stretches (2026-09-29: all 10 training sheets of a
 *  rap-metal album scored runaway, and its step-200 render raps throughout). */
export type Yue2Check = 'silence' | 'thin-vocal' | 'section-loop' | 'ins-loop' | 'vocal-loop' | 'few-pitches' | 'few-chords' | 'sections';

/** The checks a dataset's own sheets fail, so an adapter trained on it is not
 *  judged for them. Built once per run from the cache's lead sheets. */
export interface Yue2StyleNorms { exempt: Yue2Check[]; sheets: number }

export interface Yue2ScoreHealth {
  verdict: 'healthy' | 'long' | 'runaway' | 'unknown';
  /** Bars in the Vocal voice (whole-bar rests counted). */
  bars: number;
  /** Bars where the Vocal voice carries at least one note. */
  vocalBars: number;
  vocalShare: number;
  /** Longest run of consecutive vocal-silent bars. */
  longestSilentRun: number;
  /** `% section` markers in order, duplicates kept. */
  sections: string[];
  tempo?: number;
  meter?: string;
  /** bars × beats ÷ tempo — what the score wants, cap or no cap. */
  estSeconds?: number;
  reason: string;
  /** Render-free legibility of a plan the verdict passed. */
  legibility: Yue2ScoreLegibility;
  /** Every verdict-level check this sheet fails, exempt or not (for norms). */
  failed: Yue2Check[];
}

// Legibility: what a "healthy" plan can still get wrong. A planner that has
// forgotten how to write a song writes one riff for the whole Ins voice, a
// melody on two pitches, one chord, or fewer sections than the lyric has.
// Each measure is a plain count so the flag can say what it saw.
//
// ponytail: thresholds are PROVISIONAL, set against the 14 lead sheets in
// server/data/audio on 2026-09-25 (one known riff loop, the rest ordinary
// base and adapter plans). They are flags with a reason, not a score, until
// a refine ladder's ear scores have been checked against them.
export interface Yue2ScoreLegibility {
  /** Distinct 4-bar phrases ÷ phrases, over bars that carry a note. 1 = never repeats. */
  vocalPhraseVariety: number;
  insPhraseVariety: number;
  /** Bars carrying a note, per voice — the denominators above. */
  vocalSoundingBars: number;
  insSoundingBars: number;
  /** Longest stretch where a voice repeats itself with a period of 1-4 bars
   *  (bar i equals bar i-p), and that period. A groove repeats for 8-16 bars;
   *  a runaway repeats one riff for the whole sheet. */
  vocalLoop: { bars: number; period: number };
  insLoop: { bars: number; period: number };
  /** Distinct sung pitches (letter + accidental + octave). */
  vocalPitches: number;
  /** Distinct chord symbols in the whole sheet. */
  chords: number;
  /** `% section` markers vs `[Section]` tags in the lyric (undefined without a lyric). */
  sections: number;
  lyricSections?: number;
  /** Worst first. Empty = nothing to say. */
  flags: string[];
  /** Every flag check this sheet fails, exempt or not (for norms). */
  failed: Yue2Check[];
}

const PHRASE_BARS = 4;
const MIN_SOUNDING_FOR_VARIETY = 16;
const MAX_PERIOD = 4;
const LOOP_BARS = 32;          // calibrated below against the 2026-09-25 sidecars
const FEW_PITCHES = 4;         // healthy sidecars: 4-9 pitches, 3-6 chords, loops of 4-19 bars
const FEW_CHORDS = 3;

function phraseVariety(bars: string[]): number {
  if (bars.length < MIN_SOUNDING_FOR_VARIETY) return 1;
  const phrases = new Set<string>();
  const n = bars.length - PHRASE_BARS + 1;
  for (let i = 0; i < n; i++) phrases.add(bars.slice(i, i + PHRASE_BARS).join('|'));
  return phrases.size / n;
}

/** Longest run of bars that repeat with a period of 1..MAX_PERIOD. Sounding
 *  bars only, in order: a two-bar riff played 90 times is 178 bars, period 2. */
function longestLoop(bars: string[]): { bars: number; period: number } {
  let best = { bars: 0, period: 0 };
  for (let p = 1; p <= MAX_PERIOD; p++) {
    let run = 0;
    for (let i = p; i < bars.length; i++) {
      run = bars[i] === bars[i - p] ? run + 1 : 0;
      if (run + p > best.bars) best = { bars: run + p, period: p };
    }
  }
  return best;
}

const LYRIC_SECTION_TAG = /^\s*\[([^\]\n]+)\]/gm;

export function lyricSectionTags(lyrics: string): string[] {
  return [...lyrics.matchAll(LYRIC_SECTION_TAG)].map(match => match[1].trim());
}

export function withoutLyricSectionTags(lyrics: string): string {
  return lyrics.replace(LYRIC_SECTION_TAG, '');
}

export function scoreLegibility(voices: { vocal: string[]; ins: string[] }, chords: Set<string>, sections: number, lyrics?: string, exempt: ReadonlySet<Yue2Check> = new Set()): Yue2ScoreLegibility {
  const norm = (bars: string[]) => bars.map(b => stripChords(b).replace(/\s+/g, '')).filter(b => hasNote(b));
  const vocal = norm(voices.vocal);
  const ins = norm(voices.ins);
  const pitches = new Set<string>();
  for (const b of vocal) for (const m of b.matchAll(/[_^=]*[A-Ga-g][,']*/g)) pitches.add(m[0]);
  const lyricSections = lyrics === undefined ? undefined : lyricSectionTags(lyrics).length;
  const out: Yue2ScoreLegibility = {
    vocalPhraseVariety: round3(phraseVariety(vocal)), insPhraseVariety: round3(phraseVariety(ins)),
    vocalSoundingBars: vocal.length, insSoundingBars: ins.length,
    vocalLoop: longestLoop(vocal), insLoop: longestLoop(ins),
    vocalPitches: pitches.size, chords: chords.size, sections, lyricSections, flags: [], failed: [],
  };
  const flag = (check: Yue2Check, text: string) => { out.failed.push(check); if (!exempt.has(check)) out.flags.push(text); };
  const loopFlag = (name: string, loop: { bars: number; period: number }) =>
    `${name} voice repeats a ${loop.period}-bar ${loop.period === 1 ? 'figure' : 'riff'} for ${loop.bars} bars`;
  if (out.insLoop.bars >= LOOP_BARS) flag('ins-loop', loopFlag('Ins', out.insLoop));
  if (out.vocalLoop.bars >= LOOP_BARS) flag('vocal-loop', loopFlag('Vocal', out.vocalLoop));
  if (vocal.length && pitches.size < FEW_PITCHES) flag('few-pitches', `Vocal melody on ${pitches.size} pitch(es)`);
  if (chords.size > 0 && chords.size < FEW_CHORDS) flag('few-chords', `${chords.size} chord${chords.size === 1 ? '' : 's'} for the whole song`);
  if (lyricSections !== undefined && lyricSections > 1 && sections < lyricSections - 1) {
    flag('sections', `${sections} section(s) planned for ${lyricSections} lyric section tags`);
  }
  return out;
}

const round3 = (n: number) => Math.round(n * 1000) / 1000;

// ponytail: fixed thresholds, tuned by eye on SheetSage sheets of 3-6 minute
// songs (roughly 90-220 bars, 40-75% vocal). Revisit against a labelled set of
// runaway plans once we have a few; until then they only need to separate
// "hundreds of bars of groove" from "a song".
const SILENT_RUN_RUNAWAY = 64;   // 64 bars of no vocal ≈ 1.5-2.5 minutes at song tempos
const LONG_SCORE_BARS    = 96;
const THIN_VOCAL_SHARE   = 0.2;

function stripChords(bar: string): string {
  return bar.replace(/"[^"]*"/g, '');
}

function hasNote(bar: string): boolean {
  // A pitch letter outside chord quotes. Rests are z/Z, which the class
  // below excludes; x is an invisible rest in some dialects and is excluded
  // too.
  return /[A-GA-Ga-g]/.test(stripChords(bar).replace(/[xzXZ]/g, ''));
}

/** Number of bars a segment stands for: `Z4` is four whole-bar rests. */
export function barsIn(segment: string): number {
  const s = stripChords(segment).trim();
  const multi = s.match(/^Z(\d+)?$/);
  if (multi) return multi[1] ? Math.max(1, parseInt(multi[1], 10)) : 1;
  return 1;
}

/** Shared SheetSage voice and bar reading for health checks and section offsets. */
export function isVocalVoice(line: string): boolean {
  return /vocal/i.test(line.slice(2).split(/\s+/).filter(Boolean)[0] ?? '') || /name="Vocal/i.test(line);
}

export function scoreBarSegments(line: string): string[] {
  return line.split('|').filter(segment => segment.trim());
}

export function classifyYue2Score(abc: string, endReason?: string, lyrics?: string, norms?: Yue2StyleNorms | null): Yue2ScoreHealth {
  const exempt = new Set<Yue2Check>(norms?.exempt ?? []);
  const sections: string[] = [];
  let tempo: number | undefined;
  let meter: string | undefined;
  let inVocal = false;
  let bars = 0, vocalBars = 0, longestSilentRun = 0, silentRun = 0;
  const voices = { vocal: [] as string[], ins: [] as string[] };
  const chords = new Set<string>();

  for (const raw of String(abc ?? '').split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('%')) {
      const name = line.slice(1).trim();
      if (name) sections.push(name.toLowerCase());
      continue;
    }
    if (/^Q:/.test(line)) {
      const m = line.match(/=\s*(\d+(?:\.\d+)?)/);
      if (m) tempo = Number(m[1]);
      continue;
    }
    if (/^M:/.test(line)) { meter = line.slice(2).trim(); continue; }
    if (/^V:/.test(line)) {
      inVocal = isVocalVoice(line);
      continue;
    }
    if (/^[A-Za-z]:/.test(line)) continue;  // any other header field (X:, T:, L:, K:...)
    for (const m of line.matchAll(/"([^"]*)"/g)) if (m[1].trim()) chords.add(m[1].trim());
    if (!inVocal) {
      for (const segment of scoreBarSegments(line)) voices.ins.push(segment);
      continue;
    }

    for (const segment of scoreBarSegments(line)) {
      voices.vocal.push(segment);
      const n = barsIn(segment);
      const sings = hasNote(segment);
      bars += n;
      if (sings) {
        vocalBars += 1;  // a multi-bar rest never sings, so n is 1 here
        silentRun = 0;
      } else {
        silentRun += n;
        if (silentRun > longestSilentRun) longestSilentRun = silentRun;
      }
    }
  }

  const vocalShare = bars ? vocalBars / bars : 0;
  let estSeconds: number | undefined;
  if (bars && tempo && tempo > 0) {
    const beats = Number((meter ?? '4/4').split('/')[0]) || 4;
    estSeconds = Math.round(bars * beats * 60 / tempo);
  }
  const legibility = scoreLegibility(voices, chords, sections.length, lyrics, exempt);
  // The intro>verse loop the card describes: many section markers, almost no
  // distinct names.
  const distinct = new Set(sections).size;
  const failed: Yue2Check[] = [];
  if (sections.length >= 8 && distinct <= 2) failed.push('section-loop');
  if (longestSilentRun >= SILENT_RUN_RUNAWAY) failed.push('silence');
  if (bars >= LONG_SCORE_BARS && vocalShare < THIN_VOCAL_SHARE) failed.push('thin-vocal');
  const base = { bars, vocalBars, vocalShare: Math.round(vocalShare * 1000) / 1000, longestSilentRun, sections, tempo, meter, estSeconds, legibility, failed };
  const fails = (check: Yue2Check) => failed.includes(check) && !exempt.has(check);

  if (!bars) return { ...base, verdict: 'unknown', reason: 'no Vocal voice bars found in the score' };
  if (fails('section-loop')) {
    return { ...base, verdict: 'runaway', reason: `${sections.length} section markers but only ${distinct} distinct name(s) — the planner is looping` };
  }
  if (fails('silence')) {
    return { ...base, verdict: 'runaway', reason: `${longestSilentRun} consecutive bars with no vocal` };
  }
  if (fails('thin-vocal')) {
    return { ...base, verdict: 'runaway', reason: `${bars} bars with only ${(vocalShare * 100).toFixed(0)}% carrying vocal` };
  }
  if (endReason === 'limit_hit') {
    return { ...base, verdict: 'long', reason: `healthy score, but ${bars} bars${estSeconds ? ` (~${estSeconds}s)` : ''} is longer than the render cap — shorten the lyric or raise the cap` };
  }
  return { ...base, verdict: 'healthy', reason: `${bars} bars, ${(vocalShare * 100).toFixed(0)}% vocal, ${distinct} distinct section(s)` };
}

/** Whether the auto re-plan keeps a plan (generation and training previews
 *  share it). 'unknown' = no vocal line: for a vocal song that plan renders as
 *  garble (2026-09-24, Oasis step 140), so it is redrawn like a runaway.
 *  'long' = the planner ran to its token cap: Rob's rule (2026-09-24) is that
 *  a cap means broken, so it is redrawn too. */
export function yue2PlanUsable(verdict: string, instrumental: boolean): boolean {
  return verdict === 'healthy' || (verdict === 'unknown' && instrumental);
}

/** Pick the plan to render from a re-plan loop's attempts (2026-09-27, Rob):
 *  the first clean one (usable and no legibility flags); failing that, the
 *  usable one with the fewest flags; failing that, the last. `clean` says
 *  whether the pick was clean, so the caller can warn loudly when not. */
export function yue2PickPlan<T extends { verdict: string; flags?: string[] }>(attempts: T[], instrumental: boolean, checkFlags = true): { pick: T; index: number; clean: boolean } | undefined {
  if (!attempts.length) return undefined;
  const usable = (a: T) => yue2PlanUsable(a.verdict, instrumental);
  const flagCount = (a: T) => (checkFlags ? a.flags?.length ?? 0 : 0);
  const clean = attempts.findIndex(a => usable(a) && flagCount(a) === 0);
  if (clean >= 0) return { pick: attempts[clean], index: clean, clean: true };
  let best = -1;
  attempts.forEach((a, i) => { if (usable(a) && (best < 0 || flagCount(a) < flagCount(attempts[best]))) best = i; });
  const index = best >= 0 ? best : attempts.length - 1;
  return { pick: attempts[index], index, clean: false };
}

/** Which songs draw a plan in one auto-replan pass, as indexes into `budgets`
 *  in draw order: round-robin over the slots, never past a song's own attempt
 *  budget. The first pass draws one per song (a pass lasts as long as its
 *  longest plan, and one clean plan is all a healthy adapter needs); later
 *  passes fill every slot. */
export function yue2PlanDraws(budgets: Array<{ used: number; max: number }>, slots: number, firstRound: boolean): number[] {
  const used = budgets.map(b => b.used);
  const draws: number[] = [];
  for (let added = true; added && draws.length < slots;) {
    added = false;
    for (let i = 0; i < budgets.length && draws.length < slots; i++) {
      if (used[i] >= budgets[i].max) continue;
      draws.push(i);
      used[i]++;
      added = true;
    }
    if (firstRound) break;
  }
  return draws;
}

/** A check is exempt when at least half of the dataset's own sheets fail it
 *  (and there are at least three sheets to judge by). */
export function yue2StyleNorms(sheets: Array<{ abc: string; lyrics?: string }>): Yue2StyleNorms {
  const counts = new Map<Yue2Check, number>();
  for (const s of sheets) {
    const h = classifyYue2Score(s.abc, 'eos', s.lyrics);
    for (const c of new Set([...h.failed, ...h.legibility.failed])) counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  const exempt = sheets.length >= 3 ? [...counts].filter(([, n]) => n * 2 >= sheets.length).map(([c]) => c).sort() : [];
  return { exempt, sheets: sheets.length };
}

const NORMS_FILE = 'style-norms.json';

/** The norms saved in a joint run's folder, or null. */
export function readYue2StyleNorms(runDir: string | undefined): Yue2StyleNorms | null {
  if (!runDir) return null;
  try {
    const j = JSON.parse(fs.readFileSync(path.join(runDir, NORMS_FILE), 'utf8')) as Yue2StyleNorms;
    return Array.isArray(j.exempt) ? j : null;
  } catch { return null; }
}

/** Norms for a joint run: saved in its folder on first use, built from the
 *  lead sheets in the latent cache beside its prepared dataset (the cache can
 *  be cleared later; the run keeps what it was judged by). */
export function yue2StyleNormsForRun(runDir: string, preparedDataset?: string): Yue2StyleNorms | null {
  const saved = readYue2StyleNorms(runDir);
  if (saved || !preparedDataset) return saved;
  try {
    const manifest = path.join(path.dirname(path.dirname(preparedDataset)), 'yue2_preprocess.json');
    const j = JSON.parse(fs.readFileSync(manifest, 'utf8')) as { sources?: Array<{ abc?: unknown; lyrics?: unknown }> };
    const sheets = (j.sources ?? []).filter(s => typeof s.abc === 'string' && s.abc)
      .map(s => ({ abc: s.abc as string, ...(typeof s.lyrics === 'string' ? { lyrics: s.lyrics } : {}) }));
    if (!sheets.length) return null;
    const norms = yue2StyleNorms(sheets);
    fs.mkdirSync(runDir, { recursive: true });
    fs.writeFileSync(path.join(runDir, NORMS_FILE), JSON.stringify(norms, null, 1));
    return norms;
  } catch { return null; }
}
