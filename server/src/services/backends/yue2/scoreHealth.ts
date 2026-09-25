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

export function scoreLegibility(voices: { vocal: string[]; ins: string[] }, chords: Set<string>, sections: number, lyrics?: string): Yue2ScoreLegibility {
  const norm = (bars: string[]) => bars.map(b => stripChords(b).replace(/\s+/g, '')).filter(b => hasNote(b));
  const vocal = norm(voices.vocal);
  const ins = norm(voices.ins);
  const pitches = new Set<string>();
  for (const b of vocal) for (const m of b.matchAll(/[_^=]*[A-Ga-g][,']*/g)) pitches.add(m[0]);
  const lyricSections = lyrics === undefined ? undefined : (lyrics.match(/^\s*\[[^\]\n]+\]/gm) ?? []).length;
  const out: Yue2ScoreLegibility = {
    vocalPhraseVariety: round3(phraseVariety(vocal)), insPhraseVariety: round3(phraseVariety(ins)),
    vocalSoundingBars: vocal.length, insSoundingBars: ins.length,
    vocalLoop: longestLoop(vocal), insLoop: longestLoop(ins),
    vocalPitches: pitches.size, chords: chords.size, sections, lyricSections, flags: [],
  };
  const loopFlag = (name: string, loop: { bars: number; period: number }) =>
    `${name} voice repeats a ${loop.period}-bar ${loop.period === 1 ? 'figure' : 'riff'} for ${loop.bars} bars`;
  if (out.insLoop.bars >= LOOP_BARS) out.flags.push(loopFlag('Ins', out.insLoop));
  if (out.vocalLoop.bars >= LOOP_BARS) out.flags.push(loopFlag('Vocal', out.vocalLoop));
  if (vocal.length && pitches.size < FEW_PITCHES) out.flags.push(`Vocal melody on ${pitches.size} pitch(es)`);
  if (chords.size > 0 && chords.size < FEW_CHORDS) out.flags.push(`${chords.size} chord${chords.size === 1 ? '' : 's'} for the whole song`);
  if (lyricSections !== undefined && lyricSections > 1 && sections < lyricSections - 1) {
    out.flags.push(`${sections} section(s) planned for ${lyricSections} lyric section tags`);
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
function barsIn(segment: string): number {
  const s = stripChords(segment).trim();
  const multi = s.match(/^Z(\d+)?$/);
  if (multi) return multi[1] ? Math.max(1, parseInt(multi[1], 10)) : 1;
  return 1;
}

export function classifyYue2Score(abc: string, endReason?: string, lyrics?: string): Yue2ScoreHealth {
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
      inVocal = /vocal/i.test(line.slice(2).split(/\s+/).filter(Boolean)[0] ?? '') || /name="Vocal/i.test(line);
      continue;
    }
    if (/^[A-Za-z]:/.test(line)) continue;  // any other header field (X:, T:, L:, K:...)
    for (const m of line.matchAll(/"([^"]*)"/g)) if (m[1].trim()) chords.add(m[1].trim());
    if (!inVocal) {
      for (const segment of line.split('|')) if (segment.trim()) voices.ins.push(segment);
      continue;
    }

    for (const segment of line.split('|')) {
      if (!segment.trim()) continue;
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
  const legibility = scoreLegibility(voices, chords, sections.length, lyrics);
  const base = { bars, vocalBars, vocalShare: Math.round(vocalShare * 1000) / 1000, longestSilentRun, sections, tempo, meter, estSeconds, legibility };

  if (!bars) return { ...base, verdict: 'unknown', reason: 'no Vocal voice bars found in the score' };

  // The intro>verse loop the card describes: many section markers, almost no
  // distinct names.
  const distinct = new Set(sections).size;
  if (sections.length >= 8 && distinct <= 2) {
    return { ...base, verdict: 'runaway', reason: `${sections.length} section markers but only ${distinct} distinct name(s) — the planner is looping` };
  }
  if (longestSilentRun >= SILENT_RUN_RUNAWAY) {
    return { ...base, verdict: 'runaway', reason: `${longestSilentRun} consecutive bars with no vocal` };
  }
  if (bars >= LONG_SCORE_BARS && vocalShare < THIN_VOCAL_SHARE) {
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
