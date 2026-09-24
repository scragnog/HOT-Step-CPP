// scoreHealth.ts — is a YuE2 lead sheet a song, or a planner that ran away?
//
// A render that hits the frame cap has two very different causes, and only
// one is a fault (becausereasons' CNZN model card, September 2026):
//
//   long     a healthy score that is simply longer than the cap — a
//            ten-section lyric at 96 BPM wants six and a half minutes. Raise
//            the cap or drop a chorus; nothing is broken.
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
}

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

export function classifyYue2Score(abc: string, endReason?: string): Yue2ScoreHealth {
  const sections: string[] = [];
  let tempo: number | undefined;
  let meter: string | undefined;
  let inVocal = false;
  let bars = 0, vocalBars = 0, longestSilentRun = 0, silentRun = 0;

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
    if (!inVocal) continue;

    for (const segment of line.split('|')) {
      if (!segment.trim()) continue;
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
  const base = { bars, vocalBars, vocalShare: Math.round(vocalShare * 1000) / 1000, longestSilentRun, sections, tempo, meter, estSeconds };

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
 *  garble (2026-09-24, Oasis step 140), so it is redrawn like a runaway. */
export function yue2PlanUsable(verdict: string, instrumental: boolean): boolean {
  return verdict === 'healthy' || verdict === 'long' || (verdict === 'unknown' && instrumental);
}
