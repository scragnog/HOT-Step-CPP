/**
 * lyricsReconcile.ts — Needleman-Wunsch lyrics reconciliation service
 *
 * Aligns Whisper's free transcription against source lyrics using
 * global sequence alignment with edit-distance, phonetic-hash, and
 * fuzzy scoring. Produces timed, word-level lyrics JSON.
 */

import type { WhisperResult, WhisperWord } from './whisperTranscribe.js';

// ──────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────

export interface LyricsWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  source: 'matched' | 'whisper' | 'ad-lib';
}

export interface LyricsLine {
  start: number;
  end: number;
  text: string;
  words: LyricsWord[];
}

export interface LyricsJson {
  version: 1;
  method: 'whisper';
  whisperModel: string;
  vocalsIsolated: boolean;
  lines: LyricsLine[];
}

// ──────────────────────────────────────────────
// Alignment pair produced by Needleman-Wunsch
// ──────────────────────────────────────────────

interface AlignedPair {
  sourceIdx: number | null;
  whisperIdx: number | null;
  score: number;
}

// ──────────────────────────────────────────────
// Helper: Levenshtein edit distance (standard DP)
// ──────────────────────────────────────────────

export function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;

  // Fast-path: one or both strings empty
  if (m === 0) return n;
  if (n === 0) return m;

  // Single-row DP to save memory
  const prev = new Uint16Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;

  for (let i = 1; i <= m; i++) {
    let diagPrev = prev[0];
    prev[0] = i;
    for (let j = 1; j <= n; j++) {
      const temp = prev[j];
      if (a[i - 1] === b[j - 1]) {
        prev[j] = diagPrev;
      } else {
        prev[j] = 1 + Math.min(diagPrev, prev[j], prev[j - 1]);
      }
      diagPrev = temp;
    }
  }

  return prev[n];
}

// ──────────────────────────────────────────────
// Helper: Phonetic hash
// Strip vowels, collapse consecutive duplicates,
// keep first 6 consonants.
// ──────────────────────────────────────────────

const VOWELS = new Set(['a', 'e', 'i', 'o', 'u']);

export function phoneticHash(word: string): string {
  const lower = word.toLowerCase().replace(/[^a-z]/g, '');
  let result = '';
  let lastChar = '';

  for (const ch of lower) {
    if (VOWELS.has(ch)) continue; // strip vowels
    if (ch === lastChar) continue; // collapse doubles
    result += ch;
    lastChar = ch;
    if (result.length >= 6) break; // first 6 consonants
  }

  return result;
}

// ──────────────────────────────────────────────
// Helper: Match scoring between two words
//   EXACT    → +2
//   PHONETIC → +1.5  (same phoneticHash, word ≥ 3 chars)
//   FUZZY    → +1    (Levenshtein ≤ 2, word ≥ 3 chars)
//   MISMATCH → -1
// ──────────────────────────────────────────────

export function matchScore(a: string, b: string): number {
  const la = a.toLowerCase();
  const lb = b.toLowerCase();

  // Exact match
  if (la === lb) return 2;

  // For short words (< 3 chars), no fuzzy/phonetic — straight mismatch
  if (la.length < 3 && lb.length < 3) return -1;

  // Phonetic match (checked before fuzzy as it's a stronger signal)
  if (la.length >= 3 && lb.length >= 3) {
    const ha = phoneticHash(la);
    const hb = phoneticHash(lb);
    if (ha.length > 0 && ha === hb) return 1.5;
  }

  // Fuzzy match (Levenshtein ≤ 2 for words ≥ 3 chars)
  if (la.length >= 3 || lb.length >= 3) {
    if (levenshtein(la, lb) <= 2) return 1;
  }

  return -1;
}

// ──────────────────────────────────────────────
// Needleman-Wunsch global sequence alignment
// ──────────────────────────────────────────────

const GAP_PENALTY = -1;

export function needlemanWunsch(
  source: string[],
  whisper: string[]
): AlignedPair[] {
  const m = source.length;
  const n = whisper.length;

  // Build score matrix F[m+1][n+1]
  const F: number[][] = [];
  for (let i = 0; i <= m; i++) {
    F[i] = new Array(n + 1);
    F[i][0] = i * GAP_PENALTY;
  }
  for (let j = 0; j <= n; j++) {
    F[0][j] = j * GAP_PENALTY;
  }

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const diag = F[i - 1][j - 1] + matchScore(source[i - 1], whisper[j - 1]);
      const up   = F[i - 1][j] + GAP_PENALTY;     // gap in whisper
      const left = F[i][j - 1] + GAP_PENALTY;      // gap in source
      F[i][j] = Math.max(diag, up, left);
    }
  }

  // Backtrace — build pairs in reverse, then reverse at end
  const pairs: AlignedPair[] = [];
  let i = m;
  let j = n;

  while (i > 0 || j > 0) {
    if (
      i > 0 &&
      j > 0 &&
      F[i][j] === F[i - 1][j - 1] + matchScore(source[i - 1], whisper[j - 1])
    ) {
      // Diagonal — matched/mismatched pair
      pairs.push({
        sourceIdx: i - 1,
        whisperIdx: j - 1,
        score: matchScore(source[i - 1], whisper[j - 1]),
      });
      i--;
      j--;
    } else if (i > 0 && F[i][j] === F[i - 1][j] + GAP_PENALTY) {
      // Up — gap in whisper (source word skipped)
      pairs.push({ sourceIdx: i - 1, whisperIdx: null, score: GAP_PENALTY });
      i--;
    } else {
      // Left — gap in source (whisper-only / ad-lib)
      pairs.push({ sourceIdx: null, whisperIdx: j - 1, score: GAP_PENALTY });
      j--;
    }
  }

  // Return in forward order
  pairs.reverse();
  return pairs;
}

// ──────────────────────────────────────────────
// Section marker regex — [Verse], [Chorus], etc.
// ──────────────────────────────────────────────

const SECTION_MARKER = /^\[.*\]$/;

// ──────────────────────────────────────────────
// Line-splitting thresholds
// ──────────────────────────────────────────────

const LINE_GAP_THRESHOLD_S = 1.5;  // seconds between words to force a new line
const LINE_MAX_WORDS = 15;         // max words per line before forced split

// ──────────────────────────────────────────────
// Main: reconcileLyrics
// ──────────────────────────────────────────────

export function reconcileLyrics(
  whisperResult: WhisperResult,
  sourceLyrics: string,
  whisperModel: string,
  vocalsIsolated: boolean
): LyricsJson {
  // 1. Flatten whisper words from all segments
  const whisperWords: WhisperWord[] = [];
  for (const segment of whisperResult.segments) {
    if (segment.words) {
      for (const w of segment.words) {
        whisperWords.push(w);
      }
    }
  }

  // 2. Extract source words, preserving line structure
  //    sourceLineIdx[i] = which source line word i belongs to
  const sourceLines = sourceLyrics
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0 && !SECTION_MARKER.test(line));

  const sourceWords: string[] = [];
  const sourceLineIdx: number[] = [];  // maps word index → source line number

  for (let lineNum = 0; lineNum < sourceLines.length; lineNum++) {
    const words = sourceLines[lineNum].split(/\s+/).filter(w => w.length > 0);
    for (const w of words) {
      sourceLineIdx.push(lineNum);
      sourceWords.push(w);
    }
  }

  // 3. Build whisper text array for alignment
  const whisperTexts = whisperWords.map(w => w.word);

  // 4. Run Needleman-Wunsch alignment
  const aligned = needlemanWunsch(sourceWords, whisperTexts);

  // 5. Build merged word list, carrying source line index
  interface MergedWord extends LyricsWord {
    srcLine: number;  // -1 for ad-lib/whisper-only words
  }
  const mergedWords: MergedWord[] = [];

  for (const pair of aligned) {
    if (pair.sourceIdx !== null && pair.whisperIdx !== null && pair.score > 0) {
      // Matched: use source spelling + whisper timing
      const ww = whisperWords[pair.whisperIdx];
      mergedWords.push({
        word: sourceWords[pair.sourceIdx],
        start: ww.start,
        end: ww.end,
        confidence: ww.probability ?? 1,
        source: 'matched',
        srcLine: sourceLineIdx[pair.sourceIdx],
      });
    } else if (pair.sourceIdx !== null && pair.whisperIdx !== null && pair.score <= 0) {
      // Mismatched pair: use whisper text + timing
      const ww = whisperWords[pair.whisperIdx];
      mergedWords.push({
        word: ww.word,
        start: ww.start,
        end: ww.end,
        confidence: ww.probability ?? 0,
        source: 'whisper',
        srcLine: sourceLineIdx[pair.sourceIdx],
      });
    } else if (pair.sourceIdx === null && pair.whisperIdx !== null) {
      // Ad-lib: whisper heard something not in source
      const ww = whisperWords[pair.whisperIdx];
      mergedWords.push({
        word: ww.word,
        start: ww.start,
        end: ww.end,
        confidence: ww.probability ?? 0,
        source: 'ad-lib',
        srcLine: -1,  // no source line
      });
    }
    // pair.whisperIdx === null → source word with no whisper match → drop
  }

  // 6. Group words into lines using source line boundaries
  //    Primary break: when source line index changes
  //    Secondary break: timing gap > threshold, or line too long
  const lines: LyricsLine[] = [];

  if (mergedWords.length === 0) {
    return { version: 1, method: 'whisper', whisperModel, vocalsIsolated, lines: [] };
  }

  let currentWords: LyricsWord[] = [stripSrcLine(mergedWords[0])];
  let currentSrcLine = mergedWords[0].srcLine;

  for (let i = 1; i < mergedWords.length; i++) {
    const prev = mergedWords[i - 1];
    const curr = mergedWords[i];
    const gap = curr.start - prev.end;

    // Break at source line boundary (when the source line changes)
    const srcLineChanged = curr.srcLine !== -1 && currentSrcLine !== -1 && curr.srcLine !== currentSrcLine;
    // Also break on large timing gaps or very long lines
    const timingBreak = gap > LINE_GAP_THRESHOLD_S;
    const lengthBreak = currentWords.length >= LINE_MAX_WORDS;

    if (srcLineChanged || timingBreak || lengthBreak) {
      lines.push(buildLine(currentWords));
      currentWords = [stripSrcLine(curr)];
      currentSrcLine = curr.srcLine;
    } else {
      currentWords.push(stripSrcLine(curr));
      // Track source line — ad-lib words (-1) inherit from the current line
      if (curr.srcLine !== -1) {
        currentSrcLine = curr.srcLine;
      }
    }
  }

  // Flush remaining words
  if (currentWords.length > 0) {
    lines.push(buildLine(currentWords));
  }

  return {
    version: 1,
    method: 'whisper',
    whisperModel,
    vocalsIsolated,
    lines,
  };
}

// ──────────────────────────────────────────────
// Strip internal srcLine field before outputting
// ──────────────────────────────────────────────

function stripSrcLine(word: LyricsWord & { srcLine?: number }): LyricsWord {
  const { srcLine, ...rest } = word as any;
  return rest;
}

// ──────────────────────────────────────────────
// Build a LyricsLine from a group of words
// ──────────────────────────────────────────────

function buildLine(words: LyricsWord[]): LyricsLine {
  return {
    start: words[0].start,
    end: words[words.length - 1].end,
    text: words.map(w => w.word).join(' '),
    words,
  };
}
