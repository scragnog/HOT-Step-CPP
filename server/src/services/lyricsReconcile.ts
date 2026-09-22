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
  section?: string;  // e.g. 'Verse 1', 'Chorus', 'Bridge'
}

export interface LyricsJson {
  version: 1;
  method: 'whisper' | 'mms-fa';
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

  // 2. Extract source words, preserving line structure + section markers
  //    sourceLineIdx[i] = which source line word i belongs to
  //    sourceLineSection[lineNum] = section name for that source line
  const allLines = sourceLyrics
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const sourceLines: string[] = [];       // lyrics lines only (no section markers)
  const sourceLineSection: string[] = []; // section name for each source line
  let currentSection = '';

  for (const line of allLines) {
    if (SECTION_MARKER.test(line)) {
      // Extract section name: "[Verse 1]" → "Verse 1"
      currentSection = line.replace(/^\[/, '').replace(/\]$/, '');
    } else {
      sourceLines.push(line);
      sourceLineSection.push(currentSection);
    }
  }

  const sourceWords: string[] = [];
  const sourceLineIdx: number[] = [];  // maps word index → source line number

  for (let lineNum = 0; lineNum < sourceLines.length; lineNum++) {
    const words = sourceLines[lineNum].split(/\s+/).filter((w: string) => w.length > 0);
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
  //    Ad-lib words (hallucinations) are EXCLUDED from output.
  interface MergedWord extends LyricsWord {
    srcLine: number;  // -1 for whisper-only words
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
      // Ad-lib: whisper heard something not in source (could be repeat or hallucination)
      const ww = whisperWords[pair.whisperIdx];
      mergedWords.push({
        word: ww.word,
        start: ww.start,
        end: ww.end,
        confidence: ww.probability ?? 0,
        source: 'ad-lib',
        srcLine: -1,
      });
    }
    // pair.whisperIdx === null → source word with no whisper match → drop
  }

  // 5b. Trim intro/outro hallucinations
  //     Drop everything before the first 'matched' word and after the last.
  //     Mid-song ad-libs (genuine repeats) are preserved.
  const firstMatched = mergedWords.findIndex((w: MergedWord) => w.source === 'matched');
  let lastMatched = -1;
  for (let i = mergedWords.length - 1; i >= 0; i--) {
    if (mergedWords[i].source === 'matched') { lastMatched = i; break; }
  }
  const trimmedWords = firstMatched >= 0
    ? mergedWords.slice(firstMatched, lastMatched + 1)
    : mergedWords;  // no matches at all — keep everything as fallback

  // 6. Group words into lines using source line boundaries
  //    Primary break: when source line index changes
  //    Secondary break: timing gap > threshold, or line too long
  const lines: LyricsLine[] = [];

  if (trimmedWords.length === 0) {
    return { version: 1, method: 'whisper', whisperModel, vocalsIsolated, lines: [] };
  }

  let currentWords: LyricsWord[] = [stripSrcLine(trimmedWords[0])];
  let currentSrcLine = trimmedWords[0].srcLine;

  for (let i = 1; i < trimmedWords.length; i++) {
    const prev = trimmedWords[i - 1];
    const curr = trimmedWords[i];
    const gap = curr.start - prev.end;

    // Break at source line boundary (when the source line changes)
    const srcLineChanged = curr.srcLine !== -1 && currentSrcLine !== -1 && curr.srcLine !== currentSrcLine;
    // Also break on large timing gaps or very long lines
    const timingBreak = gap > LINE_GAP_THRESHOLD_S;
    const lengthBreak = currentWords.length >= LINE_MAX_WORDS;

    if (srcLineChanged || timingBreak || lengthBreak) {
      const section = currentSrcLine >= 0 ? sourceLineSection[currentSrcLine] : undefined;
      lines.push(buildLine(currentWords, section));
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
    const section = currentSrcLine >= 0 ? sourceLineSection[currentSrcLine] : undefined;
    lines.push(buildLine(currentWords, section));
  }

  // 7. Post-process: compress sparse leading words at section transitions
  //    When there's a gap between lines (instrumental break), whisper may place
  //    the first words of the new line during the break. Detect and compress.
  compressSparseLeading(lines);

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

function buildLine(words: LyricsWord[], section?: string): LyricsLine {
  const line: LyricsLine = {
    start: words[0].start,
    end: words[words.length - 1].end,
    text: words.map(w => w.word).join(' '),
    words,
  };
  if (section) line.section = section;
  return line;
}

// ──────────────────────────────────────────────
// Post-process: compress sparse leading words
//
// At section transitions, whisper may place the first few words of a
// new line during the instrumental break before the singing actually
// starts. This creates a "slow start" where the lyrics highlight
// crawls through words during the gap, then catches up.
//
// Detection: for each line that follows a >2s gap, check if the
// leading words are spaced much wider than the rest of the line.
// Fix: push those leading words forward to cluster with the dense
// vocal content.
// ──────────────────────────────────────────────

const GAP_THRESHOLD_S = 2.0;      // min gap between lines to trigger compression
const SPARSE_RATIO = 3.0;         // word gap must be this many times the median to be "sparse"

function compressSparseLeading(lines: LyricsLine[]): void {
  for (let li = 1; li < lines.length; li++) {
    const prevEnd = lines[li - 1].end;
    const line = lines[li];
    const words = line.words;

    // Only process lines after a significant gap
    const gap = words[0].start - prevEnd;
    if (gap < GAP_THRESHOLD_S || words.length < 4) continue;

    // Calculate inter-word gaps
    const gaps: number[] = [];
    for (let i = 1; i < words.length; i++) {
      gaps.push(words[i].start - words[i - 1].end);
    }

    // Find median gap (represents normal singing pace)
    const sorted = [...gaps].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    if (median <= 0) continue;

    // Find where the "dense zone" starts — first word where the gap
    // to the next word is within normal singing pace
    let denseStart = 0;
    for (let i = 0; i < gaps.length; i++) {
      if (gaps[i] <= median * SPARSE_RATIO) {
        denseStart = i;
        break;
      }
    }

    // If no sparse leading words found, skip
    if (denseStart === 0) continue;

    // Compress: push sparse leading words to just before the dense zone
    // Each word gets a small offset before the dense zone start
    const denseStartTime = words[denseStart].start;
    const wordSpacing = Math.min(median, 0.15); // max 150ms between compressed words

    for (let i = denseStart - 1; i >= 0; i--) {
      const offset = (denseStart - i) * wordSpacing;
      const newStart = denseStartTime - offset;
      // Don't push earlier than the previous line's end
      words[i].start = Math.max(newStart, prevEnd + 0.1);
      words[i].end = Math.max(words[i].start + 0.1, words[i].end);
      // Ensure end doesn't exceed next word's start
      if (i < words.length - 1) {
        words[i].end = Math.min(words[i].end, words[i + 1].start);
      }
    }

    // Update line start time
    line.start = words[0].start;
  }
}
