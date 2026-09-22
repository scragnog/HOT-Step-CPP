// backends/yue2/align.ts — turn POST /yue2/align's word spans into the
// `.lyrics.json` the player already reads.
//
// YuE2's DiT has no lyric-alignment head, so there is no LRC to read out of a
// render the way ACE reads one out of its DiT's cross-attention. What YuE2 has
// instead is the MMS_FA forced aligner it trains its own lyric cursor with
// (engine/src/yue2/yue2-mmsfa.h + yue2-ctc-align.h, exposed as POST
// /yue2/align). It is handed the finished audio and the EXACT lyrics string the
// render was given, and returns one span per sung word.
//
// WHY THIS FILE IS SHORT, and lyricsReconcile.ts is not: Whisper returns what
// it THINKS it heard, so that path has to align two different word sequences
// with Needleman-Wunsch before it can time anything. The aligner returns
// CODEPOINT offsets into our own lyrics string (`yue2_cursor_words_of`), so a
// word's line — and therefore its [Section] — is a lookup, not a guess. There
// is nothing to reconcile and nothing to get wrong.
//
// Codepoints, not UTF-16 code units: the engine counts the way Python does, so
// every offset here is indexed against `Array.from(lyrics)`. A lyric with an
// emoji or an astral character would land one unit off per character on a
// naive `lyrics[i]`, which is exactly the kind of drift nobody notices until
// the karaoke bar is a line behind halfway through.

import type { LyricsJson, LyricsLine, LyricsWord } from '../../lyricsReconcile.js';

/** One sung word, as POST /yue2/align returns it. `char0`/`char1` are a
 *  half-open CODEPOINT span in the lyrics string that was sent. */
export interface Yue2AlignWord {
  start: number;
  end: number;
  score: number;
  char0: number;
  char1: number;
}

/** A line of the lyrics, located in codepoints. Section-marker lines
 *  (`[Chorus]`) are not lines here — they name the ones that follow. */
interface LyricLine {
  c0: number;
  c1: number;
  text: string;
  section?: string;
}

/** Split the lyrics into locatable lines, carrying each one's section label.
 *  The `[tag]` test matches the engine's (yue2_cursor_words_of): a line whose
 *  trimmed form opens with `[` and closes with `]`. */
function lyricLines(chars: string[]): LyricLine[] {
  const out: LyricLine[] = [];
  let section: string | undefined;
  let start = 0;
  for (let i = 0; i <= chars.length; i++) {
    if (i < chars.length && chars[i] !== '\n') continue;
    const text = chars.slice(start, i).join('');
    const trimmed = text.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']') && trimmed.length > 1) {
      section = trimmed.slice(1, -1).trim() || undefined;
    } else if (trimmed) {
      out.push({ c0: start, c1: i, text: trimmed, section });
    }
    start = i + 1;
  }
  return out;
}

/**
 * Build `.lyrics.json` from the aligner's word spans.
 *
 * `lyrics` MUST be byte-for-byte the string that was posted to /yue2/align —
 * the offsets index into it and nothing re-derives them.
 *
 * Words the aligner dropped (a line of punctuation, a word with no alignable
 * character) simply have no span, so a line keeps the words that were timed
 * and takes its own start/end from them. A line with no timed word at all is
 * omitted rather than guessed at: the bar showing nothing is honest, and a
 * fabricated timestamp is not.
 */
export function yue2LyricsJson(lyrics: string, words: Yue2AlignWord[]): LyricsJson {
  const chars = Array.from(lyrics);
  const lines = lyricLines(chars);
  const sorted = [...words].sort((a, b) => a.char0 - b.char0);

  const out: LyricsLine[] = [];
  let w = 0;
  for (const line of lines) {
    // The spans arrive in lyric order, so each line consumes the run that
    // falls inside it — no search, and a stray span outside every line (which
    // should not happen) is skipped rather than attached to a neighbour.
    while (w < sorted.length && sorted[w].char0 < line.c0) w++;
    const picked: LyricsWord[] = [];
    while (w < sorted.length && sorted[w].char0 < line.c1) {
      const s = sorted[w++];
      picked.push({
        word: chars.slice(s.char0, s.char1).join(''),
        start: s.start,
        end: s.end,
        confidence: s.score,
        source: 'matched',
      });
    }
    if (picked.length === 0) continue;
    // Monotonic by construction (the Viterbi path cannot go backwards), but
    // min/max costs nothing and keeps a bad engine response from producing a
    // line the player scrubs to and never leaves.
    const entry: LyricsLine = {
      start: Math.min(...picked.map(p => p.start)),
      end: Math.max(...picked.map(p => p.end)),
      text: line.text,
      words: picked,
    };
    if (line.section) entry.section = line.section;
    out.push(entry);
  }

  return { version: 1, method: 'mms-fa', whisperModel: '', vocalsIsolated: false, lines: out };
}
