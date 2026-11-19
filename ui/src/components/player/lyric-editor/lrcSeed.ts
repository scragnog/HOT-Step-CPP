// lrcSeed.ts — word chips from a line-level `.lrc`.
//
// A track can have timed lyrics and no word timings: the engine writes a `.lrc`
// with a start for every line, and Whisper never ran on it. The editor used to
// refuse that track, which sent people to transcribe a song whose words they
// already had. This turns each line's start into somewhere to begin — the words
// of the line are spread evenly across the time the `.lrc` gives it, capped so a
// line before a long instrumental does not stretch across it — and the person
// drags them from there.
//
// Nothing here hears the audio. Every word is a guess inside a line that really
// does start where the `.lrc` says, which is why the editor's badge says where
// the chips came from instead of implying a machine placed them.

import { parseAllLrcLines } from '@/utils/lyrics/lrcUtils';
import { MIN_WORD_SEC, round3, withBounds, type EditLine } from './timingModel';
import { isPresent } from '@/utils/guards';

/** Seconds per word a line is normally spread over. */
const WORD_SEC = 0.28;
/** The shortest span a line's words are spread over, however few there are. */
const MIN_LINE_SEC = 0.4;
/** Room given to the last line when the track's length is unknown. */
const TAIL_SEC = 3;

/** `Verse 1 - whispered` from `[Verse 1 - whispered]`, without the style hint. */
function cleanLabel(text: string): string {
  const inner = text.slice(1, -1);
  const dash = inner.indexOf(' - ');
  const base = (dash === -1 ? inner : inner.slice(0, dash)).trim();
  return base.charAt(0).toUpperCase() + base.slice(1);
}

/**
 * The lines an editor opens on when the track has a `.lrc` and no
 * `.lyrics.json`. Empty when the LRC has no timed lyric lines.
 *
 * `total` is the track's length, used only for the last line: its words are
 * spread over what is left of the track rather than past the end of it.
 */
export function seedLinesFromLrc(raw: string, total = 0): EditLine[] {
  if (!raw.trim()) return [];
  const entries = parseAllLrcLines(raw);
  const out: EditLine[] = [];
  /** The label of the last section marker seen, for the next lyric line. */
  let section = '';

  entries.forEach((entry, i) => {
    if (entry.sectionLabel !== null) {
      section = cleanLabel(entry.text);
      return;
    }
    const words = entry.text.split(/\s+/).filter(isPresent);
    if (words.length === 0) return;

    // The line ends where the next timed entry starts. Past the last one it
    // ends at the end of the track, or a few seconds after it when that is not
    // known, so the last line still has room to be placed.
    const nextEntry = entries.slice(i + 1).find(e => e.time > entry.time);
    const next = nextEntry?.time ?? (total > entry.time ? total : entry.time + TAIL_SEC);
    const gap = Math.max(MIN_WORD_SEC, next - entry.time);
    const span = Math.max(MIN_LINE_SEC, Math.min(gap, WORD_SEC * words.length));
    const step = span / words.length;

    out.push(withBounds({
      start: round3(entry.time),
      end: round3(entry.time + span),
      text: entry.text,
      ...(section ? { section } : {}),
      words: words.map((word, w) => ({
        word,
        start: round3(entry.time + step * w),
        end: round3(entry.time + step * (w + 1)),
      })),
    }));
    // A label belongs to the line it starts, not to every line after it.
    section = '';
  });

  return out;
}
