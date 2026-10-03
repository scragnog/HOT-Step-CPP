import { instrumental, lyricBlocks, sectionKind } from './coverDrift.js';
import { scoreBarClock, scoreVocalNoteOnsets } from './scoreClock.js';
import { scoreSections } from './scoreSections.js';

/** C6 wire entry: a timed section the engine may hide from the codec stream. */
export interface Yue2LyricScheduleSection {
  start_sec: number;
  lyric: [number, number];
  abc?: [number, number];
}

export interface Yue2LyricScheduleWire {
  mode: 'bias' | 'mask';
  bias?: number;
  abc?: boolean;
  lead_sec?: number;
  behind?: number;
  sections: Yue2LyricScheduleSection[];
}

export interface Yue2LyricScheduleOptions {
  mode: 'bias' | 'mask';
  bias: number;
  abc: boolean;
  leadSec: number;
  behind: number;
}

const codepoints = (text: string) => Array.from(text).length;

/**
 * Tie each sung lyric block to its score section's first Vocal note, on the
 * same clock and with the same in-order label matching as the S13 drift
 * metric (coverDrift.ts). Spans are codepoint offsets into the exact `abc`
 * and `lyrics` strings sent to the engine. Blocks with no timed section are
 * left out, so the engine never hides them; they are reported in `untimed`.
 */
export function buildYue2LyricSchedule(abc: string, lyrics: string, options: Yue2LyricScheduleOptions):
  { wire: Yue2LyricScheduleWire; untimed: string[] } {
  if (lyrics.includes('\r')) throw new Error('lyric schedule needs \\n line endings in the lyrics');
  if (!/^Q:/m.test(abc)) throw new Error('lyric schedule needs a score tempo (Q:); free-tempo covers have no clock');
  const bars = scoreBarClock(abc);
  const onsets = scoreVocalNoteOnsets(abc, bars);
  const sections = scoreSections(abc);
  if (!sections.length) throw new Error('lyric schedule needs a score with labelled sections');
  // Codepoint start of each "% label" line, in the order scoreSections reads them.
  const labelStarts: number[] = [];
  let at = 0;
  for (const line of abc.split('\n')) {
    if (line.trim().startsWith('%') && line.trim().slice(1).trim()) labelStarts.push(at);
    at += codepoints(line) + 1;
  }
  if (labelStarts.length !== sections.length) throw new Error('lyric schedule could not locate the score section lines');
  const abcEnd = codepoints(abc);
  const lyricChars = Array.from(lyrics);
  const tags = lyricBlocks(lyrics);
  let nextTag = 0;
  const used = new Set<number>();
  const wire: Yue2LyricScheduleSection[] = [];
  sections.forEach((section, index) => {
    const match = instrumental(section.label) ? -1 : tags.findIndex((tag, tagIndex) =>
      tagIndex >= nextTag && tag.hasLyrics && !instrumental(tag.label) &&
      sectionKind(tag.label) === sectionKind(section.label));
    if (match < 0) return;
    nextTag = match + 1;
    const endBar = index + 1 < sections.length ? sections[index + 1].startBar - 1 : bars.length;
    const onset = onsets.slice(section.startBar - 1, endBar).find(value => value !== null) ?? null;
    if (onset === null) return;
    used.add(match);
    let end = tags[match].end;
    while (end > tags[match].start && /\s/.test(lyricChars[end - 1])) end--;
    wire.push({
      start_sec: Math.round(onset * 1000) / 1000,
      lyric: [tags[match].start, end],
      ...(options.abc ? { abc: [labelStarts[index], labelStarts[index + 1] ?? abcEnd] as [number, number] } : {}),
    });
  });
  if (!wire.length) throw new Error('lyric schedule found no sung lyric block with a timed score section');
  const untimed = tags.flatMap((tag, index) => tag.hasLyrics && !used.has(index) ? [tag.label] : []);
  return {
    wire: {
      mode: options.mode,
      ...(options.mode === 'bias' ? { bias: options.bias } : {}),
      ...(options.abc ? { abc: true } : {}),
      ...(options.leadSec ? { lead_sec: options.leadSec } : {}),
      ...(options.behind >= 0 ? { behind: options.behind } : {}),
      sections: wire,
    },
    untimed,
  };
}
