// Score sections as the ABC marks them: a whole-line `%Label` comment opens a
// section. The server's scoreSections.ts reads the same lines in the same
// order, so mark i here is score section i there. One known difference: the
// server would also count a `%%` directive line, which nothing in this repo
// writes into a score (a hand-pasted ABC could carry one).

export interface AbcSectionMark { label: string; offset: number }

/** Each section label with the character offset of its comment line, counted
 *  as abcjs counts its startChar: after turning CRLF and CR into LF. */
export function abcSectionMarks(abc: string): AbcSectionMark[] {
  const marks: AbcSectionMark[] = [];
  let offset = 0;
  for (const raw of abc.replace(/\r\n?/g, '\n').split('\n')) {
    const line = raw.trim();
    // %% lines are directives (the preview's own %%MIDI included), not labels.
    if (line.startsWith('%') && !line.startsWith('%%')) {
      const label = line.slice(1).trim();
      if (label) marks.push({ label, offset });
    }
    offset += raw.length + 1;
  }
  return marks;
}

/** Index of the section containing character `char`, or -1 before the first. */
export function sectionAtChar(marks: AbcSectionMark[], char: number): number {
  let index = -1;
  while (index + 1 < marks.length && marks[index + 1].offset <= char) index++;
  return index;
}

// Mid-tone hues that read on the score's white box and on the dark chips.
const PALETTE = ['#2563eb', '#db2777', '#16a34a', '#ea580c', '#7c3aed', '#0891b2', '#ca8a04', '#dc2626'];

/** One colour per section, shared by sections with the same label, so every
 *  chorus is the same colour and a repeat is easy to compare by eye. */
export function sectionColours(labels: string[]): string[] {
  const byLabel = new Map<string, string>();
  return labels.map(label => {
    const key = label.toLowerCase();
    if (!byLabel.has(key)) byLabel.set(key, PALETTE[byLabel.size % PALETTE.length]);
    return byLabel.get(key)!;
  });
}

/** Each `[Tag]` lyric block as a character range, tag line included. Block i
 *  pairs with score section i, as the server's section lint pairs them. */
export function lyricBlockRanges(lyrics: string): Array<{ start: number; end: number }> {
  // The server's LYRIC_SECTION_TAG; a block starts at its `[`, since `^\s*`
  // can swallow the blank lines before it.
  const starts = [...lyrics.matchAll(/^\s*\[([^\]\n]+)\]/gm)].map(match => match.index + match[0].indexOf('['));
  return starts.map((start, i) => ({ start, end: starts[i + 1] ?? lyrics.length }));
}

// ── Lyric side of the section edits in abcSectionMove.ts ──
// Tags are found line by line as the server's section lint finds them. The
// edits below only add or remove tag lines and rewrite tag text, so every
// lyric line survives. They expect lyrics that sectionPairingError passed.

const LYRIC_TAG = /^\s*\[([^\]\n]+)\]/;
const linesOf = (lyrics: string) => lyrics.split(/\r?\n/);
const joinLike = (lyrics: string, lines: string[]) => lines.join(lyrics.includes('\r\n') ? '\r\n' : '\n');
const tagText = (line: string) => LYRIC_TAG.exec(line)![1].trim();
const retag = (line: string, label: string) => line.replace(/\[[^\]\n]+\]/, () => `[${label}]`);

/** The line index of each `[Tag]` line, or an error when a tag shares its
 *  line with words: the lint counts that tag and the planner does not. */
export function lyricTagLines(lines: string[]): number[] | string {
  const out: number[] = [];
  for (const [i, line] of lines.entries()) {
    const match = LYRIC_TAG.exec(line);
    if (!match) continue;
    if (line.slice(match[0].length).trim())
      return `Lyric line ${i + 1} has words after its [${match[1].trim()}] tag; put the tag on a line of its own first.`;
    out.push(i);
  }
  return out;
}

/** Whether any lyric line opens with a `[Tag]`. */
export const hasLyricTags = (lyrics: string) => linesOf(lyrics).some(line => LYRIC_TAG.test(line));

/** Why score section i and lyric block i do not pair one to one, or null
 *  when every label matches its tag, ignoring case, as the lint compares. */
export function sectionPairingError(labels: string[], lyrics: string): string | null {
  const lines = linesOf(lyrics);
  const tags = lyricTagLines(lines);
  if (typeof tags === 'string') return tags;
  // The planner sings words above the first tag as a block of their own.
  if (lines.slice(0, tags[0] ?? lines.length).some(line => line.trim()))
    return 'The lyrics have words above their first tag; tag them or remove them first.';
  if (tags.length !== labels.length)
    return `The score has ${labels.length} sections but the lyrics have ${tags.length} tags; match them first.`;
  const i = labels.findIndex((label, k) => label.trim().toLowerCase() !== tagText(lines[tags[k]]).toLowerCase());
  return i < 0 ? null
    : `Score section ${i + 1} is ${labels[i].trim()} but lyric tag ${i + 1} is ${tagText(lines[tags[i]])}; match them first.`;
}

/** Joins block i+1 into block i under the tag of block i, or of block i+1
 *  when `keepSecond`. */
export function mergeLyricBlocks(lyrics: string, index: number, keepSecond: boolean): string {
  const lines = linesOf(lyrics);
  const tags = lyricTagLines(lines) as number[];
  if (keepSecond) lines[tags[index]] = retag(lines[tags[index]], tagText(lines[tags[index + 1]]));
  lines.splice(tags[index + 1], 1);
  return joinLike(lyrics, lines);
}

/** Block i's tag rewritten to `label`. */
export function relabelLyricBlock(lyrics: string, index: number, label: string): string {
  const lines = linesOf(lyrics);
  const tags = lyricTagLines(lines) as number[];
  lines[tags[index]] = retag(lines[tags[index]], label);
  return joinLike(lyrics, lines);
}

/** Block i's non-blank lines. A split's line boundary runs from 0 (all of
 *  them move to the new block) to their count (none move). */
export function lyricBlockLines(lyrics: string, index: number): string[] {
  const lines = linesOf(lyrics);
  const tags = lyricTagLines(lines) as number[];
  return lines.slice(tags[index] + 1, tags[index + 1] ?? lines.length).filter(line => line.trim());
}

/** Block i split before its `line`-th non-blank line: the lines from there
 *  on go under a new `[label]` tag. Null when the block has no such boundary. */
export function splitLyricBlock(lyrics: string, index: number, line: number, label: string): string | null {
  const lines = linesOf(lyrics);
  const tags = lyricTagLines(lines) as number[];
  const tag = tags[index];
  const words = lines.map((_, i) => i).slice(tag + 1, tags[index + 1] ?? lines.length).filter(i => lines[i].trim());
  if (!Number.isInteger(line) || line < 0 || line > words.length) return null;
  // The new tag sits right above the first line it takes, after a blank
  // line when it follows lyrics; with nothing to take, after the last line.
  const at = line < words.length ? words[line] : (words.at(-1) ?? tag) + 1;
  const blank = line === words.length || (at - 1 > tag && lines[at - 1].trim());
  lines.splice(at, 0, ...(blank ? [''] : []), `[${label}]`);
  return joinLike(lyrics, lines);
}

/** `abc` with abcjs numbering every bar: the directive goes after the X: line, in the header. */
export function withBarNumbers(abc: string): string {
  const lines = abc.split('\n');
  const at = lines.findIndex(line => /^X:/.test(line.trim()));
  return [...lines.slice(0, at + 1), '%%barnumbers 1', ...lines.slice(at + 1)].join('\n');
}
