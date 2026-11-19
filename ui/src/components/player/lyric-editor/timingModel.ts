import { byNumber } from '@/utils/sort';
import { isPresent } from '@/utils/guards';
// timingModel.ts — the lyric editor's data and the rules for changing it.
//
// Everything here is a pure function over `EditLine[]`: no React, no DOM, no
// audio engine. The modal renders what these return, the hooks store it, and
// the test beside this file exercises it without a browser. That split is what
// makes the clamps testable at all: "a word cannot overlap its neighbour" is a
// statement about this file, not about a pointer event.

/** Which vocal stream a word or line belongs to. Absent means the lead. */
export type VocalLayer = 'lead' | 'backing' | 'adlib';

export interface EditWord {
  word: string;
  start: number;
  end: number;
  confidence?: number;
  source?: string;
  layer?: VocalLayer;
}

export interface EditLine {
  start: number;
  end: number;
  text: string;
  section?: string;
  /** The person listened to this line and placed it by ear. A fixture holds
   *  only checked lines, so this flag is what makes a fixture ground truth. */
  checked?: boolean;
  /** Untagged is the lead; a backing line may overlap the lead in time. */
  layer?: VocalLayer;
  words: EditWord[];
}

/** A word by position in the editor's line array. */
export interface WordRef {
  line: number;
  word: number;
}

/** One selected word. Kept as its own name because the word actions act on it. */
export type Selection = WordRef;

/**
 * A multi-word selection. `anchor` is where it started, `focus` is the end last
 * touched: a plain click resets both, a shift-click only moves `focus`. The
 * word actions always act on `focus`, so extending a selection never changes
 * which word a nudge would move.
 */
export interface SelectionSpan {
  anchor: WordRef;
  focus: WordRef;
}

export interface ResizeResult {
  lines: EditLine[];
  start: number;
  end: number;
  /** A lengthened word had nowhere to go before the audio ended. */
  noRoom: boolean;
}

export interface MergeResult {
  lines: EditLine[];
  at: number;
}

export const ROW_H = 30;
export const RULER_H = 22;
/** Empty space under the last row, so the final line can be scrolled upwards. */
export const BOTTOM_PAD = 160;
/** Room past the end of the track, so the last seconds can be panned clear of
 *  the right edge instead of the view stopping flush against the final word. */
export const PAN_END_PAD_PX = 160;
/** Waveform height in the editor. Tall enough to place a syllable by eye. */
export const PEAKS_H = 84;
export const MIN_PX = 30;
/** The zoom ceiling. At 1000 px/s a 1200 px window holds 1.2 s, which is about
 *  as far as a selection is still drawable by hand; past that a syllable is
 *  wider than the pointer travel it has to be dragged across. */
export const MAX_PX = 1000;
/** Quiet period after the last change before autosave writes. */
export const AUTOSAVE_MS = 1200;
export const MAX_HISTORY = 60;
/** A pointer that travels less than this is a click, not a pan. */
export const PAN_THRESHOLD_PX = 3;
/** How close to the edge a drag has to be before the view follows it. */
export const EDGE_SCROLL_PX = 48;
export const EDGE_SCROLL_STEP = 12;
/** The shortest word the editor will produce or store. */
export const MIN_WORD_SEC = 0.05;
/** How long a word the editor inserts is, when there is room for it. */
export const NEW_WORD_SEC = 0.25;

/** The nudge and length step, in milliseconds, and the box's allowed range. */
export const STEP_MIN = 1;
export const STEP_MAX = 1000;
export const STEP_DEFAULT = 20;

/** One zoom notch. The buttons and Ctrl+plus and Ctrl+minus share it. */
export const ZOOM_FACTOR = 1.25;
/** Pixels per second when the browser has no remembered zoom. */
export const DEFAULT_PX = 120;

/** Framing the selected word: this many times its length, and never shorter. */
export const FRAME_SPAN_FACTOR = 8;
export const FRAME_MIN_SPAN = 0.3;
/** The narrowest span a zoom will fill, so it cannot divide by zero. */
export const MIN_SPAN_SEC = 0.1;

/** Lead-ins for checking a word by ear, and how far past it a loop may run. */
export const PLAY_LEAD_SEC = 0.25;
export const LOOP_LEAD_SEC = 0.2;
export const LOOP_SLACK_SEC = 0.05;

/** How much of the window sits left of the anchor after a zoom or a jump. */
export const ANCHOR_RATIO = 0.35;
/** Follow moves the window only when the playhead is this near an edge. */
export const FOLLOW_LEAD_PX = 20;
export const FOLLOW_TAIL_PX = 80;
export const FOLLOW_REVEAL_RATIO = 0.3;
/** Space left before a paged-to word, so it is not flush against the edge. */
export const PAGE_PAD_PX = 4;

/** Ruler banding: px/s at which labels tighten, and the spacing each band uses. */
export const RULER_WIDE_PX = 90;
export const RULER_DENSE_PX = 240;
/** Past this zoom a second is wider than a label, so ticks tighten again. */
export const RULER_FINE_PX = 800;
export const RULER_STEP_WIDE = 15;
export const RULER_STEP_MEDIUM = 5;
export const RULER_STEP_DENSE = 1;
export const RULER_STEP_FINE = 0.25;

/**
 * The confidence below which a word is worth checking by ear.
 *
 * Not a correctness rule: it orders the review. A word the recogniser gave a
 * 0.2 is where a minute of listening buys the most, and the queue is what turns
 * "listen to the track" into a list of places to listen.
 */
export const DOUBT_CONFIDENCE = 0.6;

export const round3 = (n: number) => Math.round(n * 1000) / 1000;
export const clampPx = (v: number) => Math.max(MIN_PX, Math.min(MAX_PX, Math.round(v)));

/** Milliseconds to seconds, for the step-driven edits. */
export const msToSec = (ms: number) => ms / 1000;

import { formatClockFine } from '@/utils/format/format';

/** Clamp a typed step into the box's range, falling back to the default. */
export function clampStep(value: number): number {
  if (!Number.isFinite(value)) return STEP_DEFAULT;
  return Math.max(STEP_MIN, Math.min(STEP_MAX, Math.round(value)));
}

/** `m:ss.d` — a tenth is the resolution that matters when placing a syllable. */
export function fmt(sec: number): string {
  return formatClockFine(sec);
}

/**
 * Set or clear one line's section label. The label is a property of the line,
 * not of a word: it marks where a section starts, and an empty value means
 * "no new section here", so the player keeps showing the one above.
 */
export function setLineSection(lines: EditLine[], index: number, section: string): EditLine[] {
  const label = section.trim();
  return lines.map((line, i) => {
    if (i !== index) return line;
    const next = { ...line };
    if (label) next.section = label;
    else delete next.section;
    return next;
  });
}

/**
 * A section as the sheet has it: a label, and the line where it starts.
 *
 * A section is not a row of its own. It is a label on the line where the singing
 * of it begins, and an unlabelled line belongs to the section above it, which is
 * what both the LRC writer and the aligner's cohort logic assume.
 */
export interface SectionRun {
  section: string;
  /** The line index the label sits on, i.e. where this run starts. */
  from: number;
}
/**
 * The track's sections, in order: one entry per run of lines sharing a label.
 *
 * A label repeating after a different one is a new run, so a chorus that comes
 * back after a verse is listed twice — once for each place it starts, which is
 * what a delete button has to act on.
 */
export function sectionRuns(lines: EditLine[]): SectionRun[] {
  const runs: SectionRun[] = [];
  // The label in effect, carried down through unlabelled lines: a line that
  // repeats the label already in force starts nothing, so it is not a second
  // place to delete from.
  let current = '';
  lines.forEach((line, index) => {
    const label = (line.section ?? '').trim();
    if (!label || label === current) return;
    runs.push({ section: label, from: index });
    current = label;
  });
  return runs;
}

/**
 * Start a section at the playhead.
 *
 * If the playhead is past the line's first word, the boundary needs a line of
 * its own, so the line is split there first: the words on either side keep the
 * times they had, because a section is a label and never a timing change. Before
 * the line's first word there is nothing to split, and the line is labelled.
 * Null means there was nothing to do — no line, or no name to give it.
 */
export function addSectionAt(
  lines: EditLine[], li: number, time: number, label: string,
): { lines: EditLine[]; at: number } | null {
  const name = label.trim();
  const line = lines[li];
  if (!line || !name) return null;
  const cut = line.words.findIndex(w => w.start >= time);
  const split = cut > 0 ? splitLineAt(lines, li, cut) : null;
  const at = split ? li + 1 : li;
  return { lines: setLineSection(split ?? lines, at, name), at };
}

/**
 * Delete the section that starts on this line: the label comes off the whole
 * run, and nothing else moves. The lines join the section above, so the words
 * keep their times.
 *
 * The run is what `sectionRuns` calls one: the starting line and every line
 * after it that repeats the same label (unlabelled lines belong to it too),
 * stopping at the next different label. Clearing only the first line left a
 * repeated label a line or two below, which then surfaced as a fresh run — the
 * reason deleting a section with a repeated label took two presses.
 */
export function removeSectionAt(lines: EditLine[], li: number): EditLine[] | null {
  const label = (lines[li]?.section ?? '').trim();
  if (!label) return null;
  const out = lines.slice();
  for (let index = li; index < lines.length; index++) {
    const at = (lines[index].section ?? '').trim();
    if (at && at !== label) break;
    if (at !== label) continue;
    const next = { ...lines[index] };
    delete next.section;
    out[index] = next;
  }
  return out;
}

/**
 * Mark a line as placed by ear, or clear the mark. The save path carries the
 * flag, and the fixture store keeps only the lines that have it: a time nobody
 * checked must not become the reference a later run is scored against.
 */
export function setLineChecked(lines: EditLine[], index: number, checked: boolean): EditLine[] {
  return lines.map((line, i) => {
    if (i !== index) return line;
    const next = { ...line };
    if (checked) next.checked = true;
    else delete next.checked;
    return next;
  });
}

/**
 * Replace one word's text, leaving its times where they are.
 *
 * A person types this because the recogniser heard the wrong word, not because
 * the syllable moved, so `source` keeps describing where the timing came from.
 * An empty value is refused: the save path drops blank words, so clearing one
 * would delete it rather than rename it.
 */
export function setWordText(lines: EditLine[], li: number, wi: number, text: string): EditLine[] {
  const current = lines[li]?.words?.[wi];
  const word = text.replace(/\s+/g, ' ').trim();
  if (!current || !word || word === current.word) return lines;
  return lines.map((line, i) => {
    if (i !== li) return line;
    return withBounds({ ...line, words: line.words.map((w, j) => (j === wi ? { ...w, word } : w)) });
  });
}

/**
 * The selection as a block: its words in reading order, the word before it and
 * the word after it. Null when nothing is selected or the selection is stale.
 */
function selectionBlock(lines: EditLine[], span: SelectionSpan | null) {
  const refs = selectedRefs(lines, span);
  if (refs.length === 0) return null;
  const order = wordOrder(lines);
  const keys = new Set(refs.map(refKeyOf));
  let first = -1;
  let last = -1;
  order.forEach((o, i) => {
    if (!keys.has(refKeyOf(o))) return;
    if (first < 0) first = i;
    last = i;
  });
  const before = first > 0 ? order[first - 1] : null;
  const after = last >= 0 && last < order.length - 1 ? order[last + 1] : null;
  return {
    refs,
    first: lines[refs[0].line].words[refs[0].word],
    last: lines[refs[refs.length - 1].line].words[refs[refs.length - 1].word],
    prevEnd: before ? lines[before.line].words[before.word].end : null,
    nextStart: after ? lines[after.line].words[after.word].start : null,
  };
}

/** Where the selection has to start to sit against the word before it, or null. */
export function selectionPrevEnd(lines: EditLine[], span: SelectionSpan | null): number | null {
  return selectionBlock(lines, span)?.prevEnd ?? null;
}

/** The selection's first and last word, for the commands that act on its ends. */
export function selectionEnds(
  lines: EditLine[], span: SelectionSpan | null,
): { first: WordRef; last: WordRef } | null {
  const refs = selectedRefs(lines, span);
  return refs.length === 0 ? null : { first: refs[0], last: refs[refs.length - 1] };
}

/**
 * Move every selected word by `delta`, keeping their spacing.
 *
 * A command acts on the selection as a block rather than on the word last
 * clicked: nudging eight words that are all early is one gesture, not eight.
 * The block stops when any of its words would collide with a word outside it,
 * so the no-overlap rule still holds across the edges.
 */
export function shiftSelection(lines: EditLine[], span: SelectionSpan | null, delta: number): EditLine[] {
  const block = selectionBlock(lines, span);
  if (!block || !delta) return lines;
  const low = Math.max(0, block.prevEnd ?? 0) - block.first.start;
  const high = block.nextStart == null ? Infinity : block.nextStart - block.last.end;
  const move = Math.min(Math.max(delta, low), high);
  if (Math.abs(move) < 1e-9) return lines;

  const moving = new Set(block.refs.map(refKeyOf));
  return lines.map((line, li) => {
    if (!line.words.some((_w, wi) => moving.has(`${li}:${wi}`))) return line;
    return withBounds({
      ...line,
      words: line.words.map((w, wi) => (moving.has(`${li}:${wi}`)
        ? { ...w, start: round3(w.start + move), end: round3(w.end + move) }
        : w)),
    });
  });
}

/**
 * A drag of a whole selection: `shiftSelection` plus the end of the audio.
 *
 * The block stops at the words outside it like any other command, and it also
 * stops before the last second of the track, because a drag has no neighbour
 * below the final word to hold it back the way a nudge does.
 */
export function dragSelection(
  lines: EditLine[], span: SelectionSpan | null, delta: number, total: number,
): EditLine[] {
  const block = selectionBlock(lines, span);
  if (!block) return lines;
  const room = total > 0 ? total - block.last.end : Infinity;
  return shiftSelection(lines, span, Math.min(delta, room));
}

/**
 * Insert a word after `wi` on line `li`; `wi = -1` inserts before the first one.
 *
 * The new word takes the gap to its neighbour, so nothing already placed is
 * overwritten. When the gap is too short it borrows the shortfall from what
 * follows — the same move Auto shift makes for a lengthened word — and it
 * refuses rather than push anything past the end of the audio.
 */
export function addWordAfter(
  lines: EditLine[], li: number, wi: number, text: string, opts: { total: number },
): { lines: EditLine[]; at: number } | null {
  const line = lines[li];
  const word = text.replace(/\s+/g, ' ').trim();
  if (!line || !word || wi < -1 || wi > line.words.length - 1) return null;

  const prevEnd = wi >= 0
    ? line.words[wi].end
    : lines[li - 1]?.words.at(-1)?.end ?? 0;
  const nextStart = wi + 1 < line.words.length
    ? line.words[wi + 1].start
    : lines[li + 1]?.words[0]?.start ?? Infinity;
  const cap = opts.total > 0 ? opts.total : Infinity;

  const gap = nextStart - prevEnd;
  let span = Math.min(NEW_WORD_SEC, gap);
  let push = 0;
  if (!(span >= MIN_WORD_SEC)) {
    span = MIN_WORD_SEC;
    push = Number.isFinite(gap) ? MIN_WORD_SEC - gap : 0;
  }

  const start = round3(Math.max(0, prevEnd));
  const end = round3(Math.min(start + span, cap));
  if (end - start < MIN_WORD_SEC) return null;
  if (push > 0) {
    const lastEnd = lines.at(-1)?.words.at(-1)?.end ?? 0;
    if (lastEnd + push > cap) return null;
  }

  const at = wi + 1;
  const pushed = push > 0 ? pushWordsAfter(lines, li, wi, push) : lines;
  const next = pushed.map((l, i) => {
    if (i !== li) return l;
    return withBounds({
      ...l,
      words: [...l.words.slice(0, at), { word, start, end }, ...l.words.slice(at)],
    });
  });
  return { lines: next, at };
}

/**
 * Remove every referenced word. A line whose words all went has nothing left to
 * say, so it goes with them rather than staying behind with a stale text.
 */
export function deleteWords(lines: EditLine[], refs: WordRef[]): EditLine[] {
  const drop = new Map<number, Set<number>>();
  for (const ref of refs) {
    const gone = drop.get(ref.line) ?? new Set<number>();
    gone.add(ref.word);
    drop.set(ref.line, gone);
  }
  return lines
    .map((line, li) => {
      const gone = drop.get(li);
      if (!gone || gone.size === 0) return line;
      const words = line.words.filter((_w, wi) => !gone.has(wi));
      return words.length > 0 ? withBounds({ ...line, words }) : { ...line, words, text: '' };
    })
    .filter(l => l.words.length > 0 || l.text.trim().length > 0);
}

/** Line bounds and text are always derived from the words, never stale. */
export function withBounds(line: EditLine): EditLine {
  if (line.words.length === 0) return line;
  return {
    ...line,
    start: line.words[0].start,
    end: line.words[line.words.length - 1].end,
    text: line.words.map(w => w.word).join(' '),
  };
}

/**
 * The nearest legal onset for one word: at or after the previous word's end, and
 * early enough to finish before the next word (or the next line) begins.
 */
export function clampStart(line: EditLine, wi: number, wanted: number, span: number, nextLineStart: number): number {
  const prevEnd = wi > 0 ? line.words[wi - 1].end : 0;
  const nextWordStart = wi + 1 < line.words.length ? line.words[wi + 1].start : Infinity;
  const hardNext = Math.min(nextWordStart, nextLineStart);
  const minStart = Math.max(0, prevEnd);
  const start = Math.max(minStart, wanted);
  if (!Number.isFinite(hardNext)) return start;
  const room = Math.max(MIN_WORD_SEC, hardNext - minStart);
  return Math.min(start, Math.max(minStart, hardNext - Math.min(span, room)));
}

/** A sidecar as it arrives: every field unknown until it is checked. */
interface RawTimingWord {
  word?: unknown;
  start?: unknown;
  end?: unknown;
  confidence?: unknown;
  source?: unknown;
  layer?: unknown;
}

interface RawTimingLine {
  start?: unknown;
  end?: unknown;
  text?: unknown;
  section?: unknown;
  checked?: unknown;
  layer?: unknown;
  words?: unknown;
}

const LAYERS: VocalLayer[] = ['lead', 'backing', 'adlib'];
const asLayer = (value: unknown): VocalLayer | undefined =>
  (LAYERS.includes(value as VocalLayer) ? value as VocalLayer : undefined);

export function cloneLines(raw: unknown): EditLine[] {
  if (!Array.isArray(raw)) return [];
  return (raw as RawTimingLine[])
    .map((l: RawTimingLine) => ({
      start: Number(l?.start) || 0,
      end: Number(l?.end) || 0,
      text: typeof l?.text === 'string' ? l.text : '',
      ...(typeof l?.section === 'string' && l.section ? { section: l.section } : {}),
      ...(l?.checked === true ? { checked: true } : {}),
      ...(asLayer(l?.layer) ? { layer: asLayer(l?.layer) } : {}),
      words: Array.isArray(l?.words)
        ? (l.words as RawTimingWord[])
            .map((w: RawTimingWord) => ({
              word: typeof w?.word === 'string' ? w.word : '',
              start: Number(w?.start) || 0,
              end: Number(w?.end) || 0,
              ...(Number.isFinite(Number(w?.confidence)) ? { confidence: Number(w.confidence) } : {}),
              ...(typeof w?.source === 'string' ? { source: w.source } : {}),
              ...(asLayer(w?.layer) ? { layer: asLayer(w?.layer) } : {}),
            }))
            .filter((w: EditWord) => w.word.trim().length > 0)
        : [],
    }))
    // A line with text but no words is kept: it is a real line the aligner timed
    // but never word-timed, and dropping it here used to erase it on the next
    // save. It shows as a plain label and cannot be selected.
    .filter((l: EditLine) => l.words.length > 0 || l.text.trim().length > 0);
}

/**
 * The lines a time span touches.
 *
 * The waveform strip hands a drag to this: the run works on whole lines, so a
 * dragged span has to become the rows it covers rather than a pair of times the
 * server would quietly round to its own idea of a line. A span that covers no
 * row is null, and nothing gets selected.
 */
export function linesInSpan(
  lines: EditLine[], start: number, end: number,
): { fromLine: number; toLine: number } | null {
  if (!(end > start)) return null;
  let from = -1;
  let to = -1;
  lines.forEach((line, index) => {
    if (!line.words.length) return;
    const lineStart = line.words[0].start;
    const lineEnd = line.words[line.words.length - 1].end;
    if (lineEnd <= start || lineStart >= end) return;
    if (from < 0) from = index;
    to = index;
  });
  return from < 0 ? null : { fromLine: from, toLine: to };
}

/** Index of the last line that has started by `time`, or -1. */
export function lineAtOrBefore(lines: EditLine[], time: number): number {
  let idx = -1;
  for (let i = 0; i < lines.length; i++) if (lines[i].start <= time) idx = i;
  return idx;
}

/**
 * The words worth checking by ear, in reading order: the ones the recogniser
 * was unsure of.
 *
 * A missing or zero confidence is an absence, not a doubt — many whisper models
 * report no probability, and zero is how that arrives. Treating it as doubtful
 * would underline the whole track and make the queue useless, so only a real
 * probability below the threshold counts.
 *
 * Ad-libs are deliberately not here. They are marked by `adlibRefs`, but a
 * word the singer really did add is usually the right word, and a queue that
 * walks them teaches the person to delete true lyrics.
 */
export function doubtfulRefs(lines: EditLine[], threshold = DOUBT_CONFIDENCE): WordRef[] {
  const out: WordRef[] = [];
  lines.forEach((line, l) => line.words.forEach((w, i) => {
    if (typeof w.confidence === 'number' && w.confidence > 0 && w.confidence < threshold) {
      out.push({ line: l, word: i });
    }
  }));
  return out;
}

/** A word shorter than this, at an instant shared with another, is not placed. */
const ROUGH_SPAN_SEC = 0.02;

/**
 * Whether the open timings are too rough to check by ear.
 *
 * After a run the editor's work is listening, so an offer to align again has to
 * be a judgement about the timings and not about how many words doubt happens to
 * report — a track the aligner placed can be full of low confidence and still be
 * the right timing. Three shapes are beyond checking, though:
 *
 *   - no word has a length at all: a lyric sheet with no times on it, which is
 *     what "every line at 0:00" looks like once it reaches the editor;
 *   - most words are stacked on one instant, after a hand edit or a run that
 *     gave up, which is the same piling in a different disguise;
 *   - the words stop long before the track does — inside a fifth of it, on a
 *     track long enough for that to mean something — which is what a run that
 *     only heard the first verse produces.
 *
 * Everything else is left to the doubtful queue. Offering a re-align on a track
 * that is already placed invites throwing good timings away.
 */
export function timingsLookRough(lines: EditLine[], total: number): boolean {
  const words = lines.flatMap(line => line.words);
  if (words.length < 2) return false;
  const timed = words.filter(w => w.end - w.start > ROUGH_SPAN_SEC).length;
  if (timed === 0) return true;
  if (timed < words.length / 2) return true;
  const lastEnd = words.reduce((last, w) => Math.max(last, w.end), 0);
  return total > 60 && lastEnd < total * 0.2;
}

/**
 * Words the recogniser heard but the lyric sheet never asked for.
 *
 * Marked on the timeline so an invented word is visible at a glance, and kept
 * out of the doubt queue for the reason above. Deleting or keeping one is a
 * judgement about the performance, not a repair.
 */
export function adlibRefs(lines: EditLine[]): WordRef[] {
  const out: WordRef[] = [];
  lines.forEach((line, l) => line.words.forEach((w, i) => {
    if (w.source === 'ad-lib') out.push({ line: l, word: i });
  }));
  return out;
}

/** Every word in reading order, as coordinates into `lines`. */
export function wordOrder(lines: EditLine[]): WordRef[] {
  const out: WordRef[] = [];
  lines.forEach((line, l) => line.words.forEach((_w, w) => out.push({ line: l, word: w })));
  return out;
}

function refKeyOf(ref: WordRef): string {
  return `${ref.line}:${ref.word}`;
}

function refIndex(order: WordRef[], ref: WordRef): number {
  return order.findIndex(o => o.line === ref.line && o.word === ref.word);
}

/** The two ends of a selection in reading order, or null when either is gone. */
export function orderSpan(lines: EditLine[], span: SelectionSpan): { from: WordRef; to: WordRef } | null {
  const order = wordOrder(lines);
  const anchor = refIndex(order, span.anchor);
  const focus = refIndex(order, span.focus);
  if (anchor < 0 || focus < 0) return null;
  return anchor <= focus
    ? { from: order[anchor], to: order[focus] }
    : { from: order[focus], to: order[anchor] };
}

/** Every selected word, in reading order. Empty when the selection is stale. */
export function selectedRefs(lines: EditLine[], span: SelectionSpan | null): WordRef[] {
  if (!span) return [];
  const order = wordOrder(lines);
  const anchor = refIndex(order, span.anchor);
  const focus = refIndex(order, span.focus);
  if (anchor < 0 || focus < 0) return [];
  return order.slice(Math.min(anchor, focus), Math.max(anchor, focus) + 1);
}

/** Membership test that is cheap enough for one Set per render. */
export function selectedKeys(lines: EditLine[], span: SelectionSpan | null): Set<string> {
  return new Set(selectedRefs(lines, span).map(refKeyOf));
}

/** The key used by `selectedKeys`. */
export function refKey(ref: WordRef): string {
  return refKeyOf(ref);
}

/** Selecting a line means selecting every word on it. */
export function lineSpan(line: EditLine, lineIndex: number): SelectionSpan | null {
  if (line.words.length === 0) return null;
  return {
    anchor: { line: lineIndex, word: 0 },
    focus: { line: lineIndex, word: line.words.length - 1 },
  };
}

/** How many words are selected. */
export function spanWordCount(lines: EditLine[], span: SelectionSpan | null): number {
  return selectedRefs(lines, span).length;
}

/** The time the selection covers, from its first onset to its last end. */
export function spanTimeRange(
  lines: EditLine[], span: SelectionSpan | null,
): { start: number; end: number } | null {
  const words = selectedRefs(lines, span).map(ref => lines[ref.line]?.words?.[ref.word]).filter(isPresent);
  if (words.length === 0) return null;
  return {
    start: Math.min(...words.map(w => w.start)),
    end: Math.max(...words.map(w => w.end)),
  };
}

/**
 * The lines a selection touches, inclusive. Re-alignment works in whole lines,
 * so a selection that starts mid-line widens to that line rather than leaving
 * half of it on the old timings.
 */
export function spanLineRange(
  lines: EditLine[], span: SelectionSpan | null,
): { fromLine: number; toLine: number } | null {
  const ordered = span ? orderSpan(lines, span) : null;
  if (!ordered) return null;
  return { fromLine: ordered.from.line, toLine: ordered.to.line };
}

function replaceWord(lines: EditLine[], li: number, wi: number, fn: (w: EditWord) => EditWord): EditLine[] {
  return lines.map((line, i) => {
    if (i !== li) return line;
    return { ...line, words: line.words.map((w, j) => (j === wi ? fn(w) : w)) };
  });
}

export function setWordRange(lines: EditLine[], li: number, wi: number, start: number, end: number): EditLine[] {
  return replaceWord(lines, li, wi, w => ({ ...w, start: round3(start), end: round3(end) }));
}

/** Move one word's onset, clamped so it cannot collide with its neighbours. */
export function moveWordTo(lines: EditLine[], li: number, wi: number, wanted: number): EditLine[] {
  const line = lines[li];
  const w = line?.words?.[wi];
  if (!w) return lines;
  const span = Math.max(MIN_WORD_SEC, w.end - w.start);
  const nextLineStart = li + 1 < lines.length ? lines[li + 1]?.words[0]?.start ?? Infinity : Infinity;
  const nextWordStart = wi + 1 < line.words.length ? line.words[wi + 1].start : Infinity;
  const hardNext = Math.min(nextWordStart, nextLineStart);
  const start = clampStart(line, wi, wanted, span, nextLineStart);
  // Both bounds, not just the line's: a long word squeezed into a short gap
  // used to be drawn over its neighbour until the save repaired it.
  const end = Math.min(start + span, Number.isFinite(hardNext) ? Math.max(start + MIN_WORD_SEC, hardNext) : start + span);
  return setWordRange(lines, li, wi, start, end);
}

/** Move every word after this one by `delta` seconds, keeping the order. */
export function pushWordsAfter(lines: EditLine[], li: number, wi: number, delta: number): EditLine[] {
  if (!delta) return lines;
  const order = wordOrder(lines);
  const from = order.findIndex(o => o.line === li && o.word === wi);
  if (from < 0 || from === order.length - 1) return lines;
  const moving = new Set(order.slice(from + 1).map(o => `${o.line}:${o.word}`));
  return lines.map((line, l) => withBounds({
    ...line,
    words: line.words.map((w, w2) => {
      if (!moving.has(`${l}:${w2}`)) return w;
      return { ...w, start: round3(w.start + delta), end: round3(w.end + delta) };
    }),
  }));
}

export interface ResizeOptions {
  autoShift: boolean;
  /** Audio duration, so a resize cannot push words past the end. */
  total: number;
}

/**
 * Resize one edge of a word. The left edge stops at the previous word. The
 * right edge stops at the next word unless Auto shift is on, in which case
 * everything after it moves by the overhang instead. Neither can push a word
 * past the end of the track: there is nothing to play there.
 */
export function resizeWordEdge(
  lines: EditLine[], li: number, wi: number, edge: 'left' | 'right', wanted: number, opts: ResizeOptions,
): ResizeResult | null {
  const line = lines[li];
  const w = line?.words?.[wi];
  if (!w) return null;

  if (edge === 'left') {
    const prevEnd = wi > 0 ? line.words[wi - 1].end : 0;
    const start = round3(Math.min(Math.max(prevEnd, wanted), w.end - MIN_WORD_SEC));
    return { lines: setWordRange(lines, li, wi, start, w.end), start, end: w.end, noRoom: false };
  }

  const nextLineStart = li + 1 < lines.length ? lines[li + 1]?.words[0]?.start ?? Infinity : Infinity;
  const nextWordStart = wi + 1 < line.words.length ? line.words[wi + 1].start : Infinity;
  const bound = Math.min(nextWordStart, nextLineStart);
  let wantedEnd = Math.max(w.start + MIN_WORD_SEC, wanted);
  if (opts.total > 0) wantedEnd = Math.min(wantedEnd, opts.total);

  let end = wantedEnd;
  let overflow = Number.isFinite(bound) && wantedEnd > bound ? wantedEnd - bound : 0;
  let noRoom = false;
  if (overflow > 0) {
    const lastWordEnd = lines[lines.length - 1]?.words.at(-1)?.end ?? 0;
    const room = opts.total > 0 ? Math.max(0, opts.total - lastWordEnd) : Infinity;
    if (!opts.autoShift || overflow > room) {
      // Off, or nowhere left to push to: stop at the neighbour rather than
      // writing words that would never play.
      end = Math.max(w.start + MIN_WORD_SEC, bound);
      noRoom = !!opts.autoShift;
      overflow = 0;
    }
  }
  let next = setWordRange(lines, li, wi, w.start, end);
  if (overflow > 0) next = pushWordsAfter(next, li, wi, overflow);
  return { lines: next, start: w.start, end, noRoom };
}

/** Every word and line by the same amount, for a uniform offset. */
export function shiftEveryWord(lines: EditLine[], delta: number): EditLine[] {
  return lines.map(line => withBounds({
    ...line,
    start: round3(Math.max(0, line.start + delta)),
    end: round3(Math.max(MIN_WORD_SEC, line.end + delta)),
    words: line.words.map(w => ({
      ...w,
      start: round3(Math.max(0, w.start + delta)),
      end: round3(Math.max(MIN_WORD_SEC, w.end + delta)),
    })),
  }));
}

/**
 * Ripple: the selected word and everything after it, by `delta`. The words
 * before it stay put, which is how a drift that starts mid-track gets closed
 * without retyping the part that is already right.
 */
export function shiftFrom(lines: EditLine[], li: number, wi: number, delta: number): EditLine[] {
  const order = wordOrder(lines);
  const from = order.findIndex(o => o.line === li && o.word === wi);
  if (from < 0) return lines;
  const moving = new Set(order.slice(from).map(o => `${o.line}:${o.word}`));
  return lines.map((line, l) => withBounds({
    ...line,
    words: line.words.map((w, w2) => {
      if (!moving.has(`${l}:${w2}`)) return w;
      return {
        ...w,
        start: round3(Math.max(0, w.start + delta)),
        end: round3(Math.max(MIN_WORD_SEC, w.end + delta)),
      };
    }),
  }));
}

/** The visible window, in the units the paging buttons work in. */
export interface PageView {
  scrollLeft: number;
  viewportPx: number;
  pxPerSec: number;
}

/**
 * The time that should land at the left edge when paging, one window per click.
 *
 * Forward moves right so the last word on screen becomes the first. Back moves
 * left so the first word on screen becomes the last, landing on a word start at
 * or before the new left edge so the strip never begins mid-word. Null when
 * there is nothing further to page to, and the button stays disabled.
 */
export function pageTargetTime(lines: EditLine[], dir: -1 | 1, view: PageView): number | null {
  if (view.pxPerSec <= 0 || view.viewportPx <= 0) return null;
  const words = lines.flatMap(l => l.words);
  const start = view.scrollLeft / view.pxPerSec;
  const end = (view.scrollLeft + view.viewportPx) / view.pxPerSec;

  if (dir === 1) {
    let lastVisible: number | null = null;
    for (const w of words) {
      if (w.start >= start - 0.001 && w.start <= end + 0.001) lastVisible = w.start;
    }
    if (lastVisible != null && lastVisible > start + 0.001) return lastVisible;
    // The rightmost word is already at the left edge: step to the next word that
    // is not on screen, or stop when the track runs out.
    const next = words.find(w => w.start > end + 0.001);
    return next ? next.start : null;
  }

  // Back always moves left. It used to return the last visible word's start,
  // which is to the right of the current left edge and scrolled forward instead.
  const left = (view.scrollLeft - view.viewportPx) / view.pxPerSec;
  if (left <= 0) return 0;
  let target = 0;
  for (const w of words) if (w.start <= left + 0.001) target = w.start;
  return target;
}

/** Label spacing at this zoom: dense when a syllable is wide, wide when not. */
function rulerStep(pxPerSec: number): number {
  if (pxPerSec >= RULER_FINE_PX) return RULER_STEP_FINE;
  if (pxPerSec >= RULER_DENSE_PX) return RULER_STEP_DENSE;
  if (pxPerSec >= RULER_WIDE_PX) return RULER_STEP_MEDIUM;
  return RULER_STEP_WIDE;
}

/**
 * Ruler labels for the current zoom: every 15 s while a phrase is a few pixels
 * wide, tightening to every second once a syllable is wide enough to read.
 */
export function rulerTicks(total: number, pxPerSec: number): number[] {
  const step = rulerStep(pxPerSec);
  const out: number[] = [];
  // Indexed rather than accumulated: the fine step is a fraction of a second,
  // and 0.25 + 0.25 + 0.25 ... drifts into 0.9999999999 labels.
  const count = Math.floor(total / step);
  for (let i = 0; i <= count; i += 1) out.push(Math.round(i * step * 1000) / 1000);
  return out;
}

/** Cut a line in two at `cut`, which becomes the first word of the new line. */
export function splitLineAt(lines: EditLine[], li: number, cut: number): EditLine[] | null {
  const line = lines[li];
  if (!line || cut <= 0 || cut >= line.words.length) return null;
  const atSec = round3(line.words[cut].start);
  const left: EditLine = {
    ...line,
    end: atSec,
    words: line.words.slice(0, cut),
    text: line.words.slice(0, cut).map(w => w.word).join(' '),
  };
  const right: EditLine = {
    ...line,
    start: atSec,
    words: line.words.slice(cut),
    text: line.words.slice(cut).map(w => w.word).join(' '),
  };
  return [...lines.slice(0, li), left, right, ...lines.slice(li + 1)];
}

/**
 * Where a split's break goes: before the selection's first word, or before the
 * first word at or after the playhead when nothing is selected. -1 when there
 * is nothing to break at.
 *
 * The first word, not the focus. A line chip selects the whole row and leaves
 * the focus on its last word, so resolving the break from the focus cut that
 * word off into a line of its own — a split that looked like it landed on the
 * wrong row. With a range selection the break belongs where the range starts,
 * because that is the word the person pointed at.
 */
export function splitCutAt(
  lines: EditLine[], li: number, span: SelectionSpan | null, time: number,
): number {
  const line = lines[li];
  if (!line) return -1;
  const ends = selectionEnds(lines, span);
  if (ends) return ends.first.line === li ? ends.first.word : -1;
  return line.words.findIndex(w => w.start >= time);
}

/** One line of the sheet a run produced, as the review receives it. */
export interface ProposedLine {
  start: number;
  end: number;
  text: string;
  section?: string;
  words: Array<{
    word: string;
    start: number;
    end: number;
    confidence?: number;
    layer?: 'lead' | 'backing';
  }>;
}

/**
 * A proposed line as an editable one.
 *
 * The tick is deliberately not carried: `checked` means a person listened to
 * this placement, which is exactly what taking the run's version replaces.
 */
export function proposedToLine(line: ProposedLine): EditLine {
  const words: EditWord[] = (line.words ?? []).map(word => ({
    word: word.word,
    start: round3(word.start),
    end: round3(word.end),
    ...(typeof word.confidence === 'number' ? { confidence: word.confidence } : {}),
    ...(word.layer ? { layer: word.layer } : {}),
  }));
  return withBounds({
    start: 0,
    end: 0,
    text: line.text?.trim() || words.map(word => word.word).join(' '),
    ...(line.section ? { section: line.section } : {}),
    words,
  });
}

/**
 * The sheet a mixed accept writes: the run's row where `take` says so, the one
 * on screen everywhere else.
 *
 * Rows the run produced and the sheet does not have are appended rather than
 * dropped, so a run that found an extra line can be taken without losing it.
 */
export function mergeProposed(
  current: EditLine[], proposed: ProposedLine[], take: boolean[],
): EditLine[] {
  const out = current.map((line, index) => (
    proposed[index] && take[index] !== false ? proposedToLine(proposed[index]) : line
  ));
  for (let index = current.length; index < proposed.length; index++) {
    if (take[index] !== false) out.push(proposedToLine(proposed[index]));
  }
  return out;
}

/**
 * Whether taking this row would write exactly the line already on screen.
 *
 * Compared as the accept would write it (`proposedToLine`), not as the two
 * payloads are shaped: empty line text normalises to its words, and a word's
 * times are rounded to milliseconds. Timings, text, section, layer and
 * confidence all count — a line whose start and text did not move still changed
 * if a word inside it did, and a new doubt mark is a different sheet.
 */
export function sameProposedLine(line: EditLine, proposed: ProposedLine): boolean {
  const fresh = proposedToLine(proposed);
  const text = line.text?.trim() || line.words.map(word => word.word).join(' ');
  return round3(line.start) === fresh.start
    && round3(line.end) === fresh.end
    && text === fresh.text
    && (line.section ?? '') === (fresh.section ?? '')
    && line.words.length === fresh.words.length
    && line.words.every((word, index) => {
      const other = fresh.words[index];
      return word.word === other.word
        && round3(word.start) === other.start
        && round3(word.end) === other.end
        && (word.layer ?? 'lead') === (other.layer ?? 'lead')
        && (word.confidence ?? null) === (other.confidence ?? null);
    });
}

/**
 * The proposed rows that are actually changes.
 *
 * A re-align rebuilds every word in its span, so most of the track's rows come
 * back identical; they are not changes and the review must not ask about them.
 * Only the run's own rows are considered: the accept cannot drop a line the run
 * did not produce, because `mergeProposed` keeps it.
 */
export function changedProposedRows(current: EditLine[], proposed: ProposedLine[]): number[] {
  const changed: number[] = [];
  proposed.forEach((fresh, index) => {
    const old = current[index];
    if (!old || !sameProposedLine(old, fresh)) changed.push(index);
  });
  return changed;
}

/** One slot of a rewritten line, as the diff shows it. */
export interface LineTextChange {
  kind: 'same' | 'renamed' | 'added' | 'removed';
  from?: string;
  to?: string;
  start?: number;
  end?: number;
}

/** What rewriting a line's text would do, before any of it is written. */
export interface LineEditPlan {
  /** The row the plan belongs to: a split or merge between planning and
   *  applying must not write the plan onto the neighbour. */
  line: number;
  /** The line as it would be written. */
  edited: EditLine;
  renamed: number;
  added: number;
  removed: number;
  changes: LineTextChange[];
}

/**
 * Rewrite a whole line's text, keeping the timings words already have.
 *
 * Words are paired by position: the first new word takes the first word's
 * times, and so on. A line the person shortens keeps its last word until the
 * old end, so the gap after the line does not grow; a line they lengthen splits
 * the last word's own interval between it and the names that follow, because
 * inventing time outside the line would push the next line rather than edit this
 * one. A line with no words at all splits its own span instead.
 *
 * Nothing is written here: the plan is what the editor shows before it asks for
 * a yes, and `null` means there is nothing to ask about — empty text or the text
 * the line already has.
 */
export function planLineText(line: EditLine, lineIndex: number, text: string): LineEditPlan | null {
  if (!line) return null;
  const tokens = text.replace(/\s+/g, ' ').trim().split(' ').filter(isPresent);
  if (!tokens.length) return null;
  const old = line.words;
  if (tokens.length === old.length && tokens.every((token, index) => token === old[index].word)) return null;

  const added = Math.max(0, tokens.length - old.length);
  const removed = Math.max(0, old.length - tokens.length);
  const tail = old.length > 0 ? old[old.length - 1] : null;
  const spanStart = tail ? tail.start : line.start;
  const spanEnd = tail ? tail.end : line.end;
  // The tail keeps the first slice, so the new words start where its own
  // syllable ended up rather than before it.
  const slices = tail ? added + 1 : Math.max(1, added);
  const slice = Math.max(MIN_WORD_SEC, (spanEnd - spanStart) / slices);

  const words: EditWord[] = [];
  const changes: LineTextChange[] = [];
  let renamed = 0;

  tokens.forEach((token, index) => {
    if (index < old.length) {
      const prev = old[index];
      const changed = token !== prev.word;
      if (changed) renamed += 1;
      words.push(changed
        // A renamed syllable is not the one the recogniser scored, so its
        // probability no longer describes this word and is dropped.
        ? (() => {
          const renamed = { ...prev, word: token };
          delete renamed.confidence;
          return renamed;
        })()
        : { ...prev, word: token });
      changes.push(changed
        ? { kind: 'renamed', from: prev.word, to: token, start: prev.start, end: prev.end }
        : { kind: 'same', from: token, to: token, start: prev.start, end: prev.end });
      return;
    }
    const start = round3(spanStart + slice * (index - old.length + (tail ? 1 : 0)));
    const end = round3(start + slice);
    words.push({ word: token, start, end });
    changes.push({ kind: 'added', to: token, start, end });
  });

  trimmedTailEnd(words, old, added, spanStart, slice);
  if (removed > 0 && words.length > 0) {
    // The words that went away hand their time to the last one kept, so the line
    // still ends where it did and does not leave a hole for the next line.
    words[words.length - 1] = {
      ...words[words.length - 1],
      end: round3(Math.max(words[words.length - 1].start + MIN_WORD_SEC, old[old.length - 1].end)),
    };
    for (let index = tokens.length; index < old.length; index++) {
      changes.push({ kind: 'removed', from: old[index].word, start: old[index].start, end: old[index].end });
    }
  }

  return {
    line: lineIndex,
    edited: withBounds({ ...line, text: tokens.join(' '), words }),
    renamed,
    added,
    removed,
    changes,
  };
}

/**
 * Give the tail word the first slice of its own interval when new words follow
 * it. `true` when it was long enough to keep at least the duration floor, the
 * one case where the added words do not need to start earlier.
 */
function trimmedTailEnd(
  words: EditWord[], old: EditWord[], added: number, spanStart: number, slice: number,
): boolean {
  if (added <= 0 || old.length === 0 || words.length < old.length) return false;
  const tail = words[old.length - 1];
  const end = round3(spanStart + slice);
  if (end - tail.start < MIN_WORD_SEC) return false;
  words[old.length - 1] = { ...tail, end };
  return true;
}

/**
 * Fold a line into its neighbour. The earlier line keeps its section label,
 * which is the rule; a merge across a chorus boundary relabels the second half.
 */
export function mergeLineAt(lines: EditLine[], idx: number, dir: -1 | 1): MergeResult | null {
  const other = idx + dir;
  if (idx < 0 || idx >= lines.length || other < 0 || other >= lines.length) return null;
  const [earlier, later] = dir === -1 ? [lines[other], lines[idx]] : [lines[idx], lines[other]];
  const merged = withBounds({
    ...earlier,
    words: [...earlier.words, ...later.words].sort(byNumber(x => x.start)),
  });
  const at = Math.min(idx, other);
  return { lines: [...lines.slice(0, at), merged, ...lines.slice(at + 2)], at };
}
