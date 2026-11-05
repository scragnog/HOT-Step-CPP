// timingModel.ts — the lyric editor's data and the rules for changing it.
//
// Everything here is a pure function over `EditLine[]`: no React, no DOM, no
// audio engine. The modal renders what these return, the hooks store it, and
// the test beside this file exercises it without a browser. That split is what
// makes the clamps testable at all: "a word cannot overlap its neighbour" is a
// statement about this file, not about a pointer event.

export interface EditWord {
  word: string;
  start: number;
  end: number;
  confidence?: number;
  source?: string;
}

export interface EditLine {
  start: number;
  end: number;
  text: string;
  section?: string;
  words: EditWord[];
}

export interface Selection {
  line: number;
  word: number;
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
/** Waveform height in the editor. Tall enough to place a syllable by eye. */
export const PEAKS_H = 84;
export const MIN_PX = 30;
export const MAX_PX = 600;
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
export const RULER_STEP_WIDE = 15;
export const RULER_STEP_MEDIUM = 5;
export const RULER_STEP_DENSE = 1;

/** The editor's view choices, remembered per browser. */
export const FOLLOW_KEY = 'hs-lyricEditorFollow';
export const ZOOM_KEY = 'hs-lyricEditorZoom';
export const AUTOSAVE_KEY = 'hs-lyricEditorAutosave';

export function readFlag(key: string, fallback: boolean): boolean {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : raw === '1';
  } catch { return fallback; }
}

export function readNumber(key: string, fallback: number): number {
  try {
    const raw = Number(localStorage.getItem(key));
    return Number.isFinite(raw) && raw > 0 ? raw : fallback;
  } catch { return fallback; }
}

export const round3 = (n: number) => Math.round(n * 1000) / 1000;
export const clampPx = (v: number) => Math.max(MIN_PX, Math.min(MAX_PX, Math.round(v)));

/** Milliseconds to seconds, for the step-driven edits. */
export const msToSec = (ms: number) => ms / 1000;

/** Clamp a typed step into the box's range, falling back to the default. */
export function clampStep(value: number): number {
  if (!Number.isFinite(value)) return STEP_DEFAULT;
  return Math.max(STEP_MIN, Math.min(STEP_MAX, Math.round(value)));
}

/** `m:ss.d` — a tenth is the resolution that matters when placing a syllable. */
export function fmt(sec: number): string {
  const safe = Number.isFinite(sec) && sec > 0 ? sec : 0;
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  const tenths = Math.floor((safe * 10) % 10);
  return `${m}:${String(s).padStart(2, '0')}.${tenths}`;
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

export function cloneLines(raw: unknown): EditLine[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((l: Partial<EditLine>) => ({
      start: Number(l?.start) || 0,
      end: Number(l?.end) || 0,
      text: typeof l?.text === 'string' ? l.text : '',
      ...(typeof l?.section === 'string' && l.section ? { section: l.section } : {}),
      words: Array.isArray(l?.words)
        ? l.words
            .map((w: Partial<EditWord>) => ({
              word: typeof w?.word === 'string' ? w.word : '',
              start: Number(w?.start) || 0,
              end: Number(w?.end) || 0,
              ...(Number.isFinite(Number(w?.confidence)) ? { confidence: Number(w.confidence) } : {}),
              ...(typeof w?.source === 'string' ? { source: w.source } : {}),
            }))
            .filter((w: EditWord) => w.word.trim().length > 0)
        : [],
    }))
    // A line with text but no words is kept: it is a real line the aligner timed
    // but never word-timed, and dropping it here used to erase it on the next
    // save. It shows as a plain label and cannot be selected.
    .filter((l: EditLine) => l.words.length > 0 || l.text.trim().length > 0);
}

/** Index of the last line that has started by `time`, or -1. */
export function lineAtOrBefore(lines: EditLine[], time: number): number {
  let idx = -1;
  for (let i = 0; i < lines.length; i++) if (lines[i].start <= time) idx = i;
  return idx;
}

/** Every word in reading order, as coordinates into `lines`. */
function wordOrder(lines: EditLine[]): Selection[] {
  const out: Selection[] = [];
  lines.forEach((line, l) => line.words.forEach((_w, w) => out.push({ line: l, word: w })));
  return out;
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
  for (let time = 0; time <= total; time += step) out.push(time);
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
 * Fold a line into its neighbour. The earlier line keeps its section label,
 * which is the rule; a merge across a chorus boundary relabels the second half.
 */
export function mergeLineAt(lines: EditLine[], idx: number, dir: -1 | 1): MergeResult | null {
  const other = idx + dir;
  if (idx < 0 || idx >= lines.length || other < 0 || other >= lines.length) return null;
  const [earlier, later] = dir === -1 ? [lines[other], lines[idx]] : [lines[idx], lines[other]];
  const merged = withBounds({
    ...earlier,
    words: [...earlier.words, ...later.words].sort((a, b) => a.start - b.start),
  });
  const at = Math.min(idx, other);
  return { lines: [...lines.slice(0, at), merged, ...lines.slice(at + 2)], at };
}
