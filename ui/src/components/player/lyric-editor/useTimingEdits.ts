// useTimingEdits.ts — every operation that changes a word or a line's timing.
//
// These are the editor's verbs: nudge, start at the playhead, snap to the
// previous word, lengthen, shift the whole track, ripple from the selection,
// split a line, merge two. Each one records its own undo entry, applies a
// `timingModel` transform, and says so when the request cannot be honoured (a
// merge with no neighbour, a split on the first word) instead of failing
// silently. That combination is the business logic, so it lives here rather than
// in the component that draws the buttons.
//
// The return is `{ data, ...handlers }` like the editor's other hooks. `data`
// carries `targetLine`, the line the line actions apply to, derived once here
// rather than in the modal and again in each button.

import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import {
  addSectionAt, addWordAfter, deleteWords, lineAtOrBefore, mergeLineAt, msToSec, removeSectionAt,
  resizeWordEdge, selectedRefs, selectionEnds, selectionPrevEnd, setLineSection, setWordText, shiftEveryWord,
  shiftFrom, shiftSelection, splitCutAt, splitLineAt,
  type EditLine, type LineEditPlan, type Selection, type SelectionSpan, type WordRef,
} from './timingModel';

type LinesUpdater = EditLine[] | ((prev: EditLine[]) => EditLine[]);

export interface TimingEditOptions {
  lines: EditLine[];
  setLines: (updater: LinesUpdater) => void;
  /** One undo entry per change. */
  record: () => void;
  selected: Selection | null;
  /** The whole selection: Add and Delete act on all of it, not just the focus. */
  selection: SelectionSpan | null;
  onSelect: (selection: Selection | null) => void;
  /** The nudge and length step, in milliseconds. */
  step: number;
  currentTime: number;
  /** Whether lengthening a word pushes the words after it along. */
  autoShift: boolean;
  /** Audio duration, so a resize cannot push words past the end. */
  total: number;
}

export function useTimingEdits({
  lines, setLines, record, selected, selection, onSelect, step, currentTime, autoShift, total,
}: TimingEditOptions) {
  const { t } = useTranslation();

  /** The line the line-actions apply to: the selected one, else the playhead's. */
  const targetLine = selected ? selected.line : lineAtOrBefore(lines, currentTime);

  // Commands act on the selection, not on the word last clicked: its two ends
  // are what a block move, a snap and a resize need.
  const ends = selectionEnds(lines, selection);
  const wordAt = (ref: WordRef | null | undefined) =>
    (ref ? lines[ref.line]?.words?.[ref.word] ?? null : null);

  /** Move the selection by the step, clamped by the words around it. */
  const wordStep = (dir: 1 | -1) => {
    if (!ends) return;
    const next = shiftSelection(lines, selection, dir * msToSec(step));
    if (next === lines) return;
    record();
    setLines(next);
  };

  /** Rename the selected word. A word the recogniser misheard keeps its times. */
  const renameWord = (text: string) => {
    if (!selected) return;
    const next = setWordText(lines, selected.line, selected.word, text);
    if (next === lines) return;
    record();
    setLines(next);
  };

  /**
   * Add a word after the selection, or after the word the playhead is on when
   * nothing is selected. It is selected and its text is a placeholder, so the
   * caller can put the cursor in the box and the person types over it.
   */
  const addWord = (): WordRef | null => {
    const li = ends ? ends.last.line : targetLine;
    if (li < 0) {
      toast.info(t('lyrics.editor.noLine', 'Select a word to split from, or move the playhead into a line'));
      return null;
    }
    let wi: number;
    if (ends) {
      wi = ends.last.word;
    } else {
      wi = -1;
      lines[li].words.forEach((w, i) => { if (w.start <= currentTime) wi = i; });
    }
    const res = addWordAfter(lines, li, wi, t('lyrics.editor.newWord', 'new'), { total });
    if (!res) {
      toast.warn(t('lyrics.editor.noRoomNewWord', 'No room for another word here: the audio ends first'));
      return null;
    }
    record();
    setLines(res.lines);
    onSelect({ line: li, word: res.at });
    return { line: li, word: res.at };
  };

  /** Remove every selected word. The last word of the track is kept. */
  const deleteWord = () => {
    const refs = selectedRefs(lines, selection);
    if (refs.length === 0) return;
    const words = lines.reduce((n, l) => n + l.words.length, 0);
    if (refs.length >= words) {
      toast.info(t('lyrics.editor.keepLastWord', 'The last word cannot be deleted: the lyrics would be empty'));
      return;
    }
    const next = deleteWords(lines, refs);
    record();
    setLines(next);
    // Stay where the deletion happened, on whatever now sits in its place.
    const li = Math.min(selected?.line ?? 0, next.length - 1);
    const line = next[li];
    onSelect(line?.words.length ? { line: li, word: Math.min(selected?.word ?? 0, line.words.length - 1) } : null);
  };

  /** Put the selection's first word on the playhead; the rest follow it. */
  const setWordAtPlayhead = () => {
    const first = wordAt(ends?.first);
    if (!first) return;
    const next = shiftSelection(lines, selection, currentTime - first.start);
    if (next === lines) return;
    record();
    setLines(next);
  };

  /**
   * Close the gap before the selection: its first word starts where the word
   * before it ended, and everything selected keeps its spacing behind it. The
   * case where the selection starts the track has nothing to snap to.
   */
  const snapToPrevious = () => {
    const first = wordAt(ends?.first);
    if (!first) return;
    const prevEnd = selectionPrevEnd(lines, selection);
    if (prevEnd == null) {
      toast.info(t('lyrics.editor.noPrevious', 'This is the first word of the track'));
      return;
    }
    const next = shiftSelection(lines, selection, prevEnd - first.start);
    if (next === lines) return;
    record();
    setLines(next);
  };

  /**
   * Lengthen or shorten the selection from its far end: the last selected
   * word's right edge moves, so a block grows or shrinks as one.
   */
  const resizeWordBy = (dir: 1 | -1) => {
    const lastRef = ends?.last;
    const w = wordAt(lastRef);
    if (!lastRef || !w) return;
    const res = resizeWordEdge(
      lines, lastRef.line, lastRef.word, 'right', w.end + (dir * msToSec(step)),
      { autoShift, total },
    );
    if (!res) return;
    record();
    setLines(res.lines);
    if (res.noRoom) toast.warn(t('lyrics.editor.noRoom', 'No room before the end of the track'));
  };

  /** Every word and line by the same amount, for a uniform offset. */
  const shiftAll = (ms: number) => {
    record();
    setLines(shiftEveryWord(lines, msToSec(ms)));
  };

  /** Ripple: the selection and everything after it, for drift that builds up. */
  const shiftFromSelection = (ms: number) => {
    if (!selected) return;
    record();
    setLines(shiftFrom(lines, selected.line, selected.word, msToSec(ms)));
  };

  /**
   * A selected word IS the cut point, so splitting does not depend on where the
   * playhead happens to be. With nothing selected, the playhead's word is used.
   *
   * The cut comes from the selection's FIRST word: a row chip selects the whole
   * line and leaves the focus on its last word, and splitting from the focus is
   * what used to strand that one word on a new line.
   */
  const splitLine = () => {
    const idx = ends ? ends.first.line : lineAtOrBefore(lines, currentTime);
    if (idx < 0) {
      toast.info(t('lyrics.editor.noLine', 'Select a word to split from, or move the playhead into a line'));
      return;
    }
    const cut = splitCutAt(lines, idx, selection, currentTime);
    const next = splitLineAt(lines, idx, cut);
    if (!next) {
      toast.info(cut <= 0
        ? t('lyrics.editor.noCutFirst', 'That word already starts its line. Pick a later word to split there')
        : t('lyrics.editor.noCut', 'Put the playhead on the word the new line should start with'));
      return;
    }
    record();
    setLines(next);
    onSelect({ line: idx + 1, word: 0 });
  };

  /** Fold a line into the one above or below it, words re-sorted by time. */
  const mergeLine = (dir: -1 | 1) => {
    const merged = mergeLineAt(lines, targetLine, dir);
    if (!merged) {
      toast.info(t('lyrics.editor.noNeighbour', 'Nothing to merge with on that side'));
      return;
    }
    record();
    setLines(merged.lines);
    onSelect({ line: merged.at, word: 0 });
  };

  /**
   * Write a line the person rewrote as a whole. `planLineText` already chose
   * which words keep their times and where the new ones fit, so this is the one
   * undoable write for the whole line.
   */
  const applyLinePlan = (plan: LineEditPlan) => {
    record();
    setLines(prev => prev.map((line, index) => (index === plan.line ? plan.edited : line)));
  };

  /** Relabel the target line: a section is renamed where it starts, not moved. */
  const renameSection = (section: string) => {
    if (targetLine < 0) return;
    record();
    setLines(prev => setLineSection(prev, targetLine, section));
  };

  /**
   * Start a section at the playhead, named by the label box.
   *
   * `addSectionAt` decides whether that needs the line split in two; this only
   * picks the line to act on — the selection's, else the playhead's — and
   * reports why nothing happened instead of failing silently.
   */
  const addSection = (name: string) => {
    // The playhead can sit before the first line, where `lineAtOrBefore` is -1
    // and no word is selected. A section can still start at the top of the
    // track, so fall back to the first line rather than doing nothing — that
    // fallback is what makes Add work without selecting a word first.
    const li = targetLine >= 0 ? targetLine : (lines.length > 0 ? 0 : -1);
    if (li < 0) {
      toast.info(t('lyrics.editor.noLine', 'Select a word to split from, or move the playhead into a line'));
      return;
    }
    if (!name.trim()) {
      toast.info(t('lyrics.editor.sectionNeedsName', 'Type a section name first, such as Chorus'));
      return;
    }
    const result = addSectionAt(lines, li, currentTime, name);
    if (!result) return;
    record();
    setLines(result.lines);
    if (result.at !== li && result.lines[result.at]?.words.length) {
      onSelect({ line: result.at, word: 0 });
    }
  };

  /** Take a section's label off, wherever in the track it starts. */
  const deleteSection = (lineIndex: number) => {
    const next = removeSectionAt(lines, lineIndex);
    if (!next) return;
    record();
    setLines(next);
  };

  return {
    data: { targetLine },
    wordStep, renameWord, addWord, deleteWord, setWordAtPlayhead, snapToPrevious, resizeWordBy,
    shiftAll, shiftFromSelection, splitLine, mergeLine, addSection, deleteSection,
    applyLinePlan, renameSection,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type TimingEdits = ReturnType<typeof useTimingEdits>;
/** What a component may call: the handlers, without the state. */
export type TimingEditHandlers = Omit<TimingEdits, 'data'>;
