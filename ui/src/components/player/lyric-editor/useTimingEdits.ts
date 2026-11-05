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
  lineAtOrBefore, mergeLineAt, moveWordTo, msToSec, resizeWordEdge, shiftEveryWord, shiftFrom,
  splitLineAt, type EditLine, type Selection,
} from './timingModel';

type LinesUpdater = EditLine[] | ((prev: EditLine[]) => EditLine[]);

export interface TimingEditOptions {
  lines: EditLine[];
  setLines: (updater: LinesUpdater) => void;
  /** One undo entry per change. */
  record: () => void;
  selected: Selection | null;
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
  lines, setLines, record, selected, onSelect, step, currentTime, autoShift, total,
}: TimingEditOptions) {
  const { t } = useTranslation();

  /** The line the line-actions apply to: the selected one, else the playhead's. */
  const targetLine = selected ? selected.line : lineAtOrBefore(lines, currentTime);

  /** Move the selected word by the step, clamped by its neighbours. */
  const wordStep = (dir: 1 | -1) => {
    if (!selected) return;
    const w = lines[selected.line]?.words?.[selected.word];
    if (!w) return;
    record();
    setLines(moveWordTo(lines, selected.line, selected.word, w.start + (dir * msToSec(step))));
  };

  const setWordAtPlayhead = () => {
    if (!selected) return;
    record();
    setLines(moveWordTo(lines, selected.line, selected.word, currentTime));
  };

  /**
   * Where the word before this one ends, across a line break: the last word of
   * the line above when this word starts its own line, and null at the start of
   * the track. One case per return, no accumulator.
   */
  const previousWordEnd = (): number | null => {
    if (!selected) return null;
    if (selected.word > 0) return lines[selected.line].words[selected.word - 1].end;
    if (selected.line === 0) return null;
    const prevLine = lines[selected.line - 1];
    if (prevLine.words.length === 0) return null;
    return prevLine.words[prevLine.words.length - 1].end;
  };

  /** Start the selected word exactly where the word before it ended. */
  const snapToPrevious = () => {
    if (!selected) return;
    const prevEnd = previousWordEnd();
    if (prevEnd == null) {
      toast.info(t('lyrics.editor.noPrevious', 'This is the first word of the track'));
      return;
    }
    record();
    setLines(moveWordTo(lines, selected.line, selected.word, prevEnd));
  };

  /** Lengthen or shorten the selected word by the step, from its end. */
  const resizeWordBy = (dir: 1 | -1) => {
    if (!selected) return;
    const w = lines[selected.line]?.words?.[selected.word];
    if (!w) return;
    record();
    const res = resizeWordEdge(
      lines, selected.line, selected.word, 'right', w.end + (dir * msToSec(step)),
      { autoShift, total },
    );
    if (!res) return;
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
   */
  const splitLine = () => {
    const idx = selected ? selected.line : lineAtOrBefore(lines, currentTime);
    if (idx < 0) {
      toast.info(t('lyrics.editor.noLine', 'Select a word to split from, or move the playhead into a line'));
      return;
    }
    const line = lines[idx];
    const cut = selected ? selected.word : line.words.findIndex(w => w.start >= currentTime);
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

  return {
    data: { targetLine },
    wordStep, setWordAtPlayhead, snapToPrevious, resizeWordBy,
    shiftAll, shiftFromSelection, splitLine, mergeLine,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type TimingEdits = ReturnType<typeof useTimingEdits>;
/** What a component may call: the handlers, without the state. */
export type TimingEditHandlers = Omit<TimingEdits, 'data'>;
