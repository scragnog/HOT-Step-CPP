// useEditHistory.ts — the editor's undo stack, kept apart from the rendering
// and from the saving. A change calls `record()` once, mutates through
// `setLines`, and undo/redo step whole snapshots. One drag is one record call,
// which is why the editor calls it on the first pointer move rather than every
// move: the granularity of history is a UI decision, so it stays in the UI.
//
// The two stacks are mirrored in refs. `undo` has to read the entry that is
// current when the pointer lands, and — the reason this matters — it must not
// write `lines` from inside a `setPast` updater. React runs that updater during
// render and a `setLines` called there is a side effect in render: React 19
// drops it, so Undo stayed enabled, emptied the stack, and left the sheet
// exactly as it was. Every mutation below updates the refs and then publishes
// them, so undo, redo and a burst of pointer-driven records all see the same
// history.
//
// The return is `{ data, ...handlers }`: `data` is what the screen reads,
// everything else is something it may call. A component takes the handler slice
// it needs and cannot touch the rest, so `const { data, ...history } =
// useEditHistory(...)` is the whole contract. The types below are inferred from
// this function, never declared by hand, so the two cannot drift.

import { useCallback, useEffect, useRef, useState } from 'react';
import { MAX_HISTORY, type EditLine } from './timingModel';

type LinesUpdater = EditLine[] | ((prev: EditLine[]) => EditLine[]);

export function useEditHistory(initial: EditLine[]) {
  const [lines, setLinesState] = useState<EditLine[]>(initial);
  const [past, setPast] = useState<EditLine[][]>([]);
  const [future, setFuture] = useState<EditLine[][]>([]);
  const linesRef = useRef(lines);
  const pastRef = useRef<EditLine[][]>([]);
  const futureRef = useRef<EditLine[][]>([]);

  /**
   * Update the ref in the same call as the state, so `record()` taken on the
   * next pointer event snapshots what is on screen, not what React last
   * rendered.
   */
  const setLines = useCallback((updater: LinesUpdater) => {
    setLinesState(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater;
      linesRef.current = next;
      return next;
    });
  }, []);

  useEffect(() => { linesRef.current = lines; }, [lines]);

  const record = useCallback(() => {
    const next = [...pastRef.current, linesRef.current];
    pastRef.current = next.length > MAX_HISTORY ? next.slice(next.length - MAX_HISTORY) : next;
    futureRef.current = [];
    setPast(pastRef.current);
    setFuture(futureRef.current);
  }, []);

  const undo = useCallback(() => {
    const entries = pastRef.current;
    if (entries.length === 0) return;
    const prev = entries[entries.length - 1];
    pastRef.current = entries.slice(0, -1);
    futureRef.current = [linesRef.current, ...futureRef.current].slice(0, MAX_HISTORY);
    linesRef.current = prev;
    setPast(pastRef.current);
    setFuture(futureRef.current);
    setLinesState(prev);
  }, []);

  const redo = useCallback(() => {
    const entries = futureRef.current;
    if (entries.length === 0) return;
    const next = entries[0];
    futureRef.current = entries.slice(1);
    pastRef.current = [...pastRef.current, linesRef.current].slice(-MAX_HISTORY);
    linesRef.current = next;
    setPast(pastRef.current);
    setFuture(futureRef.current);
    setLinesState(next);
  }, []);

  /**
   * Take a new array as one undoable step: the file after an Apply, or a
   * Reload. `reset` is for a first load, where there is nothing to go back to;
   * this is for a replacement the person might want back.
   */
  const replace = useCallback((next: EditLine[]) => {
    pastRef.current = [...pastRef.current, linesRef.current].slice(-MAX_HISTORY);
    futureRef.current = [];
    linesRef.current = next;
    setPast(pastRef.current);
    setFuture(futureRef.current);
    setLinesState(next);
  }, []);

  /** Start over from a fresh file: new lines, no history. */
  const reset = useCallback((next: EditLine[]) => {
    linesRef.current = next;
    pastRef.current = [];
    futureRef.current = [];
    setLinesState(next);
    setPast(pastRef.current);
    setFuture(futureRef.current);
  }, []);

  return {
    data: {
      lines,
      linesRef,
      canUndo: past.length > 0,
      canRedo: future.length > 0,
    },
    setLines, record, undo, redo, reset, replace,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type EditHistory = ReturnType<typeof useEditHistory>;
/** What a component may call: the handlers, without the data. */
export type EditHistoryHandlers = Omit<EditHistory, 'data'>;
