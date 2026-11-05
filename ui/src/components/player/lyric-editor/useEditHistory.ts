// useEditHistory.ts — the editor's undo stack, kept apart from the rendering
// and from the saving. A change calls `record()` once, mutates through
// `setLines`, and undo/redo step whole snapshots. One drag is one record call,
// which is why the editor calls it on the first pointer move rather than every
// move: the granularity of history is a UI decision, so it stays in the UI.
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
    setPast(p => {
      const next = [...p, linesRef.current];
      return next.length > MAX_HISTORY ? next.slice(next.length - MAX_HISTORY) : next;
    });
    setFuture([]);
  }, []);

  const undo = useCallback(() => {
    setPast(p => {
      if (p.length === 0) return p;
      const prev = p[p.length - 1];
      setFuture(f => [linesRef.current, ...f].slice(0, MAX_HISTORY));
      linesRef.current = prev;
      setLinesState(prev);
      return p.slice(0, -1);
    });
  }, []);

  const redo = useCallback(() => {
    setFuture(f => {
      if (f.length === 0) return f;
      const next = f[0];
      setPast(p => [...p, linesRef.current].slice(-MAX_HISTORY));
      linesRef.current = next;
      setLinesState(next);
      return f.slice(1);
    });
  }, []);

  /** Start over from a fresh file: new lines, no history. */
  const reset = useCallback((next: EditLine[]) => {
    linesRef.current = next;
    setLinesState(next);
    setPast([]);
    setFuture([]);
  }, []);

  return {
    data: {
      lines,
      linesRef,
      canUndo: past.length > 0,
      canRedo: future.length > 0,
    },
    setLines, record, undo, redo, reset,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type EditHistory = ReturnType<typeof useEditHistory>;
/** What a component may call: the handlers, without the state. */
export type EditHistoryHandlers = Omit<EditHistory, 'data'>;
