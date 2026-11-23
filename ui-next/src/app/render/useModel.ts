// A view's subscription to the desk: it asks the core for its model (plain data) on every change of the store, and
// redraws only when the model is not what it was. So a knob that turns redraws that knob and whatever shows its value,
// and nothing else (docs/09, "Rendering"). `key` names what the model depends on besides the store (a definition,
// a node, a surface): a new key asks afresh.

import { useCallback, useRef, useSyncExternalStore } from 'react';
import { useDesk, type DeskStore } from '../store/desk.ts';

interface Last<T> { s: DeskStore | null; key: string; json: string; v: T }

export function useModel<T>(fn: (s: DeskStore) => T, key: string): T {
  const last = useRef<Last<T> | null>(null);
  const get = useCallback(() => {
    const s = useDesk.getState(), l = last.current;
    if (l && l.s === s && l.key === key) return l.v;
    const v = fn(s), json = JSON.stringify(v);
    if (l && l.key === key && l.json === json) { last.current = { ...l, s }; return l.v; }
    last.current = { s, key, json, v };
    return v;
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps -- fn is the same function of the store for one key
  return useSyncExternalStore(useDesk.subscribe, get);
}
