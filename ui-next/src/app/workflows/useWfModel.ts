// A view's subscription to the desk and the saved records together: as render/useModel.ts, it asks the core for its
// model again when the records change or the desk changes something the model read (render/reads.ts), and redraws only
// when the model is not what it was.

import { useCallback, useRef, useSyncExternalStore } from 'react';
import { useDesk, type DeskStore } from '../store/desk.ts';
import { useWf, type WfStore } from '../store/workflows.ts';
import { useWfView, type WfView } from './state.ts';
import { readsOf, touched, type Reads } from '../render/reads.ts';

const subscribe = (f: () => void) => { const a = useDesk.subscribe(f), b = useWf.subscribe(f), c = useWfView.subscribe(f); return () => { a(); b(); c(); }; };

export function useWfModel<T>(fn: (d: DeskStore, w: WfStore, v: WfView) => T, key = ''): T {
  const last = useRef<{ d: DeskStore; w: WfStore; v: WfView; key: string; json: string; out: T; reads: Reads } | null>(null);
  const get = useCallback(() => {
    const d = useDesk.getState(), w = useWf.getState(), v = useWfView.getState(), l = last.current;
    if (l && l.w === w && l.v === v && l.key === key && !touched(l.reads, l.d, d)) { l.d = d; return l.out; }
    const { v: out, reads } = readsOf(s => fn(s, w, v), d), json = JSON.stringify(out, (_k, x: unknown) => (x instanceof HTMLElement ? null : x));
    if (l && l.key === key && l.json === json) { last.current = { ...l, d, w, v, reads }; return l.out; }
    last.current = { d, w, v, key, json, out, reads };
    return out;
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps -- fn is the same function of the stores for one key
  return useSyncExternalStore(subscribe, get);
}
