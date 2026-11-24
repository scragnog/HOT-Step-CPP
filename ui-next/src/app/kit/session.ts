// View state kept with the working session (the concept's sessionState, and the quiet keys of its desk save: S.qAll,
// S.songInfoOpen): what a reload brings back as it was, such as Quick's fold standing open or Song info open. It is
// the page's, not the desk's: never an undo step, never part of the desk's record, a workflow or a revision. A
// feature, the shell or a part registers each with `keptState(name, fallback, take)`, reads it (`useKept` in a
// component) and sets it through what that returns; all of them are kept together under one key in this browser's
// convenience store (kit/storage.ts).

import { useSyncExternalStore } from 'react';
import { readPref, writePref } from './storage.ts';

const KEY = 'hotstep.create.session';

export interface Kept<T> {
  get(): T;
  set(v: T): void;
  subscribe(fn: () => void): () => void;
}

const KEPT = new Map<string, Kept<unknown>>();
const saved = (): Record<string, unknown> => readPref<Record<string, unknown>>(KEY, {}, v => (v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : undefined));

/** Registers a piece of view state kept with the session: `take` reads a kept value (undefined: not usable, so
 *  `fallback`). A name registered twice fails at load (not on the dev server's reload). */
export function keptState<T>(name: string, fallback: T, take: (v: unknown) => T | undefined): Kept<T> {
  /* a module the dev server reloads registers again: it takes the one it had */
  if (KEPT.has(name)) { if (import.meta.hot) return KEPT.get(name) as Kept<T>; throw new Error(`the kept state "${name}" is defined twice`); }
  let now: T = take(saved()[name]) ?? fallback;
  const subs = new Set<() => void>();
  const k: Kept<T> = {
    get: () => now,
    set(v) {
      if (Object.is(v, now)) return;
      now = v;
      writePref(KEY, { ...saved(), [name]: v });
      subs.forEach(f => f());
    },
    subscribe(fn) { subs.add(fn); return () => { subs.delete(fn); }; },
  };
  KEPT.set(name, k as Kept<unknown>);
  return k;
}

/** A kept value, in a component: it redraws when the value changes. */
export const useKept = <T,>(k: Kept<T>): T => useSyncExternalStore(k.subscribe, k.get);

/** Every piece of kept state by name (the test layer reads and sets them as the concept's S does). */
export const keptStates = (): ReadonlyMap<string, Kept<unknown>> => KEPT;

/** A kept switch: true or false, false unless kept true. */
export const keptFlag = (name: string): Kept<boolean> => keptState(name, false, v => (typeof v === 'boolean' ? v : undefined));
