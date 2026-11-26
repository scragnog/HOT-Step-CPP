// The reference tracks the page has been given (the timbre reference, Mastering's), newest first: what the Timbre
// reference tile offers as recent references (the concept's S.refs, 25-module-timbre.js; the existing frontend keeps
// them on the server: ui/src/queries/global-models useReferenceTracks). Server-owned, so no step names them and Undo
// never takes one back: a file loaded on the desk is remembered, and stays remembered when the load is undone.
//
// This is the simulated server's list (the concept's three demo files to start with), kept under the page's own key;
// the real adapter, over the server's reference tracks, is package 10's.

import type { ReferencesService } from '../references.ts';

const KEY = 'hotstep.create.refs', MAX = 8, START = ['reference-mix.wav', 'my-band-master.flac', 'live-room-take.wav'];
const read = (): string[] => { try { const v = JSON.parse(localStorage.getItem(KEY) ?? 'null'); return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string').slice(0, MAX) : START; } catch { return START; } };
let list: readonly string[] = read();
const subs = new Set<() => void>();
const write = (next: readonly string[]) => { list = next; try { localStorage.setItem(KEY, JSON.stringify(list)); } catch { /* storage refused: kept for this visit */ } subs.forEach(f => f()); };

export const references: ReferencesService & { replace(next: readonly string[]): void } = {
  /** The recent reference files, newest first. */
  list: (): readonly string[] => list,
  /** A file the page was given: it comes first, once. */
  remember(name: string): void { if (name && list[0] !== name) write([name, ...list.filter(x => x !== name)].slice(0, MAX)); },
  /** The whole list (the test layer's scenes). */
  replace(next: readonly string[]): void { write(next.filter(x => typeof x === 'string').slice(0, MAX)); },
  subscribe(f: () => void): () => void { subs.add(f); return () => subs.delete(f); },
};
