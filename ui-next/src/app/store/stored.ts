// The page's storage (the concept's 13-stored.js and the read-out of 10-working-state.js): reading a stored key,
// copying aside what cannot be used before anything writes over it (F-16), and writing with the result reported, so a
// refused write is never shown as saved (F-15). Every store the page keeps (the desk, the workflows, the drafts) reads
// and writes through here. What the page says about it is the core's (src/derive/workflows.ts, "storage words").

import { create } from 'zustand';
import type { Msg, StoreWhy } from '../../derive.ts';

/** What a key holds: its parsed value (null when empty), or `bad` when it is not JSON (copied aside already). */
export interface Read { v: unknown; raw: string | null; bad?: true; off?: true }
export function readStored(key: string): Read {
  let raw: string | null = null;
  try { raw = localStorage.getItem(key); } catch { return { v: null, raw: null, off: true }; }
  if (raw == null) return { v: null, raw };
  try { return { v: JSON.parse(raw) as unknown, raw }; } catch { keepAside(key, raw); return { v: null, raw, bad: true }; }
}

/** The keys whose copies are kept aside, and the suffix the copies go under. */
const ASIDE = new Set<string>();
export const ASIDE_SUFFIX = '-unreadable';
/** Copies a stored value aside (`<key>-unreadable`; a later, different one beside it with its time), once per value. */
export function keepAside(key: string, raw0: string | null = null): void {
  try {
    const raw = raw0 ?? localStorage.getItem(key); if (raw == null) return;
    const k = key + ASIDE_SUFFIX, had = localStorage.getItem(k);
    const kept = had === raw || Object.keys(localStorage).some(x => x.startsWith(k + '-') && localStorage.getItem(x) === raw);
    if (!kept) localStorage.setItem(had == null ? k : `${k}-${Date.now()}`, raw);
    ASIDE.add(key);
  } catch { /* storage off: nothing can be kept, and nothing will be written over either */ }
}
/** Every copy kept aside, as one file to download. */
export function asideFile(): { name: string; text: string } {
  const o: Record<string, string | null> = {};
  try { Object.keys(localStorage).filter(k => new RegExp(`${ASIDE_SUFFIX}(-\\d+)?$`).test(k)).sort().forEach(k => { o[k] = localStorage.getItem(k); }); } catch { /* nothing to read */ }
  return { name: 'hot-step-unread-copies.json', text: JSON.stringify(o, null, 2) };
}
export const hasAside = (): boolean => ASIDE.size > 0;

/** The stores' last writes: a key that failed stays failed until a write to it succeeds. `failed` is the last failure,
 *  for the page to say (`n` counts them, so the same failure twice is said twice). */
interface StoredState { fails: { [key: string]: StoreWhy }; failed: { key: string; n: number } | null; told: Msg[] }
export const useStored = create<StoredState>()(() => ({ fails: {}, failed: null, told: [] }));

/** Writes a key and reports how it went. */
export function writeStored(key: string, text: string): boolean {
  try {
    localStorage.setItem(key, text);
    const s = useStored.getState();
    if (key in s.fails) { const fails = { ...s.fails }; delete fails[key]; useStored.setState({ fails }); }
    return true;
  } catch (e) {
    const why: StoreWhy = /quota/i.test((e as { name?: string })?.name ?? '') ? 'full' : 'blocked', s = useStored.getState();
    useStored.setState({ fails: { ...s.fails, [key]: why }, failed: { key, n: (s.failed?.n ?? 0) + 1 } });
    return false;
  }
}

/** What boot could not read, said once after the page is up. */
export const tellOnBoot = (m: Msg): void => { useStored.setState({ told: [...useStored.getState().told, m] }); };
