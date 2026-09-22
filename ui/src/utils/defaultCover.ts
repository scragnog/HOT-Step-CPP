// defaultCover.ts — the generic artwork a song falls back to when it has none.
//
// Two sets of 50 ship in ui/public/covers/ (generated with the app's own sd.exe
// + FLUX.2-klein-4B). A song picks one image by hashing its id, so the same
// track always shows the same art — across reloads, sessions and machines —
// without storing anything. Which SET is in play is a user preference.

import { useSyncExternalStore } from 'react';

export const DEFAULT_COVER_COUNT = 50;

export const COVER_SETS = [
  { id: 'cyber', label: 'Cyberpunk' },
  { id: 'scenic', label: 'Scenic' },
] as const;

export type CoverSet = typeof COVER_SETS[number]['id'];

const STORAGE_KEY = 'hs-coverSet';
const FALLBACK_SET: CoverSet = 'cyber';

// One module-level value with one listener, rather than a persisted-state hook
// per thumbnail — a library page mounts hundreds of these.
let currentSet: CoverSet = readSet();
const subscribers = new Set<() => void>();

function readSet(): CoverSet {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return COVER_SETS.some(s => s.id === stored) ? stored as CoverSet : FALLBACK_SET;
  } catch {
    return FALLBACK_SET;
  }
}

function publish() {
  const next = readSet();
  if (next === currentSet) return;
  currentSet = next;
  subscribers.forEach(fn => fn());
}

if (typeof window !== 'undefined') {
  // Covers the other tab and, via setCoverSet below, this one.
  window.addEventListener('storage', (e) => {
    if (e.key === STORAGE_KEY) publish();
  });
}

function subscribe(fn: () => void): () => void {
  subscribers.add(fn);
  return () => { subscribers.delete(fn); };
}

/** The set currently in use. Re-renders the caller when it changes. */
export function useCoverSet(): CoverSet {
  return useSyncExternalStore(subscribe, () => currentSet, () => FALLBACK_SET);
}

/** Switch sets. Every mounted cover updates immediately. */
export function setCoverSet(set: CoverSet): void {
  try { localStorage.setItem(STORAGE_KEY, set); } catch { /* storage unavailable */ }
  publish();
}

/** Read the set once, outside React. */
export function getCoverSet(): CoverSet {
  return currentSet;
}

/** FNV-1a. Small, stable, and does not vary between browsers or runs. */
function hash(seed: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Pick this seed's generic cover from the given set. */
export function defaultCoverUrl(seed: string, set: CoverSet = currentSet): string {
  const n = hash(seed || 'untitled') % DEFAULT_COVER_COUNT;
  return `/covers/${set}/${String(n).padStart(2, '0')}.webp`;
}

/**
 * The artwork to display: the song's own cover when it has one, otherwise its
 * generic stand-in. Never use this where the caller needs to know whether real
 * art exists (the metadata editor, the "generate cover art" menu) — check the
 * raw field there.
 */
export function coverSrc(url: string | null | undefined, seed: string, set: CoverSet = currentSet): string {
  return url?.trim() || defaultCoverUrl(seed, set);
}
