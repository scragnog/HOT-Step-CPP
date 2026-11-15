// usePersistedState.ts — useState with localStorage persistence
//
// Drop-in replacement for useState that survives page refreshes.
// Also listens for external storage writes (e.g. from queue stores)
// so the top bar stays in sync with Lyric Studio's adapter changes.

import { useState, useEffect } from 'react';

/** `revive` adjusts what was read back (merge in defaults added since it was
 *  saved, drop one-shot fields). A new `key` (another dataset, say) reads that
 *  key's value instead of carrying the old one over and saving it there. */
export function usePersistedState<T>(key: string, defaultValue: T, revive: (stored: T) => T = v => v): [T, (value: T | ((prev: T) => T)) => void] {
  const read = (): T => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? revive(JSON.parse(stored)) : defaultValue;
    } catch {
      return defaultValue;
    }
  };
  const [state, setState] = useState<T>(read);
  const [stateKey, setStateKey] = useState(key);
  if (stateKey !== key) {
    setStateKey(key);
    setState(read());
  }

  useEffect(() => {
    if (stateKey !== key) return;
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // Storage full or unavailable
    }
  }, [key, stateKey, state]);

  // Listen for writes from other tabs or from non-React code (queue stores)
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try { setState(JSON.parse(e.newValue)); } catch { /* ignore */ }
      }
    };
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }, [key]);

  return [state, setState];
}

/** Write a value to localStorage AND dispatch a StorageEvent so
 *  usePersistedState hooks in the same tab pick up the change. */
export function writePersistedState(key: string, value: unknown): void {
  const serialized = JSON.stringify(value);
  const oldValue = localStorage.getItem(key);
  localStorage.setItem(key, serialized);
  // StorageEvent only fires cross-tab; dispatch manually for same-tab listeners
  window.dispatchEvent(new StorageEvent('storage', { key, oldValue, newValue: serialized }));
}
