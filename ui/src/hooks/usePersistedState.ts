// usePersistedState.ts — useState with localStorage persistence
//
// Drop-in replacement for useState that survives page refreshes.
// Also listens for external storage writes (e.g. from queue stores)
// so the top bar stays in sync with Lyric Studio's adapter changes.

import { useState, useEffect } from 'react';

export function usePersistedState<T>(key: string, defaultValue: T): [T, (value: T | ((prev: T) => T)) => void] {
  const [state, setState] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // Storage full or unavailable
    }
  }, [key, state]);

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
