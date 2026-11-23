// useTheme.ts — theme state hook with localStorage persistence
//
// The host owns light/dark: the `dark` class on <html>, the `hs-theme` key and
// <meta name="theme-color">. `mode` is what the user picked (system, light or
// dark); `theme` is what is showing. `system` follows prefers-color-scheme live.
// One module-level store, so the classic toggle, a shell's picker and every
// other caller agree. public/theme-boot.js applies the same rule before first paint.
// Defaults to dark (preserves existing behaviour).

import { useCallback, useSyncExternalStore } from 'react';

const STORAGE_KEY = 'hs-theme';

export type Theme = 'light' | 'dark';
export type ThemeMode = 'system' | 'light' | 'dark';

const META_COLORS: Record<Theme, string> = {
  dark: '#000000',
  light: '#fafafa',
};

const MODES: ThemeMode[] = ['system', 'light', 'dark'];
const query = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function readMode(): ThemeMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'system' || stored === 'light' || stored === 'dark') return stored;
  } catch { /* ignore */ }
  return 'dark';
}

const resolve = (mode: ThemeMode): Theme => mode === 'system' ? (query?.matches === false ? 'light' : 'dark') : mode;

let mode: ThemeMode = readMode();
let theme: Theme = resolve(mode);
const listeners = new Set<() => void>();

function applyTheme(next: Theme): void {
  document.documentElement.classList.toggle('dark', next === 'dark');
  const meta = document.getElementById('theme-color-meta') as HTMLMetaElement | null;
  if (meta) meta.content = META_COLORS[next];
}

function commit(): void {
  const next = resolve(mode);
  if (next !== theme) { theme = next; applyTheme(theme); }
  listeners.forEach(l => l());
}

query?.addEventListener('change', () => { if (mode === 'system') commit(); });

export function setThemeMode(next: ThemeMode): void {
  mode = next;
  try { localStorage.setItem(STORAGE_KEY, next); } catch { /* ignore */ }
  commit();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  applyTheme(theme); // the first subscriber makes the DOM match
  return () => { listeners.delete(l); };
}

export function useTheme(): {
  theme: Theme; mode: ThemeMode; setMode: (m: ThemeMode) => void; cycleMode: () => void; toggleTheme: () => void;
} {
  const m = useSyncExternalStore(subscribe, () => mode);
  const t = useSyncExternalStore(subscribe, () => theme);
  const cycleMode = useCallback(() => setThemeMode(MODES[(MODES.indexOf(mode) + 1) % MODES.length]), []);
  // Classic's button: flip what is showing, and make it an explicit choice.
  const toggleTheme = useCallback(() => setThemeMode(theme === 'dark' ? 'light' : 'dark'), []);
  return { theme: t, mode: m, setMode: setThemeMode, cycleMode, toggleTheme };
}
