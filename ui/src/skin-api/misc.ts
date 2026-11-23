// misc.ts — theme, navigation and skin storage.

import { useCallback, useState } from 'react';
import { useTheme as useHostTheme } from '../hooks/useTheme';
import { switchShell, resolveShell, CLASSIC } from '../app/shell';
import { urlForView } from '../app/viewRoutes';
import { useSkin, themesFor } from '../themes/registry';
import type { StudioKind, ThemeMode } from './types';

/** Light and dark belong to the host (`hs-theme`, the `.dark` class on <html>, theme-boot.js).
 *  Style from `.dark`; never set it yourself. `mode` is the pick, `resolved` what is showing. */
export function useTheme(): { mode: ThemeMode; resolved: 'light' | 'dark'; setMode: (m: ThemeMode) => void; cycle: () => void } {
  const t = useHostTheme();
  return { mode: t.mode, resolved: t.theme, setMode: t.setMode, cycle: t.cycleMode };
}

/** The kit colour skin (a theme from `src/themes/`), for a shell that wears the kit. A shell
 *  with its own look can ignore this. */
export function useThemeSkin(): { skin: string | null; setSkin: (id: string | null) => void; available: { id: string; name: string }[] } {
  const { skin, setSkin } = useSkin(true);
  return { skin, setSkin, available: themesFor('studio').map(t => ({ id: t.id, name: t.name })) };
}

const VIEW: Record<StudioKind, string> = { library: 'library', cover: 'cover-studio', stem: 'stem-studio', midi: 'midi-studio', repaint: 'repaint' };

/** Open a song's studio. Reloads into the classic UI for now (it does not carry the song:
 *  the studio opens and the user picks it). Your own shell can replace this later. */
export function openInStudio(kind: StudioKind, songId: string): void {
  void songId;
  switchShell(CLASSIC, urlForView(VIEW[kind]));
}

/** Reload into the classic UI (the way back from any shell). */
export function openClassic(): void { switchShell(CLASSIC); }

const KEY_RE = /^[a-z0-9-]{1,32}$/;
/** About 64 KB per value. Layout and preferences only: never songs, drafts, presets or seeds. */
export const SKIN_STORAGE_LIMIT = 64 * 1024;

/** A JSON value kept in this shell's own corner of localStorage (`hs-skin-<shell id>-<key>`). */
export function useSkinStorage<T>(key: string, initial: T): [T, (v: T) => void] {
  if (!KEY_RE.test(key)) throw new Error(`skin storage key "${key}" must be 1-32 lowercase letters, digits or dashes`);
  const full = `hs-skin-${resolveShell()}-${key}`;
  const [value, setValue] = useState<T>(() => {
    try { const raw = localStorage.getItem(full); return raw === null ? initial : JSON.parse(raw) as T; } catch { return initial; }
  });
  const set = useCallback((v: T) => {
    const raw = JSON.stringify(v);
    if (raw === undefined || raw.length > SKIN_STORAGE_LIMIT) { console.warn(`[skin-api] "${key}" not saved: value is not JSON or is over ${SKIN_STORAGE_LIMIT} bytes`); return; }
    try { localStorage.setItem(full, raw); } catch { /* storage blocked: kept for this session */ }
    setValue(v);
  }, [full, key]);
  return [value, set];
}
