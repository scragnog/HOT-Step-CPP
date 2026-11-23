// registry.ts — discovers theme skins (docs/dev/skins.md). A theme is a folder
// `ui/src/themes/<id>/` with `theme.css` and `manifest.json`; the globs find it,
// so adding one never edits a central list. The CSS is scoped to
// html[data-skin="<id>"], so loading every theme is inert until one is selected.
// Themes are studio and kit only: classic never calls useSkin, so it never gets the attribute.

import { useCallback, useEffect, useState } from 'react';

export interface ThemeManifest {
  id: string;
  name: string;
  author: string;
  version: string;
  api: 0;
  kind: 'theme';
  /** 2 to 4 hex colours shown as the picker swatch. */
  preview: string[];
}

import.meta.glob('./*/theme.css', { eager: true });
const manifests = import.meta.glob<ThemeManifest>('./*/manifest.json', { eager: true, import: 'default' });

export const THEMES: ThemeManifest[] = Object.values(manifests).sort((a, b) => a.name.localeCompare(b.name));
export const SKIN_KEY = 'hs-skin';

function stored(): string | null {
  try {
    const id = localStorage.getItem(SKIN_KEY);
    return THEMES.some(t => t.id === id) ? id : null;
  } catch { return null; }
}

/** The active skin id (null = the default kit) and a setter. Sets html[data-skin]
 *  while the calling page is mounted and removes it on unmount, so classic stays untouched. */
export function useSkin(): { skin: string | null; setSkin: (id: string | null) => void } {
  const [skin, setSkinState] = useState<string | null>(stored);
  useEffect(() => {
    const html = document.documentElement;
    if (skin) html.setAttribute('data-skin', skin); else html.removeAttribute('data-skin');
    return () => html.removeAttribute('data-skin');
  }, [skin]);
  const setSkin = useCallback((id: string | null) => {
    try { if (id) localStorage.setItem(SKIN_KEY, id); else localStorage.removeItem(SKIN_KEY); } catch { /* storage blocked: session only */ }
    setSkinState(id);
  }, []);
  return { skin, setSkin };
}
