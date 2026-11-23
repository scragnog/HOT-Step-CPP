// registry.ts — discovers theme skins (docs/dev/skins.md). A theme is a folder
// `ui/src/themes/<id>/` with `theme.css` and `manifest.json`; the globs find it,
// so adding one never edits a central list. The CSS is scoped to
// html[data-skin="<id>"], so loading every theme is inert until one is selected.
// data-skin is set early by public/theme-boot.js; main.tsx drops it when it does not target the running shell.

import { useCallback, useState } from 'react';

export interface ThemeManifest {
  id: string;
  name: string;
  author: string;
  version: string;
  api: 0;
  kind: 'theme';
  /** Which UIs the theme applies to. Studio themes set kit tokens only. */
  targets: Array<'studio' | 'classic'>;
  /** 2 to 4 hex colours shown as the picker swatch. */
  preview: string[];
}

import.meta.glob('./*/theme.css', { eager: true });
const manifests = import.meta.glob<ThemeManifest>('./*/manifest.json', { eager: true, import: 'default' });

export const THEMES: ThemeManifest[] = Object.values(manifests).sort((a, b) => a.name.localeCompare(b.name));
export const SKIN_KEY = 'hs-skin';

/** Themes that apply to a shell (the picker lists these). */
export const themesFor = (shell: 'studio' | 'classic'): ThemeManifest[] => THEMES.filter(t => t.targets.includes(shell));

/** Remove html[data-skin] when the saved skin does not target this shell (or is unknown). */
export function dropForeignSkin(shell: 'studio' | 'classic'): void {
  const html = document.documentElement, id = html.getAttribute('data-skin');
  if (id && !themesFor(shell).some(t => t.id === id)) html.removeAttribute('data-skin');
}

function stored(): string | null {
  try {
    const id = localStorage.getItem(SKIN_KEY);
    return THEMES.some(t => t.id === id) ? id : null;
  } catch { return null; }
}

/** The saved skin id (null = the default kit) and a setter. The attribute itself is set at boot;
 *  this only changes it on a pick. */
export function useSkin(): { skin: string | null; setSkin: (id: string | null) => void } {
  const [skin, setSkinState] = useState<string | null>(stored);
  const setSkin = useCallback((id: string | null) => {
    const html = document.documentElement;
    if (id) html.setAttribute('data-skin', id); else html.removeAttribute('data-skin');
    try { if (id) localStorage.setItem(SKIN_KEY, id); else localStorage.removeItem(SKIN_KEY); } catch { /* storage blocked: session only */ }
    setSkinState(id);
  }, []);
  return { skin, setSkin };
}
