// shell.ts — which UI shell boots: the classic layout or the studio one.
// `?ui=classic|studio` wins and is remembered in `hs-shell`; otherwise the saved
// choice; otherwise classic. A switch is a full reload, never a live swap: the
// two shells never share a React tree. See docs/dev/ui-design.md#shells.

export type Shell = 'classic' | 'studio';
export const SHELL_KEY = 'hs-shell';

const isShell = (v: unknown): v is Shell => v === 'classic' || v === 'studio';

// Written like usePersistedState writes (JSON), so either reader can open it.
function save(shell: Shell): void {
  try { localStorage.setItem(SHELL_KEY, JSON.stringify(shell)); } catch { /* storage blocked: the URL still wins this load */ }
}

export function resolveShell(): Shell {
  const fromUrl = new URLSearchParams(window.location.search).get('ui');
  if (isShell(fromUrl)) { save(fromUrl); return fromUrl; }
  try {
    const stored = JSON.parse(localStorage.getItem(SHELL_KEY) ?? 'null');
    if (isShell(stored)) return stored;
  } catch { /* unreadable: fall through */ }
  return 'classic';
}

/** Save the choice and reload. Drops `?ui=` so it cannot override the new choice. */
export function switchShell(shell: Shell): void {
  save(shell);
  const url = new URL(window.location.href);
  url.searchParams.delete('ui');
  window.location.replace(url);
}
