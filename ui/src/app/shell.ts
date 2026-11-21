// shell.ts — which UI shell boots: the classic layout or the studio one.
// `?ui=classic|studio` wins and is remembered in `hs-shell`; otherwise the saved
// choice; otherwise classic. A switch is a full reload, never a live swap: the
// two shells never share a React tree. See docs/dev/ui-design.md#shells.

export type Shell = 'classic' | 'studio';
export const SHELL_KEY = 'hs-shell';

const isShell = (v: unknown): v is Shell => v === 'classic' || v === 'studio';

// Written like usePersistedState writes (JSON), so either reader can open it.
// False when storage is blocked.
function save(shell: Shell): boolean {
  try { localStorage.setItem(SHELL_KEY, JSON.stringify(shell)); return true; } catch { return false; }
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

/** Save the choice and reload. Drops `?ui=` once it is saved; if storage is
 *  blocked, `?ui=` stays in the URL so the reload still boots the chosen shell. */
export function switchShell(shell: Shell): void {
  const url = new URL(window.location.href);
  if (save(shell)) url.searchParams.delete('ui');
  else url.searchParams.set('ui', shell);
  window.location.replace(url);
}
