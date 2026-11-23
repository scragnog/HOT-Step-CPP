// shell.ts — which UI shell boots: the classic layout or a registered shell skin.
// `?ui=<id>` wins and is remembered in `hs-shell`; otherwise the saved choice;
// otherwise classic. The id must be `classic` or in the shell registry
// (app/shellRegistry.ts), so a stale or hand-edited value falls back to classic.
// A switch is a full reload, never a live swap: the shells never share a React
// tree. See docs/dev/ui-design.md#shells and docs/dev/skins.md.

import { isRegisteredShell } from './shellRegistry';

export type Shell = string;
export const SHELL_KEY = 'hs-shell';
export const CLASSIC = 'classic';

const isShell = (v: unknown): v is Shell => v === CLASSIC || isRegisteredShell(v);

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
  return CLASSIC;
}

/** Save the choice and reload, optionally at another path. Drops `?ui=` once it is saved; if
 *  storage is blocked, `?ui=` stays in the URL so the reload still boots the chosen shell. */
export function switchShell(shell: Shell, path?: string): void {
  if (!isShell(shell)) return;
  const url = new URL(path ?? window.location.href, window.location.href);
  if (save(shell)) url.searchParams.delete('ui');
  else url.searchParams.set('ui', shell);
  window.location.replace(url);
}
