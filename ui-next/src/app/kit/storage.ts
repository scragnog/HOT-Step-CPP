// What the page remembers in this browser for the user's convenience (the theme, the panels, the languages chosen
// last, the engines seen, the songs rendering when the page closed): read and written without a word when storage is
// off or holds something else. The user's records (the desk, the workflows, the drafts) go through store/stored.ts
// instead, which keeps aside what it cannot read and reports a refused write. The one place for each.

/** What `key` holds, as `take` reads it (undefined: not usable), else `fallback`. A value that is not JSON is given as
 *  its text. */
export function readPref<T>(key: string, fallback: T, take: (v: unknown) => T | undefined): T {
  let raw: string | null = null;
  try { raw = localStorage.getItem(key); } catch { return fallback; }
  if (raw == null) return fallback;
  let v: unknown = raw;
  try { v = JSON.parse(raw); } catch { /* its text */ }
  return take(v) ?? fallback;
}
/** Remembers `v` under `key` (a string as its text, anything else as JSON); null forgets it. */
export function writePref(key: string, v: unknown): void {
  try { if (v === null) localStorage.removeItem(key); else localStorage.setItem(key, typeof v === 'string' ? v : JSON.stringify(v)); } catch { /* storage off: not remembered */ }
}
