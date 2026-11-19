/**
 * storageJson.ts — reading and writing JSON in localStorage, with the failure
 * modes answered once.
 *
 * Every caller used to wrap `JSON.parse(localStorage.getItem(key) || '{}')` in
 * its own try/catch and disagree about the seam cases: a missing key, a value
 * an older build wrote in another shape, a browser in private mode, a full
 * quota. Those are all the same three answers — fall back, or report whether
 * the write landed.
 */

/** Parse `key` as JSON, returning `fallback` when it is missing or unreadable. */
export function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null || raw === '') return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

/** Store `value` as JSON. False when storage refused (quota, private mode). */
export function writeJson(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

/** Read a plain string stored as JSON. Values written without JSON quoting by
 *  an older build are taken as-is rather than counting as corrupt. */
export function readJsonString(key: string, fallback = ''): string {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null || raw === '') return fallback;
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === 'string' ? parsed : raw;
  } catch {
    return fallback;
  }
}

/** Read a plain string setting. Missing and unreadable both give `fallback`,
 *  which is what every caller's `|| ''` was reaching for. */
export function readString(key: string, fallback = ''): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}
