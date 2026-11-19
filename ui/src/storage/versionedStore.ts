import { writeJson } from '@/utils/storage/storageJson';
// versionedStore.ts — one shape for every value this app keeps in the browser.
//
// A key carries its version (`realign-<scope>:1`), so a new shape is a new key
// and the old value is still there to read from. `read()` tries today's key,
// then each older version of the same namespace, then any unversioned keys the
// module names; each older value is handed to `migrate(value, fromVersion)`,
// and a value that migrates is written to today's key so the work happens once.
//
// Keeping this in `storage/` rather than inside the lyric editor is deliberate:
// the editor is the first caller, not the only one. A studio that wants a
// versioned per-item setting should import this instead of inventing another
// `hs-something` key and another try/catch block.
//
// Nothing here touches React, and nothing reads `localStorage` at import time:
// the module is safe to import in a Node test or on the server.

export interface VersionedStore<T> {
  /** Today's key, exposed so callers can inspect or clear it. */
  readonly key: string;
  read(): T | null;
  write(value: T): void;
  remove(): void;
}

export interface VersionedStoreOptions<T> {
  /** The setting's family, e.g. `realign` or `lyric-editor-zoom`. */
  namespace: string;
  /** Bumped when the stored shape changes. It is part of the key. */
  version: number;
  /** What the key is scoped to, e.g. one song's audio URL key. Omit for a
   *  browser-wide setting. */
  scope?: string;
  /** Read a value written by an older version, `0` meaning an unversioned
   *  legacy key. Return null when the value is not usable. */
  migrate?: (value: unknown, fromVersion: number) => T | null;
  /** Read today's value. Defaults to `migrate` at the current version. */
  parse?: (value: unknown) => T | null;
  /** Unversioned keys checked after the older versions, in order. */
  legacyKeys?: readonly string[];
}

/** `realign-<scope>:1`, or `lyric-editor-zoom:1` with no scope. */
export function storageKey(namespace: string, version: number, scope?: string): string {
  return `${namespace}${scope ? `-${scope}` : ''}:${version}`;
}

/** JSON when it parses, the raw string when it does not, null when absent. */
function readJson(key: string): unknown {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null) return null;
    try { return JSON.parse(raw); } catch { return raw; }
  } catch {
    // Storage blocked (private mode, disabled cookies): behave as a miss.
    return null;
  }
}

export function createVersionedStore<T>(options: VersionedStoreOptions<T>): VersionedStore<T> {
  const { namespace, version, scope, migrate, legacyKeys = [] } = options;
  const key = storageKey(namespace, version, scope);
  const parseCurrent = options.parse
    ?? (migrate
      ? (value: unknown) => migrate(value, version)
      : (value: unknown) => value as T);

  const write = (value: T) => {
    try { writeJson(key, value); } catch { /* full or blocked */ }
  };

  const read = (): T | null => {
    try {
      const current = parseCurrent(readJson(key));
      if (current != null) return current;

      // Newest older version first: a 3 is closer to a 4 than a 1 is.
      if (migrate) {
        for (let older = version - 1; older >= 1; older--) {
          const raw = readJson(storageKey(namespace, older, scope));
          if (raw == null) continue;
          const value = migrate(raw, older);
          if (value != null) { write(value); return value; }
        }
        for (const legacyKey of legacyKeys) {
          const raw = readJson(legacyKey);
          if (raw == null) continue;
          const value = migrate(raw, 0);
          if (value != null) { write(value); return value; }
        }
      }
      return null;
    } catch {
      // A value that makes a migration throw is a miss, not a broken editor.
      return null;
    }
  };

  return {
    key,
    read,
    write,
    remove() { try { localStorage.removeItem(key); } catch { /* blocked */ } },
  };
}
