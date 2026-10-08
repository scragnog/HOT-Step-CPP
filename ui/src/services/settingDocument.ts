// settingDocument.ts — one installation setting from /api/preferences/settings
// (the AI continue template), as the UI edits it. The modal renders it and
// the tests drive it.
//
// Rules (Batch 4, 7a):
// - Hydration and saves run on one queue, so a slow first read can never
//   land after, and undo, a save.
// - The local copy (`localStorage`, read synchronously by other components)
//   is written only from the user's own value, or from the server's when the
//   user has not edited: hydration checks for an edit before touching it.
// - Saves are coalesced: the latest edit is what gets sent.
// - A save refused because the setting changed elsewhere pauses saving.
//   The user then takes the server's value or reapplies theirs over it.
import type { ImportPreferenceItem, SettingsFamily } from './preferencesApi';
import { hashImportValue, preferencesApi } from './preferencesApi';

export type SettingStatus = 'loading' | 'idle' | 'saving' | 'conflict' | 'failed';

export interface SettingSnapshot {
  value: string;
  status: SettingStatus;
  /** The server's value, when it differs from ours in a conflict. */
  serverValue: string | null;
  error: string | null;
}

type SettingsApi = typeof preferencesApi.settings;

export interface SettingDocumentOptions {
  family: SettingsFamily;
  /** The setting as a string, and back to a body. */
  read: (body: Record<string, unknown>) => string;
  write: (value: string) => Record<string, unknown>;
  /** The browser copy: read once at start, written as described above. */
  localKey: string;
  fallback: string;
  /** Set once the browser copy has been offered for import. */
  migratedFlag: string;
  api?: SettingsApi;
}

const message = (err: unknown) => err instanceof Error ? err.message : String(err);

export class SettingDocument {
  private snap: SettingSnapshot;
  private readonly listeners = new Set<() => void>();
  private readonly api: SettingsApi;
  /** The value the user started editing from (the browser copy). */
  private readonly base: string;
  private dirty = false;
  private revision: number | undefined;
  private pending: string | undefined;
  private queue: Promise<void> = Promise.resolve();
  private hydrated: Promise<void> | null = null;

  private readonly opts: SettingDocumentOptions;

  constructor(opts: SettingDocumentOptions) {
    this.opts = opts;
    this.api = opts.api ?? preferencesApi.settings;
    this.base = readLocal(opts.localKey) ?? opts.fallback;
    this.snap = { value: this.base, status: 'loading', serverValue: null, error: null };
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  getSnapshot = (): SettingSnapshot => this.snap;

  private set(patch: Partial<SettingSnapshot>): void {
    this.snap = { ...this.snap, ...patch };
    for (const l of this.listeners) l();
  }

  private enqueue(task: () => Promise<void>): Promise<void> {
    return this.queue = this.queue.then(task, task);
  }

  /** Read the server's value once, in order with any saves. */
  hydrate(): Promise<void> {
    return this.hydrated ??= this.enqueue(async () => {
      try {
        const { document } = await this.api.get<Record<string, unknown>>(this.opts.family);
        if (document) {
          this.revision = document.revision;
          const server = this.opts.read(document.body);
          if (!this.dirty) {
            writeLocal(this.opts.localKey, server);
            this.set({ value: server, status: 'idle' });
          } else if (server !== this.base) {
            // The user edited a stale copy: theirs is not sent over the newer one.
            this.set({ status: 'conflict', serverValue: server,
              error: 'This setting was changed elsewhere while you edited. Use theirs, or reapply yours over it.' });
          } else {
            this.set({ status: this.pending === undefined ? 'idle' : 'saving' });
          }
          return;
        }
        if (!this.dirty && !readFlag(this.opts.migratedFlag) && this.base !== this.opts.fallback) await this.importLocal();
        else this.set({ status: this.pending === undefined ? 'idle' : 'saving' });
      } catch (err) {
        this.set({ status: 'failed', error: `Could not load the saved setting: ${message(err)}` });
      }
    });
  }

  private async importLocal(): Promise<void> {
    const item: ImportPreferenceItem = { storageKey: this.opts.localKey, sourceHash: await hashImportValue(this.base), body: this.opts.write(this.base) };
    const result = await this.api.import(this.opts.family, item);
    writeFlag(this.opts.migratedFlag);
    if (result.outcome === 'name-conflict') { await this.refreshConflict(); return; }
    const { document } = await this.api.get<Record<string, unknown>>(this.opts.family);
    this.revision = document?.revision;
    this.set({ status: 'idle' });
  }

  /** The user's edit: shown and kept locally at once, sent in order. */
  edit(value: string): void {
    this.dirty = true;
    writeLocal(this.opts.localKey, value);
    this.set({ value });
    if (this.snap.status === 'conflict') return; // paused until reapply
    this.pending = value;
    if (this.snap.status !== 'loading') this.set({ status: 'saving', error: null });
    void this.enqueue(() => this.flush());
  }

  private async flush(): Promise<void> {
    if (this.pending === undefined || this.snap.status === 'conflict') return;
    const value = this.pending;
    this.pending = undefined;
    try {
      const { document } = await this.api.upsert(this.opts.family, this.revision, this.opts.write(value));
      this.revision = document.revision;
      if (this.pending === undefined) this.set({ status: 'idle', serverValue: null, error: null });
    } catch (err) {
      this.pending = value;
      // Moved on elsewhere (or created by another client first): a conflict.
      let current: { revision: number; body: Record<string, unknown> } | null = null;
      try { current = (await this.api.get<Record<string, unknown>>(this.opts.family)).document; } catch { /* unknown: treat as a plain failure */ }
      if (current && current.revision !== this.revision) {
        this.revision = current.revision;
        this.pending = undefined;
        this.set({ status: 'conflict', serverValue: this.opts.read(current.body),
          error: 'This setting was changed elsewhere. Use theirs, or reapply yours over it.' });
      } else {
        this.set({ status: 'failed', error: `Save failed: ${message(err)}` });
      }
    }
  }

  private async refreshConflict(): Promise<void> {
    const { document } = await this.api.get<Record<string, unknown>>(this.opts.family);
    this.revision = document?.revision;
    this.set({ status: 'conflict', serverValue: document ? this.opts.read(document.body) : null,
      error: 'This setting was changed elsewhere. Use theirs, or reapply yours over it.' });
  }

  /** Send our value again, over the server's current revision. */
  reapply(): Promise<void> {
    this.pending = this.snap.value;
    this.set({ status: 'saving', serverValue: null, error: null });
    return this.enqueue(() => this.flush());
  }

  /** Take the server's value and drop ours. */
  useServer(): void {
    if (this.snap.serverValue === null) return;
    this.dirty = false;
    this.pending = undefined;
    writeLocal(this.opts.localKey, this.snap.serverValue);
    this.set({ value: this.snap.serverValue, status: 'idle', serverValue: null, error: null });
  }

  /** Resolves when everything queued so far has run (tests, unmount). */
  settled(): Promise<void> {
    return this.enqueue(async () => {});
  }
}

function readLocal(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeLocal(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}
function readFlag(key: string): boolean {
  try { return !!localStorage.getItem(key); } catch { return false; }
}
function writeFlag(key: string): void {
  try { localStorage.setItem(key, '1'); } catch { /* storage unavailable */ }
}
