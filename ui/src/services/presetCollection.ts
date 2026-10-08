// presetCollection.ts — one named-preset family from /api/preferences, as the
// UI edits it. The VST chain store, the scale-override presets and the AI
// continue preset modal all render one of these and call its methods; the
// tests drive the same objects.
//
// Rules (Batch 4, 7a):
// - Entries are keyed by document id (a temporary key until the create
//   lands), never by name: two presets may share a name.
// - A failed create, update or delete keeps the user's content in the entry,
//   with the error, until they reapply or discard it.
// - A write refused because the document changed elsewhere (409, or 404 for
//   deleted) puts the entry in 'conflict': its writes pause, the current
//   server version is fetched, and nothing is sent until reapply.
// - Legacy browser presets are imported automatically when nothing collides;
//   a same-name collision waits for an explicit keep-both or replace.
import type { TypedDocument } from '../../../server/src/contracts/workflow';
import { preferencesApi, type ImportPreferenceItem, type PresetFamily } from './preferencesApi';
import { WorkflowRequestError } from './workflowApi';

export type PresetEntryStatus = 'saved' | 'creating' | 'saving' | 'deleting' | 'failed' | 'conflict';

export interface PresetEntry<B> {
  /** Stable key for the UI: the document id, or a temporary key before the
   *  create lands. */
  key: string;
  id: string | null;
  /** The server revision this entry's next write is based on (0: none yet). */
  revision: number;
  /** What the user sees: their edit when it is not saved yet. */
  body: B;
  /** The last body known on the server, or null if there is none (not
   *  created yet, or deleted elsewhere). */
  serverBody: B | null;
  status: PresetEntryStatus;
  /** What reapply will send, for a 'failed' or 'conflict' entry. */
  pendingOp: 'create' | 'update' | 'delete' | null;
  error: string | null;
}

export interface ImportConflict {
  storageKey: string;
  name: string;
  /** The stored document the browser value collides with. */
  existingId: string;
  error: string | null;
}

export interface PresetCollectionSnapshot<B> {
  entries: PresetEntry<B>[];
  importConflicts: ImportConflict[];
  loaded: boolean;
  loadError: string | null;
}

type PresetsApi = typeof preferencesApi.presets;

export interface PresetCollectionOptions {
  family: PresetFamily;
  /** The legacy browser values to import, or [] when there are none. */
  legacyItems: () => Promise<ImportPreferenceItem[]>;
  /** localStorage flag set once every legacy value is imported or resolved. */
  migratedFlag: string;
  api?: PresetsApi;
}

const message = (err: unknown) => err instanceof Error ? err.message : String(err);

export class PresetCollection<B extends Record<string, unknown>> {
  private snap: PresetCollectionSnapshot<B> = { entries: [], importConflicts: [], loaded: false, loadError: null };
  private readonly listeners = new Set<() => void>();
  private readonly api: PresetsApi;
  private pendingImports = new Map<string, ImportPreferenceItem>();
  /** Keys whose save is in flight, and keys edited again meanwhile. */
  private inFlight = new Set<string>();
  private editedDuringSave = new Set<string>();
  /** Temporary keys deleted while their create was in flight. */
  private deleteAfterCreate = new Set<string>();
  private loading: Promise<void> | null = null;
  private tempSeq = 0;

  private readonly opts: PresetCollectionOptions;

  constructor(opts: PresetCollectionOptions) {
    this.opts = opts;
    this.api = opts.api ?? preferencesApi.presets;
  }

  // ── Store protocol (useSyncExternalStore / zustand subscribe) ─────────────

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  getSnapshot = (): PresetCollectionSnapshot<B> => this.snap;

  private set(patch: Partial<PresetCollectionSnapshot<B>>): void {
    this.snap = { ...this.snap, ...patch };
    for (const l of this.listeners) l();
  }

  private entry(key: string): PresetEntry<B> | undefined {
    return this.snap.entries.find(e => e.key === key);
  }

  private patch(key: string, patch: Partial<PresetEntry<B>>): void {
    this.set({ entries: this.snap.entries.map(e => e.key === key ? { ...e, ...patch } : e) });
  }

  private drop(key: string): void {
    this.set({ entries: this.snap.entries.filter(e => e.key !== key) });
  }

  // ── Loading and import ────────────────────────────────────────────────────

  /** Import what can be imported without a choice, then list. Runs once;
   *  later calls return the same promise. */
  load(): Promise<void> {
    return this.loading ??= (async () => {
      try {
        await this.importLegacy();
        await this.refresh();
        this.set({ loaded: true, loadError: null });
      } catch (err) {
        this.set({ loaded: true, loadError: message(err) });
        this.loading = null; // a later load() may retry
      }
    })();
  }

  private async importLegacy(): Promise<void> {
    if (readFlag(this.opts.migratedFlag)) return;
    const items = await this.opts.legacyItems();
    if (items.length === 0) { writeFlag(this.opts.migratedFlag); return; }
    const { results } = await this.api.import(this.opts.family, items);
    const conflicts: ImportConflict[] = [];
    for (const r of results) {
      if (r.outcome !== 'name-conflict') continue;
      const item = items.find(i => i.storageKey === r.storageKey);
      if (!item) continue;
      this.pendingImports.set(item.storageKey, item);
      conflicts.push({ storageKey: item.storageKey, name: r.storedName ?? item.name ?? '', existingId: r.documentId ?? '', error: null });
    }
    this.set({ importConflicts: conflicts });
    if (conflicts.length === 0) writeFlag(this.opts.migratedFlag);
  }

  /** The user's choice for one colliding legacy preset. */
  async resolveImport(storageKey: string, resolution: 'keep-both' | 'replace'): Promise<void> {
    const item = this.pendingImports.get(storageKey);
    if (!item) return;
    try {
      const { results } = await this.api.import(this.opts.family, [{ ...item, resolution }]);
      if (results[0]?.outcome === 'name-conflict') throw new Error('It still collides; refresh and choose again');
      this.pendingImports.delete(storageKey);
      const left = this.snap.importConflicts.filter(c => c.storageKey !== storageKey);
      this.set({ importConflicts: left });
      if (left.length === 0) writeFlag(this.opts.migratedFlag);
      await this.refresh();
    } catch (err) {
      this.set({ importConflicts: this.snap.importConflicts.map(c => c.storageKey === storageKey ? { ...c, error: message(err) } : c) });
    }
  }

  /** Re-read the family. Entries with unsaved local state keep it (their
   *  revision and server copy are brought up to date for reapply); saved
   *  entries take the server's version. */
  async refresh(): Promise<void> {
    const { documents } = await this.api.list<B>(this.opts.family);
    const byId = new Map(documents.map(d => [d.id, d]));
    const local = this.snap.entries.filter(e => e.status !== 'saved');
    const localById = new Map(local.filter(e => e.id).map(e => [e.id!, e]));
    const entries: PresetEntry<B>[] = documents.map(d => {
      const mine = localById.get(d.id);
      return mine ? { ...mine, revision: d.revision, serverBody: d.body } : fromDocument(d);
    });
    for (const e of local) {
      if (!e.id) entries.push(e);
      else if (!byId.has(e.id)) entries.push({ ...e, serverBody: null, error: e.error ?? 'This preset was deleted elsewhere' });
    }
    this.set({ entries });
  }

  // ── Edits ─────────────────────────────────────────────────────────────────

  /** Add a preset. Resolves to its key once the create settles (the
   *  document id on success). */
  async create(body: B): Promise<string> {
    const key = `tmp-${++this.tempSeq}`;
    this.set({ entries: [...this.snap.entries, {
      key, id: null, revision: 0, body, serverBody: null, status: 'creating', pendingOp: 'create', error: null,
    }] });
    return this.runCreate(key);
  }

  private async runCreate(key: string): Promise<string> {
    const e = this.entry(key)!;
    this.patch(key, { status: 'creating', error: null });
    let doc: TypedDocument<B>;
    try {
      ({ document: doc } = await this.api.create<B>(this.opts.family, e.body));
    } catch (err) {
      if (this.deleteAfterCreate.delete(key)) { this.drop(key); return key; }
      this.patch(key, { status: 'failed', pendingOp: 'create', error: `Save failed: ${message(err)}` });
      return key;
    }
    // Re-key by the document id now that it exists.
    this.set({ entries: this.snap.entries.map(x => x.key === key ? {
      ...x, key: doc.id, id: doc.id, revision: doc.revision, serverBody: doc.body, status: 'saved', pendingOp: null, error: null,
    } : x) });
    if (this.deleteAfterCreate.delete(key)) await this.remove(doc.id);
    return doc.id;
  }

  /** Change a preset's body. While its save is in flight the latest edit
   *  waits and is sent next; while it is in conflict nothing is sent. */
  update(key: string, body: B): void {
    const e = this.entry(key);
    if (!e) return;
    if (!e.id) { // not created yet: the create will carry the new body
      this.patch(key, { body });
      if (e.status === 'failed') this.patch(key, { pendingOp: 'create' });
      return;
    }
    this.patch(key, { body });
    if (e.status === 'conflict') { this.patch(key, { pendingOp: 'update' }); return; }
    if (this.inFlight.has(key)) { this.editedDuringSave.add(key); return; }
    void this.runUpdate(key);
  }

  private async runUpdate(key: string): Promise<void> {
    this.inFlight.add(key);
    try {
      for (;;) {
        const e = this.entry(key);
        if (!e?.id || e.status === 'conflict') return;
        this.editedDuringSave.delete(key);
        this.patch(key, { status: 'saving', pendingOp: 'update', error: null });
        try {
          const { document } = await this.api.update<B>(this.opts.family, e.id, e.revision, e.body);
          this.patch(key, { revision: document.revision, serverBody: document.body, status: 'saved', pendingOp: null });
        } catch (err) {
          await this.failed(key, 'update', err);
          return;
        }
        if (!this.editedDuringSave.has(key)) return;
      }
    } finally {
      this.inFlight.delete(key);
    }
  }

  /** Delete a preset. One whose create is still in flight is deleted once
   *  that create lands; a failure there is shown on the entry. */
  async remove(key: string): Promise<void> {
    const e = this.entry(key);
    if (!e || e.status === 'deleting') return;
    if (!e.id) {
      if (e.status === 'creating') { this.deleteAfterCreate.add(key); this.patch(key, { status: 'deleting' }); }
      else this.drop(key); // a create that failed: nothing on the server
      return;
    }
    this.patch(key, { status: 'deleting', pendingOp: 'delete', error: null });
    try {
      await this.api.remove(this.opts.family, e.id, e.revision);
      this.drop(key);
    } catch (err) {
      await this.failed(key, 'delete', err);
    }
  }

  /** Send the entry's pending create, update or delete again, at the
   *  revision last read from the server. */
  async reapply(key: string): Promise<void> {
    const e = this.entry(key);
    if (!e?.pendingOp) return;
    if (e.pendingOp === 'create' || (e.pendingOp === 'update' && e.serverBody === null)) {
      // Never created, or deleted elsewhere: create it again from the edit.
      this.patch(key, { id: null, revision: 0, status: 'creating', pendingOp: 'create' });
      await this.runCreate(key);
      return;
    }
    if (e.pendingOp === 'delete') {
      if (e.serverBody === null) { this.drop(key); return; } // already gone
      await this.remove(key);
      return;
    }
    this.patch(key, { status: 'saving' });
    await this.runUpdate(key);
  }

  /** Drop the entry's unsaved change and show the server's version. */
  discard(key: string): void {
    const e = this.entry(key);
    if (!e) return;
    if (!e.id || e.serverBody === null) { this.drop(key); return; }
    this.patch(key, { body: e.serverBody, status: 'saved', pendingOp: null, error: null });
  }

  /** A failed write: a conflict when the document moved on (or is gone),
   *  read back so reapply starts from the current revision; otherwise a
   *  retryable failure. Either way the user's body stays. */
  private async failed(key: string, op: 'update' | 'delete', err: unknown): Promise<void> {
    const e = this.entry(key);
    if (!e?.id) return;
    const status = err instanceof WorkflowRequestError ? err.status : 0;
    if (status !== 409 && status !== 404) {
      this.patch(key, { status: 'failed', pendingOp: op, error: `${op === 'delete' ? 'Delete' : 'Save'} failed: ${message(err)}` });
      return;
    }
    let current: TypedDocument<B> | undefined;
    try { current = (await this.api.list<B>(this.opts.family)).documents.find(d => d.id === e.id); } catch { /* keep what we know */ }
    this.patch(key, {
      status: 'conflict', pendingOp: op,
      ...(current ? { revision: current.revision, serverBody: current.body } : { serverBody: null }),
      error: current ? 'Changed elsewhere. Refresh to see it, or reapply your change over it.' : 'This preset was deleted elsewhere.',
    });
  }
}

function fromDocument<B>(d: TypedDocument<B>): PresetEntry<B> {
  return { key: d.id, id: d.id, revision: d.revision, body: d.body, serverBody: d.body, status: 'saved', pendingOp: null, error: null };
}

function readFlag(key: string): boolean {
  try { return !!localStorage.getItem(key); } catch { return false; }
}

function writeFlag(key: string): void {
  try { localStorage.setItem(key, '1'); } catch { /* storage unavailable: the import is retried, and is idempotent */ }
}
