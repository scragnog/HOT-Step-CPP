import { createHash } from 'node:crypto';
import type Database from 'better-sqlite3';
import { importDraftSchema, playlistCommandSchema, playlistSchema, studioDraftSchema, type PlaylistBody, type PlaylistItem, type StudioDraftBody } from '../../contracts/studioDrafts.js';
import { DRAFT_SOURCE_FIELDS, DRAFT_SOURCE_RESULT_FIELDS, RAW_STRING_DRAFT_KEYS, studioForDraftKey, validDraftField } from '../../contracts/studioDraftFields.js';
import { TypedDocuments, WorkflowDocuments, registerDocumentKind } from '../workflows/revisions.js';
import { WorkflowError } from '../workflows/workflowJobs.js';
import { resolveAudioAsset } from '../assets/audioAssets.js';

export const playlistKind = registerDocumentKind({ kind: 'studio.playlist', scope: 'user', schemaVersion: 1, schema: playlistSchema });
export const draftKind = registerDocumentKind({ kind: 'studio.draft', scope: 'user', schemaVersion: 1, schema: studioDraftSchema });

/** The studio a draft key belongs to (contracts/studioDraftFields.ts). */
export const studioForKey = studioForDraftKey;

export function validateDraft(body: StudioDraftBody): void {
  for (const key of Object.keys(body.fields)) {
    if (studioForKey(key) !== body.studio) throw new WorkflowError(400, `Unsupported ${body.studio} draft field: ${key}`);
    if (!validDraftField(key, body.fields[key])) throw new WorkflowError(400, `Invalid value for ${key}`);
  }
  const assetKey = body.studio === 'cover' ? 'cover-studio-sourceAssetId' : body.studio === 'repaint' ? 'hs-repaint-sourceAssetId' : null;
  if (assetKey && body.sourceAssetId !== undefined && body.fields[assetKey] !== undefined && body.sourceAssetId !== body.fields[assetKey])
    throw new WorkflowError(400, 'Source asset identity disagrees with the draft fields');
  if (body.studio === 'cover' && body.sourceSongId !== undefined && body.fields['cover-studio-sourceSongId'] !== undefined &&
    body.sourceSongId !== body.fields['cover-studio-sourceSongId'])
    throw new WorkflowError(400, 'Source song identity disagrees with the draft fields');
}

/** A source switch makes analysis and generated caption text from the old
 * source inapplicable. Keep the user's own lyrics and tuning choices. */
export function withoutStaleSourceResults(current: StudioDraftBody, next: StudioDraftBody): StudioDraftBody {
  const sourceKeys = DRAFT_SOURCE_FIELDS[current.studio] ?? [];
  const changed = current.sourceAssetId !== next.sourceAssetId || current.sourceSongId !== next.sourceSongId ||
    current.sourceRevision !== next.sourceRevision || sourceKeys.some(key =>
      JSON.stringify(current.fields[key]) !== JSON.stringify(next.fields[key]));
  if (!changed) return next;
  const fields = { ...next.fields };
  for (const key of DRAFT_SOURCE_RESULT_FIELDS[next.studio] ?? []) delete fields[key];
  return { ...next, fields };
}

export type { PlaylistCommand } from '../../contracts/studioDrafts.js';

export class StudioDrafts {
  readonly playlists: TypedDocuments<PlaylistBody>;
  readonly drafts: TypedDocuments<StudioDraftBody>;
  constructor(private readonly db: Database.Database, private readonly workflowDocs = new WorkflowDocuments(db)) {
    this.playlists = new TypedDocuments(workflowDocs, db, playlistKind);
    this.drafts = new TypedDocuments(workflowDocs, db, draftKind);
  }

  playlist(userId: string) { return this.playlists.list(userId)[0] ?? null; }

  /** Existing Cover and training-to-Create handoff IDs stay readable. The
   * original revisioned document remains the authoritative workflow input. */
  handoff(userId: string, id: string) {
    const document = this.workflowDocs.get(id, userId);
    if (document.kind !== 'cover-draft' && document.kind !== 'training-audition-create')
      throw new WorkflowError(404, `Handoff ${id} not found`);
    return document;
  }

  sourceError(userId: string, body: StudioDraftBody, dataDir: string): string | null {
    const id = body.sourceAssetId || (body.studio === 'cover' ? body.fields['cover-studio-sourceAssetId'] :
      body.studio === 'repaint' ? body.fields['hs-repaint-sourceAssetId'] : undefined);
    if (typeof id !== 'string' || !id) return null;
    try { resolveAudioAsset(this.db, id, userId, dataDir); return null; }
    catch (err) {
      if (err instanceof WorkflowError && err.status === 404) return `Source asset ${id} is unavailable. Reupload the source before continuing.`;
      throw err;
    }
  }

  command(userId: string, expectedRevision: number, input: unknown) {
    const parsed = playlistCommandSchema.safeParse(input);
    if (!parsed.success) throw new WorkflowError(400, 'Invalid playlist command');
    const cmd = parsed.data;
    return this.db.transaction(() => {
      const current = this.playlist(userId);
      if ((current?.revision ?? 0) !== expectedRevision) throw new WorkflowError(409, 'Playlist changed elsewhere. Reload and reapply your edit.', { currentRevision: current?.revision ?? 0 });
      const items: PlaylistItem[] = [...(current?.body.items ?? [])];
      if (cmd.operation === 'add') { if (!items.some(i => i.id === cmd.item.id)) items.push(cmd.item); }
      if (cmd.operation === 'remove') { const at = items.findIndex(i => i.id === cmd.id); if (at >= 0) items.splice(at, 1); }
      if (cmd.operation === 'clear') items.length = 0;
      if (cmd.operation === 'reorder') {
        if (cmd.ids.length !== items.length || new Set(cmd.ids).size !== items.length || cmd.ids.some(id => !items.some(i => i.id === id)))
          throw new WorkflowError(400, 'Reorder must name every current item exactly once');
        const byId = new Map(items.map(i => [i.id, i]));
        items.splice(0, items.length, ...cmd.ids.map(id => byId.get(id)!));
      }
      if (cmd.operation === 'update') {
        const at = items.findIndex(i => i.id === cmd.id);
        if (at >= 0) items[at] = { ...items[at], ...cmd.patch, id: cmd.id } as PlaylistItem;
      }
      return current
        ? this.playlists.update(current.id, userId, expectedRevision, { items }, { origin: 'client' })
        : this.playlists.create(userId, { items }, { origin: 'client' });
    })();
  }

  importValue(userId: string, input: unknown) {
    const parsed = importDraftSchema.safeParse(input);
    if (!parsed.success) throw new WorkflowError(400, 'Invalid import request');
    const { storageKey, raw, sourceHash, expectedRevision, resolution } = parsed.data;
    if (`sha256:${createHash('sha256').update(raw).digest('hex')}` !== sourceHash) throw new WorkflowError(400, 'Import hash does not match source value');
    const source = { storageKey, sourceHash };
    let value: unknown;
    try { value = RAW_STRING_DRAFT_KEYS.has(storageKey) ? raw : JSON.parse(raw); } catch { throw new WorkflowError(400, 'Invalid browser value'); }
    if (storageKey === 'lireek-playQueue') {
      const body = playlistSchema.safeParse({ items: value });
      if (!body.success) throw new WorkflowError(400, 'Invalid playlist snapshot');
      const known = this.playlists.receipts(userId).find(r => r.storageKey === storageKey && r.sourceHash === sourceHash);
      if (known) return this.playlists.importOnce(userId, source, body.data);
      const current = this.playlist(userId);
      if ((current?.revision ?? 0) !== expectedRevision) throw new WorkflowError(409, 'Playlist changed elsewhere', { currentRevision: current?.revision ?? 0 });
      if (current) {
        if (resolution !== 'replace') throw new WorkflowError(409, 'Playlist exists. Choose replace to import this snapshot.');
        return this.replaceImport(this.playlists, userId, source, current.id, expectedRevision, body.data);
      }
      return this.playlists.importOnce(userId, source, body.data);
    }
    const studio = studioForKey(storageKey);
    if (!studio) throw new WorkflowError(400, `Unsupported import key: ${storageKey}`);
    const body = { studio, fields: { [storageKey]: value } };
    validateDraft(body);
    const receipts = this.drafts.receipts(userId).filter(r => r.storageKey === storageKey);
    if (receipts.some(r => r.sourceHash === sourceHash)) return this.drafts.importOnce(userId, source, body);
    const existing = this.drafts.list(userId).find(d => Object.hasOwn(d.body.fields, storageKey));
    if (existing) {
      if (existing.revision !== expectedRevision) throw new WorkflowError(409, 'Draft changed elsewhere', { currentRevision: existing.revision });
      if (JSON.stringify(existing.body.fields[storageKey]) === JSON.stringify(value))
        return this.receiptForExisting(this.drafts, userId, source, existing);
      if (!resolution) throw new WorkflowError(409, 'This browser key was imported with different content. Choose keep-both or replace.');
      if (resolution === 'replace') {
        return this.replaceImport(this.drafts, userId, source, existing.id, expectedRevision, body);
      }
    }
    if (!existing && expectedRevision !== 0) throw new WorkflowError(409, 'New draft import requires expectedRevision 0');
    return this.drafts.importOnce(userId, source, body);
  }

  private replaceImport<T extends Record<string, unknown>>(store: TypedDocuments<T>, userId: string,
    source: { storageKey: string; sourceHash: string }, id: string, expectedRevision: number, body: T) {
    return this.db.transaction(() => {
      const document = store.update(id, userId, expectedRevision, body, { origin: 'import', importedFrom: source });
      const importedAt = Date.now();
      this.db.prepare('INSERT INTO document_import_receipts (user_id, kind, storage_key, source_hash, document_id, imported_at) VALUES (?, ?, ?, ?, ?, ?)')
        .run(store.owner(userId), store.def.kind, source.storageKey, source.sourceHash, document.id, importedAt);
      return { receipt: { kind: store.def.kind, ...source, documentId: id, importedAt }, document, created: false };
    })();
  }

  private receiptForExisting<T extends Record<string, unknown>>(store: TypedDocuments<T>, userId: string,
    source: { storageKey: string; sourceHash: string }, document: { id: string; revision: number; body: T }) {
    return this.db.transaction(() => {
      const current = store.get(document.id, userId);
      if (current.revision !== document.revision) throw new WorkflowError(409, 'Draft changed elsewhere', { currentRevision: current.revision });
      const importedAt = Date.now();
      this.db.prepare('INSERT INTO document_import_receipts (user_id, kind, storage_key, source_hash, document_id, imported_at) VALUES (?, ?, ?, ?, ?, ?)')
        .run(store.owner(userId), store.def.kind, source.storageKey, source.sourceHash, document.id, importedAt);
      return { receipt: { kind: store.def.kind, ...source, documentId: document.id, importedAt }, document: current, created: false };
    })();
  }
}
