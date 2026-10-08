import { createHash } from 'node:crypto';
import type Database from 'better-sqlite3';
import { z } from 'zod/v4';
import { importDraftSchema, playlistSchema, studioDraftSchema, type PlaylistBody, type PlaylistItem, type StudioDraftBody } from '../../contracts/studioDrafts.js';
import { TypedDocuments, WorkflowDocuments, registerDocumentKind } from '../workflows/revisions.js';
import { WorkflowError } from '../workflows/workflowJobs.js';
import { resolveAudioAsset } from '../assets/audioAssets.js';

export const playlistKind = registerDocumentKind({ kind: 'studio.playlist', scope: 'user', schemaVersion: 1, schema: playlistSchema });
export const draftKind = registerDocumentKind({ kind: 'studio.draft', scope: 'user', schemaVersion: 1, schema: studioDraftSchema });

const CREATE = new Set('caption lyrics negative-prompt instrumental lora-trigger beat-intro intro-bars title artist subject bpm keyScale timeSignature duration vocalLanguage vocalGender sourceLatentUrl'.split(' ').map(x => `hs-${x}`));
const COVER = new Set('sourceFileName sourceAudioUrl sourceAssetId sourceSongId metadata analysis songArtist songTitle lyrics lyricsSource datasetAnalysis selectedArtistId selectedPreset artistCaption audioCoverStrength coverNoiseStrength coverNoiseMethod tempoScale pitchShift bpmCorrection bpmOverride keyOverride noFsq coverInstrumental sourceLatentUrl coverVocalLanguage coverTimbreOverride sepLevel'.split(' ').map(x => `cover-studio-${x}`));
const REPAINT = new Set('sourceSong sourceAssetId sourceAudioUrl sourceName regionStart regionEnd lyrics repaintMode crossfadeFrames styleCaption'.split(' ').map(x => `hs-repaint-${x}`));
const STORM = new Set('caption lyrics neg lora instrumental beat-intro intro-bars bpm duration-v2 deck-a-caption deck-a-lyrics deck-a-bpm deck-b-caption deck-b-lyrics deck-b-bpm'.split(' ').map(x => `hs-storm-${x}`));
const STEM = new Set('hs-stem-sourceUrl hs-stem-sourceFile hs-stem-sepLevel hs-stem-extractModel'.split(' '));
const BUILDER = new Set('hs-sb-sourceUrl hs-sb-sourceRef hs-sb-sourceFile hs-sb-model'.split(' '));
const CAPTION = /^(hs-mm3CaptionSource:[^:]+|hs-mm3CaptionSources|hs-yue2CaptionSource:ds:[^:]+|hs-yue2CaptionSource:ds:song:[^:]+:[^:]+|hs-yue2CaptionDataset|hs-yue2CaptionSources)$/;
const RAW = new Set([...STEM, ...BUILDER].filter(x => x !== 'hs-sb-sourceRef'));
const BOOLEAN = new Set('hs-instrumental hs-beat-intro cover-studio-datasetAnalysis cover-studio-noFsq cover-studio-coverInstrumental hs-storm-instrumental hs-storm-beat-intro'.split(' '));
const NUMBER = new Set('hs-bpm hs-duration hs-intro-bars cover-studio-audioCoverStrength cover-studio-coverNoiseStrength cover-studio-tempoScale cover-studio-pitchShift cover-studio-bpmCorrection cover-studio-sepLevel hs-repaint-regionStart hs-repaint-regionEnd hs-repaint-crossfadeFrames hs-storm-intro-bars hs-storm-bpm hs-storm-duration-v2 hs-storm-deck-a-bpm hs-storm-deck-b-bpm'.split(' '));
const NULLABLE_NUMBER = new Set('cover-studio-selectedArtistId cover-studio-bpmOverride'.split(' '));
const NULLABLE_STRING = new Set('cover-studio-lyricsSource cover-studio-keyOverride'.split(' '));
const OBJECT = new Set('cover-studio-metadata cover-studio-analysis cover-studio-selectedPreset hs-repaint-sourceSong hs-sb-sourceRef'.split(' '));

function validField(key: string, value: unknown): boolean {
  if (BOOLEAN.has(key)) return typeof value === 'boolean';
  if (NUMBER.has(key)) return typeof value === 'number' && Number.isFinite(value);
  if (NULLABLE_NUMBER.has(key)) return value === null || (typeof value === 'number' && Number.isFinite(value));
  if (NULLABLE_STRING.has(key)) return value === null || typeof value === 'string';
  if (OBJECT.has(key) || key.startsWith('hs-mm3CaptionSource:') || key === 'hs-mm3CaptionSources' ||
    key.startsWith('hs-yue2CaptionSource:') || key === 'hs-yue2CaptionSources')
    return value === null || (typeof value === 'object' && !Array.isArray(value));
  return typeof value === 'string';
}

export function studioForKey(key: string): StudioDraftBody['studio'] | null {
  if (CREATE.has(key) || CAPTION.test(key)) return 'create';
  if (COVER.has(key)) return 'cover';
  if (REPAINT.has(key)) return 'repaint';
  if (STORM.has(key)) return 'storm';
  if (STEM.has(key)) return 'stem-studio';
  if (BUILDER.has(key)) return 'stem-builder';
  return null;
}

export function validateDraft(body: StudioDraftBody): void {
  for (const key of Object.keys(body.fields)) {
    if (studioForKey(key) !== body.studio) throw new WorkflowError(400, `Unsupported ${body.studio} draft field: ${key}`);
    if (!validField(key, body.fields[key])) throw new WorkflowError(400, `Invalid value for ${key}`);
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
  const sourceKeys = current.studio === 'cover'
    ? ['cover-studio-sourceAssetId', 'cover-studio-sourceSongId', 'cover-studio-sourceAudioUrl']
    : current.studio === 'repaint'
      ? ['hs-repaint-sourceAssetId', 'hs-repaint-sourceSong', 'hs-repaint-sourceAudioUrl']
      : current.studio === 'create' ? ['hs-sourceLatentUrl'] : [];
  const changed = current.sourceAssetId !== next.sourceAssetId || current.sourceSongId !== next.sourceSongId ||
    current.sourceRevision !== next.sourceRevision || sourceKeys.some(key =>
      JSON.stringify(current.fields[key]) !== JSON.stringify(next.fields[key]));
  if (!changed) return next;
  const fields = { ...next.fields };
  if (next.studio === 'cover') {
    for (const key of ['cover-studio-analysis', 'cover-studio-metadata', 'cover-studio-artistCaption', 'cover-studio-lyricsSource']) delete fields[key];
  }
  if (next.studio === 'create') {
    for (const key of ['hs-mm3CaptionSources', 'hs-yue2CaptionSources']) delete fields[key];
  }
  return { ...next, fields };
}

const commandSchema = z.discriminatedUnion('operation', [
  z.object({ operation: z.literal('add'), item: playlistSchema.shape.items.element }),
  z.object({ operation: z.literal('remove'), id: z.string().min(1) }),
  z.object({ operation: z.literal('clear') }),
  z.object({ operation: z.literal('reorder'), ids: z.array(z.string().min(1)) }),
  z.object({ operation: z.literal('update'), id: z.string().min(1), patch: z.record(z.string(), z.unknown()) }),
]);
export type PlaylistCommand = z.infer<typeof commandSchema>;

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
    const parsed = commandSchema.safeParse(input);
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
    try { value = RAW.has(storageKey) ? raw : JSON.parse(raw); } catch { throw new WorkflowError(400, 'Invalid browser value'); }
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
