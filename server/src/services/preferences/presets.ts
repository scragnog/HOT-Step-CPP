// services/preferences/presets.ts — import logic shared by every family in
// routes/preferences.ts: a named-preset collection (VST chain, scale
// override, AI continue, YuE2 joint) or a singleton current-value document
// (AI continue template, STORM tuning).
//
// All of it runs as the fixed LOCAL_OWNER: every kind here is
// installation-scoped (contracts/preferences.ts), so TypedDocuments.owner()
// maps any caller to the one shared owner regardless of this value.
import type { TypedDocuments } from '../workflows/revisions.js';
import { WorkflowError } from '../workflows/workflowJobs.js';
import type { ImportPreferenceItem, ImportPreferenceResult } from '../../contracts/preferences.js';

export const LOCAL_OWNER = 'local';

const deepEqual = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);

/** Import one browser value into a named-preset family.
 *
 *  Dedup is (storageKey, sourceHash): retrying the same source returns
 *  'unchanged' without writing again. A different value under a name that
 *  already exists needs an explicit `resolution` — 'replace' updates that
 *  document in place (atomic, guarded by its current revision); 'keep-both'
 *  imports it as a second, separate document sharing the display name.
 *  Without a resolution the import is refused and nothing is written, so a
 *  partial batch can be retried safely. */
export function importPreset<T extends Record<string, unknown>>(
  td: TypedDocuments<T>, nameOf: (body: T) => string, item: ImportPreferenceItem,
): ImportPreferenceResult {
  const name = item.name ?? nameOf(item.body as T);
  const existing = td.list(LOCAL_OWNER).find(d => nameOf(d.body) === name);
  const provenance = { origin: 'import' as const, importedFrom: { storageKey: item.storageKey, sourceHash: item.sourceHash } };
  if (existing && !deepEqual(existing.body, item.body)) {
    if (!item.resolution) return { storageKey: item.storageKey, outcome: 'name-conflict', documentId: existing.id, storedName: name };
    if (item.resolution === 'replace') {
      const updated = td.update(existing.id, LOCAL_OWNER, existing.revision, item.body, provenance);
      return { storageKey: item.storageKey, outcome: 'replaced', documentId: updated.id, storedName: name };
    }
    // 'keep-both' falls through to importOnce below, landing as a second document under the same name.
  }
  const { receipt, document, created } = td.importOnce(LOCAL_OWNER, { storageKey: item.storageKey, sourceHash: item.sourceHash }, item.body, provenance);
  return { storageKey: item.storageKey, outcome: created ? 'imported' : 'unchanged', documentId: document?.id ?? receipt.documentId, storedName: name };
}

/** The installation's one document for a singleton family, or null if none
 *  has been created yet. */
export function getSingleton<T extends Record<string, unknown>>(td: TypedDocuments<T>) {
  return td.list(LOCAL_OWNER)[0] ?? null;
}

/** Create the singleton if none exists, else update it. `expectedRevision`
 *  is required once a document exists (the usual stale-write guard); it is
 *  ignored for the first create. */
export function upsertSingleton<T extends Record<string, unknown>>(
  td: TypedDocuments<T>, expectedRevision: number | undefined, body: unknown,
) {
  const existing = getSingleton(td);
  if (!existing) return td.create(LOCAL_OWNER, body, { origin: 'client' });
  if (expectedRevision === undefined) throw new WorkflowError(400, 'expectedRevision is required to update an existing document');
  return td.update(existing.id, LOCAL_OWNER, expectedRevision, body, { origin: 'client' });
}

/** Import a browser value for a singleton family. A singleton has no "keep
 *  both": a different value than what is already stored needs 'replace',
 *  anything else is a conflict and nothing is written. */
export function importSingleton<T extends Record<string, unknown>>(
  td: TypedDocuments<T>, item: ImportPreferenceItem,
): ImportPreferenceResult {
  const existing = getSingleton(td);
  const provenance = { origin: 'import' as const, importedFrom: { storageKey: item.storageKey, sourceHash: item.sourceHash } };
  if (existing && !deepEqual(existing.body, item.body)) {
    if (item.resolution !== 'replace') return { storageKey: item.storageKey, outcome: 'name-conflict', documentId: existing.id };
    const updated = td.update(existing.id, LOCAL_OWNER, existing.revision, item.body, provenance);
    return { storageKey: item.storageKey, outcome: 'replaced', documentId: updated.id };
  }
  const { receipt, document, created } = td.importOnce(LOCAL_OWNER, { storageKey: item.storageKey, sourceHash: item.sourceHash }, item.body, provenance);
  return { storageKey: item.storageKey, outcome: created ? 'imported' : 'unchanged', documentId: document?.id ?? receipt.documentId };
}
