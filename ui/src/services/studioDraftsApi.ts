// studio-drafts domain client: reads go through the shared query cache
// (readSnapshot), writes hit the transport directly and then invalidate the
// domain's keys (changed). The bearer comes from the auth session accessor.

import type { PlaylistBody, StudioDraftBody } from '../../../server/src/contracts/studioDrafts';
import type { DocumentImportReceipt, TypedDocument, WorkflowDocument } from '../../../server/src/contracts/workflow';
import { apiClient, type ClientOptions, type ReadOptions } from './httpClient';
import { readSnapshot, changed } from '@/queries/cover-lyrics/reads';
import { getSessionEpoch } from '@/queries/keys';

async function request<T>(url: string, method = 'GET', body?: unknown, options: ClientOptions = {}): Promise<T> {
  const target = { session: getSessionEpoch(), api: apiClient.baseUrl };
  const path = url.replace(/^\/api/, '');
  if (method === 'GET') return readSnapshot<T>('drafts', path, options);
  // POST/PUT writes are silent: every caller renders its own inline error —
  // the draft mirror's onError line and the playlist store's saveError status.
  // `remove` has no production caller yet, so it keeps the global toast until
  // a caller without inline UI is added.
  const write = method === 'DELETE' ? options : { ...options, silent: true };
  const result = method === 'DELETE' ? await apiClient.delete<T>(path, write)
    : method === 'PUT' ? await apiClient.put<T>(path, body, write)
    : await apiClient.post<T>(path, body, write);
  await changed('drafts', target);
  return result;
}

const path = '/api/studio-drafts';
const id = (value: string) => encodeURIComponent(value);
export type PlaylistDocument = TypedDocument<PlaylistBody>;
export type StudioDraftDocument = TypedDocument<StudioDraftBody>;

export const studioDraftsApi = {
  playlist: (options: ReadOptions = {}) => request<{ document: PlaylistDocument | null }>(`${path}/playlist`, 'GET', undefined, options),
  playlistCommand: (expectedRevision: number, command: unknown) =>
    request<{ document: PlaylistDocument }>(`${path}/playlist/commands`, 'POST', { expectedRevision, command }),
  handoff: (documentId: string, options: ReadOptions = {}) =>
    request<{ document: WorkflowDocument }>(`${path}/handoffs/${id(documentId)}`, 'GET', undefined, options),
  list: (studio?: StudioDraftBody['studio'], options: ReadOptions = {}) =>
    request<{ documents: StudioDraftDocument[] }>(`${path}/drafts${studio ? `?studio=${id(studio)}` : ''}`, 'GET', undefined, options),
  create: (body: StudioDraftBody) =>
    request<{ document: StudioDraftDocument }>(`${path}/drafts`, 'POST', { body }),
  get: (documentId: string, options: ReadOptions = {}) =>
    request<{ document: StudioDraftDocument; sourceError: string | null }>(`${path}/drafts/${id(documentId)}`, 'GET', undefined, options),
  // Silent: the autosave mirror handles every failure itself (a 404 or 409
  // just starts a new draft) and shows its own message when it can't.
  update: (documentId: string, expectedRevision: number, body: StudioDraftBody) =>
    request<{ document: StudioDraftDocument }>(`${path}/drafts/${id(documentId)}`, 'PUT', { expectedRevision, body }, { silent: true }),
  remove: (documentId: string, expectedRevision: number) =>
    request<{ removed: true }>(`${path}/drafts/${id(documentId)}?expectedRevision=${expectedRevision}`, 'DELETE'),
  importValue: (input: { storageKey: string; raw: string; sourceHash: string; schemaVersion: 1; expectedRevision: number; resolution?: 'keep-both' | 'replace' }) =>
    request<{ receipt: DocumentImportReceipt; document: PlaylistDocument | StudioDraftDocument | null; created: boolean }>(`${path}/import`, 'POST', input),
};
