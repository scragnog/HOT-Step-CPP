import type { PlaylistBody, StudioDraftBody } from '../../../server/src/contracts/studioDrafts';
import type { DocumentImportReceipt, TypedDocument, WorkflowDocument } from '../../../server/src/contracts/workflow';
import { revisionedRequest } from './workflowApi';

const path = '/api/studio-drafts';
const id = (value: string) => encodeURIComponent(value);
export type PlaylistDocument = TypedDocument<PlaylistBody>;
export type StudioDraftDocument = TypedDocument<StudioDraftBody>;

export const studioDraftsApi = {
  playlist: (token: string) => revisionedRequest<{ document: PlaylistDocument | null }>(token, `${path}/playlist`),
  playlistCommand: (token: string, expectedRevision: number, command: unknown) =>
    revisionedRequest<{ document: PlaylistDocument }>(token, `${path}/playlist/commands`, 'POST', { expectedRevision, command }),
  handoff: (token: string, documentId: string) =>
    revisionedRequest<{ document: WorkflowDocument }>(token, `${path}/handoffs/${id(documentId)}`),
  list: (token: string, studio?: StudioDraftBody['studio']) =>
    revisionedRequest<{ documents: StudioDraftDocument[] }>(token, `${path}/drafts${studio ? `?studio=${id(studio)}` : ''}`),
  create: (token: string, body: StudioDraftBody) =>
    revisionedRequest<{ document: StudioDraftDocument }>(token, `${path}/drafts`, 'POST', { body }),
  get: (token: string, documentId: string) =>
    revisionedRequest<{ document: StudioDraftDocument; sourceError: string | null }>(token, `${path}/drafts/${id(documentId)}`),
  update: (token: string, documentId: string, expectedRevision: number, body: StudioDraftBody) =>
    revisionedRequest<{ document: StudioDraftDocument }>(token, `${path}/drafts/${id(documentId)}`, 'PUT', { expectedRevision, body }),
  remove: (token: string, documentId: string, expectedRevision: number) =>
    revisionedRequest<{ removed: true }>(token, `${path}/drafts/${id(documentId)}?expectedRevision=${expectedRevision}`, 'DELETE'),
  importValue: (token: string, input: { storageKey: string; raw: string; sourceHash: string; schemaVersion: 1; expectedRevision: number; resolution?: 'keep-both' | 'replace' }) =>
    revisionedRequest<{ receipt: DocumentImportReceipt; document: PlaylistDocument | StudioDraftDocument | null; created: boolean }>(token, `${path}/import`, 'POST', input),
};
