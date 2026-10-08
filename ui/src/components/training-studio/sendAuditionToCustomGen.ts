// The audition handoff creates a server draft before navigating to Create.
// Create loads that draft by id only after the user chooses to apply it.
import type { AuditionPreview, AuditionSideResult } from '../../services/trainingApi';
import { snapshotFor, trainingOperation } from '../../services/trainingOperations';

export type AuditionRenderCell = 'bare' | 'adapter';

export async function sendAuditionToCustomGen(preview: AuditionPreview, side: AuditionSideResult,
  cell: AuditionRenderCell, token: string): Promise<string> {
  const response = await fetch(`/api/training/datasets/${encodeURIComponent(preview.datasetId)}`);
  const dataset = await response.json();
  if (!response.ok || typeof dataset.updatedAt !== 'string')
    throw new Error(dataset.error || 'Could not load the training dataset revision');
  const result = await trainingOperation<{ draftId: string }>('review', '/draft', {
    token,
    body: snapshotFor({ kind: 'review:audition-draft', idempotencyKey: crypto.randomUUID(),
      worker: { kind: 'local' }, dataset: { id: preview.datasetId, revision: dataset.updatedAt },
      sources: [{ kind: 'audition-preview', id: preview.previewId, revision: preview.createdAt }],
      payload: { datasetId: preview.datasetId, previewId: preview.previewId, slot: side.slot, cell } }),
  });
  window.history.pushState({}, '', `/?trainingDraft=${encodeURIComponent(result.draftId)}`);
  window.dispatchEvent(new PopStateEvent('popstate'));
  return result.draftId;
}
