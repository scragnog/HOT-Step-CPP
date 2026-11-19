// The audition handoff creates a server draft before navigating to Create.
// Create loads that draft by id only after the user chooses to apply it.
import type { AuditionPreview, AuditionSideResult } from '@/services/trainingApi';
import { snapshotFor, trainingOperation } from '@/services/trainingOperations';
import { queryClient } from '@/queries/client';
import { trainingQueries } from '@/queries/training-data/queries';
import { newId } from '@/utils/id';

export type AuditionRenderCell = 'bare' | 'adapter';

export async function sendAuditionToCustomGen(preview: AuditionPreview, side: AuditionSideResult,
  cell: AuditionRenderCell): Promise<string> {
  // The dataset's current revision, always from this PC: audition previews
  // and the draft they feed are local, whatever "Train on" says.
  const dataset = await queryClient.query({ ...trainingQueries.dataset(null, preview.datasetId), staleTime: 0 });
  if (typeof dataset.updatedAt !== 'string') throw new Error('Could not load the training dataset revision');
  const result = await trainingOperation<{ draftId: string }>('review', '/draft', {
    body: snapshotFor({ kind: 'review:audition-draft', idempotencyKey: newId(),
      worker: { kind: 'local' }, dataset: { id: preview.datasetId, revision: dataset.updatedAt },
      sources: [{ kind: 'audition-preview', id: preview.previewId, revision: preview.createdAt }],
      payload: { datasetId: preview.datasetId, previewId: preview.previewId, slot: side.slot, cell } }),
  });
  window.history.pushState({}, '', `/?trainingDraft=${encodeURIComponent(result.draftId)}`);
  window.dispatchEvent(new PopStateEvent('popstate'));
  return result.draftId;
}
