import { changed, songsChanged } from '../queries/cover-lyrics/reads';
import { waitForSnapshot } from '../queries/cover-lyrics/wait';
import { workflowOptions } from '../queries/cover-lyrics/workflows';
import { apiClient } from './httpClient';
import type { WorkflowReplay } from '../../../server/src/contracts/workflow';
import { coverLyricWorkflowApi as workflowApi } from '../queries/cover-lyrics/workflows';
import type { WorkflowJob } from '../../../server/src/contracts/workflow';
import type { CoverLeadSheetSource, CoverLeadSheetResult } from '../../../server/src/contracts/studioWorkflows';

export interface CoverDraftResult {
  documentId: string;
  revision: number;
  metadata?: { artist: string; title: string; album: string; duration: number | null };
  analysis?: { bpm: number; key: string; scale?: string } | null;
  abc?: string;
  scoreSource?: string;
  caption?: string;
  leadSheetSource?: CoverLeadSheetSource;
  leadSheet?: CoverLeadSheetResult;
}

export async function waitForCoverJob(jobId: string, cancelled: () => boolean,
  progress?: (job: WorkflowJob) => void): Promise<WorkflowJob> {
  const result = await waitForSnapshot(workflowOptions(jobId).queryKey,
    signal => apiClient.get<WorkflowReplay>(`/workflows/jobs/${encodeURIComponent(jobId)}?after=0`, { signal }),
    data => !['succeeded', 'failed', 'cancelled', 'interrupted'].includes(data.job.status),
    data => {
      if (cancelled()) {
        void workflowApi.cancel(jobId).catch(() => {});
        throw new Error('Cover operation cancelled');
      }
      progress?.(data.job);
    }, { deliverRepeats: true });
  if (result.job.status !== 'succeeded') throw new Error(result.job.error || `Cover operation ${result.job.status}`);
  if (result.job.kind === 'cover-render') await songsChanged();
  return result.job;
}

export const coverWorkflowApi = {
  open: (assetId: string, cached?: {
    metadata: { artist: string; title: string; album: string; duration: number | null };
    analysis: { bpm: number; key: string; scale?: string };
  }, leadSheetSource?: CoverLeadSheetSource) => workflowApi.submit('cover-open', crypto.randomUUID(), { assetId, cached, leadSheetSource }),
  source: (documentId: string, revision: number, leadSheetSource: CoverLeadSheetSource, assetId?: string) =>
    workflowApi.submit('cover-source', crypto.randomUUID(), { documentId, revision, leadSheetSource, assetId }),
  transcribe: (documentId: string, revision: number, force: boolean) =>
    workflowApi.submit('cover-transcribe', crypto.randomUUID(), { documentId, revision, force }),
  caption: (documentId: string, revision: number, artistId: number,
    provider: string, model: string, force: boolean) =>
    workflowApi.submit('cover-caption', crypto.randomUUID(),
      { documentId, revision, artistId, provider, model, force }),
  render: (input: Record<string, unknown>) =>
    workflowApi.submit('cover-render', crypto.randomUUID(), input),
  approveScore: async (documentId: string, revision: number, abc: string) => {
    const data = await apiClient.post<{ document: { id: string; revision: number }; sourceId: string; sourceLabel: string; abc: string }>(
      `/yue2-cover/drafts/${encodeURIComponent(documentId)}/approve-score`, { revision, abc });
    await Promise.all([changed('drafts'), changed('cover')]);
    return data;
  },
  // Silent: CoverStudio's score sheet renders the save failure itself (its
  // "Save to dataset:" line is the sole notifier).
  saveScoreDetails: async (documentId: string, revision: number,
    abc: string, factor: number) => {
    const data = await apiClient.post<{ saved: true; bpm: number; key: string }>(
      `/yue2-cover/drafts/${encodeURIComponent(documentId)}/save-score-details`, { revision, abc, factor }, { silent: true });
    await changed('cover');
    return data;
  },
};
