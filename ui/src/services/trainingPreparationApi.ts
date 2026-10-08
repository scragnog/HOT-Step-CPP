import type { TrainingSnapshot } from '../../../server/src/contracts/trainingOperation';
import type { TrainingDatasetRef, TrainingSourceRef } from '../../../server/src/contracts/trainingOperation';
import type { Yue2PreparationPayload, Yue2PreparationSummary } from '../../../server/src/contracts/trainingPreparation';
import { trainingOperation } from './trainingOperations';

export type { Yue2PreparationPayload, Yue2PreparationSummary } from '../../../server/src/contracts/trainingPreparation';

const domain = 'preparation';
type Snapshot = TrainingSnapshot<Yue2PreparationPayload>;

export async function getYue2PreparationContext(datasetId: string): Promise<{ dataset: TrainingDatasetRef; source: TrainingSourceRef }> {
  return trainingOperation(domain, `/context/${encodeURIComponent(datasetId)}`);
}

export async function startYue2Preparation(snapshot: Snapshot): Promise<Yue2PreparationSummary> {
  return (await trainingOperation<{ pipeline: Yue2PreparationSummary }>(domain, '/', { body: snapshot })).pipeline;
}

export async function listYue2Preparations(datasetId?: string): Promise<Yue2PreparationSummary[]> {
  const suffix = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
  return (await trainingOperation<{ pipelines: Yue2PreparationSummary[] }>(domain, `/${suffix}`)).pipelines;
}

export async function getYue2Preparation(id: string): Promise<Yue2PreparationSummary> {
  return (await trainingOperation<{ pipeline: Yue2PreparationSummary }>(domain, `/${encodeURIComponent(id)}`)).pipeline;
}

export async function getYue2PreparationJob<T>(id: string): Promise<T | null> {
  return (await trainingOperation<{ job: T | null }>(domain, `/${encodeURIComponent(id)}/job`)).job;
}

export async function commandYue2Preparation(id: string, action: 'pause' | 'resume' | 'cancel' | 'retry'): Promise<Yue2PreparationSummary> {
  return (await trainingOperation<{ pipeline: Yue2PreparationSummary }>(domain,
    `/${encodeURIComponent(id)}/${action}`, { method: 'POST' })).pipeline;
}
