import type { Yue2CleanupChoice, Yue2CleanupPlan } from './trainingApi';
import { snapshotFor, trainingOperation, type TrainingSnapshot } from './trainingOperations';

export interface ReviewRung {
  step: number;
  dir: string;
  blindLabel: string;
  likeness: number | null;
  corruption: number | null;
  overall: number | null;
  replansPerTake: number;
  previewCount: number;
  planFlags: number;
}
export interface ReviewLadder {
  dataset: { id: string; revision: string };
  run: string;
  runRevision: number;
  reviewRevision: string;
  status: string;
  finished: boolean;
  origin: string | null;
  method?: string;
  freezePlannerNow: boolean;
  facts: ReviewRung[];
  best: ReviewRung | null;
}
export interface ReviewPick {
  datasetId: string;
  datasetRevision: string;
  runId: string;
  runRevision: number;
  reviewRevision: string;
  step: number;
  checkpointDir: string;
  blindLabel?: string;
}

export const getReviewLadder = (datasetId: string, runId: string) =>
  trainingOperation<ReviewLadder>('review', `/ladder/${encodeURIComponent(datasetId)}/${encodeURIComponent(runId)}`);

export function pickFromLadder(ladder: ReviewLadder, step: number): ReviewPick {
  const fact = ladder.facts.find(item => item.step === step);
  if (!fact) throw new Error('The selected checkpoint is no longer on this ladder. Reload and try again.');
  return { datasetId: ladder.dataset.id, datasetRevision: ladder.dataset.revision,
    runId: ladder.run, runRevision: ladder.runRevision, reviewRevision: ladder.reviewRevision,
    step, checkpointDir: fact.dir, ...(fact.blindLabel ? { blindLabel: fact.blindLabel } : {}) };
}

function snapshot<P>(kind: string, payload: P, picks: ReviewPick[]): TrainingSnapshot<P> {
  return snapshotFor({ kind, idempotencyKey: crypto.randomUUID(), worker: { kind: 'local' },
    ...(picks.length === 1 ? { dataset: { id: picks[0].datasetId, revision: picks[0].datasetRevision } } : {}),
    sources: picks.map(pick => ({ kind: 'yue2-run', id: pick.runId, revision: String(pick.runRevision) })), payload });
}

export function finishReviewBatch(picks: ReviewPick[], knee: boolean) {
  return trainingOperation<{ batch: unknown }>('review', '/finish-batch', { body: snapshot('review:finish-batch',
    { entries: picks, knee }, picks) });
}
export function selectReviewRung(pick: ReviewPick) {
  const { datasetRevision: _datasetRevision, ...payload } = pick;
  return trainingOperation<{ linked: boolean; alreadyLinked: boolean; plan: Yue2CleanupPlan }>('review', '/select',
    { body: snapshot('review:select', payload, [pick]) });
}
export function cleanupReviewRung(pick: ReviewPick, choice: Yue2CleanupChoice) {
  const { datasetRevision: _datasetRevision, ...payload } = pick;
  return trainingOperation<{ freedBytes: number; done: string[]; finishError?: string }>('review', '/cleanup',
    { body: snapshot('review:cleanup', { ...payload, choice }, [pick]) });
}

export function decideReviewUse(pick: ReviewPick, narFurther: boolean, nar: {
  budget: number; lrScale: number; keepDelta: number; target: number | null; knee: boolean;
}) {
  const { datasetRevision: _datasetRevision, ...payload } = pick;
  return trainingOperation<{ outcome: 'skip' | 'reject' | 'chain'; worker?: string; jobId?: string }>('review', '/decision',
    { body: snapshot('review:decision', { ...payload, narFurther, nar }, [pick]) });
}
