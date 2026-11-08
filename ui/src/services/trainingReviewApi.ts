import type { Mm3AlbumScore, Mm3CleanupPlan, Mm3RungScore, Yue2CleanupChoice, Yue2CleanupPlan, Yue2TrainedDirection } from './trainingApi';
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

export const getReviewLadder = (datasetId: string, runId: string, opts: { signal?: AbortSignal } = {}) =>
  trainingOperation<ReviewLadder>('review', `/ladder/${encodeURIComponent(datasetId)}/${encodeURIComponent(runId)}`, opts);

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

// All four write commands are silent: each has a single component caller that
// renders the failure in its own error line (ReviewPanel's finish line,
// Yue2LadderReview's rung/cleanup lines, RefinePanel's NAR-use line).
export function finishReviewBatch(picks: ReviewPick[], knee: boolean) {
  return trainingOperation<{ batch: unknown }>('review', '/finish-batch', { body: snapshot('review:finish-batch',
    { entries: picks, knee }, picks), silent: true });
}
export function selectReviewRung(pick: ReviewPick) {
  const { datasetRevision: _datasetRevision, ...payload } = pick;
  return trainingOperation<{ linked: boolean; alreadyLinked: boolean; plan: Yue2CleanupPlan }>('review', '/select',
    { body: snapshot('review:select', payload, [pick]), silent: true });
}
export function cleanupReviewRung(pick: ReviewPick, choice: Yue2CleanupChoice) {
  const { datasetRevision: _datasetRevision, ...payload } = pick;
  return trainingOperation<{ freedBytes: number; done: string[]; finishError?: string }>('review', '/cleanup',
    { body: snapshot('review:cleanup', { ...payload, choice }, [pick]), silent: true });
}

export function decideReviewUse(pick: ReviewPick, narFurther: boolean, nar: {
  budget: number; lrScale: number; keepDelta: number; target: number | null; knee: boolean;
}) {
  const { datasetRevision: _datasetRevision, ...payload } = pick;
  return trainingOperation<{ outcome: 'skip' | 'reject' | 'chain'; worker?: string; jobId?: string }>('review', '/decision',
    { body: snapshot('review:decision', { ...payload, narFurther, nar }, [pick]), silent: true });
}

/** "Use this checkpoint" on an MM3 ladder: the dataset's album presets move to
 *  `step`. Returns the adapter reference the global LM adapter should take. */
export function selectMm3Checkpoint(dataset: { id: string; revision: string }, runName: string, step: number) {
  return trainingOperation<{ ref: string; presets: number; plan: Mm3CleanupPlan | null }>('mm3', '/select', { body: snapshotFor({
    kind: 'mm3:select', idempotencyKey: crypto.randomUUID(), worker: { kind: 'local' }, dataset,
    sources: [], payload: { datasetId: dataset.id, runName, step } }), silent: true });
}
/** Cleanup around the MM3 checkpoint just chosen, while it is still linked. */
export function cleanupMm3Checkpoint(dataset: { id: string; revision: string }, runName: string, step: number,
  choice: Yue2CleanupChoice, blindLabel?: string) {
  return trainingOperation<{ freedBytes: number; done: string[] }>('mm3', '/cleanup', { body: snapshotFor({
    kind: 'mm3:cleanup', idempotencyKey: crypto.randomUUID(), worker: { kind: 'local' }, dataset,
    sources: [], payload: { datasetId: dataset.id, runName, step, choice, ...(blindLabel ? { blindLabel } : {}) } }), silent: true });
}
/** A rung's likeness, corruption or notes; a field not sent keeps its value. */
export function scoreMm3Rung(body: { datasetId: string; runName: string; step: number; likeness?: number | null;
  corruption?: number | null; notes?: string; blind: boolean; blindLabel: string }) {
  return trainingOperation<{ score: Mm3RungScore }>('mm3', '/score', { body, silent: true });
}
export function scoreMm3Album(body: { datasetId: string; runName: string; score?: number | null;
  instruments?: Yue2TrainedDirection | null; vocals?: Yue2TrainedDirection | null; notes?: string }) {
  return trainingOperation<{ score: Mm3AlbumScore }>('mm3', '/album', { body, silent: true });
}
