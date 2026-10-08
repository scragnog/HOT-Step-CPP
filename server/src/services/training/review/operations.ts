import path from 'node:path';
import type { Router } from 'express';
import { auditionDraftSchema, reviewCleanupSchema, reviewFinishSchema, reviewPickSchema, reviewUseSchema } from '../../../contracts/trainingReview.js';
import { getDataset } from '../datasetsRepo.js';
import { activeJobForDataset } from '../labelingQueue.js';
import { hasActivePipeline } from '../pipelineRunner.js';
import { readPreview } from '../auditionStore.js';
import { createTrainingCreateDraft, getTrainingCreateDraft, mirroredGenerationDraft } from './auditionDraft.js';
import { getUserId } from '../../../routes/auth.js';
import { narContinuationRequest, reviewUseDecision } from './reviewPolicy.js';
import { listYue2AitkRuns, yue2RunFinished } from '../yue2AitkRuns.js';
import { listYue2JointPreviews } from '../yue2JointPreview.js';
import { listYue2RungScores } from '../yue2RungScores.js';
import { bestReviewFact, reviewRevision, reviewRungFacts, reviewSelectionError } from '../yue2BestRung.js';
import { finishScoredLadders, listBatches } from '../yue2BatchRunner.js';
import { readYue2Linked, refreshYue2PresetsForJointCheckpoint } from '../lyricStudioExport.js';
import { planYue2Cleanup, runYue2Cleanup } from '../yue2Cleanup.js';
import { acceptTrainingSnapshot, assertDatasetCurrent, registerTrainingOperations, trainingOp, TrainingOperationFailure,
  type TrainingOperationDeps } from '../operations.js';
import { admitYue2JointTrain } from '../yue2JointAdmission.js';

const stale = (message: string) => new TrainingOperationFailure(409, { error: message });
const missing = (message: string) => new TrainingOperationFailure(404, { error: message });
// Selection changes the link that cleanup trusts. Hold the same dataset lock
// across cleanup's asynchronous deletion of other runs.
const reviewMutations = new Set<string>();
export async function withReviewMutation<T>(datasetId: string, action: () => Promise<T> | T): Promise<T> {
  if (reviewMutations.has(datasetId)) throw stale('A review selection or cleanup is already in progress');
  reviewMutations.add(datasetId);
  try { return await action(); }
  finally { reviewMutations.delete(datasetId); }
}

function ladder(datasetId: string, runId: string) {
  const dataset = getDataset(datasetId);
  if (!dataset) throw missing('Dataset not found');
  const run = listYue2AitkRuns(dataset.id, dataset.slug).find(r => r.jobId === runId);
  if (!run) throw missing('Review run not found for this dataset');
  const scores = listYue2RungScores(dataset.id, run.jobId);
  const previews = listYue2JointPreviews(run.output);
  const facts = reviewRungFacts(run, scores, previews);
  return { dataset, run, facts, revision: reviewRevision(run, scores, previews), best: bestReviewFact(facts) };
}

type Pick = {
  datasetId: string; runId: string; runRevision: number; reviewRevision: string;
  step: number; checkpointDir: string; blindLabel?: string;
};

/** The picked run and checkpoint are checked again at the mutation boundary. */
function chosen(pick: Pick, requireScore: boolean) {
  const current = ladder(pick.datasetId, pick.runId);
  if (current.run.updatedAt !== pick.runRevision)
    throw stale('Run changed since this rung was selected; reload the ladder');
  const selectionError = reviewSelectionError(current.facts, current.revision, pick, requireScore);
  if (selectionError) throw stale(`${selectionError}; reload the ladder`);
  if (current.run.status === 'running' || yue2RunFinished(current.run.output))
    throw stale('This run is running or already finished');
  const checkpoint = current.run.checkpoints.find(c => c.step === pick.step && c.arPath && c.narPath);
  if (!checkpoint || path.resolve(checkpoint.dir).toLowerCase() !== path.resolve(pick.checkpointDir).toLowerCase())
    throw stale('The selected checkpoint changed; reload the ladder');
  if (pick.blindLabel && pick.blindLabel !== current.run.blindLabels?.[pick.step])
    throw stale('Blind label no longer identifies the selected rung');
  if (requireScore && (current.best?.step !== pick.step || current.best.overall === null))
    throw stale('The best-scored rung changed or has no complete human score');
  return { ...current, checkpoint };
}

function assertRunSource(sources: Array<{ kind: string; id: string; revision: string }>, pick: Pick) {
  if (!sources.some(s => s.kind === 'yue2-run' && s.id === pick.runId && s.revision === String(pick.runRevision)))
    throw new TrainingOperationFailure(400, { error: 'The selected run revision is required in sources' });
}

export function mountTrainingReview(router: Router, deps: TrainingOperationDeps) {
  router.get('/ladder/:datasetId/:runId', trainingOp(async req => {
    const result = ladder(String(req.params.datasetId), String(req.params.runId));
    return { dataset: { id: result.dataset.id, revision: deps.datasetRevision(result.dataset.id) },
      run: result.run.jobId, runRevision: result.run.updatedAt, reviewRevision: result.revision,
      status: result.run.status, finished: yue2RunFinished(result.run.output), origin: result.run.origin?.worker ?? null,
      method: result.run.options.method, freezePlannerNow: result.run.options.freezePlannerNow === true,
      facts: result.facts, best: result.best };
  }));
  router.post('/finish-batch', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, reviewFinishSchema, 'review', deps);
    const seen = new Set<string>();
    for (const pick of snapshot.payload.entries) {
      if (seen.has(pick.runId)) throw stale('A ladder was selected twice for finish');
      seen.add(pick.runId);
      if (deps.datasetRevision(pick.datasetId) !== pick.datasetRevision) throw stale('Dataset changed since finish was prepared');
      assertRunSource(snapshot.sources, pick);
      const current = chosen(pick, true);
      if (current.run.origin && current.run.options.method !== 'base-matched')
        throw stale('NAR further training is unavailable for a mirrored worker run');
      if (listBatches().some(batch => (batch.status === 'running' || batch.status === 'paused')
          && batch.items.some(item => item.refineRun === pick.runId && (item.status === 'pending' || item.status === 'running'))))
        throw stale('This ladder is already queued for finish');
    }
    const result = finishScoredLadders(snapshot.payload.entries.map(pick => ({ datasetId: pick.datasetId,
      refineRun: pick.runId, pickStep: pick.step })), { knee: snapshot.payload.knee });
    if ('error' in result) throw stale(result.error);
    return { batch: result, selections: snapshot.payload.entries.map(pick => ({ runId: pick.runId, step: pick.step })) };
  }));
  router.post('/decision', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, reviewUseSchema, 'review', deps);
    const pick = snapshot.payload;
    return withReviewMutation(pick.datasetId, async () => {
    if (snapshot.dataset?.id !== pick.datasetId) throw stale('Dataset selection changed');
    assertRunSource(snapshot.sources, pick);
    const current = chosen(pick, false);
    const outcome = reviewUseDecision(current.run, pick.narFurther);
    if (outcome !== 'chain') return { outcome, ...(outcome === 'reject' ? { worker: current.run.origin?.worker } : {}) };
    const admitted = await admitYue2JointTrain({ datasetId: pick.datasetId,
      body: narContinuationRequest(pick.runId, pick.step, pick.nar),
      idempotencyKey: `review-decision:${snapshot.operation.idempotencyKey}`,
      revalidate: () => {
        assertDatasetCurrent(snapshot, deps);
        const again = chosen(pick, false);
        if (reviewUseDecision(again.run, pick.narFurther) !== 'chain')
          throw stale('This rung no longer takes a decoder pass');
      } });
    if (admitted.status !== 200)
      throw new TrainingOperationFailure(admitted.status === 400 ? 400 : 409,
        { error: typeof admitted.body.error === 'string' ? admitted.body.error : 'Could not start decoder training' });
    if (typeof admitted.body.jobId !== 'string') throw stale('Decoder training did not return a job id');
    return { outcome, jobId: admitted.body.jobId };
    });
  }));
  router.post('/select', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, reviewPickSchema, 'review', deps);
    const pick = snapshot.payload;
    return withReviewMutation(pick.datasetId, () => {
    if (snapshot.dataset?.id !== pick.datasetId) throw stale('Dataset selection changed');
    assertRunSource(snapshot.sources, pick);
    const current = chosen(pick, false);
    if (deps.datasetRevision(pick.datasetId) !== snapshot.dataset.revision) throw stale('Dataset changed since this rung was selected');
    if (activeJobForDataset(pick.datasetId) || hasActivePipeline()) throw stale('A job or pipeline is active for this dataset');
    const ckpt = current.checkpoint;
    const linked = readYue2Linked()[current.dataset.slug.toLowerCase()];
    const alreadyLinked = !!linked && path.resolve(linked.arPath).toLowerCase() === path.resolve(ckpt.arPath!).toLowerCase()
      && path.resolve(linked.narPath).toLowerCase() === path.resolve(ckpt.narPath!).toLowerCase();
    if (!alreadyLinked) {
      const known = listYue2AitkRuns(current.dataset.id, current.dataset.slug).flatMap(r => r.checkpoints)
        .flatMap(c => [c.arPath, c.narPath].filter((v): v is string => !!v));
      const result = refreshYue2PresetsForJointCheckpoint({ slug: current.dataset.slug, lyricsSetId: current.dataset.lyricsSetId },
        ckpt.arPath!, ckpt.narPath!, known);
      if (!result.linked) throw stale(`Could not link checkpoint: ${result.error ?? 'unknown error'}`);
    }
    return { linked: true, alreadyLinked, plan: planYue2Cleanup(current.dataset, pick.runId, pick.step) };
    });
  }));
  router.post('/draft', trainingOp(async req => {
    const user = getUserId(req);
    if (!user) throw new TrainingOperationFailure(400, { error: 'Sign in before sending an audition to Create' });
    const { snapshot } = await acceptTrainingSnapshot(req.body, auditionDraftSchema, 'review', deps);
    const { datasetId, previewId, slot, cell } = snapshot.payload;
    if (snapshot.dataset?.id !== datasetId) throw stale('Dataset selection changed');
    const preview = readPreview(previewId);
    if (!preview || preview.datasetId !== datasetId) throw missing('Audition preview not found for this dataset');
    if (!snapshot.sources.some(source => source.kind === 'audition-preview' && source.id === previewId
        && source.revision === preview.createdAt)) throw stale('Audition preview changed since this draft was prepared');
    const side = preview.sides.find(candidate => candidate.slot === slot);
    if (!side?.ok) throw missing('Audition side has no successful render');
    if (cell === 'adapter' && !preview.renderDitAdapter) throw stale('This audition has no DiT-adapter render');
    const draft = mirroredGenerationDraft(preview, side, cell);
    const doc = createTrainingCreateDraft(user, snapshot.operation.idempotencyKey, draft);
    return { draftId: doc.id, revision: doc.revision };
  }));
  router.get('/draft/:id', trainingOp(async req => {
    const user = getUserId(req);
    if (!user) throw new TrainingOperationFailure(400, { error: 'Sign in before opening an audition draft' });
    try { return { draft: getTrainingCreateDraft(user, String(req.params.id)) }; }
    catch { throw missing('Audition draft not found'); }
  }));
  router.post('/cleanup', trainingOp(async req => {
    const { snapshot } = await acceptTrainingSnapshot(req.body, reviewCleanupSchema, 'review', deps);
    const pick = snapshot.payload;
    return withReviewMutation(pick.datasetId, async () => {
    if (snapshot.dataset?.id !== pick.datasetId) throw stale('Dataset selection changed');
    assertRunSource(snapshot.sources, pick);
    const current = chosen(pick, false);
    if (deps.datasetRevision(pick.datasetId) !== snapshot.dataset.revision) throw stale('Dataset changed since this rung was selected');
    if (activeJobForDataset(pick.datasetId) || hasActivePipeline()) throw stale('A job or pipeline is active for this dataset');
    const linked = readYue2Linked()[current.dataset.slug.toLowerCase()];
    if (!linked || path.resolve(linked.arPath).toLowerCase() !== path.resolve(current.checkpoint.arPath!).toLowerCase()
        || path.resolve(linked.narPath).toLowerCase() !== path.resolve(current.checkpoint.narPath!).toLowerCase())
      throw stale('This checkpoint is no longer the linked adapter');
    return runYue2Cleanup(current.dataset, pick.runId, pick.step, pick.choice,
      { blind: !!pick.blindLabel, blindLabel: pick.blindLabel });
    });
  }));
}

registerTrainingOperations('review', mountTrainingReview);
