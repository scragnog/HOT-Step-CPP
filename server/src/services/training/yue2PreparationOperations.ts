import type { Router } from 'express';
import { trainingSnapshotSchema } from '../../contracts/trainingOperation.js';
import { yue2PreparationPayloadSchema } from '../../contracts/trainingPreparation.js';
import { recipeFor, resolveRecipe } from './recipes/operations.js';
import { getDataset } from './datasetsRepo.js';
import { workerJson } from './trainingWorkers.js';
import { acceptTrainingSnapshot, registerTrainingOperations, trainingOp, TrainingOperationFailure,
  type TrainingOperationDeps } from './operations.js';
import { getYue2PreparationPipeline, sourceRevision, Yue2PreparationConflict } from './yue2PreparationPipeline.js';

const family = { nar: 'yue2-nar', ar: 'yue2-ar', joint: 'yue2-joint' } as const;

function requireStageShape(stages: string[], mode: string, forms: Record<string, unknown>): void {
  if (new Set(stages).size !== stages.length) throw new TrainingOperationFailure(400, { error: 'A stage appears twice' });
  const trainStages = stages.filter(s => s === 'nar' || s === 'ar' || s === 'joint');
  if (mode === 'prepare-only' && trainStages.length) throw new TrainingOperationFailure(400, { error: 'Preparation only cannot start a trainer' });
  if (mode === 'train-after-preparation' && !trainStages.length) throw new TrainingOperationFailure(400, { error: 'Train mode needs a training stage' });
  if (trainStages.includes('joint') && trainStages.length > 1) throw new TrainingOperationFailure(400, { error: 'Joint training cannot be combined with separate NAR or AR training' });
  for (const stage of trainStages) if (!forms[stage]) throw new TrainingOperationFailure(400, { error: `Missing accepted ${stage} form` });
  // Joint alone is direct Start-training on prepared inputs (the pipeline checks them).
  if (stages.includes('joint') && stages.length > 1 && !stages.includes('latents')) throw new TrainingOperationFailure(400, { error: 'Joint training requires a latent preparation stage' });
}

export function mountYue2Preparation(router: Router, deps: TrainingOperationDeps): void {
  const yue2PreparationPipeline = getYue2PreparationPipeline();
  router.get('/context/:datasetId', trainingOp(async req => {
    const dataset = getDataset(String(req.params.datasetId));
    if (!dataset) throw new TrainingOperationFailure(404, { error: 'Dataset not found' });
    return { dataset: { id: dataset.id, revision: dataset.updatedAt },
      source: { kind: 'dataset-sources', id: dataset.id, revision: sourceRevision(dataset.sourceDir) } };
  }));
  router.post('/', trainingOp(async req => {
    // An accepted idempotency key remains readable after a worker goes offline
    // or the dataset changes. A different body with the same key is a conflict.
    const parsed = trainingSnapshotSchema(yue2PreparationPayloadSchema).safeParse(req.body);
    if (!parsed.success) throw new TrainingOperationFailure(400, { error: 'Invalid preparation command',
      issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
    const input = parsed.data;
    if (input.operation.kind !== 'yue2-preparation') throw new TrainingOperationFailure(400, { error: 'Wrong operation kind' });
    const old = yue2PreparationPipeline.byKey(input.operation.kind, input.operation.idempotencyKey);
    if (old) {
      if (JSON.stringify(old.snapshot) !== JSON.stringify(input)) throw new TrainingOperationFailure(409, { error: 'Idempotency key belongs to another command' });
      return { pipeline: old.summary };
    }
    const capability = input.payload.mode === 'prepare-only' ? 'prepare' : 'train';
    const { snapshot, worker } = await acceptTrainingSnapshot(input, yue2PreparationPayloadSchema, capability, deps);
    if (!snapshot.dataset) throw new TrainingOperationFailure(400, { error: 'Preparation requires a dataset reference' });
    const { stages, mode, recipes } = snapshot.payload;
    requireStageShape(stages, mode, recipes);
    const trainBodies: Record<string, Record<string, unknown>> = {};
    for (const stage of ['nar', 'ar', 'joint'] as const) {
      if (!stages.includes(stage)) continue;
      const form = recipes[stage]!;
      const remoteRecipe = worker.remote ? await workerJson<ReturnType<typeof recipeFor>>(
        worker.remote, `/api/training/ops/recipes/${family[stage]}${form.preset ? `?preset=${encodeURIComponent(form.preset)}` : ''}`) : undefined;
      const resolved = resolveRecipe(family[stage], form.overrides, form.preset, remoteRecipe);
      trainBodies[stage] = resolved.execution as Record<string, unknown>;
    }
    // Recheck after remote recipe fetch, before the durable claim.
    if (deps.datasetRevision(snapshot.dataset.id) !== snapshot.dataset.revision) {
      throw new TrainingOperationFailure(409, { reason: 'stale-dataset', error: 'Dataset changed while accepting preparation',
        currentRevision: deps.datasetRevision(snapshot.dataset.id) ?? undefined });
    }
    try { return { pipeline: await yue2PreparationPipeline.start(snapshot, trainBodies, worker.remote?.url ?? null) }; }
    catch (err) { if (err instanceof Yue2PreparationConflict) throw new TrainingOperationFailure(409, { error: err.message }); throw err; }
  }));
  router.get('/', trainingOp(async req => ({ pipelines: yue2PreparationPipeline.list(
    typeof req.query.datasetId === 'string' ? req.query.datasetId : undefined) })));
  router.get('/:id', trainingOp(async req => {
    const pipeline = yue2PreparationPipeline.get(String(req.params.id));
    if (!pipeline) throw new TrainingOperationFailure(404, { error: 'Preparation not found' });
    return { pipeline };
  }));
  router.get('/:id/job', trainingOp(async req => {
    const id = String(req.params.id);
    if (!yue2PreparationPipeline.get(id)) throw new TrainingOperationFailure(404, { error: 'Preparation not found' });
    return { job: await yue2PreparationPipeline.currentJob(id) };
  }));
  for (const action of ['pause', 'resume', 'retry', 'cancel'] as const) {
    router.post(`/:id/${action}`, trainingOp(async req => {
      const id = String(req.params.id);
      try {
        const pipeline = action === 'pause' ? yue2PreparationPipeline.pause(id)
          : action === 'resume' ? yue2PreparationPipeline.resume(id)
            : action === 'retry' ? await yue2PreparationPipeline.retry(id)
              : await yue2PreparationPipeline.cancel(id);
        if (!pipeline) throw new TrainingOperationFailure(404, { error: 'Preparation not found' });
        return { pipeline };
      } catch (err) {
        if (err instanceof Yue2PreparationConflict) throw new TrainingOperationFailure(409, { error: err.message });
        throw err;
      }
    }));
  }
}

registerTrainingOperations('preparation', mountYue2Preparation);
