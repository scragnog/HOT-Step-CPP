import type { Router } from 'express';
import {
  ACE_DIT_FORM_DEFAULTS, ACE_DIT_LOKR_FORM_DEFAULTS, ACE_LM_FORM_DEFAULTS, YUE2_JOINT_FORM_DEFAULTS,
  TRAINING_RECIPE_VERSION, aceDitExecutionBody, aceLmExecutionBody,
  trainingRecipeFamilySchema, trainingRecipePayloadSchema, type TrainingRecipeFamily,
} from '../../../contracts/trainingRecipes.js';
import { MM3_LM_DEFAULTS, applyMm3Preset } from '../mm3Train.js';
import { YUE2_NAR_DEFAULTS, applyYue2Preset } from '../yue2Train.js';
import { YUE2_AR_DEFAULTS } from '../yue2ArTrain.js';
import { applyBaseMatchedRecipe } from '../yue2JointTrainRunner.js';
import { getTrainingDefaults, resolveTrainingDefaultLayers } from '../trainingDefaults.js';
import { workerJson } from '../trainingWorkers.js';
import { acceptTrainingSnapshot, registerTrainingOperations, resolveTrainingWorker, trainingOp, TrainingOperationFailure, type TrainingOperationDeps } from '../operations.js';

/** The form seed, stored partial and execution projection are distinct. */
export function recipeFor(family: TrainingRecipeFamily, preset?: string) {
  switch (family) {
    case 'ace-lm': return { builtin: ACE_LM_FORM_DEFAULTS, stored: getTrainingDefaults().trainLm,
      deferred: ['lmModel', 'maxLen'] };
    case 'ace-dit': return { builtin: preset === 'lora' ? ACE_DIT_FORM_DEFAULTS : ACE_DIT_LOKR_FORM_DEFAULTS,
      stored: getTrainingDefaults().trainDit, deferred: ['crop', 'layers'] };
    case 'mm3-lm': return { builtin: { ...applyMm3Preset(MM3_LM_DEFAULTS, preset), trigger: '', previewCaption: '' }, stored: {},
      deferred: ['basePrecision', 'rank'] };
    case 'yue2-nar': return { builtin: { ...applyYue2Preset(YUE2_NAR_DEFAULTS, preset), trigger: '' }, stored: {},
      deferred: ['lmType'] };
    case 'yue2-ar': return { builtin: { ...YUE2_AR_DEFAULTS, trigger: '', style: '', lyrics: '' }, stored: {}, deferred: ['device'] };
    case 'yue2-joint': return { builtin: YUE2_JOINT_FORM_DEFAULTS, stored: {},
      deferred: ['device', 'base', 'warmup'] };
  }
}

export function resolveRecipe(family: TrainingRecipeFamily, overrides: Record<string, unknown>, preset?: string,
  recipe = recipeFor(family, preset)) {
  const builtin = recipe.builtin as Record<string, unknown>;
  // Own-key overlay preserves explicit zero, false, empty string and null.
  // Undefined cannot cross JSON; omit it rather than clobbering a default.
  const applied = Object.fromEntries(Object.entries(overrides).filter(([, value]) => value !== undefined));
  const stored = recipe.stored as Record<string, unknown>;
  const storedForm = family === 'ace-lm' || family === 'ace-dit'
    ? { ...stored, ...(typeof stored.initAdapter === 'string'
      ? { resumeFromLatest: stored.initAdapter === 'latest' } : {}) }
    : stored;
  const layers = resolveTrainingDefaultLayers(builtin, storedForm, applied);
  const resolved = layers.resolved;
  const stringFields = family === 'ace-lm' || family === 'ace-dit' ? ['adapterName']
    : family === 'mm3-lm' ? ['trigger', 'previewCaption']
      : family === 'yue2-nar' ? ['trigger']
        : family === 'yue2-ar' ? ['trigger', 'style', 'lyrics'] : [];
  const bad = stringFields.filter(key => typeof resolved[key] !== 'string');
  if (bad.length) throw new TrainingOperationFailure(400, { error: 'Invalid recipe form',
    issues: bad.map(key => ({ path: `payload.overrides.${key}`, message: 'Expected a string' })) });
  const projected = family === 'ace-lm'
    ? aceLmExecutionBody(resolved as typeof ACE_LM_FORM_DEFAULTS, String(resolved.variantKey ?? '') || undefined)
    : family === 'ace-dit'
      ? aceDitExecutionBody(resolved as typeof ACE_DIT_FORM_DEFAULTS, String(resolved.variantKey ?? '') || undefined)
      : family === 'mm3-lm'
        ? mm3ExecutionBody(resolved, String(overrides.activePreset ?? 'custom'), overrides.trainLaunder === true,
          overrides.flashSupported !== false)
      : family === 'yue2-nar'
        ? yue2NarExecutionBody(resolved, String(overrides.activePreset ?? 'custom'))
      : family === 'yue2-ar'
        ? yue2ArExecutionBody(resolved, overrides.overtrain === true, overrides.mintedMissing === true)
      : resolved;
  const execution = projected;
  return { recipeVersion: TRAINING_RECIPE_VERSION, family, builtin, stored: recipe.stored,
    resolved, execution, provenance: layers.provenance, deferred: recipe.deferred,
    ...(family === 'yue2-joint' && resolved.method === 'base-matched'
      ? { effective: applyBaseMatchedRecipe(resolved) } : {}) };
}

export function mountTrainingRecipes(router: Router, deps: TrainingOperationDeps) {
  router.get('/', trainingOp(async () => ({ version: TRAINING_RECIPE_VERSION,
    families: ['ace-lm', 'ace-dit', 'mm3-lm', 'yue2-nar', 'yue2-ar', 'yue2-joint'] })));
  router.post('/resolve', trainingOp(async (req) => {
    const { snapshot, worker } = await acceptTrainingSnapshot(req.body, trainingRecipePayloadSchema, 'train', deps);
    const family = snapshot.payload.family;
    const preset = snapshot.payload.preset;
    const suffix = preset ? `?preset=${encodeURIComponent(preset)}` : '';
    const remoteRecipe = worker.remote
      ? await workerJson<ReturnType<typeof recipeFor>>(worker.remote, `/api/training/ops/recipes/${family}${suffix}`)
      : undefined;
    return { ...resolveRecipe(family, snapshot.payload.overrides, preset, remoteRecipe),
      worker: snapshot.worker, operation: snapshot.operation };
  }));
  router.get('/:family', trainingOp(async (req) => {
    const parsed = trainingRecipeFamilySchema.safeParse(req.params.family);
    if (!parsed.success) throw new TrainingOperationFailure(400, { error: 'Unknown training recipe family' });
    const preset = typeof req.query.preset === 'string' ? req.query.preset : undefined;
    const workerName = typeof req.query.worker === 'string' ? req.query.worker : '';
    if (workerName) {
      const worker = await resolveTrainingWorker({ kind: 'remote', name: workerName }, 'train', deps);
      if (worker.remote) {
        const suffix = preset ? `?preset=${encodeURIComponent(preset)}` : '';
        return workerJson(worker.remote, `/api/training/ops/recipes/${parsed.data}${suffix}`);
      }
    }
    return resolveRecipe(parsed.data, {}, preset);
  }));
}

registerTrainingOperations('recipes', mountTrainingRecipes);

/** Project the complete MM3 form into the legacy trainer request. */
export function mm3ExecutionBody(form: Record<string, any>, activePreset: string, trainLaunder: boolean, flashSupported: boolean) {
  return {
        steps: form.steps, saveEvery: form.saveEvery, keepResumeState: form.keepResumeState,
        longTracks: form.longTracks,
        rank: form.rank, alpha: form.alpha,
        lr: form.lr, maxFrames: form.maxFrames, cropMode: form.cropMode,
        preset: activePreset === 'custom' ? undefined : activePreset,
        cropStartFrac: form.cropStartFrac, cropEndFrac: form.cropEndFrac,
        cropStartTiles: form.cropStartTiles,
        depthLossWeight: form.depthLossWeight, depthLossFrames: form.depthLossFrames,
        optimizer: form.optimizer, muonLrScale: form.muonLrScale,
        adapterType: form.adapterType, lokrFactor: form.lokrFactor,
        gradAccum: form.gradAccum, seed: form.seed,
        basePrecision: form.basePrecision, holdout: form.holdout, evalEvery: form.evalEvery,
        cropAnchor: form.cropAnchor,
        stopMode: form.stopMode,
        ...(form.stopMode === 'loss' ? {
          targetLoss: form.targetLoss,
          targetLossMetric: form.targetLossMetric,
          targetLossEpochs: form.targetLossEpochs,
        } : {}),
        prefixFrames: form.cropAnchor === 'song' ? Math.max(0, form.prefixFrames) : 0,
        ...(form.trigger.trim()
          ? { trigger: form.trigger.trim(), triggerPrepend: form.triggerPrepend }
          : {}),
        ...(form.regDatasetId ? {
          regularisation: {
            datasetId: form.regDatasetId,
            every: form.regEvery,
            topK: form.regTopK,
          },
        } : {}),
        ...(trainLaunder ? { launder: true } : {}),
        ...(form.previewEverySteps > 0 || form.previewEveryMinutes > 0 ? {
          preview: {
            everySteps: form.previewEverySteps,
            everyMinutes: form.previewEveryMinutes,
            seconds: form.previewSeconds,
            seed: form.previewSeed,
            control: form.previewControl,
            baseline: form.previewBaseline,
            scaleMlp: form.previewScaleMlp,
            ...(form.previewCaption.trim() ? { caption: form.previewCaption.trim() } : {}),
            ...(!form.previewCaption.trim() && form.previewSongId
              ? { previewSongId: form.previewSongId } : {}),
          },
        } : {}),
        attnBackend: flashSupported ? form.attnBackend : 'exact',
        dora:   form.adapterType === 'lora' && form.dora,
        hira:   form.adapterType === 'lora' && form.hira,
        loha:   form.adapterType === 'lora' && form.loha,
        rslora: form.adapterType === 'lora' && form.rslora,
        pissa:    form.adapterType === 'lora' && form.pissa
                  && !form.dora && !form.hira && !form.loha,
        hotPizza: form.adapterType === 'lora' && form.pissa && form.hotPizza
                  && !form.dora && !form.hira && !form.loha,
        hra: form.adapterType === 'lora' && form.hra
             && !form.dora && !form.hira && !form.loha && !form.pissa,
        loraPlusRatio: form.adapterType === 'lora' ? form.loraPlusRatio : 1,
        artistToken: form.adapterType === 'lora' && form.artistTokenOn ? form.artistToken : '',
        artistTokenK: form.artistTokenK,
        artistTokenLr: form.artistTokenLr,
        prefixN: form.adapterType === 'lora' && !form.regDatasetId
          ? Math.max(0, form.prefixN) : 0,
      };
}

export function yue2NarExecutionBody(form: Record<string, any>, activePreset: string) {
  return {
        ...(activePreset === 'custom' ? {} : { preset: activePreset }),
        lmType: form.lmType,
        rank: form.rank, alpha: form.alpha, target: form.target,
        optimizer: form.optimizer, prodigyD0: form.prodigyD0,
        muonLrScale: form.muonLrScale, muonNsSteps: form.muonNsSteps,
        lr: form.lr, lrScheduler: form.lrScheduler,
        steps: form.steps, warmup: form.warmup,
        saveEvery: form.saveEvery, logEvery: form.logEvery,
        gradAccum: form.gradAccum, maxGradNorm: form.maxGradNorm,
        weightDecay: form.weightDecay, captionDropout: form.captionDropout,
        abcDropout: form.abcDropout,
        tSampling: form.tSampling, seed: form.seed, kvCache: form.kvCache,
        clipBlock: form.clipBlock,
        ...(form.trigger.trim() ? { trigger: form.trigger.trim() } : {}),
      };
}

export function yue2ArExecutionBody(form: Record<string, any>, overtrain: boolean, mintedMissing: boolean) {
  return {
        lmType: form.lmType,
        ...(form.trigger.trim() ? { trigger: form.trigger.trim() } : { allowNoTrigger: true }),
        styleTemplate: form.styleTemplate,
        ...(form.style.trim() ? { style: form.style.trim() } : {}),
        ...(form.lyrics.trim() ? { lyrics: form.lyrics.trim() } : {}),
        sidecars: form.sidecars,
        target: form.target,
        rank: form.rank, alpha: form.alpha,
        optimizer: form.optimizer, prodigyD0: form.prodigyD0,
        muonLrScale: form.muonLrScale, muonNsSteps: form.muonNsSteps,
        lr: form.lr, lrScheduler: form.lrScheduler,
        schedSteps: form.schedSteps, warmup: form.warmup,
        steps: form.steps,
        ...(overtrain ? { allowOvertrain: true } : {}),
        gradAccum: form.gradAccum, maxGradNorm: form.maxGradNorm,
        weightDecay: form.weightDecay,
        artistFrac: form.artistFrac,
        adamBeta1: form.adamBeta1, adamBeta2: form.adamBeta2,
        captionDropout: form.captionDropout,
        attn: form.attn, maxLen: form.maxLen, chunk: form.chunk,
        cursorWeight: form.cursorWeight, abcDropout: form.abcDropout, seed: form.seed,
        ckptFrom: form.ckptFrom, saveEvery: form.saveEvery,
        evalEvery: form.evalEvery, logEvery: form.logEvery,
        ...(mintedMissing ? { allowNoMinted: true } : {}),
      };
}
