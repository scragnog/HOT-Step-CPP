import { YUE2_JOINT_LADDER_PREVIEW } from '../../../../server/src/contracts/trainingRecipes';
import type { Yue2AitkPrepareRequest, Yue2JointTrainRequest, Yue2JointPreviewOptions } from '../../services/trainingApi';
import { TRAINING_RECIPE_VERSION } from '../../../../server/src/contracts/trainingRecipes';
import { currentWorkerRef, snapshotFor } from '../../services/trainingOperations';
import type { TrainingDatasetRef, TrainingSourceRef, TrainingWorkerRef } from '../../../../server/src/contracts/trainingOperation';
import type { Yue2PreparationPayload } from '../../services/trainingPreparationApi';
import type { Yue2PreparationSummary } from '../../services/trainingPreparationApi';

/** Capture the visible joint form before submitting a preparation chain. */
export function captureJointOverrides(
  form: Yue2JointTrainRequest, prepare: Yue2AitkPrepareRequest,
  lyricTiming: boolean, defaultDevice: string, resumeChoice: string,
): Record<string, unknown> {
  const timingWeight = lyricTiming
    ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08)
    : 0;
  const [resumeRunId, resumeStepText] = resumeChoice.split('|');
  const selectedResume = resumeRunId && resumeStepText ? { resumeRunId, resumeStep: Number(resumeStepText) } : {};
  const request = { ...form, device: defaultDevice === 'CUDA0' ? 'CUDA0' : form.device,
    autoCaption: undefined, lyricTiming, alignmentEnabled: lyricTiming, cursorWeight: timingWeight,
    autoPrepare: !resumeChoice && !form.resume?.trim(), preparation: prepare,
    checkpoint: '', output: '',
    ...(form.preview ? { preview: { ...(YUE2_JOINT_LADDER_PREVIEW as Yue2JointPreviewOptions), ...form.preview,
      everySteps: form.preview.parallel ? 0 : form.saveEvery, previewMaxFrames: Math.max(8, Math.min(360, form.preview.seconds || 300)) * 25 } } : {}),
    ...(form.resume?.trim() && !resumeChoice ? { resume: form.resume.trim() } : {}),
    ...selectedResume };
  return structuredClone(request);
}

/** A direct Start uses prepared inputs and admits only the joint train stage. */
export function captureJointStart(dataset: TrainingDatasetRef, source: TrainingSourceRef,
  overrides: Record<string, unknown>, idempotencyKey: string, worker: TrainingWorkerRef) {
  return snapshotFor({ kind: 'yue2-preparation', idempotencyKey, dataset, sources: [source],
    worker,
    payload: { mode: 'train-after-preparation', stages: ['joint'], trigger: '',
      lyricTiming: Boolean(overrides.lyricTiming),
      recipes: { joint: { version: TRAINING_RECIPE_VERSION, overrides: structuredClone(overrides) } } } as Yue2PreparationPayload });
}

export async function submitJointStart(datasetId: string, overrides: Record<string, unknown>,
  runner: {
    getContext: (id: string) => Promise<{ dataset: TrainingDatasetRef; source: TrainingSourceRef }>;
    start: (snapshot: ReturnType<typeof captureJointStart>) => Promise<Yue2PreparationSummary>;
  }, idempotencyKey = crypto.randomUUID(), worker = currentWorkerRef()): Promise<Yue2PreparationSummary> {
  const capturedOverrides = structuredClone(overrides);
  const { dataset, source } = await runner.getContext(datasetId);
  return runner.start(captureJointStart(dataset, source, capturedOverrides, idempotencyKey, worker));
}
