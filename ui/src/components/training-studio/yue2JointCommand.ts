import { YUE2_JOINT_LADDER_PREVIEW } from '../../../../server/src/contracts/trainingRecipes';
import type { Yue2AitkPrepareRequest, Yue2JointTrainRequest, Yue2JointPreviewOptions } from '../../services/trainingApi';

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
