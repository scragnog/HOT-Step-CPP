import type { Yue2AitkRunRecord } from '../yue2AitkRuns.js';

/** Whether a selected rung may start another decoder pass. */
export function reviewUseDecision(run: Pick<Yue2AitkRunRecord, 'origin' | 'options'>, narFurther: boolean): 'skip' | 'reject' | 'chain' {
  if (!narFurther) return 'skip';
  if (run.origin) return 'reject';
  if (run.options.freezePlannerNow === true) return 'skip';
  return 'chain';
}

export function narContinuationRequest(runId: string, step: number, options: {
  budget: number; lrScale: number; keepDelta: number; target: number | null; knee: boolean;
}): Record<string, unknown> {
  return { trainingMethod: 'aitk', refine: true, resumeRunId: runId, resumeStep: step,
    steps: step + options.budget, saveEvery: 10, stopMode: 'kl', narExtraSteps: step + options.budget,
    freezePlannerNow: true, narLrScale: options.lrScale,
    reconStop: options.knee ? 0.005 : 0, reconStopWindow: 10, reconKeepDelta: options.keepDelta,
    ...(options.target === null ? {} : { reconTarget: options.target }), stopEngine: false,
    lyricTiming: true, alignmentEnabled: true, autoPrepare: false, checkpoint: '', output: '',
    preview: { enabled: false, everySteps: 0, seconds: 90, seed: 424242, previewMaxFrames: 2250,
      baseline: false, control: false } };
}