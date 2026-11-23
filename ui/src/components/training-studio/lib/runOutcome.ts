/** How a finished training run reads, shared by the run lists. */
export type RunOutcome = 'completed' | 'target-reached' | 'halted' | 'failed' | 'unknown';

export const RUN_OUTCOME: Record<RunOutcome, { label: string; tone: string }> = {
  completed: { label: 'Finished', tone: 'text-emerald-500' },
  'target-reached': { label: 'Target reached', tone: 'text-emerald-500' },
  halted: { label: 'Stopped early', tone: 'text-amber-500' },
  failed: { label: 'Failed', tone: 'text-rose-500' },
  unknown: { label: 'Unknown', tone: 'text-zinc-500' },
};
