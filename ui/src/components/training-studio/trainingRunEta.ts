// Time left in a training run: the steps still to train at the run's mean step time,
// plus the checkpoint previews still to come at what a preview has cost so far.
//
// A preview's cost is measured, not guessed. Before the first mid-run take only the
// base take exists (rendered before training began), so its render time stands in.
// After one, its pause is used instead: trainer exit, engine restart, the render,
// engine stop and model reload, which is what each later take will cost too.

export interface TrainingRunEtaInput {
  step: number;
  totalSteps: number;
  msPerStep: number | null;
  /** Steps between previews; 0 = previews off. */
  everySteps: number;
  previews: Array<{ kind: string; step: number; status: string; renderMs: number | null; pauseMs: number | null }>;
}

export interface TrainingRunEta {
  ms: number;
  /** Previews still to render, the final take included. */
  previewsLeft: number;
  /** Cost per preview used, ms; null when no take has finished yet. */
  previewMs: number | null;
}

export function trainingRunEta(i: TrainingRunEtaInput): TrainingRunEta | null {
  if (!i.msPerStep || i.totalSteps <= 0 || i.step >= i.totalSteps) return null;
  const done = i.previews.filter(p => p.status === 'done');
  const paused = done.filter(p => p.kind === 'step' && p.pauseMs).sort((a, b) => b.step - a.step)[0];
  const base = done.find(p => p.kind === 'base' && p.renderMs);
  const previewMs = paused?.pauseMs ?? base?.renderMs ?? null;
  // Checkpoint takes at every multiple of everySteps after now and before the end, then the final take.
  const previewsLeft = i.everySteps > 0
    ? Math.max(0, Math.floor((i.totalSteps - 1) / i.everySteps) - Math.floor(i.step / i.everySteps)) + 1
    : 0;
  return { ms: (i.totalSteps - i.step) * i.msPerStep + previewsLeft * (previewMs ?? 0), previewsLeft, previewMs };
}

export function formatDuration(ms: number): string {
  const min = Math.round(ms / 60_000);
  if (min < 1) return '<1 min';
  const h = Math.floor(min / 60), m = min % 60;
  return h ? `${h} h ${String(m).padStart(2, '0')} min` : `${m} min`;
}

/** MM3 preview rows (one per spec, so a checkpoint has an artist and a control
 *  take) folded to one entry per checkpoint: the renders add up, the pause is
 *  carried once on the last take of the set. */
export function previewsFromRows(rows: Array<{ step: number; base: boolean; ms: number; pauseMs?: number }>): TrainingRunEtaInput['previews'] {
  const byStep = new Map<number, { kind: string; step: number; status: string; renderMs: number; pauseMs: number | null }>();
  for (const r of rows) {
    const e = byStep.get(r.step) ?? { kind: r.base ? 'base' : 'step', step: r.step, status: 'done', renderMs: 0, pauseMs: null };
    e.renderMs += r.ms;
    e.pauseMs = r.pauseMs ?? e.pauseMs;
    byStep.set(r.step, e);
  }
  return [...byStep.values()];
}
