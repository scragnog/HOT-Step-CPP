import { formatDecimal } from '@/utils/format/format';
// trainingEta.ts — "when will this run hit the target loss?"
//
// THE SINGLE SOURCE OF TRUTH for a training run's remaining time. Exactly one
// component renders it (TrainingRunStats, directly above the loss curve); the
// store memoises it (selectTrainingEta) so the object identity is stable.
//
// History — the two bugs this file was rewritten to kill:
//  1. TWO ETAs for one run. JobProgress rendered computeTrainingEta ("6m to
//     target") while the metric tile under the chart rendered the ENGINE's own
//     `etaMs`, which is a completely different quantity: mean-epoch-ms ×
//     epochs-remaining-to-the-CAP (dit-train-run.h / lm-train-run.h). On the LM
//     run that reads 24 s × 48 = "19m" while the loss curve is 6 minutes from
//     the target it will actually stop on. Both were labelled "ETA".
//  2. DiT said "<1m" forever. The old rate was the mean of the POSITIVE deltas
//     of the RAW epoch loss over a 5-window. DiT's loss drops 1.27 → 0.96 in
//     five epochs and then plateaus at ~0.93 with ±0.05 of per-epoch noise, so:
//       • MIN_EPOCHS 3 published a rate built from the initial cliff
//         (0.116/epoch) against a target of 0.4 → 11 epochs → 43 s → "<1m";
//       • once flat, the noise still produced positive deltas every window, so
//         the mean-of-positives never fell to the stall threshold;
//       • and because the cap was far away (100, now 400 epochs) the
//         "target unreachable" branch — which compares epochs-to-target against
//         epochs-to-cap — could never fire either.
//
// The fixes: rate off the MA5 series (the same quantity the auto-stop compares
// to target, and the same violet line the chart draws), a MEDIAN over 8 deltas
// instead of a mean over 5, a per-kind minimum epoch count, and a one-epoch
// floor on the displayed number.
//
// Everything here is derived from the epoch series the SSE `metric:'epoch'`
// events already carry (epoch, loss, ms) — no new engine or server field.

/** The slice of TrainLmEpoch / TrainDitEpoch this module needs. Both are
 *  structurally identical, so one shape covers LM and DiT. */
export interface EtaEpochPoint {
  epoch: number;
  loss: number;
  ms: number;
}

/** The two job kinds whose progress is a loss curve rather than an item count. */
export type TrainingEtaKind = 'train-lm' | 'train-dit';

export type TrainingEta =
  /** Too few epochs for this kind, or no usable epoch duration yet. */
  | { kind: 'estimating' }
  /** MA5 is descending fast enough to reach `target` before the cap. */
  | { kind: 'target'; etaMs: number; target: number }
  /** Still descending, but the cap arrives first — `etaMs` is time to the cap. */
  | { kind: 'unlikely'; etaMs: number; target: number }
  /** MA5 has stopped falling (median delta <= 0) — `etaMs` is time to the cap. */
  | { kind: 'stalled'; etaMs: number; target: number }
  /** No target set, or the target is already met: time to the epoch cap. */
  | { kind: 'cap'; etaMs: number }
  /** Nothing sensible to say (no cap known and no target to chase). */
  | { kind: 'none' };

/** Window of the moving average the auto-stop watches — must match
 *  TrainingChart's MA_WINDOW and the engine's own ma5. */
const MA_WINDOW = 5;
/** How many trailing MA5 deltas the median is taken over. */
const RATE_WINDOW = 8;
/** Below this many descending deltas, keep the whole window instead. */
const MIN_DESCENDING = 3;
/** Epochs needed before a target ETA is worth showing. DiT's per-epoch loss is
 *  far noisier than the LM's and its first five epochs are a cliff, so it needs
 *  more evidence before anyone should read a number off it. */
const MIN_EPOCHS: Record<TrainingEtaKind, number> = { 'train-lm': 3, 'train-dit': 5 };
/** Trailing epochs the wall-clock average is taken over. */
const MS_WINDOW = 5;
/** A projection past this many epochs is not an estimate, it is a plateau. */
const MAX_PROJECTED_EPOCHS = 100000;

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * Trailing moving average, partial window at the head — identical to
 * TrainingChart's own `movingAverage`, so the rate is measured off the exact
 * line the user is looking at (and the exact quantity the engine's auto-stop
 * compares against target).
 */
export function movingAverage(values: number[], window = MA_WINDOW): number[] {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(i + 1, window));
  }
  return out;
}

/**
 * Median per-epoch descent of the MA5 series over the last RATE_WINDOW deltas.
 * Positive = falling. Returns 0 for "not moving / not enough data".
 *
 * MEDIAN, not mean: one noisy epoch on a plateau is a ±0.05 delta on a curve
 * whose real slope is ~0.001, and a mean lets that single sample set the whole
 * estimate. The median ignores it.
 *
 * Negative deltas are KEPT unless the window is dominated by descent: on a
 * plateau the up-ticks are the signal, and dropping them is precisely what made
 * the old estimator promise a target the run was never going to reach. The drop
 * only happens when at least MIN_DESCENDING descending deltas remain, i.e. when
 * the curve really is going down and the odd up-tick is noise.
 *
 * The median is then capped by the window's MEAN delta, which is exactly the NET
 * descent across the window divided by its length. A plateau produces a healthy
 * median every time the noise happens to line up, but its net movement over
 * eight epochs is ~0 — taking the smaller of the two means a run only gets
 * credit for ground it has actually covered. This is what carries the DiT
 * far-target case into the stalled/unlikely states instead of flapping back to
 * a cheerful "9m to target 0.40" on a curve that has been flat for 60 epochs.
 */
export function descentRate(ma5: number[]): number {
  if (ma5.length < 2) return 0;
  const deltas: number[] = [];
  for (let i = 1; i < ma5.length; i++) deltas.push(ma5[i - 1] - ma5[i]);
  const window = deltas.slice(-RATE_WINDOW);
  const descending = window.filter(d => d > 0);
  // Fewer than MIN_DESCENDING descending deltas means the window is flat or
  // rising — the full window (negatives and all) is the honest basis then.
  const typical = median(descending.length >= MIN_DESCENDING && descending.length > window.length / 2
    ? descending
    : window);
  return Math.min(typical, mean(window));
}

/** Mean wall-clock duration of the last MS_WINDOW epochs that reported one. */
export function avgEpochMs(epochs: EtaEpochPoint[]): number {
  const durations = epochs
    .map(e => e.ms)
    .filter(ms => Number.isFinite(ms) && ms > 0)
    .slice(-MS_WINDOW);
  return mean(durations);
}

/** Total wall-clock of every epoch that reported a duration. Used for the
 *  done-state "time taken", where no job.startedAt survives. */
export function totalEpochMs(epochs: EtaEpochPoint[]): number {
  return epochs.reduce((a, e) => a + (Number.isFinite(e.ms) && e.ms > 0 ? e.ms : 0), 0);
}

/**
 * Estimate remaining time from the live epoch series.
 *
 * @param epochs     live epoch series, ascending by epoch (the store keeps it sorted)
 * @param targetLoss target loss; <= 0 means auto-stop is disabled
 * @param maxEpochs  the run's epoch cap; 0 = unknown
 * @param kind       which training kind — sets the minimum epoch count
 */
export function computeTrainingEta(
  epochs: EtaEpochPoint[],
  targetLoss: number,
  maxEpochs: number,
  kind: TrainingEtaKind = 'train-lm',
): TrainingEta {
  const usable = epochs.filter(e => Number.isFinite(e.loss));
  if (usable.length < MIN_EPOCHS[kind]) return { kind: 'estimating' };

  const perEpochMs = avgEpochMs(usable);
  if (perEpochMs <= 0) return { kind: 'estimating' };

  const last = usable[usable.length - 1];
  const epochsLeft = maxEpochs > 0 ? Math.max(0, maxEpochs - last.epoch) : 0;
  const capMs = epochsLeft * perEpochMs;

  // The engine auto-stops on MA5, not on the raw epoch loss, so MA5 is what the
  // countdown has to be measured against — otherwise a lucky epoch reads as
  // "arrived" while the run keeps going.
  const ma5 = movingAverage(usable.map(e => e.loss));
  const lastMa = ma5[ma5.length - 1];

  // No target (or already met) — the epoch cap is the only honest deadline.
  if (!(targetLoss > 0) || lastMa <= targetLoss) {
    return capMs > 0 ? { kind: 'cap', etaMs: capMs } : { kind: 'none' };
  }

  const rate = descentRate(ma5);
  // Flat or rising: there is no arrival time to quote, only the cap.
  if (!(rate > 0)) return { kind: 'stalled', etaMs: capMs, target: targetLoss };

  const epochsToTarget = (lastMa - targetLoss) / rate;
  if (!Number.isFinite(epochsToTarget) || epochsToTarget > MAX_PROJECTED_EPOCHS) {
    return { kind: 'stalled', etaMs: capMs, target: targetLoss };
  }
  // A target the cap will arrive at first is not a target this run will hit, and
  // quoting a time past the cap would be a promise the job cannot keep.
  if (maxEpochs > 0 && epochsToTarget > epochsLeft) {
    return { kind: 'unlikely', etaMs: capMs, target: targetLoss };
  }
  // Floor at one epoch. Without it a single well-behaved window on a fast-epoch
  // run projects a fraction of an epoch and the header flaps on "<1m" while
  // there are still dozens of epochs to go.
  return {
    kind: 'target',
    etaMs: Math.max(epochsToTarget * perEpochMs, perEpochMs),
    target: targetLoss,
  };
}

/** Coarse duration label — minutes and hours, for forward-looking estimates. */
export function formatEtaMs(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '';
  const m = Math.round(ms / 60000);
  if (m < 1) return '<1m';
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** Precise duration label — for elapsed / total time, where seconds matter at
 *  the start of a run and would be silly at the end of one. */
export function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '';
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}

/** Seconds-per-epoch label for the run stats strip. */
export function formatEpochPace(msPerEpoch: number): string {
  if (!Number.isFinite(msPerEpoch) || msPerEpoch <= 0) return '';
  const s = msPerEpoch / 1000;
  return s >= 100 ? `${Math.round(s)}s` : `${formatDecimal(s, 1)}s`;
}
