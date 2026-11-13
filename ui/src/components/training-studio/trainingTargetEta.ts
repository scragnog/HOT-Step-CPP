// trainingTargetEta.ts — the loss-target ETA text shared by the YuE2 joint and MiniMax-Music3 cards.
import { descentRate, formatDurationMs } from '../../utils/trainingEta';

/** Descent per step of a trailing-mean series; null until there are enough readings to trust. */
export function lossDescentRate(means: number[]): number | null {
  return means.length >= 9 ? descentRate(means) : null;
}

/** estimating / reached / stalled / unlikely before the cap / ETA, each with the cap time. */
export function lossTargetEta(i: { current: number | undefined; rate: number | null; target: number; remaining: number; paceMs: number; capMs: number }): string {
  const cap = formatDurationMs(i.capMs);
  if (i.rate === null || i.current === undefined) return `target ETA estimating · cap ${cap}`;
  if (i.current <= i.target) return `target reached · cap ${cap}`;
  if (!(i.rate > 0)) return `target trend stalled · cap ${cap}`;
  const stepsToTarget = (i.current - i.target) / i.rate;
  if (stepsToTarget > i.remaining) return `target unlikely before cap · cap ${cap}`;
  return `target ETA ${formatDurationMs(Math.max(1, stepsToTarget) * i.paceMs)}`;
}
