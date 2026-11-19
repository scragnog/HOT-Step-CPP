/**
 * `x` confined to 0..1 for progress, fades and drag fractions. A non-finite
 * value becomes 0, so a missing measurement cannot put a bar or a handle at
 * `NaN%` and strand it there.
 */
export function clamp01(x: number): number {
  return Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0;
}
