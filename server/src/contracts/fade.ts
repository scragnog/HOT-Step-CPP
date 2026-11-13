// fade.ts — the playbar's manual fade-in / fade-out shapes.
//
// Shared by the server, which writes the fade into the WAV, and the UI, which
// draws it over the waveform and previews it on the player volume. One curve
// function, so what you hear before applying is what gets written.

export type FadeKind = 'in' | 'out';
export type FadeCurve = 'linear' | 'exponential' | 'logarithmic' | 'scurve';

export const FADE_CURVES: readonly FadeCurve[] = ['linear', 'exponential', 'logarithmic', 'scurve'];

/** Fade-in gain at progress `p` (0 = start of the fade, 1 = end), 0 → 1.
 *  A fade-out uses the mirror, `fadeInGain(curve, 1 - p)`. */
export function fadeInGain(curve: FadeCurve, p: number): number {
  const x = Math.max(0, Math.min(1, p));
  switch (curve) {
    case 'exponential': return x * x;                          // slow start, fast finish
    case 'logarithmic': return 1 - (1 - x) * (1 - x);          // fast start, gentle finish
    case 'scurve': return 0.5 - 0.5 * Math.cos(Math.PI * x);   // eased at both ends
    default: return x;
  }
}

/** Gain at time `t` (seconds) for a fade of `length` seconds on a track of
 *  `duration` seconds. 1 outside the fade. */
export function fadeGainAt(kind: FadeKind, curve: FadeCurve, length: number, duration: number, t: number): number {
  if (!(length > 0)) return 1;
  if (kind === 'in') return t >= length ? 1 : fadeInGain(curve, t / length);
  const start = duration - length;
  return t <= start ? 1 : fadeInGain(curve, (duration - t) / length);
}
