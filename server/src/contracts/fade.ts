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

/** A pending crop and fade of both ends, in seconds on the current file. */
export interface EndEdit {
  cropIn: number;
  /** null = keep the end of the track. */
  cropOut: number | null;
  fadeIn: number;
  fadeOut: number;
  curveIn: FadeCurve;
  curveOut: FadeCurve;
}

/** Gain at time `t` under a pending edit: 0 where it is cut, the fade curve
 *  inside each fade, 1 elsewhere. The waveform draws this and the player
 *  previews it, so what you see is what you hear. */
export function endEditGainAt(e: EndEdit, duration: number, t: number): number {
  const a = e.cropIn;
  const b = e.cropOut ?? duration;
  if (t < a || t > b) return 0;
  let g = 1;
  if (e.fadeIn > 0 && t < a + e.fadeIn) g *= fadeInGain(e.curveIn, (t - a) / e.fadeIn);
  if (e.fadeOut > 0 && t > b - e.fadeOut) g *= fadeInGain(e.curveOut, (b - t) / e.fadeOut);
  return g;
}
