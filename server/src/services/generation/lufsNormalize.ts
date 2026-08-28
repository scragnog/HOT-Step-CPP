// lufsNormalize.ts — LUFS normalization with peak limiting
//
// ITU-R BS.1770-4 integrated loudness measurement + gain adjustment. Runs as
// the final audio-modifying stage in the post-processing chain, after
// reference-based mastering.
//
//   1. Measure integrated loudness (audioLevel.integratedLufs)
//   2. Apply the gain needed to reach the target
//   3. Look-ahead limit to the ceiling
//
// The measurement and the WAV I/O live in audioLevel.ts, which owns level for
// the whole chain. This file used to carry its own copy of the BS.1770
// implementation, its own WAV parser, and a "true-peak limiter" that was a
// per-sample hard clip. All three are gone.
//
// Ceiling semantics, stated plainly: the ceiling is enforced on the sample
// peak, and the true peak is measured and reported but not enforced. Enforcing
// it would mean limiting in the oversampled domain. The previous code claimed
// true-peak limiting and did neither — it clipped. If `truePeakAfter` comes
// back above the ceiling, that is an inter-sample over surviving, and the log
// will say so.
//
// See docs/plans/2026-08-28-post-processing-gain-staging.md.

import {
  readWav, writeWav, measureLevels, integratedLufs, limitPeaks,
  applyGain, truePeak, linToDb, type WavSampleFormat,
} from './audioLevel.js';

export interface LufsResult {
  measuredLufs: number;    // integrated LUFS before normalization
  targetLufs: number;      // requested target
  appliedGainDb: number;   // actual gain applied
  limiterActive: boolean;  // true if the limiter engaged
  peakBefore: number;      // max absolute sample before gain
  peakAfter: number;       // max absolute sample after gain + limiting
  /** True peak after limiting, linear. Reported, not enforced — see above. */
  truePeakAfter: number;
  /** Largest gain reduction the limiter applied, in dB. */
  limiterReductionDb: number;
}

export interface LufsOptions {
  /** Peak ceiling in dBFS. Default -1. */
  ceilingDb?: number;
  /** Output sample format. Defaults to the input's, so this is safe to call
   *  on a 16-bit file; pass 'f32' when the chain is carrying float. */
  outFormat?: WavSampleFormat;
}

/**
 * Normalize a WAV file to a target integrated LUFS level, in place.
 *
 * @param wavPath     WAV file to modify
 * @param targetLufs  Target integrated LUFS (e.g. -14)
 * @param ceilingDb   Peak ceiling in dBFS (default -1)
 */
export function normalizeLufs(
  wavPath: string,
  targetLufs: number,
  ceilingDb: number = -1.0,
  opts: LufsOptions = {},
): LufsResult {
  const wav = readWav(wavPath);
  const ceiling = opts.ceilingDb ?? ceilingDb;

  const measuredLufs = integratedLufs(wav.channels, wav.sampleRate);

  if (!Number.isFinite(measuredLufs)) {
    // Silent or near-silent — nothing to normalize, and dividing into it would
    // produce an absurd gain.
    return {
      measuredLufs: -Infinity,
      targetLufs,
      appliedGainDb: 0,
      limiterActive: false,
      peakBefore: 0,
      peakAfter: 0,
      truePeakAfter: 0,
      limiterReductionDb: 0,
    };
  }

  const before = measureLevels(wav, { skipTruePeak: true });
  const gainDb = targetLufs - measuredLufs;

  const limit = applyGain(wav, gainDb, 'limit', { ceilingDb: ceiling });
  const after = measureLevels(wav, { skipTruePeak: true });

  writeWav(wavPath, wav, opts.outFormat ?? wav.sourceFormat);

  return {
    measuredLufs,
    targetLufs,
    appliedGainDb: gainDb,
    limiterActive: limit?.engaged ?? false,
    peakBefore: before.samplePeak,
    peakAfter: after.samplePeak,
    truePeakAfter: truePeak(wav.channels),
    limiterReductionDb: limit?.maxReductionDb ?? 0,
  };
}

/** One-line log summary, including whether an inter-sample over survived. */
export function formatLufsLog(r: LufsResult, label: string): string {
  if (!Number.isFinite(r.measuredLufs)) return `[LUFS] ${label}: silent, skipped`;
  const tpDb = linToDb(r.truePeakAfter);
  return `[LUFS] ${label}: ${r.measuredLufs.toFixed(1)} -> ${r.targetLufs.toFixed(1)} LUFS `
    + `(${r.appliedGainDb > 0 ? '+' : ''}${r.appliedGainDb.toFixed(1)} dB`
    + `${r.limiterActive ? `, limiter -${r.limiterReductionDb.toFixed(1)} dB` : ''})`
    + ` | peak ${linToDb(r.peakBefore).toFixed(1)} -> ${linToDb(r.peakAfter).toFixed(1)} dBFS`
    + `, true peak ${tpDb.toFixed(1)} dBTP`;
}

export { limitPeaks };
