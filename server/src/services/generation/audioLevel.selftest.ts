// audioLevel.selftest.ts — verification for the gain-staging module
//
//   cd server && npx tsx src/services/generation/audioLevel.selftest.ts
//
// The repo has no test runner, so this is a standalone tsx script. It exists
// because audioLevel.ts guards audio correctness for the whole post-processing
// chain: if the limiter's ceiling is not a guarantee, or f32 encode quietly
// clamps, the chain is back to clipping and nothing downstream would notice.
//
// See docs/plans/2026-08-28-post-processing-gain-staging.md.

import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  parseWav, encodeWav, measureLevels, truePeak, limitPeaks, applyGain,
  integratedLufs, linToDb, dbToLin, formatLevels, FLAT_TOP_RUN, type FloatWav,
} from './audioLevel.js';
import { normalizeLufs } from './lufsNormalize.js';

let failures = 0;
function check(name: string, ok: boolean, detail = ''): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
}

const SR = 48000;

function makeTone(freq: number, sec: number, amp: number, phase = 0): FloatWav {
  const n = Math.round(SR * sec);
  const l = new Float32Array(n), r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = amp * Math.sin((2 * Math.PI * freq * i) / SR + phase);
    l[i] = v; r[i] = v;
  }
  return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' };
}

const peakOf = (w: FloatWav) => measureLevels(w, { skipTruePeak: true }).samplePeak;

// ── 1. Round-trip through each format ───────────────────────────────────────

for (const fmt of ['s16', 's24', 'f32'] as const) {
  const src = makeTone(440, 0.5, 0.5);
  const rt = parseWav(encodeWav(src, fmt));
  let maxErr = 0;
  for (let i = 0; i < src.channels[0].length; i++) {
    maxErr = Math.max(maxErr, Math.abs(src.channels[0][i] - rt.channels[0][i]));
  }
  // s16 tolerance is one quantisation step (1/32768 = 3.05e-5). Encoding
  // scales by 32767 and decoding by 32768 — the standard asymmetric int16
  // convention, which the rest of this codebase also uses — so the error is a
  // half-step of rounding plus a systematic 1/32768 of the sample value.
  const tol = fmt === 's16' ? 3.2e-5 : fmt === 's24' ? 2e-7 : 1e-9;
  check(`round-trip ${fmt}`,
    maxErr < tol && rt.sourceFormat === fmt && rt.sampleRate === SR && rt.numChannels === 2,
    `maxErr=${maxErr.toExponential(2)} fmt=${rt.sourceFormat}`);
}

// ── 2. f32 preserves above-full-scale peaks ─────────────────────────────────
// The load-bearing claim for the whole float chain: a stage may hand on peaks
// above 0 dBFS and the next stage gets them intact.

const hot = makeTone(440, 0.2, 1.8);
check('f32 keeps +5 dBFS peaks', peakOf(parseWav(encodeWav(hot, 'f32'))) > 1.79,
  `peak=${peakOf(parseWav(encodeWav(hot, 'f32'))).toFixed(3)}`);
check('s16 clamps them, as it must', peakOf(parseWav(encodeWav(hot, 's16'))) <= 1.0,
  `peak=${peakOf(parseWav(encodeWav(hot, 's16'))).toFixed(3)}`);

// ── 3. True peak ────────────────────────────────────────────────────────────

// Exhaustive reference: same kernel, every sample, no candidate threshold.
// If the fast path ever misses a peak this disagrees.
function truePeakExhaustive(ch: Float32Array): number {
  const PHASES = 4, TAPS = 16, total = TAPS * PHASES, center = (total - 1) / 2;
  const h = new Float64Array(total);
  for (let i = 0; i < total; i++) {
    const x = (i - center) / PHASES;
    const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    h[i] = sinc * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (total - 1)));
  }
  const branches: Float64Array[] = [];
  for (let p = 0; p < PHASES; p++) {
    const b = new Float64Array(TAPS);
    let sum = 0;
    for (let k = 0; k < TAPS; k++) { b[k] = h[k * PHASES + p]; sum += b[k]; }
    for (let k = 0; k < TAPS; k++) b[k] /= sum;
    branches.push(b);
  }
  const half = TAPS >> 1;
  let peak = 0;
  for (let i = 0; i < ch.length; i++) peak = Math.max(peak, Math.abs(ch[i]));
  for (let i = 0; i < ch.length; i++) {
    for (let p = 0; p < PHASES; p++) {
      let acc = 0;
      for (let k = 0; k < TAPS; k++) {
        const idx = i + k - half;
        if (idx >= 0 && idx < ch.length) acc += branches[p][k] * ch[idx];
      }
      peak = Math.max(peak, Math.abs(acc));
    }
  }
  return peak;
}

// A tone at SR/4 on its zero crossings is the worst realistic inter-sample
// over: sample peak sits ~3 dB below the real amplitude.
const isp = makeTone(SR / 4, 0.2, 0.9, Math.PI / 4);
const ispM = measureLevels(isp);
check('truePeak >= samplePeak', ispM.truePeak >= ispM.samplePeak - 1e-6,
  `tp=${ispM.truePeakDb.toFixed(2)} sp=${ispM.samplePeakDb.toFixed(2)} dB`);
check('inter-sample over is found', ispM.truePeak > ispM.samplePeak * 1.3,
  `${(ispM.truePeakDb - ispM.samplePeakDb).toFixed(2)} dB above sample peak`);

// The candidate threshold must not cost accuracy.
for (const [label, sig] of [
  ['SR/4 zero-crossing tone', isp],
  ['mixed tones', (() => {
    const n = SR / 4;
    const l = new Float32Array(n), r = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = 0.4 * Math.sin(2 * Math.PI * 137 * i / SR)
              + 0.3 * Math.sin(2 * Math.PI * 4001 * i / SR)
              + 0.2 * Math.sin(2 * Math.PI * 11003 * i / SR + 1.1);
      l[i] = v; r[i] = v * 0.8;
    }
    return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' } as FloatWav;
  })()],
] as Array<[string, FloatWav]>) {
  const fast = truePeak(sig.channels);
  const slow = Math.max(...sig.channels.map(truePeakExhaustive));
  check(`candidate threshold loses nothing (${label})`, Math.abs(fast - slow) < 1e-6,
    `fast=${linToDb(fast).toFixed(4)} exhaustive=${linToDb(slow).toFixed(4)} dB`);
}

const quiet = measureLevels(makeTone(440, 0.2, 0.25));
check('truePeak sane on an ordinary tone', Math.abs(quiet.truePeakDb - linToDb(0.25)) < 0.5,
  `tp=${quiet.truePeakDb.toFixed(2)} dB, expected ~${linToDb(0.25).toFixed(2)}`);

// ── 4. The limiter's ceiling is a guarantee ─────────────────────────────────

for (const ceilingDb of [-1, -0.3, -6]) {
  const w = makeTone(220, 1.0, 1.6);
  const res = limitPeaks(w, { ceilingDb });
  check(`limiter holds ${ceilingDb} dBFS ceiling`,
    res.engaged && peakOf(w) <= dbToLin(ceilingDb) + 1e-6,
    `peak after ${linToDb(peakOf(w)).toFixed(3)} dBFS, GR ${res.maxReductionDb.toFixed(2)} dB`);
}

// Program material, not a steady tone: level steps and transients are where a
// smoothed envelope is most likely to overshoot its clamp.
const dynamic = (() => {
  const n = SR * 2;
  const l = new Float32Array(n), r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const env = i < n / 3 ? 0.2 : i < (2 * n) / 3 ? 1.9 : 0.35;
    const hit = i % 9600 < 60 ? 2.4 : 0;
    const v = (env + hit) * Math.sin((2 * Math.PI * 180 * i) / SR);
    l[i] = v; r[i] = v * 0.6;
  }
  return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' } as FloatWav;
})();
const dynRes = limitPeaks(dynamic, { ceilingDb: -1 });
check('ceiling holds on dynamic program material',
  peakOf(dynamic) <= dbToLin(-1) + 1e-6,
  `peak after ${linToDb(peakOf(dynamic)).toFixed(3)} dBFS, GR ${dynRes.maxReductionDb.toFixed(2)} dB`);

// ── 5. Transparent below the ceiling ────────────────────────────────────────

const soft = makeTone(220, 0.3, 0.4);
const softBefore = Float32Array.from(soft.channels[0]);
const softRes = limitPeaks(soft, { ceilingDb: -1 });
let softErr = 0;
for (let i = 0; i < softBefore.length; i++) {
  softErr = Math.max(softErr, Math.abs(softBefore[i] - soft.channels[0][i]));
}
check('limiter is bit-transparent below the ceiling', !softRes.engaged && softErr === 0,
  `maxErr=${softErr}`);

// ── 6. Same ceiling, no flat-topping ────────────────────────────────────────
// The point of the whole exercise: hold 0 dBFS without hard-clipped runs.

function burst(): FloatWav {
  const n = SR;
  const l = new Float32Array(n), r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const base = 0.3 * Math.sin((2 * Math.PI * 200 * i) / SR);
    const b = i > 20000 && i < 20400 ? 1.5 * Math.sin((2 * Math.PI * 900 * i) / SR) : 0;
    l[i] = base + b; r[i] = base + b;
  }
  return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' };
}

const limited = burst(); limitPeaks(limited, { ceilingDb: 0 });
const clipped = burst(); applyGain(clipped, 0, 'clip', { ceilingDb: 0 });
const mL = measureLevels(limited, { skipTruePeak: true });
const mC = measureLevels(clipped, { skipTruePeak: true });
// Pinning a sample exactly on a 0 dBFS ceiling counts as a full-scale sample,
// so the limiter is allowed a short run there — formatLevels only calls it
// clipping at 3 or more. What must not happen is the clipper's long flat top.
check('clip flat-tops, limiter does not, both at 0 dBFS',
  mC.longestClipRun > 5 && mL.longestClipRun < 3 && mL.samplePeak <= 1 + 1e-6,
  `clipped run ${mC.longestClipRun}, limited run ${mL.longestClipRun}, `
  + `limited peak ${mL.samplePeakDb.toFixed(3)} dB`);

// At the ceiling this chain actually uses, nothing should sit at full scale.
const limited1 = burst(); limitPeaks(limited1, { ceilingDb: -1 });
const m1 = measureLevels(limited1, { skipTruePeak: true });
check('limiter at -1 dBFS leaves no flat top',
  m1.longestClipRun < FLAT_TOP_RUN, `longest run ${m1.longestClipRun}`);

// ── 7. Flat-top detection ───────────────────────────────────────────────────
// Measured against the signal's own peak, not full scale, so it means the same
// thing on a finished 16-bit file and on a float file carrying +3 dBFS
// mid-chain. A clipper leaves runs; a limiter leaves isolated samples.

const hardClipped = makeTone(220, 0.3, 1.5);
applyGain(hardClipped, 0, 'clip', { ceilingDb: 0 });
const hcM = measureLevels(hardClipped, { skipTruePeak: true });
check('flat tops detected', hcM.longestClipRun > 10 && hcM.clippedSamples > 100,
  `${hcM.clippedSamples} samples, longest run ${hcM.longestClipRun}`);

const cleanM = measureLevels(makeTone(220, 0.3, 0.5), { skipTruePeak: true });
// Every signal has a sample or two sitting on its own peak; that is not a
// flat top, which is why the reporting threshold is a run length.
check('no false positive on a clean tone', cleanM.longestClipRun < FLAT_TOP_RUN,
  `longest run ${cleanM.longestClipRun}`);

// Regression: the first version of this counted everything at or above 0.9999,
// so a float file legitimately carrying peaks above full scale mid-chain
// reported tens of thousands of "clipped" samples that were merely loud. That
// warning fired on a real render before it was caught.
const hotClean = makeTone(220, 1.0, 1.43);   // ~+3.1 dBFS, no clipping anywhere
const hotM = measureLevels(hotClean, { skipTruePeak: true });
check('a hot but unclipped float signal is not called clipped',
  hotM.longestClipRun < FLAT_TOP_RUN,
  `peak ${hotM.samplePeakDb.toFixed(1)} dBFS, longest run ${hotM.longestClipRun}`);

// And a limiter's output: samples touch the ceiling but never in runs.
// Heavy sustained limiting on a steady tone is the worst case for this: the
// safety clamp pins samples exactly on the ceiling. It must still stay clear
// of a clipper's signature by a wide margin.
const limOut = makeTone(220, 1.0, 1.6);
limitPeaks(limOut, { ceilingDb: -1 });
const limM = measureLevels(limOut, { skipTruePeak: true });
check('limiter output is not called flat-topped', limM.longestClipRun < FLAT_TOP_RUN,
  `longest run ${limM.longestClipRun} vs a clipper's ${hcM.longestClipRun}`);

// A 16-bit file clipped upstream reads back a hair under 1.0 — detection has
// to see that, or the chain cannot tell it inherited damage.
const s16Clipped = parseWav(encodeWav(hardClipped, 's16'));
const s16M = measureLevels(s16Clipped, { skipTruePeak: true });
check('clipping survives an s16 round-trip and is still detected',
  s16M.longestClipRun > 10, `longest run ${s16M.longestClipRun}`);

// ── 8. Integrated LUFS ──────────────────────────────────────────────────────
// BS.1770 sums L and R at weight 1.0, so dual mono is +3 dB over one channel,
// which cancels the sine's -3 dB RMS. With K-weighting adding a little under
// +1 dB at 1 kHz, a -20 dBFS dual-mono sine lands near -20 LUFS. This is the
// standard alignment point.

const lufs = integratedLufs(makeTone(1000, 3.0, dbToLin(-20)).channels, SR);
check('LUFS of a -20 dBFS 1 kHz sine ~= -20', Math.abs(lufs - -20) < 1.0, `${lufs.toFixed(2)} LUFS`);

const lufsQuiet = integratedLufs(makeTone(1000, 3.0, dbToLin(-30)).channels, SR);
check('LUFS tracks a 10 dB level drop', Math.abs(lufsQuiet - lufs - -10) < 0.2,
  `${lufsQuiet.toFixed(2)} vs ${lufs.toFixed(2)} LUFS`);

check('LUFS of silence is -Infinity',
  integratedLufs(makeTone(1000, 1.0, 0).channels, SR) === -Infinity);

// The BS.1770 code here was lifted out of lufsNormalize.ts, which now calls
// into it, so the two agreeing proves nothing any more. What is worth pinning
// is the value itself: -11.934 LUFS is what lufsNormalize.ts's own
// implementation returned for this exact signal before the extraction
// (verified at the time, commit 9aa28586). Any drift here is a regression in
// the loudness measurement, not a new opinion about it.
{
  const PINNED_LUFS = -11.934;
  const music = (() => {
    const n = SR * 4;
    const l = new Float32Array(n), r = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      const env = 0.25 + 0.2 * Math.sin(2 * Math.PI * 0.7 * t);
      const v = env * (Math.sin(2 * Math.PI * 110 * i / SR)
                     + 0.5 * Math.sin(2 * Math.PI * 523 * i / SR)
                     + 0.25 * Math.sin(2 * Math.PI * 2637 * i / SR + 0.9));
      l[i] = v * 0.9; r[i] = v * 0.7;
    }
    return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' } as FloatWav;
  })();

  const tmp = path.join(os.tmpdir(), `audiolevel-selftest-${process.pid}.wav`);
  fs.writeFileSync(tmp, encodeWav(music, 's16'));
  try {
    const mine = integratedLufs(parseWav(fs.readFileSync(tmp)).channels, SR);
    check('LUFS matches the pre-extraction implementation',
      Math.abs(mine - PINNED_LUFS) < 0.01,
      `${mine.toFixed(3)} vs pinned ${PINNED_LUFS} LUFS`);

    // End-to-end: normalizeLufs should hit its target and hold the ceiling.
    const res = normalizeLufs(tmp, -14, -1);
    const out = measureLevels(parseWav(fs.readFileSync(tmp)), { skipTruePeak: true });
    check('normalizeLufs reaches its target',
      Math.abs(integratedLufs(parseWav(fs.readFileSync(tmp)).channels, SR) - -14) < 0.5,
      `applied ${res.appliedGainDb.toFixed(2)} dB`);
    check('normalizeLufs holds its ceiling', out.samplePeak <= dbToLin(-1) + 1e-6,
      `peak ${linToDb(out.samplePeak).toFixed(3)} dBFS`);
    check('normalizeLufs no longer flat-tops', out.longestClipRun < FLAT_TOP_RUN,
      `longest run ${out.longestClipRun}`);
  } finally {
    try { fs.unlinkSync(tmp); } catch { /* best effort */ }
  }
}

// ── 9. Gain arithmetic and overflow modes ───────────────────────────────────

const g = makeTone(440, 0.2, 0.25);
applyGain(g, 6.0206, 'passthrough');
check('+6 dB doubles amplitude', Math.abs(peakOf(g) - 0.5) < 1e-3, `peak=${peakOf(g).toFixed(4)}`);

const gp = makeTone(440, 0.2, 0.9);
applyGain(gp, 6, 'passthrough');
check('passthrough leaves overs alone', peakOf(gp) > 1.7, `peak=${peakOf(gp).toFixed(3)}`);

const gl = makeTone(440, 0.2, 0.9);
const glRes = applyGain(gl, 6, 'limit', { ceilingDb: -1 });
check('limit mode brings the same gain back under the ceiling',
  glRes?.engaged === true && peakOf(gl) <= dbToLin(-1) + 1e-6,
  `peak=${linToDb(peakOf(gl)).toFixed(3)} dBFS`);

// ── 10. Stereo image survives limiting ──────────────────────────────────────

const st = (() => {
  const n = SR / 2;
  const l = new Float32Array(n), r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    l[i] = 1.4 * Math.sin((2 * Math.PI * 300 * i) / SR);
    r[i] = 0.7 * Math.sin((2 * Math.PI * 300 * i) / SR);
  }
  return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' } as FloatWav;
})();
limitPeaks(st, { ceilingDb: -1 });
let ratioErr = 0;
for (let i = 0; i < st.channels[0].length; i++) {
  if (Math.abs(st.channels[0][i]) > 1e-4) {
    ratioErr = Math.max(ratioErr, Math.abs(st.channels[1][i] / st.channels[0][i] - 0.5));
  }
}
check('L/R ratio preserved through limiting', ratioErr < 1e-4,
  `maxRatioErr=${ratioErr.toExponential(2)}`);

// ── 11. Metering cost at a realistic length ─────────────────────────────────
// Per-stage logging is only affordable if this stays well under a second.

const long = makeTone(440, 240, 0.8);
const t0 = Date.now();
const longM = measureLevels(long);
const ms = Date.now() - t0;
check('4 min stereo meters in under 3 s', ms < 3000, `${ms} ms`);
console.log(`\n  ${formatLevels(longM, '4 min reference tone')}`);

console.log(`\n${failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`}`);
process.exit(failures === 0 ? 0 : 1);
