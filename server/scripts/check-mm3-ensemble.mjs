#!/usr/bin/env node
// check-mm3-ensemble.mjs — the ensemble's correctness bar.
//
// Ensemble mode decodes K independent songs from one prompt in a single batched
// AR pass (engine/src/minimax/mm3-ar-loop.h). Two things have to be true, and
// neither can be checked by eye:
//
//   1. NO REGRESSION AT ONE TAKE. A single-track render must be exactly the
//      render it was before ensembles existed — same seed, same song.
//   2. WIDENING THE BATCH MUST NOT DISTURB THE MATH. Take 0's forward at K = 4
//      has to match the reference dump as well as it does at K = 1.
//
// WHY NOT JUST COMPARE THE CODES ACROSS K: you cannot. The AR sampler is a
// top-k multinomial, so a 1-ulp difference flips one draw and the sequences
// diverge forever after. Different K genuinely produces a different (equally
// valid) song from the same seed — the upstream block documents the same thing.
// So correctness is judged on the FORWARD, against mm3-weights/fixtures, and
// only same-K comparisons are made on codes.
//
// Likewise do NOT use `prefill_sum` as an equality test — it sums 4096 signed
// activations that nearly cancel, so it swings wildly for rows that agree to
// corr 0.9994. It is a tripwire, not a verdict.
//
// Usage:
//   node server/scripts/check-mm3-ensemble.mjs [--url http://127.0.0.1:8085] [--fixtures DIR]
//
// The engine must be warm (POST /mm3/warm) with the LM and depth resident.

import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const URL_BASE = arg('--url', 'http://127.0.0.1:8085').replace(/\/$/, '');
const FIX = arg('--fixtures', path.resolve('../mm3-weights/fixtures'));
const H = 4096;
const TAKES = [1, 2, 3, 4];

// A correlation this far below the reference is a real defect; the bf16 dump's
// own noise floor sits around 1.7e-2 relative RMSE, so the gate is deliberately
// loose on RMSE and tight on correlation.
const MIN_CORR = 0.999;

let failures = 0;
const fail = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };
const pass = (msg) => console.log(`  ok    ${msg}`);

const readI32 = (p) => {
  const b = fs.readFileSync(p);
  return Array.from(new Int32Array(b.buffer, b.byteOffset, b.length / 4));
};
const readF32 = (p) => {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
};

function compare(a, b) {
  let sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / a.length, mb = sb / a.length;
  let sab = 0, saa = 0, sbb = 0, se = 0, sr = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    sab += x * y; saa += x * x; sbb += y * y;
    const d = a[i] - b[i];
    se += d * d; sr += b[i] * b[i];
  }
  return { corr: sab / Math.sqrt(saa * sbb), relRmse: Math.sqrt(se / a.length) / Math.sqrt(sr / a.length) };
}

async function plan(body, { dump = 0 } = {}) {
  const url = `${URL_BASE}/mm3/lm-plan${dump ? `?dump=${dump}` : ''}`;
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  if (dump) {
    const buf = Buffer.from(await r.arrayBuffer());
    // The binary body opens with prefill_hidden = [2, H] floats.
    return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + 2 * H * 4));
  }
  return r.json();
}

const PROMPT = [
  '[Structured Caption]',
  'Global Metadata: Genre: electronic/synthwave. BPM: 110. Key: A minor. Mood: driving, nocturnal.',
  'Vocal Details: Male vocals, mid register, slight rasp.',
  'Arrangement: Analog synth bass, gated drums, arpeggiated pads, wide reverb.',
  '[Lyrics]',
  '[verse]',
  'Neon runs the length of the street',
].join('\n');

console.log(`engine   ${URL_BASE}`);
console.log(`fixtures ${FIX}`);

// ── 1. the forward, against the reference, at every take count ──────────────
console.log('\n[1] take 0 forward vs the reference bf16 dump, per take count');
const refPath = path.join(FIX, 'lm_i0_last_hidden.bin');
if (!fs.existsSync(refPath)) {
  console.log(`  SKIP  no fixtures at ${FIX} (pass --fixtures DIR)`);
} else {
  const ref = readF32(refPath);
  const idsCond = readI32(path.join(FIX, 'tok_ids_cond.bin'));
  const idsUncond = readI32(path.join(FIX, 'tok_ids_uncond.bin'));
  const base = {};
  for (const k of TAKES) {
    const ph = await plan(
      { ids_cond: idsCond, ids_uncond: idsUncond, max_frames: 4, seed: 4242, takes: k },
      { dump: 1 },
    );
    const c = compare(ph.subarray(0, H), ref.subarray(0, H));
    const u = compare(ph.subarray(H, 2 * H), ref.subarray(H, 2 * H));
    if (k === 1) { base.c = c.corr; base.u = u.corr; }
    const line = `K=${k} cond corr ${c.corr.toFixed(7)} rmse ${c.relRmse.toExponential(3)} | ` +
                 `uncond corr ${u.corr.toFixed(7)} rmse ${u.relRmse.toExponential(3)}`;
    // Absolute floor, plus a guard that widening the batch has not DEGRADED it
    // relative to the single-take baseline by more than dump-noise.
    if (c.corr < MIN_CORR || u.corr < MIN_CORR) fail(`${line}  (below ${MIN_CORR})`);
    else if (base.c - c.corr > 1e-3 || base.u - u.corr > 1e-3) fail(`${line}  (degraded vs K=1)`);
    else pass(line);
  }
}

// ── 2. takes must be different songs, and reproducible ──────────────────────
console.log('\n[2] takes are distinct songs, and the batch is deterministic');
for (const k of TAKES.filter((k) => k > 1)) {
  const a = await plan({ prompt: PROMPT, max_frames: 120, seed: 4242, takes: k });
  const sigs = a.per_take.map((t) => t.semantic.join(','));
  const uniq = new Set(sigs).size;
  if (uniq !== k) fail(`K=${k}: only ${uniq}/${k} distinct code sequences — takes are duplicating`);
  else pass(`K=${k}: ${k}/${k} distinct code sequences`);

  const b = await plan({ prompt: PROMPT, max_frames: 120, seed: 4242, takes: k });
  const same = b.per_take.every((t, i) => t.semantic.join(',') === sigs[i]);
  if (!same) fail(`K=${k}: re-running the same seed gave different takes — not deterministic`);
  else pass(`K=${k}: identical on a re-run (deterministic)`);

  const seeds = a.per_take.map((t) => Number(t.seed));
  const want = seeds.map((_, i) => 4242 + i);
  if (seeds.join(',') !== want.join(',')) fail(`K=${k}: seeds ${seeds} (expected ${want})`);
  else pass(`K=${k}: seeds are base + take index`);
}

// ── 3. the take count must not change a single-track render ─────────────────
console.log('\n[3] one take is unaffected by ensemble support');
{
  const solo = await plan({ prompt: PROMPT, max_frames: 200, seed: 909, takes: 1 });
  const again = await plan({ prompt: PROMPT, max_frames: 200, seed: 909, takes: 1 });
  if (solo.semantic.join(',') !== again.semantic.join(','))
    fail('a K=1 render is not reproducible across calls');
  else pass(`K=1 reproducible (${solo.frames} frames, eos=${solo.eos})`);

  // A K=1 render sandwiched between ensemble renders must still match, i.e.
  // the graph teardown/rebuild that a take-count change forces is clean.
  await plan({ prompt: PROMPT, max_frames: 60, seed: 1, takes: 4 });
  const after = await plan({ prompt: PROMPT, max_frames: 200, seed: 909, takes: 1 });
  if (after.semantic.join(',') !== solo.semantic.join(','))
    fail('a K=1 render changed after an ensemble render — graph rebuild is not clean');
  else pass('K=1 identical after an intervening K=4 render');
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
