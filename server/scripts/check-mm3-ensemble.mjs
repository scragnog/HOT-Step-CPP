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
//   node server/scripts/check-mm3-ensemble.mjs [--url http://127.0.0.1:8085] [--fixtures DIR] [--adapter PATH]
//
// --adapter adds the plan-dial checks (section 4) with that MM3 LM adapter.
//
// Reference fixtures are optional and are not distributed with the app. To
// regenerate them, instrument the original bf16 MM3 reference on one prompt:
// save its conditional/unconditional token IDs as little-endian int32
// tok_ids_{cond,uncond}.bin; save the conditional then unconditional final LM
// hidden rows as float32 lm_i0_last_hidden.bin [2,4096]. For decode parity,
// capture four AR iterations, including the un-emitted iteration 0. Save
// codes_semantic_all.bin [I] and codes_acoustic_all.bin [I,7] as int32, plus
// float32 lm_i{1,2,3}_{last_hidden,semantic_logits,depth_hidden}.bin with shapes
// [2,4096], [2,16384] (semantic span only), and [7,4096]. Capture on that same
// sampled trajectory, then pass its directory with --fixtures DIR. An engine
// dump is not an independent reference. Missing sections are reported as skips.
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
let skipped = 0;
const fail = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };
const pass = (msg) => console.log(`  ok    ${msg}`);
const skip = (msg) => { console.log(`  SKIP  ${msg}`); skipped++; };
const missingFixtures = (files) => files.filter((f) => !fs.existsSync(path.join(FIX, f)));

const readI32 = (p) => {
  const b = fs.readFileSync(p);
  return Array.from(new Int32Array(b.buffer, b.byteOffset, b.length / 4));
};
const readF32 = (p) => {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
};

function compare(a, b) {
  if (!a.length || a.length !== b.length || !a.every(Number.isFinite) || !b.every(Number.isFinite)) {
    throw new Error(`invalid reference comparison: lengths ${a.length}/${b.length}, expected equal finite arrays`);
  }
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
  if (!saa || !sbb || !sr) throw new Error('invalid reference comparison: zero variance or reference energy');
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
const missingForward = missingFixtures(['lm_i0_last_hidden.bin', 'tok_ids_cond.bin', 'tok_ids_uncond.bin']);
if (missingForward.length) {
  skip(`reference prefill unavailable: ${missingForward.join(', ')}; see the capture recipe in this script, then pass --fixtures DIR`);
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

// ── 1b. the FOLDED DECODE kernel must be correct, not merely fast ───────────
//
// The fold (mm3_lm_mm / mm3_depth_block) swaps which CUDA kernel runs for every
// decode-step matmul, so it must be checked on the DECODE path — section [1]
// only exercises the prefill, where no fold happens. Forced replay feeds the
// fixture's own code sequence so the graph sees exactly the tokens the
// reference saw, which makes the hidden states and logits directly comparable
// even though the reference's multinomial draw is not reproducible.
//
// This, not seed reproducibility, is the bar the fold has to clear: a different
// song from the same seed is harmless (nobody heard the other one), but a
// WRONG forward is not.
console.log('\n[1b] folded decode path vs the reference, via forced replay');
{
  const need = ['codes_semantic_all.bin', 'codes_acoustic_all.bin', 'tok_ids_cond.bin', 'tok_ids_uncond.bin',
    ...[1, 2, 3].flatMap((it) => ['last_hidden', 'semantic_logits', 'depth_hidden'].map((name) => `lm_i${it}_${name}.bin`))];
  const missingDecode = missingFixtures(need);
  if (missingDecode.length) {
    skip(`reference decode unavailable: ${missingDecode.join(', ')}; see the capture recipe in this script, then pass --fixtures DIR`);
  } else {
    const H_ = H, SV = 16384, NC = 7;
    const semAll = readI32(path.join(FIX, 'codes_semantic_all.bin'));
    const acAll = readI32(path.join(FIX, 'codes_acoustic_all.bin'));
    const idsCond = readI32(path.join(FIX, 'tok_ids_cond.bin'));
    const idsUncond = readI32(path.join(FIX, 'tok_ids_uncond.bin'));
    const ITERS = 4;

    const url = `${URL_BASE}/mm3/lm-plan?dump=${ITERS}`;
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        ids_cond: idsCond, ids_uncond: idsUncond,
        forced_semantic: semAll, forced_acoustic: acAll,
        max_frames: ITERS + 2, seed: 4242, takes: 1,
      }),
    });
    if (!r.ok) {
      fail(`forced replay: ${r.status} ${(await r.text()).slice(0, 200)}`);
    } else {
      const buf = Buffer.from(await r.arrayBuffer());
      const all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
      // Body layout (mm3-server.h): prefill_hidden, then per iteration
      // last_hidden[2H] sem_logits[2SV] guided[SV] feedback[2H] depth_hidden[NC*H].
      const per = 2 * H_ + 2 * SV + SV + 2 * H_ + NC * H_;
      for (let it = 1; it < ITERS; it++) {
        const base = 2 * H_ + it * per;
        if (base + per > all.length) { fail(`dump truncated at iteration ${it}`); break; }
        const lh = all.subarray(base, base + 2 * H_);
        const sl = all.subarray(base + 2 * H_, base + 2 * H_ + 2 * SV);
        const dh = all.subarray(base + 2 * H_ + 2 * SV + SV + 2 * H_,
                                base + 2 * H_ + 2 * SV + SV + 2 * H_ + NC * H_);
        const checks = [
          ['last_hidden', lh, `lm_i${it}_last_hidden.bin`],
          ['sem_logits ', sl, `lm_i${it}_semantic_logits.bin`],
          ['depth_hidden', dh, `lm_i${it}_depth_hidden.bin`],
        ];
        for (const [name, got, file] of checks) {
          const fp = path.join(FIX, file);
          const ref = readF32(fp);
          const st = compare(got, ref);
          const line = `iter ${it} ${name}: corr ${st.corr.toFixed(7)} rmse ${st.relRmse.toExponential(3)}`;
          if (st.corr < MIN_CORR) fail(`${line}  (below ${MIN_CORR})`);
          else pass(line);
        }
      }
    }
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

// ── 4. the plan dial (lm_adapter_plan), with --adapter PATH ─────────────────
// At 1 it must be today's render exactly (no base rows, same draws); a dial
// below 1 in between must leave that intact. The 16-row LM budget supports
// four takes with base rows and CFG, matching the depth decoder's take cap.
const ADAPTER = arg('--adapter', '');
if (ADAPTER) {
  console.log('\n[4] plan dial');
  const body = { prompt: PROMPT, max_frames: 200, seed: 77, takes: 3, lm_adapter: ADAPTER };
  const sig = (r) => r.per_take.map((t) => t.semantic.join(',')).join('|');
  const ref = await plan(body);
  const one = await plan({ ...body, lm_adapter_plan: 1 });
  if (sig(one) !== sig(ref)) fail('plan 1.0 differs from an omitted plan dial');
  else pass('plan 1.0 identical to no plan dial');
  const zero = await plan({ ...body, lm_adapter_plan: 0 });
  if (Number(zero.takes) !== 3 || zero.per_take.length !== 3) fail(`plan 0 planned ${zero.takes} takes of 3; expected all 3 within the 16-row LM budget`);
  else pass('plan 0: 3 takes asked, 3 planned (within the 16-row LM budget)');
  if (sig(zero) === sig(ref))
    fail('plan 0 drew the same codes as plan 1');
  else pass('plan 0 draws a different song from plan 1');
  const after = await plan(body);
  if (sig(after) !== sig(ref)) fail('a plan-1 render changed after a plan-0 render; the graph rebuild is not clean');
  else pass('plan 1 identical after an intervening plan-0 render');
  const capped = await plan({ ...body, max_frames: 4, takes: 5, lm_adapter_plan: 0 });
  if (Number(capped.takes) !== 4 || capped.per_take.length !== 4) fail(`plan 0 planned ${capped.takes} takes of 5; expected the 4-take cap`);
  else pass('plan 0: 5 takes asked, 4 planned (16-row LM and 8-row depth budgets)');
  await plan({ prompt: PROMPT, max_frames: 4, seed: 1, lm_adapter_clear: true });
} else {
  skip('plan dial: pass --adapter PATH to a runtime MM3 LM adapter without an artist token or KV prefix');
}

console.log(failures ? `\n${failures} FAILURE(S); ${skipped} section(s) skipped` : `\nPASS: all available ensemble checks passed; ${skipped} section(s) skipped`);
process.exitCode = failures ? 1 : 0;
