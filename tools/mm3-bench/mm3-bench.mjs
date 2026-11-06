// mm3-bench.mjs - fixed-seed MiniMax-Music3 renders with per-stage timings.
//
// Talks to the engine directly (default http://127.0.0.1:8085) so the numbers
// are the engine's own: warm (model load), AR planner, flow, vocoder, total,
// plus a SHA-256 of the WAV so a change that should be bit-neutral can be
// proven so. Run it with the GPU otherwise idle.
//
// Usage:
//   node tools/mm3-bench/mm3-bench.mjs [--engine URL] [--runs N] [--duration S] [--steps N]
//        [--seed N] [--renderer ggml|tensorrt] [--lm Q] [--depth Q] [--dit Q]
//        [--extra '{"json":"merged into the request"}'] [--wav-dir DIR] [--label TEXT]
//
// --lm/--depth/--dit select per-role quants for the run (every role is sent,
// see the select-model trap in the mm3-backend skill) and the previous
// selection is restored afterwards. --extra lets an experiment pass any
// /mm3/synth field without editing this file.

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const ENGINE = opt('engine', 'http://127.0.0.1:8085');
const RUNS = Number(opt('runs', '2'));
const DURATION = Number(opt('duration', '60'));
const STEPS = Number(opt('steps', '30'));
const SEED = Number(opt('seed', '4242'));
const RENDERER = opt('renderer', 'tensorrt');
const EXTRA = JSON.parse(opt('extra', '{}'));
const WAV_DIR = opt('wav-dir', '');
const LABEL = opt('label', '');
const ROLES = ['lm', 'depth', 'cond', 'dit', 'voc'];

const CAPTION = [
  'Global Metadata',
  'Basic Attributes: bpm is 120. key is A, and scale is minor. Synth Pop.',
  'Sonics & Production Profile: Clean, punchy mix with warm analog synths and a tight drum machine.',
  'Vocal Details',
  'Vocal Gender & Timbre: Female singer with a clear, bright timbre.',
  'Arrangement',
  'Driving bassline, gated pads, four-on-the-floor kick, chorus lifts with layered synths.',
].join('\n');
const LYRICS = [
  '[verse]', 'City lights are calling out my name', 'Every street a different kind of flame',
  '[chorus]', 'We run until the morning comes', 'Hearts in time with the drums',
].join('\n');

const sleep = ms => new Promise(r => setTimeout(r, ms));
const getJson = async (path, init) => {
  const r = await fetch(`${ENGINE}${path}`, init);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
};
const quantOf = name => (name || '').replace(/^mm3-[a-z]+-/, '').replace(/\.gguf$/, '');

async function selectModels(sel) {
  await getJson('/mm3/select-model', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(sel),
  });
}

async function render(run) {
  const t0 = Date.now();
  const sub = await getJson('/mm3/synth', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      caption: CAPTION, lyrics: LYRICS, duration: DURATION, seed: SEED, steps: STEPS,
      get_wav_bits: 16, get_lrc: false, reuse_ar: false, dit_backend: RENDERER, ...EXTRA,
    }),
  });
  let d;
  for (;;) {
    d = await getJson(`/mm3/job?id=${sub.job_id}`);
    if (d.status === 'done' || d.status === 'failed' || d.status === 'cancelled') break;
    await sleep(300);
  }
  if (d.status !== 'done') throw new Error(`job ${d.status}: ${d.error || ''}`);
  const wav = Buffer.from(await (await fetch(`${ENGINE}/job?id=${sub.job_id}&result=1`)).arrayBuffer());
  const sha = createHash('sha256').update(wav).digest('hex').slice(0, 16);
  if (WAV_DIR) {
    mkdirSync(WAV_DIR, { recursive: true });
    writeFileSync(join(WAV_DIR, `${LABEL || 'run'}-${run}.wav`), wav);
  }
  const ms = d.result?.ms || {};
  return {
    run, wall: (Date.now() - t0) / 1000, warm: d.warm_ms / 1000, ar: ms.ar / 1000, flow: ms.flow / 1000,
    voc: ms.voc / 1000, total: ms.total / 1000, audio: d.result?.duration_sec, frames: d.result?.frames, sha,
  };
}

const props = await getJson('/mm3/props');
const before = Object.fromEntries(ROLES.map(r => [r, quantOf(props.files?.[r]?.name)]));
const wanted = { ...before };
for (const r of ['lm', 'depth', 'dit']) if (opt(r)) wanted[r] = opt(r);
const changed = ROLES.some(r => wanted[r] !== before[r]);
if (changed) await selectModels(wanted);

console.log(`mm3-bench ${LABEL} renderer=${RENDERER} steps=${STEPS} seed=${SEED} duration=${DURATION}s ` +
            `models=${ROLES.map(r => `${r}:${wanted[r]}`).join(',')} extra=${JSON.stringify(EXTRA)}`);
console.log('run   wall    warm     ar   flow    voc  total  audio  frames  sha256/16');
const rows = [];
try {
  for (let i = 1; i <= RUNS; i++) {
    const r = await render(i);
    rows.push(r);
    const f = v => (v ?? NaN).toFixed(1).padStart(6);
    console.log(`${String(i).padStart(3)} ${f(r.wall)} ${f(r.warm)} ${f(r.ar)} ${f(r.flow)} ${f(r.voc)} ${f(r.total)} ` +
                `${f(r.audio)} ${String(r.frames).padStart(7)}  ${r.sha}`);
  }
} finally {
  if (changed) await selectModels(before);
}
const same = rows.every(r => r.sha === rows[0].sha);
console.log(`MM3 BENCH: ${rows.length} run(s), output ${same ? 'identical across runs' : 'DIFFERS across runs'}`);
