#!/usr/bin/env node
// YuE2 adapter ladder: render a joint run's checkpoints with the planner (AR)
// and decoder (NAR) taken from DIFFERENT steps, through the app's own
// generate path, then score each render's diction with the engine's forced
// aligner (POST /yue2/align, the MMS_FA model the trainer builds its lyric
// cursor with). Writes numbered WAVs plus results.jsonl for the ear test.
//
//   node server/scripts/yue2-ladder.mjs <config.json>
//
// config: { runDir, outDir, caption, lyrics, seed, pairs: [[arStep, narStep], ...], takes? }
//   takes > 1 renders each pair that many times (seed, seed+1, ...) as -t1, -t2 files:
//   renders are not reproducible from the seed, so one take per rung is noisy.
//   a pair of [0, 0] renders the base model with no adapter (diction reference).
// Idempotent: a pair whose WAV already exists is skipped, so a crashed or
// interrupted ladder resumes by running it again.
//
// Needs the app running (Node :3001, engine :8085) with YuE2 as the active
// backend. It changes the YuE2 adapter picks and restores the originals at the end.
import fs from 'node:fs';
import path from 'node:path';

const APP = process.env.HOTSTEP_URL || 'http://localhost:3001';
const ENGINE = process.env.ACE_URL || 'http://127.0.0.1:8085';
const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
fs.mkdirSync(cfg.outDir, { recursive: true });
const results = path.join(cfg.outDir, 'results.jsonl');

const ckpt = (step, half) => step > 0 ? path.join(cfg.runDir, `checkpoint-step${step}`, `native-${half}.safetensors`) : '';
const meters = (step) => {
  try { return JSON.parse(fs.readFileSync(path.join(cfg.runDir, `checkpoint-step${step}`, 'meters.json'), 'utf8')); }
  catch { return {}; }
};
// Single-user local auth: /api/auth/auto hands out a bearer token.
const { token } = await fetch(`${APP}/api/auth/auto`).then(r => r.json());
const auth = { Authorization: `Bearer ${token}` };
const json = async (res) => { const t = await res.text(); if (!res.ok) throw new Error(`${res.status} ${t.slice(0, 300)}`); return JSON.parse(t); };
const select = (ar, nar) => fetch(`${APP}/api/backends/models`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ backend: 'yue2', selection: { lmAdapterAr: ar, lmAdapterNar: nar } }) }).then(json);

async function render(title, seed) {
  const { jobId } = await json(await fetch(`${APP}/api/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...auth },
    body: JSON.stringify({ prompt: cfg.caption, lyrics: cfg.lyrics, seed, randomSeed: false, batchSize: 1, title }) }));
  // An engine restart mid-render (training stages stop and start it) leaves
  // the app polling a job the new engine never heard of: it sits at 0% for
  // ever instead of failing. So watch the engine, and cap a stalled render.
  const deadline = Date.now() + 10 * 60_000;
  let enginePaused = false;
  const cancel = () => fetch(`${APP}/api/generate/cancel/${jobId}`, { method: 'POST', headers: auth }).catch(() => {});
  for (;;) {
    if (Date.now() > deadline) { await cancel(); throw new Error(`render ${jobId} stalled for 10 min; cancelled`); }
    const ready = (await fetch(`${APP}/api/health`).then(r => r.json()).catch(() => null))?.engine?.ready;
    if (!ready) enginePaused = true;
    else if (enginePaused) { await cancel(); throw new Error(`render ${jobId} was cut off by an engine restart; cancelled`); }
    const s = await json(await fetch(`${APP}/api/generate/status/${jobId}`, { headers: auth }));
    if (s.status === 'succeeded') {
      const url = (s.result?.audioUrls ?? s.audioUrls ?? [])[0];
      if (!url) throw new Error(`render ${jobId} succeeded without audio`);
      const res = await fetch(url.startsWith('http') ? url : `${APP}${url}`, { headers: auth });
      if (!res.ok) throw new Error(`audio download ${res.status}`);
      return { jobId, bytes: Buffer.from(await res.arrayBuffer()), name: path.basename(url.split('?')[0]) };
    }
    if (s.status === 'failed' || s.status === 'cancelled') throw new Error(`render ${jobId} ${s.status}: ${s.error ?? ''}`);
    await new Promise(r => setTimeout(r, 3000));
  }
}

// Diction: the aligner's per-word score is the mean frame probability of that
// word's characters (0..1). Garbled or missing words score low.
async function diction(bytes, name) {
  const form = new FormData();
  form.append('audio', new Blob([bytes]), name);
  form.append('lyrics', cfg.lyrics);
  const out = await json(await fetch(`${ENGINE}/yue2/align`, { method: 'POST', body: form }));
  const words = Array.isArray(out) ? out : out.words ?? [];
  const scores = words.map(w => w.score).filter(Number.isFinite);
  if (!scores.length) return { words: 0 };
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  return { words: scores.length, mean: +mean.toFixed(4), low: +(scores.filter(s => s < 0.2).length / scores.length).toFixed(4) };
}

async function engineReady() {
  for (let waited = 0; ; waited += 15) {
    const h = await fetch(`${APP}/api/health`).then(r => r.json()).catch(() => null);
    if (h?.engine?.ready) return;
    if (waited % 300 === 0) console.error(`engine not ready (${h?.engine?.bootStatus ?? 'app unreachable'}); waiting`);
    await new Promise(r => setTimeout(r, 15_000));
  }
}

const original = await fetch(`${APP}/api/backends/models?backend=yue2`).then(r => r.ok ? r.json() : null).catch(() => null);
let n = 0;
try {
  for (const [ar, nar] of cfg.pairs) {
    const label = ar === 0 && nar === 0 ? 'base' : `ar${String(ar).padStart(4, '0')}-nar${String(nar).padStart(4, '0')}`;
    const num = String(++n).padStart(2, '0');
    const takes = ar === 0 && nar === 0 ? 1 : Math.max(1, cfg.takes ?? 1);
    for (let take = 1; take <= takes; take++) {
    const file = path.join(cfg.outDir, `${num}-${label}${takes > 1 ? `-t${take}` : ''}.wav`);
    const seed = cfg.seed + take - 1;
    if (fs.existsSync(file)) { console.log(`skip ${path.basename(file)}`); continue; }
    for (const [step, half] of [[ar, 'ar'], [nar, 'nar']]) {
      if (step > 0 && !fs.existsSync(ckpt(step, half))) throw new Error(`missing ${ckpt(step, half)}`);
    }
    const started = Date.now();
    // Training's data-prep stages pause the engine for minutes at a time, and
    // a render caught by a pause fails. Wait for the engine, then retry.
    let out;
    for (let attempt = 1; ; attempt++) {
      await engineReady();
      try {
        await select(ckpt(ar, 'ar'), ckpt(nar, 'nar'));
        out = await render(`ladder ${path.basename(cfg.outDir)} ${label} t${take}`, seed);
        break;
      } catch (err) {
        if (attempt >= 4) throw err;
        console.error(`attempt ${attempt} failed (${err.message}); waiting for the engine and retrying`);
        await new Promise(r => setTimeout(r, 30_000));
      }
    }
    fs.writeFileSync(file, out.bytes);
    const d = await diction(out.bytes, out.name).catch(err => ({ error: String(err.message || err) }));
    const row = { file: path.basename(file), ar, nar, seed, take, seconds: Math.round((Date.now() - started) / 1000),
      ar_kl: meters(ar).ar_kl_mean20 ?? null, nar_drift: meters(nar).nar_drift ?? null, diction: d, jobId: out.jobId };
    fs.appendFileSync(results, JSON.stringify(row) + '\n');
    console.log(JSON.stringify(row));
    }
  }
} finally {
  // Put the user's picks back, or clear them if they cannot be read.
  const d = original?.defaults ?? {};
  await select(d.lmAdapterAr ?? '', d.lmAdapterNar ?? '').catch(err => console.error(`restore failed: ${err.message}`));
}
