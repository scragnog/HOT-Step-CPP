#!/usr/bin/env node
// YuE2 planner health sweep: for each planner (AR) checkpoint, write N plans
// (score preview: lead sheet only, no audio, seconds each) and record how each
// one ended and what the score checker made of it. A render-free way to see
// where a planner starts to break, for comparing against ear scores.
//
//   node server/scripts/yue2-plan-sweep.mjs <sweep.json>
//
// sweep.json: { out, plans, seed, semantic?, albums: [{ name, runDir, caption, lyrics, steps: [..] }] }
//   semantic: true runs the semantic stage too (about a minute a song instead
//   of seconds) and records the planner's codec id stream per plan.
//   step 0 = the base planner. Appends one JSON line per plan to <out>/<name>.jsonl
//   and skips plans already recorded there, so an interrupted sweep resumes.
import fs from 'node:fs';
import path from 'node:path';

const APP = process.env.HOTSTEP_URL || 'http://localhost:3001';
const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
fs.mkdirSync(cfg.out, { recursive: true });
const { token } = await fetch(`${APP}/api/auth/auto`).then(r => r.json());
const auth = { Authorization: `Bearer ${token}` };
const json = async (res) => { const t = await res.text(); if (!res.ok) throw new Error(`${res.status} ${t.slice(0, 300)}`); return JSON.parse(t); };
const select = (ar) => fetch(`${APP}/api/backends/models`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ backend: 'yue2', selection: { lmAdapterAr: ar, lmAdapterNar: '' } }) }).then(json);

const original = await fetch(`${APP}/api/backends/models?backend=yue2`).then(r => r.ok ? r.json() : null).catch(() => null);
try {
  for (const a of cfg.albums) {
    const file = path.join(cfg.out, `${a.name}.jsonl`);
    const done = new Set(fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(r => !r.error).map(r => `${r.step}:${r.seed}`) : []);
    for (const step of a.steps) {
      const seeds = Array.from({ length: cfg.plans }, (_, i) => cfg.seed + i).filter(s => !done.has(`${step}:${s}`));
      if (!seeds.length) continue;
      await select(step > 0 ? path.join(a.runDir, `checkpoint-step${step}`, 'native-ar.safetensors') : '');
      let kl = null;
      try { kl = JSON.parse(fs.readFileSync(path.join(a.runDir, `checkpoint-step${step}`, 'meters.json'), 'utf8')).ar_kl_mean20 ?? null; } catch {}
      for (const seed of seeds) {
        const t0 = Date.now();
        let row;
        try {
          const p = await json(await fetch(`${APP}/api/generate/yue2/plan`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...auth },
            body: JSON.stringify({ prompt: a.caption, lyrics: a.lyrics, seed, randomSeed: false, ...(cfg.semantic ? { semantic: true } : {}) }) }));
          row = { album: a.name, step, kl, seed, end_reason: p.end_reason, stage_end_reasons: p.stage_end_reasons, ...p.health, abc: p.abc, ms: Date.now() - t0,
            ...(p.semantic_ids ? { semantic_ids: p.semantic_ids } : {}) };
        } catch (err) {
          row = { album: a.name, step, kl, seed, error: String(err.message || err), ms: Date.now() - t0 };
        }
        fs.appendFileSync(file, JSON.stringify(row) + '\n');
        console.log(`${a.name} step ${step} seed ${seed}: ${row.verdict ?? row.error} ${row.end_reason ?? ''} (${Math.round(row.ms / 1000)} s)`);
      }
    }
  }
} finally {
  const d = original?.defaults ?? {};
  await select(d.lmAdapterAr ?? '').catch(() => {});
  await fetch(`${APP}/api/backends/models`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ backend: 'yue2', selection: { lmAdapterAr: d.lmAdapterAr ?? '', lmAdapterNar: d.lmAdapterNar ?? '' } }) }).catch(() => {});
}
