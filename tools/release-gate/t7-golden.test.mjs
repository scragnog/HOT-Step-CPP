// Tier 7 — golden fingerprints. Same seed and request as the last time the
// goldens were written; the spectral shape must still match. This catches a
// numerical regression that completes "successfully". Warn-only by default
// (run.mjs --strict-golden makes it block), because goldens are tied to the
// models on the box that wrote them.
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  MIN, HERE, api, generate, wavFile, fingerprint, fingerprintSimilarity, setBackend, activeBackend, backendModels,
  CAPTION, LYRICS, MM3_CAPTION,
} from './lib.mjs';

const GOLDENS = path.join(HERE, 'goldens');
const UPDATE = process.env.GATE_UPDATE_GOLDENS === '1';
const THRESHOLD = 0.97;

const CASES = [
  { key: 'ace-text2music', backend: 'ace', need: ['dit', 'vae', 'lm'],
    body: { prompt: CAPTION, lyrics: LYRICS, instrumental: false, duration: 10, inferenceSteps: 8, seed: 31337, randomSeed: false, batchSize: 1, skipLm: false } },
  { key: 'ace-skiplm', backend: 'ace', need: ['dit', 'vae'],
    body: { prompt: 'minimal ambient techno, warm analog pads, slow', instrumental: true, duration: 6, inferenceSteps: 6, seed: 31337, randomSeed: false, batchSize: 1, skipLm: true } },
  { key: 'mm3', backend: 'minimax-m3', need: ['lm', 'dit'], timeoutMs: 25 * MIN,
    body: { prompt: MM3_CAPTION, lyrics: LYRICS, instrumental: false, duration: 30, mm3Steps: 8, seed: 31337, randomSeed: false, batchSize: 1 } },
  { key: 'yue2', backend: 'yue2', need: [], timeoutMs: 30 * MIN,
    body: { prompt: CAPTION, lyrics: LYRICS, instrumental: false, duration: 30, yue2OdeSteps: 8, seed: 31337, randomSeed: false, batchSize: 1 } },
];

let original = 'ace';
before(async () => { original = await activeBackend(); fs.mkdirSync(GOLDENS, { recursive: true }); });
after(async () => { await setBackend(original); });

for (const c of CASES) {
  test(`golden: ${c.key}`, { timeout: (c.timeoutMs ?? 15 * MIN) + 5 * MIN }, async (t) => {
    const m = await backendModels(c.backend, c.need);
    const any = Object.values(m.buckets ?? {}).some((b) => Array.isArray(b) && b.length);
    if (!m.ok || !any) return t.skip(`${c.backend} models not installed`);
    const file = path.join(GOLDENS, `${c.key}.json`);
    const have = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
    if (!have && !UPDATE) return t.skip('no golden yet; run with --update-goldens to write one');

    await setBackend(c.backend);
    const g = await generate(`t7-${c.key}`, c.body, { timeoutMs: c.timeoutMs ?? 15 * MIN });
    const w = wavFile(g.file);
    const fp = fingerprint(w.mono, w.rate);
    const models = (await api('GET', `/api/backends/models?backend=${c.backend}`)).defaults ?? {};

    if (UPDATE) {
      fs.writeFileSync(file, JSON.stringify({ key: c.key, backend: c.backend, writtenAt: new Date().toISOString(), rate: w.rate, seconds: w.seconds, models, body: c.body, fingerprint: fp }));
      t.diagnostic(`golden written: ${file} (${fp.length} frames)`);
      return;
    }
    const sim = fingerprintSimilarity(have.fingerprint, fp);
    const drift = JSON.stringify(have.models) !== JSON.stringify(models);
    t.diagnostic(`similarity ${sim.toFixed(4)} vs threshold ${THRESHOLD}; golden from ${have.writtenAt}${drift ? '; engine defaults differ from when the golden was written' : ''}`);
    assert.ok(Math.abs(have.fingerprint.length - fp.length) <= 5, `length changed: golden ${have.fingerprint.length} frames, now ${fp.length}`);
    assert.ok(sim >= THRESHOLD, `${c.key} drifted from its golden (similarity ${sim.toFixed(4)} < ${THRESHOLD}). Listen to ${g.file}; if it is fine, rerun with --update-goldens.`);
  });
}
