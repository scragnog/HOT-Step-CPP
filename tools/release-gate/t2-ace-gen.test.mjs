// Tier 2 — every ACE generation mode completes mechanically and yields audio.
import test, { before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { MIN, CAPTION, LYRICS, generate, assertAudible, generationLog, setBackend, state } from './lib.mjs';

const SEED = 20260925;
const common = { prompt: CAPTION, duration: 12, inferenceSteps: 8, seed: SEED, randomSeed: false, batchSize: 1 };

before(async () => { await setBackend('ace'); });

test('text2music with the LM and lyrics', { timeout: 20 * MIN }, async (t) => {
  const g = await generate('t2-text2music', { ...common, lyrics: LYRICS, instrumental: false, skipLm: false });
  assertAudible(assert, g.file, 'text2music');
  state.set('sourceAudioUrl', g.audioUrls[0]);
  state.set('songId', g.songIds[0] ?? null);
  state.set('t2JobId', g.jobId);
  t.diagnostic(`${g.seconds} s, ${g.audioUrls.length} take(s), song ${g.songIds[0] ?? 'none'}`);
});

test('text2music instrumental with the LM skipped', { timeout: 15 * MIN }, async (t) => {
  const g = await generate('t2-skiplm', { ...common, instrumental: true, skipLm: true, duration: 8, inferenceSteps: 4 });
  assertAudible(assert, g.file, 'skipLm');
  state.set('songId2', g.songIds[0] ?? null);
  t.diagnostic(`${g.seconds} s`);
});

const TASKS = [
  ['cover', { instrumental: true, audioCoverStrength: 0.5 }],
  ['repaint', { lyrics: LYRICS, instrumental: false, repaintingStart: 3, repaintingEnd: 8 }],
  ['extract', { lyrics: LYRICS, instrumental: false, trackName: 'vocals' }],
  ['lego', { lyrics: LYRICS, instrumental: false, trackName: 'drums' }],
  ['complete', { lyrics: LYRICS, instrumental: false }],
];

for (const [task, extra] of TASKS) {
  test(`${task} from the text2music render`, { timeout: 20 * MIN }, async (t) => {
    const sourceAudioUrl = state.get('sourceAudioUrl');
    if (!sourceAudioUrl) return t.skip('no source render (text2music failed)');
    const g = await generate(`t2-${task}`, { ...common, ...extra, taskType: task, sourceAudioUrl });
    assertAudible(assert, g.file, task);
    t.diagnostic(`${g.seconds} s`);
  });
}

test('per-generation log ends with GENERATION COMPLETED', async (t) => {
  const jobId = state.get('t2JobId');
  if (!jobId) return t.skip('no text2music job');
  const log = generationLog(jobId);
  if (!log) return t.skip('generation log not on this disk (remote app?)');
  const text = fs.readFileSync(log, 'utf8');
  assert.ok(/GENERATION COMPLETED/.test(text), `${log} does not end with GENERATION COMPLETED`);
});
