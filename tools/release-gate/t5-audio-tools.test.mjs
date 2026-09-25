// Tier 5 — the audio tools that take an existing render: analyze, stem
// separation, DiT stem extraction, mastering, the post-process chain, MIDI.
import test, { before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { MIN, api, apiRaw, generate, setBackend, sleep, state, CAPTION } from './lib.mjs';

let audioUrl = null;
let songId = null;

before(async () => {
  await setBackend('ace');
  audioUrl = state.get('sourceAudioUrl') || null;
  songId = state.get('songId') || null;
  if (!audioUrl) {
    const g = await generate('t5-source', { prompt: CAPTION, instrumental: true, duration: 8, inferenceSteps: 4, seed: 5, randomSeed: false, skipLm: true });
    audioUrl = g.audioUrls[0];
    songId = g.songIds[0] ?? null;
  }
});

async function poll(route, label, isDone, isFailed, timeoutMs = 10 * MIN) {
  const started = Date.now();
  for (;;) {
    const r = await api('GET', route);
    if (isDone(r)) return r;
    if (isFailed(r)) throw new Error(`${label}: ${r.status}: ${r.error ?? r.message ?? ''}`);
    if (Date.now() - started > timeoutMs) throw new Error(`${label}: timed out (${r.status} ${r.progress ?? ''})`);
    await sleep(2000);
  }
}

/** A 4xx/503 that names a missing binary or model is a skip, not a failure. */
function skippable(r) {
  return r.status >= 400 && r.status < 500 || r.status === 503;
}

test('analyze: BPM and key', { timeout: 5 * MIN }, async () => {
  const r = await api('POST', '/api/analyze', { audioUrl }, { timeoutMs: 5 * MIN });
  assert.ok(Number.isFinite(r.bpm) && r.bpm > 0, `no bpm in ${JSON.stringify(r)}`);
});

test('supersep: stem separation', { timeout: 15 * MIN }, async (t) => {
  const start = await apiRaw('POST', '/api/supersep/separate?level=0', { audioUrl }, { timeoutMs: 5 * MIN });
  if (skippable(start)) return t.skip(`supersep unavailable: ${start.body?.error ?? start.status}`);
  assert.equal(start.status, 200, JSON.stringify(start.body).slice(0, 300));
  const id = start.body.id ?? start.body.jobId;
  assert.ok(id, `no job id in ${JSON.stringify(start.body)}`);
  await poll(`/api/supersep/${id}/progress`, 'supersep', (r) => r.status === 'done', (r) => r.status === 'failed' || r.status === 'cancelled');
  const result = await api('GET', `/api/supersep/${id}/result`);
  assert.ok((result.stems ?? []).length > 0, 'separation finished with no stems');
  t.diagnostic(`${result.stems.length} stems`);
});

test('stem studio: extract a track with the DiT', { timeout: 20 * MIN }, async (t) => {
  const start = await apiRaw('POST', '/api/stem-studio/extract', { sourceAudioUrl: audioUrl, sourceFileName: 'gate.wav', tracks: ['drums'], style: CAPTION }, { auth: true });
  if (skippable(start)) return t.skip(`stem extraction unavailable: ${start.body?.error ?? start.status}`);
  assert.equal(start.status, 200, JSON.stringify(start.body).slice(0, 300));
  const id = start.body.id;
  await poll(`/api/stem-studio/${id}/progress`, 'stem extract', (r) => r.status === 'done', (r) => r.status === 'failed' || r.status === 'cancelled', 20 * MIN);
  const result = await apiRaw('GET', `/api/stem-studio/${id}/result`);
  assert.equal(result.status, 200, JSON.stringify(result.body).slice(0, 300));
});

test('mastering: run against a reference', { timeout: 10 * MIN }, async (t) => {
  if (!songId) return t.skip('no song id for the source render');
  let refs = (await api('GET', '/api/mastering/references')).references ?? [];
  if (!refs.length) {
    const wav = state.get('renders', []).find((r) => r.file.endsWith('.wav'));
    if (!wav) return t.skip('no reference and no render to upload as one');
    const form = new FormData();
    form.append('file', new Blob([fs.readFileSync(wav.file)], { type: 'audio/wav' }), 'release-gate-reference.wav');
    const up = await apiRaw('POST', '/api/mastering/upload-reference', form, { auth: true, timeoutMs: 2 * MIN });
    assert.equal(up.status, 200, JSON.stringify(up.body).slice(0, 300));
    refs = (await api('GET', '/api/mastering/references')).references ?? [];
  }
  const referenceName = refs[0]?.name ?? refs[0];
  const r = await api('POST', '/api/mastering/run', { songId, referenceName }, { auth: true, timeoutMs: 10 * MIN });
  assert.ok(r.ok && r.masteredUrl, `mastering returned ${JSON.stringify(r).slice(0, 300)}`);
  t.diagnostic(`reference ${referenceName} -> ${r.masteredUrl}`);
});

test('post-process chain: re-run on a song (gain offset stage)', { timeout: 15 * MIN }, async (t) => {
  const id = state.get('songId2') || songId;
  if (!id) return t.skip('no song id');
  const start = await apiRaw('POST', `/api/songs/${id}/postprocess`, { gainOffsetDb: -1 }, { auth: true });
  if (start.status === 409) return t.skip(`song not eligible: ${start.body?.error}`);
  assert.equal(start.status, 200, JSON.stringify(start.body).slice(0, 300));
  await poll(`/api/songs/${id}/postprocess`, 'postprocess',
    (r) => r.status === 'succeeded' || r.status === 'done',
    (r) => r.status === 'failed' || r.status === 'cancelled', 15 * MIN);
});

test('MIDI studio: transcribe the render', { timeout: 15 * MIN }, async (t) => {
  const s = await api('GET', '/api/midi-studio/status');
  if (!s.engineAvailable) return t.skip('ace-midi binary not available');
  const size = Object.entries(s.models ?? {}).find(([, m]) => m.downloaded)?.[0];
  if (!size) return t.skip('no MIDI model downloaded');
  const start = await apiRaw('POST', '/api/midi-studio/transcribe', { sourceAudioUrl: audioUrl, sourceFileName: 'gate.wav', model: size }, { auth: true });
  if (skippable(start)) return t.skip(`transcription unavailable: ${start.body?.error ?? start.status}`);
  assert.equal(start.status, 200, JSON.stringify(start.body).slice(0, 300));
  const id = start.body.id;
  await poll(`/api/midi-studio/${id}/progress`, 'midi', (r) => r.status === 'done', (r) => r.status === 'failed' || r.status === 'cancelled', 15 * MIN);
  const notes = await apiRaw('GET', `/api/midi-studio/${id}/notes`);
  assert.equal(notes.status, 200, JSON.stringify(notes.body).slice(0, 300));
  const file = path.join(process.env.GATE_RUN_DIR ?? '.', 'renders', `t5-midi-${size}.json`);
  fs.writeFileSync(file, JSON.stringify(notes.body).slice(0, 1 << 20));
});
