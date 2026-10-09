// The published stream wire shapes against what routes/generate.ts reads and
// writes. A field added to the control route, or renamed in status, fails here.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { STREAM_SESSION_HEADER, type Mm3StreamStatus, type StormControl } from './streaming.js';

const source = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'routes', 'generate.ts'), 'utf8');
const between = (from: string, to: string) => source.slice(source.indexOf(from), source.indexOf(to, source.indexOf(from)));

// Compile-time key lists: adding a key to the type without listing it here fails tsc.
const CONTROL_KEYS = ['streamId', 'guidance_scale', 'lss_strength', 'inference_steps', 'duration', 'bpm', 'next_bpm', 'next_duration',
  'seed_lock', 'seed', 'prompt', 'lyrics', 'stick_prompt', 'stick_lyrics', 'stream_pause', 'infer_method', 'scheduler',
  'guidance_mode', 'plugin_params'] as const satisfies ReadonlyArray<keyof StormControl>;
type MissingControl = Exclude<keyof StormControl, typeof CONTROL_KEYS[number]>;
const _controlComplete: MissingControl extends never ? true : never = true;
const MM3_KEYS = ['mm3_streaming', 'mm3_interleaved', 'mm3_duration', 'mm3_takes', 'mm3_take_seeds'] as const satisfies ReadonlyArray<keyof Mm3StreamStatus>;
type MissingMm3 = Exclude<keyof Mm3StreamStatus, typeof MM3_KEYS[number]>;
const _mm3Complete: MissingMm3 extends never ? true : never = true;
void _controlComplete; void _mm3Complete;

test('StormControl lists exactly the fields POST /storm/control reads', () => {
  const handler = between("router.post('/storm/control'", "router.get('/storm/control'");
  const read = new Set([...handler.matchAll(/\bb\.(\w+)/g)].map(m => m[1]));
  assert.deepEqual([...read].sort(), [...CONTROL_KEYS].sort());
});

test('the stream routes set the published session header, and status carries the MM3 fields', () => {
  assert.equal(STREAM_SESSION_HEADER, 'X-Stream-Session');
  for (const route of ["router.get('/mm3/stream/:id'", "router.post('/storm/stream'"]) {
    const start = source.indexOf(route);
    assert.ok(start >= 0, route);
    const body = source.slice(start, source.indexOf('\nrouter.', start + 1));
    assert.ok(body.includes(`res.setHeader('${STREAM_SESSION_HEADER}', session.id)`), route);
    assert.ok(body.includes("res.setHeader('Content-Type', 'audio/wav')"), route);
  }
  const status = between("router.get('/status/:id'", '\nrouter.');
  for (const key of MM3_KEYS) assert.ok(new RegExp(`\\b${key}:`).test(status), key);
});

test('STORM start and stop read the published fields', () => {
  const start = between("router.post('/storm/stream'", '\nrouter.');
  for (const key of ['streamId', 'seed', 'coResident', 'pluginParams']) assert.ok(start.includes(`baseParams.${key}`), key);
  const stop = between("router.post('/storm/stop'", '\nrouter.');
  assert.ok(stop.includes('b.streamId'));
  assert.ok(stop.includes('res.json({ ok: true })'));
});
