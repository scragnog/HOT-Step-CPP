import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { config } from '../../../config.js';
import { buildMm3CodesArgs } from '../../training/mm3Train.js';
import { mm3PlankProvenance } from '../../workflows/coverWorkflow.js';
import { readMm3PlankMeta, writeMm3PlankFromCodes } from './plank.js';

/** A plank dir in a temp root, plus a 3-row [rows, 8] int32 codes file. */
function setup(t: { mock: { getter: Function } }): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-plank-'));
  fs.mkdirSync(path.join(root, 'audio'));
  t.mock.getter(config.data, 'audioDir', () => path.join(root, 'audio'));
  const codes = path.join(root, 'source.codes');
  fs.writeFileSync(codes, Buffer.alloc(3 * 32));
  return codes;
}

test('cover encode: the launder flag picks the CLI subcommand, off stays mm3-codes', () => {
  const plain = buildMm3CodesArgs({ datasetJson: 'd.json', outDir: 'out', maxDuration: 359 });
  assert.equal(plain[0], 'mm3-codes');
  assert.ok(!plain.includes('--rec7'));
  const laundered = buildMm3CodesArgs({ datasetJson: 'd.json', outDir: 'out', maxDuration: 359, launder: true, launderSteps: 40, launderSeed: 123 });
  assert.equal(laundered[0], 'mm3-launder');
  assert.ok(laundered.includes('--rec7'));
  assert.equal(laundered[laundered.indexOf('--steps') + 1], '40');
  assert.equal(laundered[laundered.indexOf('--seed') + 1], '123');
  assert.ok(!plain.includes('--steps'));
  assert.ok(!plain.includes('--seed'));
  const defaults = buildMm3CodesArgs({ datasetJson: 'd.json', outDir: 'out', launder: true });
  assert.equal(defaults[defaults.indexOf('--steps') + 1], '30');
  assert.equal(defaults[defaults.indexOf('--seed') + 1], '42');
  assert.equal(laundered[laundered.indexOf('--max-duration') + 1], '359');
});

test('plank sidecar records a launder and the cover request carries it to the song facts', (t) => {
  const codes = setup(t);
  const file = writeMm3PlankFromCodes(codes, 'lau', { jobId: 'lau', caption: '', lyrics: '', source: 'a.wav',
    laundered: true, launderSteps: 40, launderSeed: 123, launderCaption: 'acoustic guitar', rec7: 'mm3-rec7-f16.gguf', encoder: 'mm3-enc-f16.gguf' });
  assert.equal(readMm3PlankMeta(file)?.laundered, true);
  assert.equal(readMm3PlankMeta(file)?.launderSteps, 40);
  assert.equal(readMm3PlankMeta(file)?.launderSeed, 123);
  assert.equal(readMm3PlankMeta(file)?.launderCaption, 'acoustic guitar');
  assert.deepEqual(mm3PlankProvenance(file),
    { mm3PlankLaundered: true, mm3PlankLaunderSteps: 40, mm3PlankLaunderSeed: 123, mm3PlankLaunderCaption: 'acoustic guitar', mm3PlankLaunderModels: 'mm3-rec7-f16.gguf, mm3-enc-f16.gguf' });

  const raw = writeMm3PlankFromCodes(codes, 'raw', { jobId: 'raw', caption: '', lyrics: '', source: 'a.wav', laundered: false });
  assert.equal(readMm3PlankMeta(raw)?.rec7, undefined);
  assert.deepEqual(mm3PlankProvenance(raw), { mm3PlankLaundered: false });
  // A plank captured from a render has no launder field: nothing is claimed.
  const captured = writeMm3PlankFromCodes(codes, 'cap', { jobId: 'cap', caption: 'c', lyrics: '' });
  assert.deepEqual(mm3PlankProvenance(captured), {});
});
