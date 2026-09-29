import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';
import { config } from '../../config.js';
import { listWorkers, resolveDatasetFile } from './trainingWorkers.js';
import { labelsDir } from './paths.js';

test('pushed file paths stay inside the dataset or its labels folder', () => {
  const root = path.resolve('/tmp/ds');
  assert.equal(resolveDatasetFile(root, 'band', 'a/song.flac'), path.join(root, 'a', 'song.flac'));
  assert.equal(resolveDatasetFile(root, 'band', '__labels/x.json'), path.join(labelsDir('band'), 'x.json'));
  for (const bad of ['../evil.flac', 'a/../../evil', '__labels/../x.json', '', '.', path.resolve('/elsewhere/x')]) {
    assert.throws(() => resolveDatasetFile(root, 'band', bad), /Refused path/, bad);
  }
});

test('TRAINING_WORKERS parses name=url pairs and skips junk', () => {
  const before = config.workers.list;
  config.workers.list = 'LivingRoom=http://192.168.50.50:3001/, bad, =http://x, Ftp=ftp://y ;Other = https://o:1';
  try {
    assert.deepEqual(listWorkers(), [
      { name: 'LivingRoom', url: 'http://192.168.50.50:3001' },
      { name: 'Other', url: 'https://o:1' },
    ]);
  } finally { config.workers.list = before; }
});
