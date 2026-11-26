import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { once } from 'node:events';
import test from 'node:test';
import express from 'express';
import modelManager from './modelManager.js';
import { modelDownloadService } from '../services/modelDownloadService.js';

test('MIDI SoundFont endpoint accepts only installed registry IDs', async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hotstep-soundfont-test-'));
  const getter = t.mock.getter(modelDownloadService, 'modelsDir', () => root);
  const app = express();
  app.use('/api/model-manager', modelManager);
  const server = app.listen(0, '127.0.0.1');
  t.after(async () => {
    server.closeAllConnections();
    server.close();
    getter.mock.restore();
    await rm(root, { recursive: true, force: true });
  });
  await once(server, 'listening');
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = 'http://127.0.0.1:' + address.port + '/api/model-manager';
  const route = base + '/midi-soundfont';
  assert.equal((await fetch(route)).status, 404);

  const dir = path.join(root, 'soundfonts');
  await mkdir(dir);
  const bank = Buffer.from('RIFF\x04\0\0\0sfbk');
  await writeFile(path.join(dir, 'GeneralUser-GS.sf2'), bank);
  assert.equal((await fetch(route)).status, 404);
  await writeFile(path.join(dir, 'LICENSE.txt'), 'licence');
  await writeFile(path.join(root, 'private.bin'), 'unrelated');
  const result = await fetch(route + '?path=../private.bin');
  assert.equal(result.status, 200);
  assert.equal(result.headers.get('content-type'), 'application/octet-stream');
  assert.deepEqual(Buffer.from(await result.arrayBuffer()), bank);
  assert.equal((await fetch(route + '/midi-generaluser-gs')).status, 200);
  assert.equal((await fetch(route + '/private.bin')).status, 404);
  assert.equal((await fetch(route + '/%2e%2e%2fprivate.bin')).status, 404);
  assert.equal((await fetch(route + '/muscriptor-small')).status, 404);

  const monoRoute = route + '/midi-fluidr3-mono-gm';
  const mono = Buffer.concat([bank, Buffer.from('mono')]);
  await writeFile(path.join(dir, 'FluidR3Mono_GM.sf3'), mono);
  assert.equal((await fetch(monoRoute)).status, 404);
  await writeFile(path.join(dir, 'FluidR3Mono_License.md'), 'licence');
  const monoResult = await fetch(monoRoute);
  assert.equal(monoResult.status, 200);
  assert.deepEqual(Buffer.from(await monoResult.arrayBuffer()), mono);
});
