import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { once } from 'node:events';
import express from 'express';
import { apiErrorHandler } from './apiErrorHandler.js';
import { createSoundFontLibraryRouter } from './soundFontLibrary.js';
import { SoundFontLibrary, type SoundFontEntry } from '../services/soundFontLibrary.js';

test('SoundFont library validates uploads and persists, deduplicates, renames and removes banks by ID', async t => {
  const directory = await fs.mkdtemp(path.resolve('.test-soundfonts-'));
  const root = path.join(directory, 'soundfonts');
  const app = express();
  app.use(express.json());
  app.use('/soundfonts', createSoundFontLibraryRouter(new SoundFontLibrary(root), req => req.get('Authorization'), 64));
  app.use(apiErrorHandler);
  const server = app.listen(0, '127.0.0.1');
  t.after(async () => {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    await fs.rm(directory, { recursive: true, force: true });
  });
  await once(server, 'listening');
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}/soundfonts`;
  const auth = { Authorization: 'Bearer test' };

  const upload = (filename: string, data: Uint8Array<ArrayBuffer>, name?: string, authorized = true) => {
    const body = new FormData();
    body.append('file', new Blob([data]), filename);
    if (name) body.append('name', name);
    return fetch(base, { method: 'POST', headers: authorized ? auth : {}, body });
  };
  const bank = (size = 64, marker = 0) => {
    const bytes = Buffer.alloc(size);
    bytes.write('RIFF');
    bytes.writeUInt32LE(size - 8, 4);
    bytes.write('sfbk', 8);
    bytes[size - 1] = marker;
    return bytes;
  };
  const status = async (response: Promise<Response>, expected: number) => {
    const res = await response;
    assert.equal(res.status, expected, res.status === expected ? undefined : await res.text());
    await res.body?.cancel();
  };
  const remove = (id: string) => fetch(`${base}/${id}`, { method: 'DELETE', headers: auth });

  await status(upload('private.sf2', bank(), undefined, false), 401);
  await status(upload('other.wav', bank()), 400);
  await status(upload('fake.sf2', Buffer.alloc(20)), 400);
  await status(upload('too-big.sf3', bank(65)), 413);
  assert.deepEqual(await (await fetch(base)).json(), []);

  const responses = await Promise.all([
    upload('one.sf2', bank(), 'Premier étage'),
    upload('duplicate.sf2', bank()),
  ]);
  assert.deepEqual(responses.map(res => res.status).sort(), [200, 201]);
  const [first, duplicate] = await Promise.all(responses.map(async res => await res.json() as SoundFontEntry));
  assert.deepEqual(duplicate, first);
  assert.equal(first.size, 64);
  assert.equal(first.format, 'sf2');
  assert.ok(Number.isFinite(Date.parse(first.addedAt)));

  const secondResponse = await upload('two.SF3', bank(64, 1));
  assert.equal(secondResponse.status, 201);
  const second = await secondResponse.json() as SoundFontEntry;
  assert.notEqual(second.id, first.id);
  assert.equal(second.format, 'sf3');
  assert.deepEqual(new Set(new SoundFontLibrary(root).list().map(entry => entry.id)), new Set([first.id, second.id]));
  assert.deepEqual((await fs.readdir(root)).sort(), [first.id, second.id].sort());

  const bytes = await fetch(`${base}/${first.id}`);
  assert.equal(bytes.status, 200);
  assert.deepEqual(Buffer.from(await bytes.arrayBuffer()), bank());
  const rename = await fetch(`${base}/${first.id}`, {
    method: 'PATCH', headers: { ...auth, 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: 'Banque révisée' }),
  });
  assert.equal(rename.status, 200);
  assert.deepEqual(await rename.json(), { ...first, name: 'Banque révisée' });

  const outside = path.join(directory, 'outside');
  await fs.mkdir(outside);
  const sentinel = path.join(outside, 'keep.sf2');
  await fs.writeFile(sentinel, bank());
  await status(remove(randomUUID()), 404);
  await status(remove('..%2Foutside%2Fkeep.sf2'), 404);
  await status(remove('%2e%2e%5coutside%5ckeep.sf2'), 404);
  await status(remove(encodeURIComponent(sentinel)), 404);
  const link = randomUUID();
  await fs.symlink(outside, path.join(root, link), 'junction');
  await status(remove(link), 404);
  assert.deepEqual(await fs.readFile(sentinel), bank());

  await status(remove(first.id), 204);
  await status(fetch(`${base}/${first.id}`), 404);
  await status(remove(first.id), 404);
  assert.deepEqual(await (await fetch(base)).json(), [second]);
});
