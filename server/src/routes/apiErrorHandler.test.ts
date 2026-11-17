import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import multer from 'multer';
import { apiErrorHandler } from './apiErrorHandler.js';

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 4 } });
app.post('/api/up', upload.single('audio'), (_req, res) => { res.json({ ok: true }); });
app.get('/api/boom', () => { throw new Error('disk on fire'); });
app.post('/api/json', express.json(), (_req, res) => { res.json({ ok: true }); });
app.use('/api', apiErrorHandler);
const server = http.createServer(app);
await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api`;
test.after(() => { server.close(); });

test('an oversized upload is a 413 with a JSON message, not the HTML error page', async () => {
  const form = new FormData();
  form.append('audio', new Blob([new Uint8Array(64)]), 'big.flac');
  const res = await fetch(`${base}/up`, { method: 'POST', body: form });
  assert.equal(res.status, 413);
  assert.deepEqual(await res.json(), { error: 'The file is too large to upload' });
});

test('a thrown error is a 500 carrying its message', async () => {
  const res = await fetch(`${base}/boom`);
  assert.equal(res.status, 500);
  assert.deepEqual(await res.json(), { error: 'disk on fire' });
});

test('a client error keeps its 4xx status', async () => {
  const res = await fetch(`${base}/json`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{nope' });
  assert.equal(res.status, 400);
  assert.equal(typeof (await res.json()).error, 'string');
});
