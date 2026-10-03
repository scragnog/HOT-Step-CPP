// Router-level tests for routes/listening.ts over real HTTP: loopback static
// serving, traversal at the Express layer, and the CSRF/body guards on the
// POST that writes scores.json.
//
// HOT_STEP_ROOT is set before the first import of anything under src/ —
// config.ts computes PROJECT_ROOT once at import time, and a static import
// of routes/listening.ts would be hoisted above any top-level assignment in
// this file, locking PROJECT_ROOT to the real repo instead of this test's
// tmp dir. Dynamic import() after setting the env var avoids that.
import assert from 'node:assert/strict';
import { test, after } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';

const httpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'listening-http-root-'));
process.env.HOT_STEP_ROOT = httpRoot;
const studyDir = path.join(httpRoot, '_experiments', '_LISTENING', 'study-a');
fs.mkdirSync(studyDir, { recursive: true });
fs.writeFileSync(path.join(studyDir, 'index.html'), '<html>ok</html>');
const scoresPath = path.join(studyDir, 'scores.json');
fs.writeFileSync(scoresPath, JSON.stringify({ scores: { seed: 1 } }));

const { default: express } = await import('express');
const { default: listeningRouter } = await import('../src/routes/listening.js');
const app = express();
app.use(express.json());
app.use('/listening', listeningRouter);
const server = http.createServer(app);
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = (server.address() as any).port;
const base = `http://127.0.0.1:${port}`;
after(() => {
  server.close();
  fs.rmSync(httpRoot, { recursive: true, force: true });
});

test('GET serves a file from the confined folder', async () => {
  const r = await fetch(`${base}/listening/study-a/index.html`);
  assert.equal(r.status, 200);
  assert.match(await r.text(), /ok/);
});

test('GET 404s for a file that does not exist', async () => {
  const r = await fetch(`${base}/listening/study-a/nope.txt`);
  assert.equal(r.status, 404);
});

// fetch()/undici normalize %2e%2e client-side before the request ever leaves
// the process (confirmed against a raw http.createServer: the URL it
// receives is already collapsed to "/package.json"), so this traversal can
// only be exercised with a client that writes the request line literally —
// a real curl/script, or here, http.request's raw `path` option.
test('GET rejects an encoded .. traversal segment', async () => {
  const status = await new Promise<number>((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, path: '/listening/study-a/%2e%2e/%2e%2e/package.json', method: 'GET' },
      (res) => { res.resume(); res.on('end', () => resolve(res.statusCode!)); },
    );
    req.on('error', reject);
    req.end();
  });
  assert.equal(status, 400);
});

test('POST with no Origin header (same-origin) saves', async () => {
  const r = await fetch(`${base}/listening/study-a/scores`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scores: { a: 1 } }),
  });
  assert.equal(r.status, 200);
  assert.deepEqual(JSON.parse(fs.readFileSync(scoresPath, 'utf8')).scores, { a: 1 });
});

test('POST with the server\'s own Origin saves', async () => {
  const r = await fetch(`${base}/listening/study-a/scores`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'http://localhost:3001' },
    body: JSON.stringify({ scores: { b: 2 } }),
  });
  assert.equal(r.status, 200);
  assert.deepEqual(JSON.parse(fs.readFileSync(scoresPath, 'utf8')).scores, { b: 2 });
});

test('POST with a foreign Origin is rejected and scores.json is untouched', async () => {
  const before = fs.readFileSync(scoresPath, 'utf8');
  const r = await fetch(`${base}/listening/study-a/scores`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://untrusted.example' },
    body: JSON.stringify({ scores: { evil: true } }),
  });
  assert.equal(r.status, 403);
  assert.equal(fs.readFileSync(scoresPath, 'utf8'), before);
});

test('POST with a non-envelope body is rejected and scores.json is untouched', async () => {
  const before = fs.readFileSync(scoresPath, 'utf8');
  const r = await fetch(`${base}/listening/study-a/scores`, {
    method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'junk=1',
  });
  assert.equal(r.status, 400);
  assert.equal(fs.readFileSync(scoresPath, 'utf8'), before);
});
