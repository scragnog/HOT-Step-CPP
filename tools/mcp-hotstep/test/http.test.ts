// http.test.ts — a loopback HTTP server standing in for the HOT-Step API,
// the same way server/src/services/training/trainingWorkers.test.ts stands
// one in for a training worker. Exercises http.ts's login/retry behaviour and
// tools.ts's request-building directly — no MCP protocol involved here (see
// test/client.test.ts for that layer).

import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';

interface Fixture {
  loginCount: number;
  currentToken: string;
  /** Per-test route logic for everything except /api/auth/auto. Receives the
   *  parsed JSON body (or undefined) and must send the response itself. */
  route: (req: http.IncomingMessage, res: http.ServerResponse, body: unknown) => void;
  generateCalls: Array<Record<string, unknown>>;
}

const fixture: Fixture = {
  loginCount: 0,
  currentToken: 'tok-0',
  route: (_req, res) => { res.writeHead(404).end('{}'); },
  generateCalls: [],
};

function freshLogin(): void {
  fixture.loginCount++;
  fixture.currentToken = `tok-${fixture.loginCount}`;
}

/** Simulates "the server no longer recognises the client's cached token"
 *  (e.g. a restart cleared the in-memory token map) without the client
 *  knowing — the next call must 401 against this new token. */
function invalidateClientToken(): void {
  freshLogin();
}

let server: http.Server;
let base: string;
let httpMod: typeof import('../src/http.js');
let toolsMod: typeof import('../src/tools.js');

before(async () => {
  server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      let body: unknown;
      try { body = raw ? JSON.parse(raw) : undefined; } catch { body = undefined; }

      if (req.url === '/api/auth/auto' && req.method === 'GET') {
        freshLogin();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ user: { id: 'u1' }, token: fixture.currentToken }));
        return;
      }

      if (req.headers.authorization !== `Bearer ${fixture.currentToken}`) {
        res.writeHead(401, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'Unauthorized' }));
        return;
      }

      if (req.url === '/api/generate' && req.method === 'POST') {
        fixture.generateCalls.push(body as Record<string, unknown>);
      }
      fixture.route(req, res, body);
    });
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  process.env.HOTSTEP_URL = base;
  httpMod = await import('../src/http.js');
  toolsMod = await import('../src/tools.js');
});

after(async () => {
  await new Promise(resolve => server.close(resolve));
});

beforeEach(() => {
  httpMod.resetToken();
  fixture.generateCalls.length = 0;
  fixture.route = (_req, res) => { res.writeHead(404).end('{}'); };
});

// ── login / 401 retry ───────────────────────────────────────────────────────

test('auto-logs in on first use and reuses the token after that', async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ depth: 0 })); };
  const before = fixture.loginCount;
  await toolsMod.genQueue();
  await toolsMod.genQueue();
  assert.equal(fixture.loginCount - before, 1, 'should log in exactly once, not per call');
});

test('retries once on a 401 after a fresh login (idempotent call)', async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ depth: 0 })); };
  await toolsMod.genQueue(); // caches a token
  invalidateClientToken(); // server-side only — the cached token is now stale
  const loginsBefore = fixture.loginCount;
  const outcome = await toolsMod.genQueue();
  assert.deepEqual(outcome, { kind: 'ok', data: { depth: 0 } });
  assert.equal(fixture.loginCount - loginsBefore, 1, 'exactly one relogin, then the retry succeeds');
});

test('gen_submit does NOT retry on 401 — no double-submit risk', async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-x', status: 'pending' })); };
  await toolsMod.genQueue(); // caches a token
  invalidateClientToken();
  const loginsBefore = fixture.loginCount;
  const outcome = await toolsMod.genSubmit({ backend: 'ace', caption: 'test' });
  assert.equal(outcome.kind, 'http_error');
  assert.equal((outcome as { status: number }).status, 401);
  assert.equal(fixture.loginCount, loginsBefore, 'must not relogin-and-resend a submit');
  assert.equal(fixture.generateCalls.length, 0, 'the submit must never have reached the server');
});

test('a server error is returned verbatim, not paraphrased', async () => {
  fixture.route = (_req, res) => { res.writeHead(500, { 'content-type': 'application/json' }).end(JSON.stringify({ error: 'engine exploded' })); };
  const outcome = await toolsMod.genQueue();
  assert.equal(outcome.kind, 'http_error');
  assert.equal((outcome as { status: number }).status, 500);
  assert.match((outcome as { text: string }).text, /engine exploded/);
});

// ── gen_submit: exact body per backend ───────────────────────────────────────

test('gen_submit builds the exact ACE body, keeping ditModel and lmModel as independent catalogue roles', async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'j1', status: 'pending' })); };
  // Deliberately distinct values: ditModel and lmModel are separate ACE
  // catalogues (translateParams.ts -> synth_model / lm_model), never the same
  // scalar duplicated into both — the engine rejects an LM name absent from
  // the LM bucket, so asserting equal values here would hide that bug.
  const outcome = await toolsMod.genSubmit({
    backend: 'ace', caption: 'synthpop', lyrics: '[Verse]\nhi', duration: 120, seed: 7, batchSize: 2, title: 'Song',
    ditModel: 'dit-f16', lmModel: 'lm-q8_0',
  });
  assert.equal(outcome.kind, 'ok');
  assert.deepEqual(fixture.generateCalls[0], {
    backend: 'ace', expectedBackend: 'ace', taskType: 'text2music', caption: 'synthpop',
    lyrics: '[Verse]\nhi', duration: 120, seed: 7, batchSize: 2, title: 'Song',
    ditModel: 'dit-f16', lmModel: 'lm-q8_0',
  });
});

test('gen_submit builds the exact MiniMax-Music3 body, flattening mm3* options to top level', async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'j2', status: 'pending' })); };
  const outcome = await toolsMod.genSubmit({
    backend: 'minimax-m3', caption: 'ambient', instrumental: true,
    options: { mm3Steps: 24, mm3CfgScale: 7.5 },
  });
  assert.equal(outcome.kind, 'ok');
  assert.deepEqual(fixture.generateCalls[0], {
    backend: 'minimax-m3', expectedBackend: 'minimax-m3', taskType: 'text2music', caption: 'ambient',
    instrumental: true, mm3Steps: 24, mm3CfgScale: 7.5,
  });
});

test('gen_submit builds the exact YuE2 body, flattening yue2* options to top level', async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'j3', status: 'pending' })); };
  const outcome = await toolsMod.genSubmit({
    backend: 'yue2', caption: 'folk guitar', options: { yue2Something: 'x' },
  });
  assert.equal(outcome.kind, 'ok');
  assert.deepEqual(fixture.generateCalls[0], {
    backend: 'yue2', expectedBackend: 'yue2', taskType: 'text2music', caption: 'folk guitar', yue2Something: 'x',
  });
});

// ── gen_submit: rejections before any HTTP call ─────────────────────────────

test('gen_submit rejects an options key that collides with a reserved field, without calling the server', async () => {
  const outcome = await toolsMod.genSubmit({ backend: 'ace', caption: 'x', options: { title: 'sneaky' } });
  assert.equal(outcome.kind, 'rejected');
  assert.equal(fixture.generateCalls.length, 0);
});

test('gen_submit rejects an operation the backend does not support, without calling the server', async () => {
  const outcome = await toolsMod.genSubmit({ backend: 'yue2', caption: 'x', operation: 'cover' });
  assert.equal(outcome.kind, 'rejected');
  assert.equal(fixture.generateCalls.length, 0);
});

// ── gen_wait ─────────────────────────────────────────────────────────────────

test('gen_wait returns the latest status once its budget runs out', { timeout: 5000 }, async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-w', status: 'running' })); };
  const outcome = await toolsMod.genWait('job-w', 1, new AbortController().signal);
  assert.equal(outcome.kind, 'ok');
  assert.equal((outcome as { data: { status: string } }).data.status, 'running');
});

test('gen_wait stops polling early when its signal is aborted, without cancelling the job', { timeout: 5000 }, async () => {
  fixture.route = (_req, res) => { res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-w2', status: 'running' })); };
  const ac = new AbortController();
  const waitPromise = toolsMod.genWait('job-w2', 30, ac.signal);
  setTimeout(() => ac.abort(), 50);
  const start = Date.now();
  const outcome = await waitPromise;
  assert.ok(Date.now() - start < 2000, 'should return promptly on abort, not ride out the 30s budget');
  assert.equal(outcome.kind, 'ok');
  assert.equal((outcome as { data: { status: string } }).data.status, 'running');
});

test('gen_wait with a slow status endpoint does not issue one more poll once the budget is already spent', { timeout: 5000 }, async () => {
  let calls = 0;
  fixture.route = (_req, res) => {
    calls++;
    const respond = () => res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-slow', status: 'running' }));
    if (calls === 1) respond(); else setTimeout(respond, 2200); // only reached if the bug regresses
  };
  const start = Date.now();
  const outcome = await toolsMod.genWait('job-slow', 1, new AbortController().signal);
  const elapsed = Date.now() - start;
  assert.equal(outcome.kind, 'ok');
  assert.ok(elapsed < 1500, `expected to return close to the 1s budget, took ${elapsed}ms`);
  assert.equal(calls, 1, 'a second poll means the budget check after waking from sleep did not fire');
});

test('gen_wait, aborted mid-sleep, does not issue one more poll after the signal fired', { timeout: 5000 }, async () => {
  let calls = 0;
  fixture.route = (_req, res) => {
    calls++;
    const respond = () => res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-slow2', status: 'running' }));
    if (calls === 1) respond(); else setTimeout(respond, 2200); // only reached if the bug regresses
  };
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 50);
  const start = Date.now();
  const outcome = await toolsMod.genWait('job-slow2', 30, ac.signal);
  const elapsed = Date.now() - start;
  assert.equal(outcome.kind, 'ok');
  assert.ok(elapsed < 1000, `expected to return promptly after abort, took ${elapsed}ms`);
  assert.equal(calls, 1, 'a second poll means the abort check after waking from sleep did not fire');
});

// ── gen_cancel ───────────────────────────────────────────────────────────────

test('gen_cancel posts to the matching route and returns the server response', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/generate/cancel/job-9');
    assert.equal(req.method, 'POST');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ success: true, jobId: 'job-9' }));
  };
  const outcome = await toolsMod.genCancel('job-9');
  assert.deepEqual(outcome, { kind: 'ok', data: { success: true, jobId: 'job-9' } });
});
