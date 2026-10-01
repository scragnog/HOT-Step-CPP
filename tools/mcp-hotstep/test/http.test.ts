// http.test.ts — a loopback HTTP server standing in for the HOT-Step API,
// the same way server/src/services/training/trainingWorkers.test.ts stands
// one in for a training worker. Exercises http.ts's login/retry behaviour and
// tools.ts's request-building directly — no MCP protocol involved here (see
// test/client.test.ts for that layer).

import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import type { TrainPrepareArgs, TrainStartArgs, TrainBackend } from '../src/tools.js';

interface Fixture {
  loginCount: number;
  currentToken: string;
  /** Delay before /api/auth/auto responds, for testing a slow/held login. */
  loginDelayMs: number;
  /** Per-test route logic for everything except /api/auth/auto. Receives the
   *  parsed JSON body (or undefined) and must send the response itself. */
  route: (req: http.IncomingMessage, res: http.ServerResponse, body: unknown) => void;
  generateCalls: Array<Record<string, unknown>>;
}

const fixture: Fixture = {
  loginCount: 0,
  currentToken: 'tok-0',
  loginDelayMs: 0,
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
        const respond = () => {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ user: { id: 'u1' }, token: fixture.currentToken }));
        };
        if (fixture.loginDelayMs > 0) setTimeout(respond, fixture.loginDelayMs);
        else respond();
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
  fixture.loginDelayMs = 0;
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
//
// "N requests" below always counts hits to the endpoint actually under test
// in that scenario: the login attempt for the slow-login case (the status
// endpoint is never reached), the status endpoint for the other three — not
// a combined total, since every status poll is preceded by an incidental
// login the first time a test's token is reset (see beforeEach).

test('gen_wait: a slow login does not blow the budget — outcome budget, no status, one request', { timeout: 5000 }, async () => {
  fixture.loginDelayMs = 2200;
  let statusCalls = 0;
  fixture.route = (_req, res) => { statusCalls++; res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ status: 'running' })); };
  const loginsBefore = fixture.loginCount;
  const start = Date.now();
  const outcome = await toolsMod.genWait('job-a', 1, new AbortController().signal);
  const elapsed = Date.now() - start;
  assert.ok(elapsed < 1300, `expected <1300ms, took ${elapsed}ms`);
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-a', outcome: 'budget', status: null } });
  assert.equal(fixture.loginCount - loginsBefore, 1, 'one login attempt');
  assert.equal(statusCalls, 0, 'the status endpoint must never be reached while login is still pending');
});

test('gen_wait: abort while the first status request is held open — outcome cancelled, no status, one request', { timeout: 5000 }, async () => {
  let statusCalls = 0;
  fixture.route = (_req, res) => { statusCalls++; /* held open: never responds */ void res; };
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 50);
  const start = Date.now();
  const outcome = await toolsMod.genWait('job-b', 30, ac.signal);
  const elapsed = Date.now() - start;
  assert.ok(elapsed < 300, `expected <300ms, took ${elapsed}ms`);
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-b', outcome: 'cancelled', status: null } });
  assert.equal(statusCalls, 1);
});

test('gen_wait: first status instant, second held open, abort 50ms into the second — outcome cancelled, status is the first response, two requests', { timeout: 5000 }, async () => {
  let statusCalls = 0;
  const ac = new AbortController();
  fixture.route = (_req, res) => {
    statusCalls++;
    if (statusCalls === 1) {
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ status: 'running', seq: 1 }));
    } else {
      setTimeout(() => ac.abort(), 50); // abort only once the second request has actually started
      // held open otherwise: never responds
    }
  };
  const start = Date.now();
  const outcome = await toolsMod.genWait('job-c', 30, ac.signal);
  const elapsed = Date.now() - start;
  assert.ok(elapsed < 3000, `expected well under the 30s budget, took ${elapsed}ms`);
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-c', outcome: 'cancelled', status: { status: 'running', seq: 1 } } });
  assert.equal(statusCalls, 2);
});

test('request() with a held-open response body and timeoutMs 500 returns status 0 instead of throwing', { timeout: 5000 }, async () => {
  fixture.route = (req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.write('{"statu'); // headers + partial body sent, then held open — never calls res.end()
    void req;
  };
  const start = Date.now();
  const result = await httpMod.request('GET', '/api/generate/status/job-d', undefined, { timeoutMs: 500 });
  const elapsed = Date.now() - start;
  assert.ok(elapsed < 1000, `expected the 500ms timeout to bound this, took ${elapsed}ms`);
  assert.equal(result.ok, false);
  assert.equal(result.status, 0);
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

// ── training tools ───────────────────────────────────────────────────────────

test('train_capabilities returns the route\'s body whole', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/capabilities');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ engine: { up: true } }));
  };
  const outcome = await toolsMod.trainCapabilities();
  assert.deepEqual(outcome, { kind: 'ok', data: { engine: { up: true } } });
});

test('train_datasets trims each row to id/name/slug/sourceDir/sampleCount/assets', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/datasets');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({
      datasets: [{
        id: 'ds1', name: 'Album', slug: 'album', sourceDir: '/x/album', sampleCount: 12,
        assets: { labeled: true, built: false }, customTag: 'secret-settings-field', status: 'labeling',
      }],
    }));
  };
  const outcome = await toolsMod.trainDatasets();
  assert.deepEqual(outcome, {
    kind: 'ok',
    data: { datasets: [{ id: 'ds1', name: 'Album', slug: 'album', sourceDir: '/x/album', sampleCount: 12, assets: { labeled: true, built: false } }] },
  });
});

test('train_dataset returns one dataset in full', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/datasets/ds1');
    assert.equal(req.method, 'GET');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ id: 'ds1', samples: [] }));
  };
  const outcome = await toolsMod.trainDataset('ds1');
  assert.deepEqual(outcome, { kind: 'ok', data: { id: 'ds1', samples: [] } });
});

test('train_dataset_create posts the exact CreateDatasetInput body', async () => {
  let posted: unknown;
  fixture.route = (req, res, body) => {
    assert.equal(req.url, '/api/training/datasets');
    assert.equal(req.method, 'POST');
    posted = body;
    res.writeHead(201, { 'content-type': 'application/json' }).end(JSON.stringify({ dataset: { id: 'ds2' } }));
  };
  const args = {
    name: 'Album', sourceDir: '/x/album', recursive: true, customTag: 'tag', tagPosition: 'prepend' as const,
    genreRatio: 50, defaultArtist: 'A', defaultAlbum: 'B', defaultGenre: 'G', defaultLanguage: 'english',
  };
  const outcome = await toolsMod.trainDatasetCreate(args);
  assert.equal(outcome.kind, 'ok');
  assert.deepEqual(posted, args);
});

test('train_dataset_rescan posts to the matching route', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/datasets/ds1/rescan');
    assert.equal(req.method, 'POST');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ id: 'ds1' }));
  };
  const outcome = await toolsMod.trainDatasetRescan('ds1');
  assert.deepEqual(outcome, { kind: 'ok', data: { id: 'ds1' } });
});

test('train_dataset_label stage="label" posts options verbatim to /label', async () => {
  let posted: unknown;
  fixture.route = (req, res, body) => {
    assert.equal(req.url, '/api/training/datasets/ds1/label');
    posted = body;
    res.writeHead(202, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-label' }));
  };
  const options = { useEssentia: true, scope: 'all' };
  const outcome = await toolsMod.trainDatasetLabel({ datasetId: 'ds1', stage: 'label', options });
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-label' } });
  assert.deepEqual(posted, options);
});

test('train_dataset_label stage="caption" posts options verbatim to /enhance/caption', async () => {
  let posted: unknown;
  fixture.route = (req, res, body) => {
    assert.equal(req.url, '/api/training/datasets/ds1/enhance/caption');
    posted = body;
    res.writeHead(202, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-caption' }));
  };
  const options = { provider: 'gemini' };
  const outcome = await toolsMod.trainDatasetLabel({ datasetId: 'ds1', stage: 'caption', options });
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-caption' } });
  assert.deepEqual(posted, options);
});

test('train_dataset_label stage="build" posts options verbatim to /build', async () => {
  let posted: unknown;
  fixture.route = (req, res, body) => {
    assert.equal(req.url, '/api/training/datasets/ds1/build');
    posted = body;
    res.writeHead(202, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-build' }));
  };
  const options = { outputPath: '/x/album/dataset.json' };
  const outcome = await toolsMod.trainDatasetLabel({ datasetId: 'ds1', stage: 'build', options });
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-build' } });
  assert.deepEqual(posted, options);
});

test('train_dataset_label surfaces a 409 ("a job is already running") verbatim', async () => {
  fixture.route = (_req, res) => {
    res.writeHead(409, { 'content-type': 'application/json' }).end(JSON.stringify({ error: 'A job is already running for this dataset' }));
  };
  const outcome = await toolsMod.trainDatasetLabel({ datasetId: 'ds1', stage: 'label', options: {} });
  assert.equal(outcome.kind, 'http_error');
  assert.equal((outcome as { status: number }).status, 409);
  assert.match((outcome as { text: string }).text, /A job is already running for this dataset/);
});

test('train_jobs lists jobs, optionally filtered by datasetId', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/jobs?datasetId=ds1');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobs: [] }));
  };
  const outcome = await toolsMod.trainJobs('ds1');
  assert.deepEqual(outcome, { kind: 'ok', data: { jobs: [] } });
});

test('train_job GETs status by default', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/jobs/job-1');
    assert.equal(req.method, 'GET');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ id: 'job-1', status: 'running' }));
  };
  const outcome = await toolsMod.trainJob('job-1');
  assert.deepEqual(outcome, { kind: 'ok', data: { id: 'job-1', status: 'running' } });
});

test('train_job with cancel:true DELETEs instead', async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/jobs/job-1');
    assert.equal(req.method, 'DELETE');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true }));
  };
  const outcome = await toolsMod.trainJob('job-1', true);
  assert.deepEqual(outcome, { kind: 'ok', data: { ok: true } });
});

// train_wait shares genWait's waitForJob loop (see the four gen_wait cases
// above, which exercise the loop itself); these two only confirm train_wait
// wires it to the right path and terminal-status set ('done', not 'succeeded').

test('train_wait: budget runs out against a non-terminal job — outcome budget, status preserved', { timeout: 5000 }, async () => {
  fixture.route = (req, res) => {
    assert.equal(req.url, '/api/training/jobs/job-t1');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ id: 'job-t1', status: 'running' }));
  };
  const outcome = await toolsMod.trainWait('job-t1', 1, new AbortController().signal);
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-t1', outcome: 'budget', status: { id: 'job-t1', status: 'running' } } });
});

test('train_wait: a "done" status ends the loop as outcome done', async () => {
  fixture.route = (_req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ id: 'job-t2', status: 'done' }));
  };
  const outcome = await toolsMod.trainWait('job-t2', 5, new AbortController().signal);
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-t2', outcome: 'done', status: { id: 'job-t2', status: 'done' } } });
});

// ── train_prepare / train_start / train_runs ────────────────────────────────

/** Records every request that reaches a route and answers it with `reply`. */
function recordTraining(reply: (req: http.IncomingMessage) => [number, unknown]) {
  const calls: Array<{ method: string; url: string; body: unknown }> = [];
  fixture.route = (req, res, body) => {
    calls.push({ method: req.method ?? '', url: req.url ?? '', body });
    const [status, data] = reply(req);
    res.writeHead(status, { 'content-type': 'application/json' }).end(JSON.stringify(data));
  };
  return calls;
}

test('train_prepare ace posts the exact preprocess body, options merged under the typed fields', async () => {
  const calls = recordTraining(() => [202, { jobId: 'job-pre' }]);
  const outcome = await toolsMod.trainPrepare({
    datasetId: 'ds1', backend: 'ace',
    ace: { ditModel: 'dit-a', maxDuration: 300, dtype: 'bf16', sampleIds: ['s1', 's2'] },
    options: { futureField: 7 },
  });
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-pre' } });
  assert.deepEqual(calls, [{
    method: 'POST', url: '/api/training/datasets/ds1/preprocess',
    body: { futureField: 7, ditModel: 'dit-a', maxDuration: 300, dtype: 'bf16', sampleIds: ['s1', 's2'] },
  }]);
});

test('train_prepare mm3 posts launder to /mm3-codes', async () => {
  const calls = recordTraining(() => [200, { jobId: 'job-codes', kind: 'mm3-codes' }]);
  const outcome = await toolsMod.trainPrepare({ datasetId: 'ds1', backend: 'mm3', mm3: { launder: true, maxDuration: 240 } });
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-codes', kind: 'mm3-codes' } });
  assert.deepEqual(calls, [{ method: 'POST', url: '/api/training/datasets/ds1/mm3-codes', body: { launder: true, maxDuration: 240 } }]);
});

test('train_prepare yue2 routes by stage: preprocess and joint-prepare', async () => {
  const calls = recordTraining(req => [req.url?.endsWith('joint-prepare') ? 202 : 200, { jobId: 'job-y' }]);
  await toolsMod.trainPrepare({ datasetId: 'ds1', backend: 'yue2', stage: 'preprocess', yue2Preprocess: { vaeVariant: 'legacy', captionMode: 'yue2', loudnessLufs: 0 } });
  await toolsMod.trainPrepare({
    datasetId: 'ds1', backend: 'yue2', stage: 'joint-prepare',
    yue2JointPrepare: { models: { vae: '/m/vae.gguf', semantic: '/m/sem.gguf', sheetsage: '/m/ss.gguf' }, lyricTiming: false },
  });
  assert.deepEqual(calls, [
    { method: 'POST', url: '/api/training/datasets/ds1/yue2-preprocess', body: { vaeVariant: 'legacy', captionMode: 'yue2', loudnessLufs: 0 } },
    { method: 'POST', url: '/api/training/datasets/ds1/yue2-joint-prepare', body: { models: { vae: '/m/vae.gguf', semantic: '/m/sem.gguf', sheetsage: '/m/ss.gguf' }, lyricTiming: false } },
  ]);
});

test('train_prepare rejects a missing or misplaced stage, another backend\'s fields, a typed key in options and an unknown backend, with no HTTP call', async () => {
  const calls = recordTraining(() => [202, { jobId: 'never' }]);
  const cases: TrainPrepareArgs[] = [
    { datasetId: 'ds1', backend: 'yue2' },
    { datasetId: 'ds1', backend: 'ace', stage: 'preprocess' },
    { datasetId: 'ds1', backend: 'ace', mm3: { launder: true } },
    { datasetId: 'ds1', backend: 'mm3', options: { launder: true } },
    { datasetId: 'ds1', backend: 'nope' as unknown as 'ace' },
  ];
  for (const args of cases) assert.equal((await toolsMod.trainPrepare(args)).kind, 'rejected', JSON.stringify(args));
  assert.equal(calls.length, 0);
});

test('train_start ace-lm posts the exact train-lm body', async () => {
  const calls = recordTraining(() => [202, { jobId: 'job-lm' }]);
  const outcome = await toolsMod.trainStart({
    datasetId: 'ds1', backend: 'ace-lm',
    aceLm: { variantKey: 'acestep-v15-base-BF16', lmSize: '1.7B', targetLossStages: [2, 1.5], artistToken: '', initAdapter: '' },
  });
  assert.deepEqual(outcome, { kind: 'ok', data: { jobId: 'job-lm' } });
  assert.deepEqual(calls, [{
    method: 'POST', url: '/api/training/datasets/ds1/train-lm',
    body: { variantKey: 'acestep-v15-base-BF16', lmSize: '1.7B', targetLossStages: [2, 1.5], artistToken: '', initAdapter: '' },
  }]);
});

test('train_start ace-dit posts the exact train-dit body, variantKey included', async () => {
  const calls = recordTraining(() => [202, { jobId: 'job-dit' }]);
  await toolsMod.trainStart({
    datasetId: 'ds1', backend: 'ace-dit',
    aceDit: { variantKey: 'acestep-v15-xl-BF16', adapterType: 'lora', rank: 64, mirror: 'f32', attnBackend: 'flash' },
  });
  assert.deepEqual(calls, [{
    method: 'POST', url: '/api/training/datasets/ds1/train-dit',
    body: { variantKey: 'acestep-v15-xl-BF16', adapterType: 'lora', rank: 64, mirror: 'f32', attnBackend: 'flash' },
  }]);
});

test('train_start mm3-lm posts launder, basePrecision and nested regularisation exactly', async () => {
  const calls = recordTraining(() => [200, { jobId: 'job-mm3', kind: 'mm3-train-lm', runName: 'r', outDir: '/o', attnBackend: 'flash' }]);
  const outcome = await toolsMod.trainStart({
    datasetId: 'ds1', backend: 'mm3-lm',
    mm3Lm: { launder: true, basePrecision: 'bf16', preset: 'thorough', steps: 800, regularisation: { datasetId: 'ds2', every: 4 } },
  });
  assert.equal(outcome.kind, 'ok');
  assert.deepEqual(calls, [{
    method: 'POST', url: '/api/training/datasets/ds1/mm3-train-lm',
    body: { launder: true, basePrecision: 'bf16', preset: 'thorough', steps: 800, regularisation: { datasetId: 'ds2', every: 4 } },
  }]);
});

test('train_start yue2-joint forces trainingMethod "aitk" on a fresh run and on a resume', async () => {
  const calls = recordTraining(() => [200, { jobId: 'job-y2', trainingMethod: 'aitk' }]);
  await toolsMod.trainStart({ datasetId: 'ds1', backend: 'yue2-joint', yue2Joint: { steps: 4000, saveEvery: 500, dataset: '/p/dataset.json', rank: 128 } });
  await toolsMod.trainStart({ datasetId: 'ds1', backend: 'yue2-joint', yue2Joint: { steps: 6000, resumeRunId: 'job-old', resumeStep: 4000, stopMode: 'kl', targetKl: 2 } });
  assert.deepEqual(calls, [
    { method: 'POST', url: '/api/training/datasets/ds1/yue2-joint-train', body: { steps: 4000, saveEvery: 500, dataset: '/p/dataset.json', rank: 128, trainingMethod: 'aitk' } },
    { method: 'POST', url: '/api/training/datasets/ds1/yue2-joint-train', body: { steps: 6000, resumeRunId: 'job-old', resumeStep: 4000, stopMode: 'kl', targetKl: 2, trainingMethod: 'aitk' } },
  ]);
});

test('train_start rejects trainingMethod, a half resume, another backend\'s fields, a typed key in options and an unknown backend, with no HTTP call', async () => {
  const calls = recordTraining(() => [202, { jobId: 'never' }]);
  const cases: TrainStartArgs[] = [
    { datasetId: 'ds1', backend: 'yue2-joint', yue2Joint: { steps: 10, saveEvery: 5 }, options: { trainingMethod: 'legacy' } },
    { datasetId: 'ds1', backend: 'yue2-joint', yue2Joint: { steps: 10, saveEvery: 5, trainingMethod: 'legacy' } as never },
    { datasetId: 'ds1', backend: 'yue2-joint', yue2Joint: { steps: 6000, resumeStep: 4000 } },
    { datasetId: 'ds1', backend: 'yue2-joint', yue2Joint: { steps: 6000, resumeRunId: 'job-old' } },
    { datasetId: 'ds1', backend: 'ace-dit', aceLm: { rank: 8 } },
    { datasetId: 'ds1', backend: 'ace-lm', options: { rank: 8 } },
    { datasetId: 'ds1', backend: 'legacy' as unknown as 'ace-lm' },
  ];
  for (const args of cases) assert.equal((await toolsMod.trainStart(args)).kind, 'rejected', JSON.stringify(args));
  assert.equal(calls.length, 0);
});

test('train_start and train_prepare pass a 400 and a 409 through verbatim', async () => {
  let status = 400;
  recordTraining(() => [status, { error: status === 400 ? 'Dataset has no preprocessed tensors — run Preprocess first' : 'A job is already running for this dataset' }]);
  const bad = await toolsMod.trainStart({ datasetId: 'ds1', backend: 'ace-lm' });
  assert.equal(bad.kind, 'http_error');
  assert.equal((bad as { status: number }).status, 400);
  assert.match((bad as { text: string }).text, /run Preprocess first/);
  status = 409;
  const busy = await toolsMod.trainPrepare({ datasetId: 'ds1', backend: 'mm3' });
  assert.equal(busy.kind, 'http_error');
  assert.equal((busy as { status: number }).status, 409);
  assert.match((busy as { text: string }).text, /A job is already running for this dataset/);
});

test('train_start does NOT retry on 401 — no second run', async () => {
  const calls = recordTraining(() => [202, { jobId: 'job-x' }]);
  await toolsMod.trainJobs(); // caches a token
  calls.length = 0;
  invalidateClientToken();
  const loginsBefore = fixture.loginCount;
  const outcome = await toolsMod.trainStart({ datasetId: 'ds1', backend: 'ace-dit' });
  assert.equal(outcome.kind, 'http_error');
  assert.equal((outcome as { status: number }).status, 401);
  assert.equal(fixture.loginCount, loginsBefore, 'must not relogin-and-resend a training start');
  assert.equal(calls.length, 0, 'the start must never have reached the route');
});

test('train_runs folds each backend\'s readiness GETs into one answer', async () => {
  const calls = recordTraining(req => [200, { from: req.url }]);
  const expected: Record<TrainBackend, { runs: string; readiness: Record<string, string> }> = {
    'ace-lm': { runs: 'train-lm', readiness: { preprocess: 'preprocess' } },
    'ace-dit': { runs: 'train-dit', readiness: { preprocess: 'preprocess' } },
    'mm3-lm': { runs: 'mm3-runs', readiness: { mm3: 'mm3' } },
    'yue2-joint': { runs: 'yue2-joint-runs', readiness: { yue2: 'yue2', jointPrepare: 'yue2-joint-prepare' } },
  };
  const p = (r: string) => `/api/training/datasets/ds1/${r}`;
  for (const [backend, want] of Object.entries(expected) as Array<[TrainBackend, typeof expected['ace-lm']]>) {
    calls.length = 0;
    const outcome = await toolsMod.trainRuns('ds1', backend);
    assert.deepEqual(outcome, {
      kind: 'ok',
      data: {
        runs: { from: p(want.runs) },
        readiness: Object.fromEntries(Object.entries(want.readiness).map(([k, r]) => [k, { from: p(r) }])),
      },
    });
    assert.ok(calls.every(c => c.method === 'GET'));
    assert.equal(calls.length, 1 + Object.keys(want.readiness).length);
  }
});
