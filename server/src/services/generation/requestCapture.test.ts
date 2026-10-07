import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import express from 'express';
import type { Server } from 'node:http';
import {
  buildGenerateCapture, boundedLabel, boundedRefererPath, checkoutState, createGenerateCapture,
  generateCaptureDir, generateCaptureMode, isCredentialKey, redactCredentials, REDACTED,
  GENERATE_CAPTURE_SCHEMA, type GenerateCaptureMode,
} from './requestCapture.js';

const ctx = {
  mode: 'record' as const, commit: 'abc123', dirty: false, activeBackendId: 'yue2',
  referer: 'http://localhost:3000/create?x=1', now: new Date('2026-10-07T20:00:00Z'), id: 'fixed-id',
};

function sampleBody() {
  return {
    source: 'create', backend: 'yue2', caption: 'a caption', lyrics: '[Verse]\nline',
    seed: 0, randomSeed: false, lmSeed: '18446744073709551615', duration: 0, instrumental: false,
    maxTokens: 4096, apiKey: 'sk-secret', nested: { hfToken: 'hf_x', authToken: '', items: [{ password: 'p' }] },
    adapters: [{ path: 'a.safetensors', scale: 1 }],
  };
}

test('capture mode is off unless a dev server names a mode', () => {
  const dev = { HOT_STEP_DEV: '1' };
  assert.equal(generateCaptureMode({}), 'off');
  assert.equal(generateCaptureMode({ ...dev }), 'off');
  assert.equal(generateCaptureMode({ ...dev, HOTSTEP_GENERATE_CAPTURE: '1' }), 'off');
  assert.equal(generateCaptureMode({ ...dev, HOTSTEP_GENERATE_CAPTURE: 'record' }), 'record');
  assert.equal(generateCaptureMode({ ...dev, HOTSTEP_GENERATE_CAPTURE: ' Capture-Only ' }), 'capture-only');
  // not under dev.bat: the end-user launchers never set HOT_STEP_DEV
  assert.equal(generateCaptureMode({ HOTSTEP_GENERATE_CAPTURE: 'record' }), 'off');
});

test('capture never alters the request body it records', () => {
  const body = sampleBody();
  const before = structuredClone(body);
  const fixture = buildGenerateCapture(body, ctx);
  assert.deepEqual(body, before);
  // and the fixture holds a copy, not a reference
  assert.notEqual(fixture.body, body);
  (fixture.body as any).caption = 'changed';
  (fixture.body as any).adapters[0].scale = 9;
  assert.deepEqual(body, before);
});

test('credential key spellings are recognised; ordinary keys are not', () => {
  for (const k of ['apiKey', 'api_key', 'API-KEY', 'x-api-key', 'hfToken', 'access_token', 'sessionToken', 'token',
    'secret', 'clientSecret', 'secretAccessKey', 'password', 'passwd', 'passphrase', 'authorization',
    'cookie', 'credentials', 'privateKey', 'private_key', 'accessKeyId', 'access_key_id', 'signingKey', 'bearer']) {
    assert.ok(isCredentialKey(k), `${k} should be a credential key`);
  }
  for (const k of ['maxTokens', 'lmMaxNewTokens', 'keyScale', 'tokenizer', 'caption', 'lyrics', 'seed',
    'source', 'backend', 'adapterPath', 'lmAdapter', 'yue2Abc', 'mm3LmAdapter']) {
    assert.ok(!isCredentialKey(k), `${k} should not be a credential key`);
  }
});

test('credential-like keys are redacted, ordinary fields and falsy values are kept', () => {
  const body = { ...sampleBody(), privateKey: 'pk', creds: { accessKeyId: 'AKIA', secretAccessKey: 's' } };
  const fixture = buildGenerateCapture(body, ctx);
  const b = fixture.body as any;
  assert.equal(b.apiKey, REDACTED);
  assert.equal(b.privateKey, REDACTED);
  assert.equal(b.creds.accessKeyId, REDACTED);
  assert.equal(b.creds.secretAccessKey, REDACTED);
  assert.equal(b.nested.hfToken, REDACTED);
  assert.equal(b.nested.items[0].password, REDACTED);
  assert.equal(b.nested.authToken, '');          // empty: nothing to hide
  assert.equal(b.maxTokens, 4096);
  assert.equal(b.seed, 0);
  assert.equal(b.randomSeed, false);
  assert.equal(b.duration, 0);
  assert.deepEqual(fixture.redactedPaths.sort(), [
    'apiKey', 'creds.accessKeyId', 'creds.secretAccessKey', 'nested.hfToken', 'nested.items[0].password', 'privateKey',
  ]);
});

test('caller metadata keeps only bounded labels and plain paths', () => {
  assert.equal(boundedLabel('lyric-studio'), 'lyric-studio');
  assert.equal(boundedLabel('harness_2.1'), 'harness_2.1');
  for (const bad of ['', 'has space', 'Bearer abc.def', 'x'.repeat(65), 'a/b', 42, null, undefined]) {
    assert.equal(boundedLabel(bad), null, String(bad));
  }
  assert.equal(boundedRefererPath('http://localhost:3000/create?token=abc#frag'), '/create');
  assert.equal(boundedRefererPath('http://localhost:3000/'), '/');
  assert.equal(boundedRefererPath(`http://h/${'a'.repeat(200)}`), null);
  assert.equal(boundedRefererPath('http://h/a%20b'), null);
  assert.equal(boundedRefererPath('not a url'), null);
  assert.equal(boundedRefererPath(undefined), null);
});

test('fixture records commit, caller, settings and seed', () => {
  const fixture = buildGenerateCapture(sampleBody(), { ...ctx, callerLabel: 'harness' });
  assert.equal(fixture.schema, GENERATE_CAPTURE_SCHEMA);
  assert.equal(fixture.id, 'fixed-id');
  assert.equal(fixture.capturedAt, '2026-10-07T20:00:00.000Z');
  assert.equal(fixture.commit, 'abc123');
  assert.deepEqual(fixture.caller, { source: 'create', label: 'harness', refererPath: '/create' });
  assert.deepEqual(fixture.settings, { activeBackendId: 'yue2', submittedBackend: 'yue2' });
  assert.deepEqual(fixture.seed, { seed: 0, randomSeed: false, lmSeed: '18446744073709551615' });
});

test('free-text caller header is dropped, not stored', () => {
  const fixture = buildGenerateCapture({ source: 'free text with a secret sk-123' }, { ...ctx, callerLabel: 'Bearer sk-123' });
  assert.deepEqual(fixture.caller, { source: null, label: null, refererPath: '/create' });
});

test('a missing or non-object body still produces a fixture', () => {
  for (const body of [undefined, null, 'text', [1, 2]]) {
    const fixture = buildGenerateCapture(body, ctx);
    assert.equal(fixture.caller.source, null);
    assert.deepEqual(fixture.seed, { seed: undefined, randomSeed: undefined, lmSeed: undefined });
  }
});

test('redactCredentials leaves its input untouched', () => {
  const input = { token: 't', list: [{ secret: 's' }] };
  const before = structuredClone(input);
  const { value } = redactCredentials(input);
  assert.deepEqual(input, before);
  assert.deepEqual(value, { token: REDACTED, list: [{ secret: REDACTED }] });
});

test('each capture reads HEAD afresh, so a commit between captures is attributed correctly', () => {
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), 'capture-head-'));
  const git = (...args: string[]) => execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', ...args],
    { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  try {
    git('init', '-q');
    fs.writeFileSync(path.join(repo, 'a.txt'), '1');
    git('add', 'a.txt');
    git('commit', '-q', '-m', 'one');
    const first = checkoutState(repo);
    fs.writeFileSync(path.join(repo, 'a.txt'), '2');
    assert.equal(checkoutState(repo).dirty, true);
    git('commit', '-q', '-am', 'two');
    const second = checkoutState(repo);
    assert.equal(first.commit, execFileSync('git', ['rev-parse', 'HEAD~1'], { cwd: repo, encoding: 'utf8' }).trim());
    assert.equal(second.commit, git('rev-parse', 'HEAD'));
    assert.notEqual(first.commit, second.commit);
    assert.equal(first.dirty, false);
    assert.equal(second.dirty, false);
  } finally {
    fs.rmSync(repo, { recursive: true, force: true });
  }
});

// ── Route level ─────────────────────────────────────────────────────────────

async function withApp(mode: GenerateCaptureMode, run: (url: string, dir: string, jobs: () => number) => Promise<void>) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'capture-route-'));
  let jobs = 0;
  let commits = 0;
  const app = express();
  app.use(express.json());
  app.post('/api/generate', createGenerateCapture({
    mode: () => mode,
    userId: req => (req.headers.authorization === 'Bearer good' ? 'user-1' : null),
    activeBackendId: () => 'ace',
    dir: () => dir,
    checkout: () => ({ commit: `c${++commits}`, dirty: false }),
  }), (_req, res) => { jobs++; res.json({ jobId: 'job-1', status: 'pending' }); });
  const server: Server = await new Promise(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const { port } = server.address() as { port: number };
  try {
    await run(`http://127.0.0.1:${port}/api/generate`, dir, () => jobs);
  } finally {
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const post = (url: string, token?: string, body: unknown = sampleBody()) => fetch(url, {
  method: 'POST',
  headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
  body: JSON.stringify(body),
});

for (const mode of ['record', 'capture-only'] as const) {
  test(`${mode}: a missing or stale token gets 401 and writes nothing`, async () => {
    await withApp(mode, async (url, dir, jobs) => {
      for (const token of [undefined, 'stale']) {
        const res = await post(url, token);
        assert.equal(res.status, 401);
        assert.deepEqual(await res.json(), { error: 'Unauthorized' });
      }
      assert.deepEqual(fs.readdirSync(dir), []);
      assert.equal(jobs(), 0);
    });
  });
}

test('capture-only with a valid token writes one fixture and creates no job', async () => {
  await withApp('capture-only', async (url, dir, jobs) => {
    const res = await post(url, 'good');
    assert.equal(res.status, 200);
    const out = await res.json() as { jobId: unknown; status: string; captureId: string };
    assert.equal(out.jobId, null);
    assert.equal(out.status, 'captured');
    assert.equal(jobs(), 0);
    const files = fs.readdirSync(dir);
    assert.equal(files.length, 1);
    const fixture = JSON.parse(fs.readFileSync(path.join(dir, files[0]), 'utf8'));
    assert.equal(fixture.id, out.captureId);
    assert.equal(fixture.body.apiKey, REDACTED);
    assert.equal(fixture.body.caption, 'a caption');
  });
});

test('record with a valid token writes a fixture and hands on to the handler', async () => {
  await withApp('record', async (url, dir, jobs) => {
    const first = await post(url, 'good');
    const second = await post(url, 'good');
    assert.deepEqual(await first.json(), { jobId: 'job-1', status: 'pending' });
    await second.json();
    assert.equal(jobs(), 2);
    const commits = fs.readdirSync(dir).map(f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')).commit).sort();
    assert.deepEqual(commits, ['c1', 'c2']);   // checkout state is read per request
  });
});

test('off passes every request straight through, token or not', async () => {
  await withApp('off', async (url, dir, jobs) => {
    assert.equal((await post(url)).status, 200);
    assert.equal(jobs(), 1);
    assert.deepEqual(fs.readdirSync(dir), []);
  });
});

test('the real generate route checks the token before capturing', async () => {
  const saved = { dev: process.env.HOT_STEP_DEV, cap: process.env.HOTSTEP_GENERATE_CAPTURE };
  process.env.HOT_STEP_DEV = '1';
  process.env.HOTSTEP_GENERATE_CAPTURE = 'capture-only';
  const dir = generateCaptureDir();
  const count = () => (fs.existsSync(dir) ? fs.readdirSync(dir).length : 0);
  const before = count();
  const { default: router } = await import('../../routes/generate.js');
  const app = express();
  app.use(express.json());
  app.use('/api/generate', router);
  const server: Server = await new Promise(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const { port } = server.address() as { port: number };
  try {
    for (const token of [undefined, 'stale']) {
      // Without the capture middleware ahead of it this route would answer 503
      // (the engine is not running in tests), not 401.
      const res = await post(`http://127.0.0.1:${port}/api/generate`, token);
      assert.equal(res.status, 401);
    }
    assert.equal(count(), before);
    // Control: with capture off the same unauthenticated request never reaches
    // a 401 here, so the 401 above came from the capture middleware.
    delete process.env.HOTSTEP_GENERATE_CAPTURE;
    const control = await post(`http://127.0.0.1:${port}/api/generate`);
    assert.notEqual(control.status, 401);
    assert.equal(count(), before);
  } finally {
    await new Promise(resolve => server.close(resolve));
    if (saved.dev === undefined) delete process.env.HOT_STEP_DEV; else process.env.HOT_STEP_DEV = saved.dev;
    if (saved.cap === undefined) delete process.env.HOTSTEP_GENERATE_CAPTURE; else process.env.HOTSTEP_GENERATE_CAPTURE = saved.cap;
  }
});
