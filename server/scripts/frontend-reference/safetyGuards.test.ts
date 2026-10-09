// safetyGuards.test.ts — proves the guard mechanism itself, independent of
// the full fakeServer.ts/client.test.ts run: a blocked call still records a
// violation even when the caller catches the thrown error, exactly the
// shape essentiaClient.ts's run() and workerUpdate.ts's startupCommit use in
// production (try/catch around a subprocess call, degrade to null/empty).
// client.test.ts's own closing "Safety" test only proves violations stayed
// at zero across a real run; it cannot by itself prove the mechanism would
// catch one if production ever did slip through — these tests manufacture
// that case directly.

import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import http from 'node:http';
import { allowedOrigins, installNetworkGuard, installSubprocessGuard, violations } from './safetyGuards.js';

const cp = createRequire(import.meta.url)('node:child_process') as typeof import('node:child_process');

test.before(() => {
  installSubprocessGuard();
  installNetworkGuard();
});

test.beforeEach(() => { violations.length = 0; });

test('Subprocess guard: a blocked execSync call still records a violation after being caught', () => {
  let caught = false;
  try { cp.execSync('echo hi'); } catch { caught = true; }
  assert.equal(caught, true, 'the guard must still throw so the caller sees an ordinary error');
  assert.equal(violations.length, 1);
  assert.match(violations[0], /execSync/);
});

test('Subprocess guard: execFileSync, spawnSync and spawn are blocked too', () => {
  assert.throws(() => cp.execFileSync('git', ['log']));
  assert.throws(() => cp.spawnSync('git', ['log']));
  // spawn() itself does not throw synchronously in the real API (errors
  // arrive on the child's 'error' event) — the guard replaces it outright,
  // so calling the guarded version throws immediately instead.
  assert.throws(() => cp.spawn('git', ['log']));
  assert.equal(violations.length, 3);
});

test('Subprocess guard: git rev-parse/status return a fixture, with no exec and no violation', () => {
  const head = cp.execFileSync('git', ['rev-parse', 'HEAD']);
  const status = cp.execFileSync('git', ['status', '--porcelain']);
  assert.equal(head, '0000000000000000000000000000000000000000');
  assert.equal(status, '');
  assert.equal(violations.length, 0);
});

test('Subprocess guard: a non-fixture git subcommand still blocks (not a blanket git allowlist)', () => {
  assert.throws(() => cp.execFileSync('git', ['push']));
  assert.equal(violations.length, 1);
});

test('Network guard: a blocked fetch to an unlisted origin still records a violation after being caught', async () => {
  let caught = false;
  try { await fetch('http://127.0.0.1:1/'); } catch { caught = true; }
  assert.equal(caught, true);
  assert.equal(violations.length, 1);
  assert.match(violations[0], /fetch/);
});

test('Network guard: a real local service on an unlisted 127.0.0.1 port is blocked, not just a different hostname', async () => {
  const server = http.createServer((_req, res) => res.end('not a fixture')).listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  const port = (server.address() as { port: number }).port;
  try {
    await assert.rejects(() => fetch(`http://127.0.0.1:${port}/`));
    assert.equal(violations.length, 1);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});

test('Network guard: an allowlisted origin is reachable once added', async () => {
  const server = http.createServer((_req, res) => res.end('ok')).listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  const port = (server.address() as { port: number }).port;
  const origin = `http://127.0.0.1:${port}`;
  allowedOrigins.add(origin);
  try {
    const res = await fetch(origin);
    assert.equal(await res.text(), 'ok');
    assert.equal(violations.length, 0);
  } finally {
    allowedOrigins.delete(origin);
    await new Promise(resolve => server.close(resolve));
  }
});

test('Network guard: a redirect off an allowlisted origin is blocked even though its target is also allowlisted', async () => {
  const target = http.createServer((_req, res) => res.end('should never be reached')).listen(0, '127.0.0.1');
  await new Promise(resolve => target.once('listening', resolve));
  const targetPort = (target.address() as { port: number }).port;
  const targetOrigin = `http://127.0.0.1:${targetPort}`;

  const redirector = http.createServer((_req, res) => {
    res.writeHead(302, { Location: targetOrigin });
    res.end();
  }).listen(0, '127.0.0.1');
  await new Promise(resolve => redirector.once('listening', resolve));
  const redirectorPort = (redirector.address() as { port: number }).port;
  const redirectorOrigin = `http://127.0.0.1:${redirectorPort}`;

  allowedOrigins.add(redirectorOrigin);
  allowedOrigins.add(targetOrigin);
  try {
    let caught = false;
    try { await fetch(redirectorOrigin); } catch { caught = true; }
    assert.equal(caught, true, 'a redirect response must block, not silently follow');
    assert.equal(violations.length, 1);
    assert.match(violations[0], /redirect/);
  } finally {
    allowedOrigins.delete(redirectorOrigin);
    allowedOrigins.delete(targetOrigin);
    await new Promise(resolve => redirector.close(resolve));
    await new Promise(resolve => target.close(resolve));
  }
});
