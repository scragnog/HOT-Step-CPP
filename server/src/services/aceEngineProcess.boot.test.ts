// Boot readiness (HOTSTEP-1149): the engine reads ready only once its own
// /health has answered. A missing binary or failed spawn stays down with the
// reason and `failed` set, which is what the update launcher's attestation reads.
// Own file: the engine lifecycle is module state, kept apart from the crash-limiter test.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import type { AddressInfo } from 'node:net';

function fakeChild(pid: number | null = 4242): any {
  const c: any = new EventEmitter();
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  c.pid = pid ?? undefined;   // null: a spawn that failed has no pid
  c.exitCode = null;
  c.signalCode = null;
  c.kill = () => true;
  return c;
}

const until = async (cond: () => boolean, ms = 3000) => {
  for (const end = Date.now() + ms; !cond() && Date.now() < end;) await new Promise(r => setTimeout(r, 25));
};

test('boot: ready only after /health; missing exe, failed spawn and no models are not ready', async () => {
  let healthUp = false;
  const srv = http.createServer((req, res) => {
    if (req.url === '/health' && healthUp) { res.setHeader('content-type', 'application/json'); res.end('{"status":"ok"}'); }
    else { res.statusCode = 503; res.end('booting'); }
  });
  await new Promise<void>(r => srv.listen(0, '127.0.0.1', r));
  process.env.ACESTEPCPP_PORT = String((srv.address() as AddressInfo).port);
  process.env.ACESTEPCPP_HOST = '127.0.0.1';
  process.env.ACESTEPCPP_EXE = process.execPath;   // only needs to exist

  const proc = await import('./aceEngineProcess.js');
  const state = await import('../engineState.js');
  const { config } = await import('../config.js');
  const kids: any[] = [];
  let nextChild: () => any = () => fakeChild();
  let restores = 0;
  proc.onEngineRestarted(() => { restores++; });
  proc.testHooks.spawn = (() => { const c = nextChild(); kids.push(c); return c; }) as any;
  proc.testHooks.lateReadyPollMs = 50;
  const reset = async () => {
    for (const k of kids) k.exitCode = 0;   // fake pid: never let stop reach taskkill
    await proc.stopAceServer('next case', 100, { suspend: false });
    state.setEngineReady(false, 'Starting engine...');
  };

  try {
    // Missing binary (the packaged acceptance case): down, failed, and the status names the path.
    const exe = config.aceServer.exe;
    config.aceServer.exe = path.join(path.dirname(process.execPath), 'deliberately-absent-engine.exe');
    state.setEngineReady(false, 'Starting engine...');
    assert.equal(proc.bootAceServer('Ready'), null);
    await new Promise(r => setTimeout(r, 200));
    assert.equal(state.engineReady, false, 'a missing ace-server must not read Ready');
    assert.equal(state.engineFailed, true);
    assert.match(state.engineBootStatus, /ace-server not found at .*deliberately-absent-engine\.exe/);
    config.aceServer.exe = exe;
    await reset();

    // Spawn throws (not an executable): same.
    nextChild = () => { throw new Error('spawn UNKNOWN'); };
    assert.equal(proc.bootAceServer('Ready'), null);
    assert.equal(state.engineReady, false);
    assert.equal(state.engineFailed, true);
    assert.match(state.engineBootStatus, /could not start: spawn UNKNOWN/);
    await reset();

    // Spawn fails asynchronously ('error', no pid, no 'exit').
    nextChild = () => fakeChild(null);
    const bad: any = proc.bootAceServer('Ready');
    assert.ok(bad);
    assert.equal(state.engineFailed, false, 'a fresh spawn clears failed');
    bad.emit('error', new Error('spawn EACCES'));
    assert.equal(state.engineReady, false);
    assert.equal(state.engineFailed, true);
    assert.match(state.engineBootStatus, /could not start: spawn EACCES/);
    assert.equal(proc.getAceProcess(), null, 'nothing keeps waiting on a child that never ran');
    await reset();

    // A good spawn is not ready until /health answers; then ready with the boot status, no restore hooks.
    nextChild = () => fakeChild();
    proc.bootAceServer('Ready (CPU only — GPU runtime missing)');
    await new Promise(r => setTimeout(r, 300));
    assert.equal(state.engineReady, false, 'spawned but /health not answering yet');
    assert.equal(state.engineBootStatus, 'Starting engine...');
    healthUp = true;
    await until(() => state.engineReady);
    assert.equal(state.engineReady, true);
    assert.equal(state.engineBootStatus, 'Ready (CPU only — GPU runtime missing)');
    assert.equal(state.engineFailed, false);
    assert.equal(restores, 0, 'boot leaves restores to maintenance.whenOpen');
    await reset();

    // Empty models folder: the binary reports it and exits 1. Not ready, not failed; noModels.
    healthUp = false;
    const empty: any = proc.bootAceServer('Ready');
    empty.stderr.emit('data', Buffer.from('ERROR: no models found in /models\n'));
    empty.exitCode = 1;
    empty.emit('exit', 1, null);
    await until(() => state.engineNoModels);
    assert.equal(state.engineNoModels, true);
    assert.equal(state.engineReady, false);
    assert.equal(state.engineFailed, false);
  } finally {
    for (const k of kids) k.exitCode = 0;
    await proc.stopAceServer('test over', 100).catch(() => {});
    srv.close();
  }
});
