// POST /api/updates/apply in isolation: fake launcher, real gate and lease, temp folders only.
import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createApplier, payloadProblems, readLastResult, MANIFEST_FILE, type ApplyDeps } from './apply.js';
import { PrepareError } from './download.js';
import { DATA_LEASE_FILE, LEASE_FILE, createGate, readLeases } from './maintenance.js';

const sha = (b: string) => crypto.createHash('sha256').update(b).digest('hex');

function fixture(t: test.TestContext, over: Partial<ApplyDeps> = {}) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-apply-'));
  t.after(() => fs.rmSync(base, { recursive: true, force: true }));
  const installRoot = path.join(base, 'install');
  const payloadDir = path.join(installRoot, 'app', 'v1.0.0-aaaaaaaaaaaa-cpu');
  const updaterDir = path.join(installRoot, 'updater');
  const dataRoot = path.join(base, 'data');
  for (const d of [payloadDir, updaterDir, dataRoot]) fs.mkdirSync(d, { recursive: true });
  const files: Record<string, string> = { 'VERSION': '1.0.0', 'server/server.mjs': 'server', 'engine/plugins/a.lua': 'lua' };
  for (const [rel, text] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(payloadDir, rel)), { recursive: true });
    fs.writeFileSync(path.join(payloadDir, rel), text);
  }
  fs.writeFileSync(path.join(payloadDir, MANIFEST_FILE), JSON.stringify({ format: 1, files: Object.fromEntries(Object.entries(files).map(([k, v]) => [k, sha(v)])) }));
  const gate = createGate();
  const sent: any[] = [];
  let stops = 0;
  let busy: string[] = [];
  let reply: (m: any) => any = () => ({ type: 'apply-ack', ok: true });
  const deps: ApplyDeps = {
    unsupported: () => null,
    revalidate: () => ({ id: 'prepare-abc123', payloadId: 'v2.0.0-bbbbbbbbbbbb-cpu', candidate: { id: 'cand-1' } }),
    payloadDir, installRoot, updaterDir, dataRoot,
    preflight: () => [], blockers: () => busy, gate,
    snapshotSet: () => ({ dbs: [path.join(dataRoot, 'library', 'hotstep.db')], files: [] }),
    ask: async (m, type) => { sent.push(m); const r = reply(m); return r?.type === type ? r : null; },
    stop: () => { stops++; },
    ...over,
  };
  return {
    base, payloadDir, updaterDir, dataRoot, gate, sent, a: createApplier(deps),
    busy: (b: string[]) => { busy = b; }, reply: (f: typeof reply) => { reply = f; }, stops: () => stops,
    leaseFiles: () => [path.join(updaterDir, LEASE_FILE), path.join(dataRoot, DATA_LEASE_FILE)].filter(f => fs.existsSync(f)),
  };
}

test('protocol 0 (or no launcher) refuses before touching the gate', async t => {
  const f = fixture(t, { unsupported: () => 'This build installs updates manually (updater protocol 0)' });
  await assert.rejects(f.a.apply('cand-1'), (e: any) => e instanceof PrepareError && e.status === 409 && /protocol 0/.test(e.message));
  assert.equal(f.gate.state.closed, false);
  assert.deepEqual(f.leaseFiles(), []);
  assert.equal(f.a.status().supported, false);
  assert.equal(f.sent.length, 0);
});

test('only the staged candidate is accepted', async t => {
  const f = fixture(t);
  await assert.rejects(f.a.apply('other'), /not the update that was downloaded/);
  assert.equal(f.gate.state.closed, false);
});

test('busy: 409 with the blockers, the gate reopens, the lease is gone and nothing is stopped', async t => {
  const f = fixture(t);
  f.busy(['a generation is running']);
  await assert.rejects(f.a.apply('cand-1'), (e: any) => e.status === 409 && e.blockers?.[0] === 'a generation is running');
  assert.equal(f.gate.state.closed, false);
  assert.deepEqual(f.leaseFiles(), []);
  assert.equal(f.sent.length, 0);
  assert.equal(f.stops(), 0);
  // Once idle the same request goes through.
  f.busy([]);
  assert.equal((await f.a.apply('cand-1')).phase, 'stopping');
});

test('a launcher refusal or silence reopens the gate and releases the lease', async t => {
  for (const r of [{ type: 'apply-ack', ok: false, error: 'the prepared payload is missing' }, null]) {
    const f = fixture(t);
    f.reply(() => r);
    await assert.rejects(f.a.apply('cand-1'), /launcher did not take the update/);
    assert.equal(f.gate.state.closed, false);
    assert.deepEqual(f.leaseFiles(), []);
    assert.equal(f.a.status().phase, 'idle');
  }
});

test('accepted: the launcher gets the lease token and snapshot list, then the server stops', async t => {
  const f = fixture(t);
  const r = await f.a.apply('cand-1');
  assert.deepEqual(r, { phase: 'stopping', from: 'v1.0.0-aaaaaaaaaaaa-cpu', to: 'v2.0.0-bbbbbbbbbbbb-cpu' });
  const [m] = f.sent;
  assert.equal(m.type, 'apply');
  assert.equal(m.preparedId, 'prepare-abc123');
  assert.equal(m.oldPayload, 'v1.0.0-aaaaaaaaaaaa-cpu');
  const [lease] = readLeases({ updaterDir: f.updaterDir, dataRoot: f.dataRoot });
  assert.equal(lease.id, m.leaseId);
  assert.equal(lease.tokenSha256, sha(m.token));
  assert.equal(f.gate.state.closed, true, 'the gate stays closed for the launcher');
  assert.equal(f.a.status().phase, 'stopping');
  await assert.rejects(f.a.apply('cand-1'), /already being installed/);
  await new Promise(r => setTimeout(r, 400));
  assert.equal(f.stops(), 1);
});

test('an edited, added or missing payload file blocks with a 400 naming it', async t => {
  const f = fixture(t);
  assert.deepEqual(await payloadProblems(f.payloadDir), []);
  fs.writeFileSync(path.join(f.payloadDir, 'engine/plugins/a.lua'), 'edited');
  fs.writeFileSync(path.join(f.payloadDir, 'engine/plugins/mine.lua'), 'user plugin');
  fs.rmSync(path.join(f.payloadDir, 'VERSION'));
  const problems = await payloadProblems(f.payloadDir);
  assert.equal(problems.length, 3);
  assert.match(problems.join('\n'), /added.*mine\.lua[\s\S]*edited.*a\.lua[\s\S]*missing.*VERSION/);
  await assert.rejects(f.a.apply('cand-1'), (e: any) => e.status === 400 && e.errors.length === 3);
  assert.equal(f.gate.state.closed, false);
  fs.rmSync(path.join(f.payloadDir, MANIFEST_FILE));
  assert.match((await payloadProblems(f.payloadDir))[0], /no readable payload-manifest/);
});

test('the last result is read from the launcher record, and ignored when malformed', async t => {
  const f = fixture(t);
  assert.equal(readLastResult(f.updaterDir), null);
  const file = path.join(f.updaterDir, 'last-result.json');
  fs.writeFileSync(file, JSON.stringify({ format: 1, outcome: 'rolled-back', from: 'a', to: 'b', at: '2026-11-11T00:00:00Z', message: 'health' }));
  assert.deepEqual(f.a.status().lastResult, { outcome: 'rolled-back', from: 'a', to: 'b', at: '2026-11-11T00:00:00Z', message: 'health' });
  fs.writeFileSync(file, JSON.stringify({ format: 1, outcome: 'maybe', at: 'x' }));
  assert.equal(readLastResult(f.updaterDir), null);
  fs.writeFileSync(file, 'not json');
  assert.equal(readLastResult(f.updaterDir), null);
});
