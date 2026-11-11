// Disposable fixtures under the OS temp folder: never the real data root.
import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import express from 'express';
import Database from 'better-sqlite3';
import {
  DATA_LEASE_FILE, LEASE_FILE, MANIFEST_FILE, TOKEN_ENV, TOKEN_HEADER, UPDATER_DIR_ENV,
  acquireLease, createGate, createSnapshot, leaseForStartup, maintenanceRouter, readLeases, readTransaction,
  releaseLease, restoreSnapshot, verifyRestored, writeTransaction, type Lease,
} from './maintenance.js';
import { resolveRuntime } from '../../dataroot/bootstrap.js';
import { POINTER_FILE } from '../../dataroot/paths.js';

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'hs-maint-'));

function dirs() {
  const base = tmp();
  const o = { base, updaterDir: path.join(base, 'updater'), installRoot: path.join(base, 'install'), dataRoot: path.join(base, 'data') };
  for (const d of [o.updaterDir, o.installRoot, o.dataRoot]) fs.mkdirSync(d, { recursive: true });
  return o;
}

function makeDb(file: string, rows: number): Database.Database {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const db = new Database(file);
  db.pragma('journal_mode = WAL');
  db.exec('CREATE TABLE IF NOT EXISTS songs (id INTEGER PRIMARY KEY, title TEXT)');
  const ins = db.prepare('INSERT INTO songs (title) VALUES (?)');
  for (let i = 0; i < rows; i++) ins.run(`song ${i} ${'x'.repeat(200)}`);
  return db;
}

const count = (file: string) => {
  const db = new Database(file, { readonly: true });
  try { return (db.prepare('SELECT COUNT(*) AS n FROM songs').get() as { n: number }).n; } finally { db.close(); }
};

// ── Inert by default ─────────────────────────────────────────────────

test('no lease on disk: the gate passes through synchronously and startup work runs at once', () => {
  const d = dirs();
  assert.equal(leaseForStartup({ updaterDir: d.updaterDir, dataRoot: d.dataRoot }, {}), null);
  assert.equal(leaseForStartup({}, {}), null);
  const gate = createGate();
  let nextCalled = false;
  const res = new Proxy({}, { get: (_t, k) => { if (k === 'once') return () => {}; throw new Error(`res.${String(k)} touched`); } });
  gate.middleware({ method: 'GET', path: '/api/songs' } as any, res as any, () => { nextCalled = true; });
  assert.ok(nextCalled, 'next() ran in the same tick');
  const order: number[] = [];
  gate.whenOpen(() => order.push(1));
  order.push(2);
  gate.whenOpen(() => order.push(3));
  assert.deepEqual(order, [1, 2, 3], 'whenOpen ran in line, so startup order is unchanged');
  assert.deepEqual(gate.state, { closed: false, leaseId: null, deferred: 0 });
});

// ── Gate ─────────────────────────────────────────────────────────────

function serve(gate: ReturnType<typeof createGate>) {
  const app = express();
  app.use(gate.middleware);
  app.get('/api/health', (_q, r) => { r.json({ status: 'ok' }); });
  app.use('/api/maintenance', maintenanceRouter(gate, () => ({ build: 'test' })));
  app.get('/api/songs', (_q, r) => { r.json([]); });
  app.post('/api/songs', (_q, r) => { setTimeout(() => r.json({ ok: true }), 300); });
  app.get('/{*splat}', (_q, r) => { r.send('<html>'); });
  const server = app.listen(0);
  const agent = new http.Agent({ keepAlive: true, maxSockets: 1 });
  const port = () => (server.address() as { port: number }).port;
  const req = (method: string, p: string, headers: Record<string, string> = {}) => new Promise<{ status: number; body: string; socket: unknown; retry: string | undefined }>((resolve, reject) => {
    const r = http.request({ host: '127.0.0.1', port: port(), path: p, method, agent, headers }, res => {
      let body = '';
      res.on('data', c => { body += c; });
      res.on('end', () => resolve({ status: res.statusCode!, body, socket: res.socket, retry: res.headers['retry-after'] as string | undefined }));
    });
    r.on('error', reject);
    r.end();
  });
  return { req, close: () => { agent.destroy(); server.close(); } };
}

function fakeLease(token: string): Lease {
  return { format: 1, id: crypto.randomUUID(), tokenSha256: crypto.createHash('sha256').update(token).digest('hex'), installRoot: 'i', dataRoot: 'd', pid: 1, createdAt: '' };
}

test('a closed gate blocks an existing keep-alive client; only token-authenticated control passes', async () => {
  const gate = createGate();
  const s = serve(gate);
  try {
    const before = await s.req('GET', '/api/songs');
    assert.equal(before.status, 200);
    gate.close(fakeLease('secret'));
    const after = await s.req('GET', '/api/songs');
    assert.equal(after.socket, before.socket, 'same connection as before the gate closed');
    assert.equal(after.status, 503);
    assert.equal(after.retry, '5');
    assert.match(JSON.parse(after.body).error, /installing an update/);
    assert.equal((await s.req('POST', '/api/songs')).status, 503);
    assert.equal((await s.req('GET', '/index.html')).status, 503, 'static and SPA routes are behind the gate too');
    assert.equal((await s.req('GET', '/api/health')).status, 503, 'health needs the token');
    assert.equal((await s.req('GET', '/api/health', { [TOKEN_HEADER]: 'wrong' })).status, 503);
    assert.equal((await s.req('GET', '/api/songs', { [TOKEN_HEADER]: 'secret' })).status, 503, 'the token opens control paths only');
    assert.equal((await s.req('GET', '/api/health', { [TOKEN_HEADER]: 'secret' })).status, 200);
    const status = await s.req('GET', '/api/maintenance/status', { [TOKEN_HEADER]: 'secret' });
    assert.equal(status.status, 200);
    assert.equal(JSON.parse(status.body).closed, true);
    assert.equal((await s.req('POST', '/api/maintenance/release', { [TOKEN_HEADER]: 'secret' })).status, 200);
    assert.equal((await s.req('GET', '/api/songs')).status, 200, 'release reopens admission');
  } finally { s.close(); }
});

test('startup hooks wait for release and then run in registration order', () => {
  const gate = createGate();
  gate.close(fakeLease('t'));
  const ran: string[] = [];
  gate.whenOpen(() => ran.push('reconcile'));
  gate.whenOpen(() => ran.push('mirror'));
  gate.whenOpen(() => { throw new Error('one failing hook'); });
  gate.whenOpen(() => ran.push('workflows'));
  assert.deepEqual(ran, []);
  assert.equal(gate.state.deferred, 4);
  gate.open();
  assert.deepEqual(ran, ['reconcile', 'mirror', 'workflows'], 'a failing hook does not stop the rest');
  gate.whenOpen(() => ran.push('late'));
  assert.equal(ran.at(-1), 'late');
});

test('begin refuses while work runs or is queued, reopens and drops the lease; never cancels', async () => {
  const d = dirs();
  const gate = createGate();
  gate.addBlocker(() => ['1 audio queue item(s) waiting']);
  await assert.rejects(gate.begin({ ...d, preflight: () => [], blockers: () => ['1 generation job(s) running or queued'] }),
    (e: any) => e.status === 409 && e.blockers.length === 2);
  assert.equal(gate.state.closed, false);
  assert.deepEqual(readLeases(d), []);

  await assert.rejects(gate.begin({ ...d, preflight: () => ['a data-folder change is scheduled'], blockers: () => [] }), (e: any) => e.status === 400);
  assert.deepEqual(readLeases(d), [], 'preflight runs before the lease is taken');
});

test('begin waits for in-flight writes, then holds the lease in both places', async () => {
  const d = dirs();
  const gate = createGate();
  const s = serve(gate);
  try {
    const slow = s.req('POST', '/api/songs');
    await new Promise(r => setTimeout(r, 50));
    let checkedAt = 0;
    const started = Date.now();
    const { lease, token } = await gate.begin({ ...d, preflight: () => [], blockers: () => { checkedAt = Date.now(); return []; } });
    assert.ok(checkedAt - started >= 150, 'blockers were checked after the write finished');
    assert.equal((await slow).status, 200, 'the in-flight write completed, not cancelled');
    assert.equal(gate.state.closed, true);
    assert.deepEqual(readLeases(d).map(l => l.id), [lease.id, lease.id]);
    assert.ok(fs.existsSync(path.join(d.updaterDir, LEASE_FILE)) && fs.existsSync(path.join(d.dataRoot, DATA_LEASE_FILE)));
    assert.ok(!JSON.stringify(readLeases(d)).includes(token), 'only the token hash is on disk');
    await assert.rejects(createGate().begin({ ...d, preflight: () => [], blockers: () => [] }), (e: any) => e.status === 409);
    releaseLease(d, lease.id);
    assert.deepEqual(readLeases(d), []);
  } finally { s.close(); }
});

// ── Lease and second process ─────────────────────────────────────────

test('a second process is refused before bootstrap; a dead PID does not end the lease', async () => {
  const d = dirs();
  const { lease, token } = acquireLease(d);
  // A PID that cannot be alive.
  const dead = { ...lease, pid: 2 ** 30 };
  fs.writeFileSync(path.join(d.updaterDir, LEASE_FILE), JSON.stringify(dead));
  fs.writeFileSync(path.join(d.dataRoot, DATA_LEASE_FILE), JSON.stringify(dead));

  assert.throws(() => leaseForStartup(d, {}), /update is being installed/);
  assert.throws(() => leaseForStartup({ dataRoot: d.dataRoot }, { [TOKEN_ENV]: 'nope' }), /update is being installed/);
  assert.equal(leaseForStartup(d, { [TOKEN_ENV]: token })?.id, lease.id, 'the candidate started with the token is let through');

  // bootstrap: a release whose data root holds the lease exits before initLayout or any transfer.
  const prof = path.join(d.base, 'profile');
  fs.mkdirSync(prof);
  fs.writeFileSync(path.join(prof, POINTER_FILE), JSON.stringify({ version: 1, root: d.dataRoot, pending: null }));
  const env = { HOT_STEP_ROOT: d.installRoot, HOT_STEP_PROFILE_DIR: prof } as NodeJS.ProcessEnv;
  const r = await resolveRuntime({ env, log: () => {}, skipEngineProbe: true });
  assert.equal(r.ok, false);
  assert.equal((r as any).lease, true);
  assert.ok(!fs.existsSync(path.join(d.dataRoot, 'library')), 'no layout was created');
  const viaOverride = await resolveRuntime({ env: { ...env, HOT_STEP_DATA_ROOT: d.dataRoot }, log: () => {}, skipEngineProbe: true });
  assert.equal((viaOverride as any).lease, true);
  const viaUpdater = await resolveRuntime({ env: { ...env, [UPDATER_DIR_ENV]: d.updaterDir, HOT_STEP_DATA_ROOT: tmp() }, log: () => {}, skipEngineProbe: true });
  assert.equal((viaUpdater as any).lease, true, 'the install-side lease alone refuses too');
  const ours = await resolveRuntime({ env: { ...env, [UPDATER_DIR_ENV]: d.updaterDir, [TOKEN_ENV]: token }, log: () => {}, skipEngineProbe: true });
  assert.equal(ours.ok, true);

  // Without any lease, bootstrap is unchanged.
  releaseLease(d, lease.id);
  const free = await resolveRuntime({ env, log: () => {}, skipEngineProbe: true });
  assert.equal(free.ok, true);
});

test('a malformed lease or an unleased transaction record fails closed', () => {
  const d = dirs();
  fs.writeFileSync(path.join(d.dataRoot, DATA_LEASE_FILE), '{not json');
  assert.throws(() => leaseForStartup(d, {}), /not valid JSON/);
  fs.rmSync(path.join(d.dataRoot, DATA_LEASE_FILE));

  const tx = {
    format: 1 as const, id: 'tx1', state: 'pending' as const, leaseId: 'l1', oldPayload: 'v1.5.1-abc-cuda13.1', newPayload: 'v1.6.0-def-cuda13.1',
    dataRoot: d.dataRoot, snapshot: { dir: path.join(d.updaterDir, 'snapshots', 'tx1'), manifestSha256: 'a'.repeat(64) }, createdAt: '', committedAt: null,
  };
  writeTransaction(d.updaterDir, tx);
  assert.deepEqual(readTransaction(d.updaterDir), tx);
  assert.throws(() => leaseForStartup(d, {}), /no maintenance lease/);
  assert.throws(() => writeTransaction(d.updaterDir, { ...tx, newPayload: '../evil' }), /not a valid/);
});

// ── Snapshot ─────────────────────────────────────────────────────────

test('committed WAL data survives a snapshot, from a live writer and from a leftover WAL', async () => {
  const d = dirs();
  const live = path.join(d.dataRoot, 'live', 'hotstep.db');
  const writer = makeDb(live, 10);
  writer.pragma('wal_autocheckpoint = 0');
  writer.prepare('INSERT INTO songs (title) VALUES (?)').run('only in the WAL');
  assert.ok(fs.statSync(live + '-wal').size > 0);

  // A crash leaves the .db and its -wal behind with no -shm.
  const left = path.join(d.dataRoot, 'left', 'hotstep.db');
  fs.mkdirSync(path.dirname(left));
  fs.copyFileSync(live, left);
  fs.copyFileSync(live + '-wal', left + '-wal');

  const a = await createSnapshot(path.join(d.updaterDir, 'a'), { dbs: [live], files: [] });
  writer.close();
  const b = await createSnapshot(path.join(d.updaterDir, 'b'), { dbs: [left], files: [] });
  for (const [dir, m] of [['a', a.manifest], ['b', b.manifest]] as const) {
    const snap = path.join(d.updaterDir, dir, m.entries[0].name!);
    assert.equal(count(snap), 11, `${dir}: the WAL row is in the snapshot`);
    assert.ok(!fs.existsSync(snap + '-wal'), `${dir}: the snapshot is one self-contained file`);
  }
  assert.equal(count(left), 11, 'the source still has its WAL rows (incorporated, never dropped)');
});

test('a failed integrity check aborts the snapshot and leaves nothing behind', async () => {
  const d = dirs();
  const file = path.join(d.dataRoot, 'hotstep.db');
  const db = makeDb(file, 200);
  db.pragma('journal_mode = DELETE');
  db.close();
  const fd = fs.openSync(file, 'r+');
  fs.writeSync(fd, Buffer.alloc(512, 0xff), 0, 512, 4096 * 2 + 16);
  fs.closeSync(fd);
  const dir = path.join(d.updaterDir, 'bad');
  await assert.rejects(createSnapshot(dir, { dbs: [file], files: [] }), /integrity_check failed/);
  assert.ok(!fs.existsSync(dir));
});

// ── Restore ──────────────────────────────────────────────────────────

async function candidateFixture() {
  const d = dirs();
  const db = path.join(d.dataRoot, 'library', 'hotstep.db');
  const legacy = path.join(d.dataRoot, 'library', 'lireek.db');
  const migrated = legacy + '.migrated';
  const env = path.join(d.dataRoot, 'settings', '.env');
  const ptr = path.join(d.base, 'profile', POINTER_FILE);
  makeDb(db, 5).close();
  makeDb(legacy, 3).close();
  fs.mkdirSync(path.dirname(env), { recursive: true });
  fs.writeFileSync(env, 'OLD=1\n');
  fs.mkdirSync(path.dirname(ptr), { recursive: true });
  fs.writeFileSync(ptr, '{"version":1}');
  const snapDir = path.join(d.updaterDir, 'snapshots', 'tx1');
  const snap = await createSnapshot(snapDir, { dbs: [db, legacy], files: [env, ptr, migrated] });

  // The candidate migrates: writes rows, leaves a WAL, renames the legacy DB, edits settings.
  const w = makeDb(db, 4);
  w.pragma('wal_autocheckpoint = 0');
  w.prepare('INSERT INTO songs (title) VALUES (?)').run('candidate row');
  const wal = fs.readFileSync(db + '-wal');
  w.close();
  fs.writeFileSync(db + '-wal', wal);
  fs.renameSync(legacy, migrated);
  fs.writeFileSync(env, 'NEW=1\n');
  return { d, db, legacy, migrated, env, ptr, snapDir, snap };
}

test('restore puts back the whole set, drops the candidate WAL and keeps a diagnosis copy', async () => {
  const f = await candidateFixture();
  await restoreSnapshot(f.snapDir, f.snap.manifestSha256);
  assert.equal(count(f.db), 5);
  assert.equal(count(f.legacy), 3, 'the legacy DB is back');
  assert.ok(!fs.existsSync(f.migrated), 'a file the snapshot did not have is removed');
  assert.ok(!fs.existsSync(f.db + '-wal'), 'the old DB never sits beside the candidate WAL');
  assert.equal(fs.readFileSync(f.env, 'utf-8'), 'OLD=1\n');
  const diag = fs.readdirSync(path.join(f.snapDir, 'candidate'));
  assert.ok(diag.some(n => n.endsWith('hotstep.db-wal')) && diag.some(n => n.endsWith('lireek.db.migrated')), 'candidate files kept for diagnosis');
  verifyRestored(f.snapDir, f.snap.manifestSha256);
});

test('restore after an interrupted restore finishes the same set; the snapshot is never changed', async () => {
  const f = await candidateFixture();
  const before = fs.readdirSync(f.snapDir).map(n => [n, fs.statSync(path.join(f.snapDir, n)).mtimeMs]);
  // Interrupted mid-swap: diagnosis copy done, settings already restored, a
  // half-written staged DB, the candidate DB and WAL still in place.
  const diag = path.join(f.snapDir, 'candidate');
  fs.mkdirSync(diag);
  fs.writeFileSync(path.join(diag, '0-hotstep.db'), 'first run copy');
  fs.writeFileSync(path.join(diag, 'done'), '');
  fs.writeFileSync(f.env, 'OLD=1\n');
  fs.writeFileSync(f.db + '.restore-tmp', 'half');
  assert.throws(() => verifyRestored(f.snapDir), 'not valid yet');

  await restoreSnapshot(f.snapDir, f.snap.manifestSha256);
  await restoreSnapshot(f.snapDir, f.snap.manifestSha256);
  assert.equal(count(f.db), 5);
  assert.equal(count(f.legacy), 3);
  assert.ok(!fs.existsSync(f.db + '.restore-tmp'));
  assert.equal(fs.readFileSync(path.join(diag, '0-hotstep.db'), 'utf-8'), 'first run copy', 'a rerun never overwrites the first diagnosis copy');
  for (const [n, t] of before) assert.equal(fs.statSync(path.join(f.snapDir, n as string)).mtimeMs, t, `${n} unchanged`);
});

test('a damaged snapshot or manifest stops restore before anything is touched', async () => {
  const f = await candidateFixture();
  const name = f.snap.manifest.entries[0].name!;
  fs.appendFileSync(path.join(f.snapDir, name), 'x');
  await assert.rejects(restoreSnapshot(f.snapDir, f.snap.manifestSha256), /damaged/);
  assert.equal(fs.readFileSync(f.env, 'utf-8'), 'NEW=1\n', 'candidate state untouched');
  assert.ok(fs.existsSync(f.db + '-wal'));
  fs.appendFileSync(path.join(f.snapDir, MANIFEST_FILE), ' ');
  await assert.rejects(restoreSnapshot(f.snapDir, f.snap.manifestSha256), /does not match the transaction record/);
});
