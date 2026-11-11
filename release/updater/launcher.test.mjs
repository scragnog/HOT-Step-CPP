// The stable launcher on fixture installs in temp folders: stub payload
// servers on free ports (never 3000, 3001 or 8085), the real snapshot/restore
// tool bundled the way release/esbuild.config.mjs ships it, a throwaway DB.
// Never the real data root.
//   node --test release/updater/launcher.test.mjs
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import esbuild from 'esbuild';
import { AlreadyRunning, RECOVERY_DOC, RecoveryRequired, failureText, launch, parseDotenv, payloadIdFor, timing } from './launcher.mjs';
import { MANIFEST_FILE, packageLayout } from './package-layout.mjs';
import { checkPackageBuildInfo } from '../check-layout.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const SQLITE = path.join(repo, 'server', 'node_modules', 'better-sqlite3', 'lib', 'index.js');
const Database = createRequire(import.meta.url)(SQLITE);
const sha = s => crypto.createHash('sha256').update(s).digest('hex');
const sleep = ms => new Promise(r => setTimeout(r, ms));

Object.assign(timing, { retryDelaysMs: [20, 40], healthMs: 4_000, pollMs: 100, stopMs: 10_000, toolMs: 60_000 });

const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-launcher-'));
let toolBundle;

before(async () => {
  // The shipped tool, with better-sqlite3 pointed at the checkout's copy.
  const out = path.join(base, 'tool');
  await esbuild.build({
    entryPoints: { 'updater-tool': path.join(repo, 'server', 'src', 'services', 'updates', 'tool.ts') },
    bundle: true, platform: 'node', target: 'node22', format: 'esm', outdir: out, outExtension: { '.js': '.mjs' }, logLevel: 'error',
    banner: { js: 'import { createRequire as __cr } from "module"; const require = __cr(import.meta.url);' },
    plugins: [{ name: 'sqlite', setup(b) { b.onResolve({ filter: /^better-sqlite3$/ }, () => ({ path: pathToFileURL(SQLITE).href, external: true })); } }],
  });
  toolBundle = fs.readFileSync(path.join(out, 'updater-tool.mjs'));
});
after(() => fs.rmSync(base, { recursive: true, force: true }));

// A payload server: opens the DB, records itself in it (a startup write, like
// a migration), honours the lease the way bootstrap does, and answers the
// launcher's health and release calls. stub.json shapes its behaviour.
const STUB = `
import crypto from 'node:crypto'; import fs from 'node:fs'; import http from 'node:http'; import path from 'node:path';
import { createRequire } from 'node:module';
const Database = createRequire(import.meta.url)(${JSON.stringify(SQLITE)});
const dir = process.env.HOT_STEP_ROOT, updater = process.env.HOT_STEP_UPDATER_DIR, token = process.env.HOT_STEP_MAINTENANCE_TOKEN;
const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'stub.json'), 'utf8'));
const build = JSON.parse(fs.readFileSync(path.join(dir, 'build-info.json'), 'utf8'));
const dataRoot = process.env.HOT_STEP_DATA_ROOT || JSON.parse(fs.readFileSync(path.join(process.env.HOT_STEP_PROFILE_DIR, 'data-location.json'), 'utf8')).root;
const hash = s => crypto.createHash('sha256').update(s).digest('hex');
const leaseFile = path.join(updater, 'maintenance-lease.json');
const lease = fs.existsSync(leaseFile) ? JSON.parse(fs.readFileSync(leaseFile, 'utf8')) : null;
if (lease && (!token || hash(token) !== lease.tokenSha256)) { console.error('stub: lease held by another process'); process.exit(3); }
if (!lease && fs.existsSync(path.join(updater, 'transaction.json'))) { console.error('stub: unfinished update'); process.exit(4); }
if (cfg.exitAtStart) process.exit(5);
fs.mkdirSync(path.join(dataRoot, 'library'), { recursive: true });
const db = new Database(path.join(dataRoot, 'library', 'hotstep.db'));
db.pragma('journal_mode = WAL');
db.exec('CREATE TABLE IF NOT EXISTS runs (who TEXT)');
db.prepare('INSERT INTO runs VALUES (?)').run(cfg.version);
let closed = !!lease;
const bootAt = Date.now();
// The server's engine attestation: cfg.engine as given, or Ready (after cfg.engineReadyAfterMs).
const engineNow = () => cfg.engine ?? (Date.now() - bootAt < (cfg.engineReadyAfterMs ?? 0) ? { ready: false, status: 'Starting engine...' } : { ready: true, status: 'Ready' });
function stop() {
  db.close();
  process.send({ type: 'stopped', clean: !cfg.unclean, problems: cfg.unclean ? ['an engine process did not exit'] : [] }, () => process.exit(0));
}
const srv = http.createServer((req, res) => {
  const json = (code, body) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(body)); };
  const authorised = req.headers['x-hotstep-maintenance-token'] === token;
  let body = ''; req.on('data', d => { body += d; });
  req.on('end', () => {
    if (req.url === '/api/health') return json(cfg.health503 ? 503 : 200, { status: 'ok' });
    if (req.url === '/api/maintenance/status') {
      if (lease && !authorised) return json(403, {});
      return json(200, { closed, leaseId: lease?.id ?? null, build, db: { ok: !cfg.badDb, errors: cfg.badDb ? ['migration 42 failed'] : [] }, ui: true, engine: engineNow() });
    }
    if (req.url === '/api/maintenance/release') {
      if (lease && !authorised) return json(403, {});
      closed = false; return json(200, {});
    }
    if (req.url === '/test/apply') {
      // What maintenance.begin() and the apply route do, minus the gate.
      const t = crypto.randomBytes(32).toString('hex');
      const l = { format: 1, id: crypto.randomUUID(), tokenSha256: hash(t), installRoot: fs.realpathSync.native(process.env.HOT_STEP_INSTALL_ROOT), dataRoot: fs.realpathSync.native(dataRoot), pid: process.pid, createdAt: new Date().toISOString() };
      fs.writeFileSync(leaseFile, JSON.stringify(l));
      fs.writeFileSync(path.join(dataRoot, '.hot-step-maintenance.json'), JSON.stringify(l));
      const onAck = m => {
        if (m?.type !== 'apply-ack') return;
        process.off('message', onAck);
        json(200, m);
        if (m.ok) setTimeout(stop, 50);
        else { fs.rmSync(leaseFile, { force: true }); fs.rmSync(path.join(dataRoot, '.hot-step-maintenance.json'), { force: true }); }
      };
      process.on('message', onAck);
      process.send({ type: 'apply', ...JSON.parse(body), oldPayload: path.basename(dir), leaseId: l.id, token: t, dataRoot });
      return;
    }
    if (req.url === '/test/quit') { json(200, {}); if (body === 'restart') fs.writeFileSync(path.join(process.env.HOT_STEP_PROFILE_DIR, 'restart-requested'), ''); setTimeout(() => { db.close(); process.exit(0); }, 50); return; }
    json(404, {});
  });
});
process.on('message', m => { if (m?.type === 'stop') stop(); });
srv.listen(0, '127.0.0.1', () => {
  const port = srv.address().port;
  process.send({ type: 'listening', port, host: '127.0.0.1' });
  fs.writeFileSync(path.join(dir, 'port'), String(port));
});
`;

const info = (tag, sha1) => ({ format: 1, tag, sha: sha1.repeat(40), channel: 'release', os: 'windows', arch: 'x64', variant: 'cpu', engineSha: 'e'.repeat(40), updaterProtocol: 1 });
const OLD = info('v1.0.0', 'a');
const NEW = info('v2.0.0', 'b');

function writePayload(dir, build, cfg) {
  fs.mkdirSync(path.join(dir, 'server'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'runtime'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'runtime', 'node.exe'), 'stand-in: tests run process.execPath');
  fs.writeFileSync(path.join(dir, 'server', 'server.mjs'), STUB);
  fs.writeFileSync(path.join(dir, 'server', 'updater-tool.mjs'), toolBundle);
  fs.writeFileSync(path.join(dir, 'build-info.json'), JSON.stringify(build));
  fs.writeFileSync(path.join(dir, 'stub.json'), JSON.stringify(cfg));
}

/** An install on payload OLD with NEW staged by slice D, a profile pointer and a one-row DB. */
function install(t, newCfg = {}) {
  const top = fs.mkdtempSync(path.join(base, 'case-'));
  const root = path.join(top, 'HOT-Step'), data = path.join(top, 'HOT-Step-Data'), profile = path.join(top, 'profile');
  const oldId = payloadIdFor(OLD), newId = payloadIdFor(NEW);
  writePayload(path.join(root, 'app', oldId), OLD, { version: 'old' });
  writePayload(path.join(root, 'updater', 'prepare-abc123', 'payload', 'app', newId), NEW, { version: 'new', ...newCfg });
  fs.writeFileSync(path.join(root, 'updater', 'prepare-abc123', 'ready.json'), '{}');
  fs.writeFileSync(path.join(root, 'current.json'), JSON.stringify({ format: 1, payload: oldId }));
  fs.mkdirSync(path.join(data, 'library'), { recursive: true });
  fs.mkdirSync(profile, { recursive: true });
  fs.writeFileSync(path.join(profile, 'data-location.json'), JSON.stringify({ version: 1, root: data, pending: null }));
  const db = path.join(data, 'library', 'hotstep.db');
  const logs = [];
  const f = {
    root, data, profile, db, oldId, newId, logs,
    env: { ...process.env, HOT_STEP_PROFILE_DIR: profile },
    start() {
      return launch({ root, env: f.env, nodeFor: () => process.execPath, openBrowser: () => {}, stdio: ['ignore', 'ignore', 'ignore'], log: m => logs.push(m) });
    },
    /** The port the next server of `id` listens on (waits for it to start). */
    async port(id, ms = 15_000) {
      const file = path.join(root, 'app', id, 'port');
      for (const end = Date.now() + ms; Date.now() < end; await sleep(50)) {
        if (fs.existsSync(file)) { const p = Number(fs.readFileSync(file, 'utf8')); fs.rmSync(file); return p; }
      }
      throw new Error(`${id} never listened; launcher log:\n${logs.join('\n')}`);
    },
    post: (port, url, body = '') => fetch(`http://127.0.0.1:${port}${url}`, { method: 'POST', body }).then(r => r.json()),
    apply: port => f.post(port, '/test/apply', JSON.stringify({ preparedId: 'prepare-abc123', newPayload: newId, snapshot: { dbs: [db], files: [path.join(data, 'settings', '.env')] } })),
    pointer: () => JSON.parse(fs.readFileSync(path.join(root, 'current.json'), 'utf8')).payload,
    result: () => { try { return JSON.parse(fs.readFileSync(path.join(root, 'updater', 'last-result.json'), 'utf8')); } catch { return null; } },
    async waitResult(ms = 20_000) {
      for (const end = Date.now() + ms; Date.now() < end; await sleep(50)) { const r = f.result(); if (r) return r; }
      throw new Error(`no result; launcher log:\n${logs.join('\n')}`);
    },
    runs: () => { const d = new Database(db, { readonly: true }); try { return d.prepare('SELECT who FROM runs').all().map(r => r.who); } finally { d.close(); } },
    leases: () => [path.join(root, 'updater', 'maintenance-lease.json'), path.join(data, '.hot-step-maintenance.json')].filter(x => fs.existsSync(x)),
    tx: () => fs.existsSync(path.join(root, 'updater', 'transaction.json')),
    retained: () => { try { return JSON.parse(fs.readFileSync(path.join(root, 'updater', 'retained.json'), 'utf8')); } catch { return null; } },
    /** Seed the DB the way the old version left it before the update. */
    seed() { const d = new Database(db); d.exec('CREATE TABLE IF NOT EXISTS runs (who TEXT)'); d.prepare('INSERT INTO runs VALUES (?)').run('before'); d.close(); },
    /** A snapshot taken by the real tool, as install() would, plus its pending record. */
    snapshotTx(state = 'pending', leaseId = 'lease-1') {
      const id = crypto.randomUUID(), dir = path.join(root, 'updater', 'snapshots', id);
      const r = spawnSync(process.execPath, [path.join(root, 'app', oldId, 'server', 'updater-tool.mjs'), 'snapshot', dir, JSON.stringify({ dbs: [db], files: [] })], { encoding: 'utf8' });
      assert.equal(r.status, 0, r.stderr);
      const tx = { format: 1, id, state, leaseId, oldPayload: oldId, newPayload: newId, dataRoot: data, snapshot: { dir, manifestSha256: JSON.parse(r.stdout).manifestSha256 }, createdAt: new Date().toISOString(), committedAt: state === 'committed' ? new Date().toISOString() : null };
      fs.writeFileSync(path.join(root, 'updater', 'transaction.json'), JSON.stringify(tx));
      const lease = { format: 1, id: leaseId, tokenSha256: sha('lost with the crashed launcher'), installRoot: fs.realpathSync.native(root), dataRoot: fs.realpathSync.native(data), pid: 1, createdAt: tx.createdAt };
      fs.writeFileSync(path.join(root, 'updater', 'maintenance-lease.json'), JSON.stringify(lease));
      fs.writeFileSync(path.join(data, '.hot-step-maintenance.json'), JSON.stringify(lease));
      return tx;
    },
    /** The candidate wrote to the DB before the crash. */
    damage() { const d = new Database(db); d.prepare('INSERT INTO runs VALUES (?)').run('new'); d.close(); },
    movePayload() { fs.renameSync(path.join(root, 'updater', 'prepare-abc123', 'payload', 'app', newId), path.join(root, 'app', newId)); },
  };
  return f;
}

async function quit(f, id, run) {
  await f.post(await f.port(id), '/test/quit');
  assert.equal(await run, 0);
}

test('clean install: snapshot, pointer, healthy candidate, committed; old payload kept until a later start', async t => {
  const f = install(t);
  f.seed();
  const run = f.start();
  const ack = await f.apply(await f.port(f.oldId));
  assert.equal(ack.ok, true, JSON.stringify(ack));
  const port = await f.port(f.newId);
  const r = await f.waitResult();
  assert.equal(r.outcome, 'installed', r.message);
  assert.equal(f.pointer(), f.newId);
  assert.equal(f.tx(), false);
  assert.deepEqual(f.leases(), []);
  assert.deepEqual(f.runs(), ['before', 'old', 'new']);
  assert.ok(fs.existsSync(path.join(f.root, 'app', f.oldId)), 'old payload kept');
  assert.deepEqual(f.retained().payloads, [f.oldId]);
  assert.equal(f.retained().snapshots.length, 1);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);

  // A later good start removes what the update retained.
  const run2 = f.start();
  const port2 = await f.port(f.newId);
  for (const end = Date.now() + 10_000; Date.now() < end && f.retained(); await sleep(50));
  assert.equal(f.retained(), null, f.logs.join('\n'));
  assert.equal(fs.existsSync(path.join(f.root, 'app', f.oldId)), false);
  assert.deepEqual(fs.readdirSync(path.join(f.root, 'updater', 'snapshots')), []);
  assert.equal(fs.existsSync(path.join(f.root, 'updater', 'prepare-abc123')), false);
  await f.post(port2, '/test/quit');
  assert.equal(await run2, 0);
});

test('failed health: candidate stopped, pointer back, DB restored to the snapshot, old version running', async t => {
  const f = install(t, { badDb: true });
  f.seed();
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  await f.port(f.newId); // the candidate did start, and wrote to the DB
  const r = await f.waitResult();
  assert.equal(r.outcome, 'rolled-back');
  assert.match(r.message, /health check.*migration 42 failed/);
  const port = await f.port(f.oldId); // old version back
  assert.equal(f.pointer(), f.oldId);
  // Snapshot contents (before, old) plus the restarted old version's own row; the candidate's write is gone.
  assert.deepEqual(f.runs(), ['before', 'old', 'old']);
  assert.equal(f.tx(), false);
  assert.deepEqual(f.leases(), []);
  const snap = f.retained().snapshots[0];
  assert.ok(fs.readdirSync(path.join(snap, 'candidate')).some(n => n.endsWith('hotstep.db')), 'candidate DB kept for diagnosis');
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('engine cannot start (ace-server missing): rolled back at once, with the engine status', async t => {
  const f = install(t, { engine: { ready: false, failed: true, noModels: false, status: 'ace-server not found at X:\nowhere.exe' } });
  f.seed();
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  await f.port(f.newId);
  const started = Date.now();
  const r = await f.waitResult();
  assert.equal(r.outcome, 'rolled-back');
  assert.match(r.message, /health check: its engine cannot start \(ace-server not found at/);
  assert.ok(Date.now() - started < timing.healthMs / 2, 'a failed engine ends the wait without the full bound');
  const port = await f.port(f.oldId);
  assert.equal(f.pointer(), f.oldId);
  assert.deepEqual(f.runs(), ['before', 'old', 'old']);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('engine that never answers /health: rolled back after the bound, with the engine status', async t => {
  const f = install(t, { engine: { ready: false, failed: false, noModels: false, status: 'Starting engine...' } });
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  const r = await f.waitResult(timing.healthMs + 15_000);
  assert.equal(r.outcome, 'rolled-back');
  assert.match(r.message, /no healthy answer within \d+ s \(its engine is not ready \(Starting engine\.\.\.\)\)/);
  await quit(f, f.oldId, run);
  assert.equal(f.pointer(), f.oldId);
});

test('engine slower than the server: the launcher waits for its answer and commits', async t => {
  const f = install(t, { engineReadyAfterMs: 1_500 });
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  const port = await f.port(f.newId);
  const r = await f.waitResult();
  assert.equal(r.outcome, 'installed', r.message);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('no models yet: the engine ran and reported an empty folder, so the update is installed', async t => {
  const f = install(t, { engine: { ready: false, failed: false, noModels: true, status: 'No models installed — download one in the Model Manager' } });
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  const port = await f.port(f.newId);
  const r = await f.waitResult();
  assert.equal(r.outcome, 'installed', r.message);
  assert.equal(f.pointer(), f.newId);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('a version that cannot prove a clean stop is restarted, and the candidate never runs', async t => {
  const f = install(t);
  fs.writeFileSync(path.join(f.root, 'app', f.oldId, 'stub.json'), JSON.stringify({ version: 'old', unclean: true }));
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  const r = await f.waitResult();
  assert.equal(r.outcome, 'failed');
  assert.match(r.message, /did not stop cleanly.*engine/);
  await quit(f, f.oldId, run);
  assert.equal(f.pointer(), f.oldId);
  assert.equal(fs.existsSync(path.join(f.root, 'app', f.newId)), false);
  assert.deepEqual(f.runs(), ['old', 'old']);
  assert.deepEqual(f.leases(), []);
});

for (const [name, after] of [['before', false], ['after', true]]) {
  test(`crash ${name} the pointer write: the next launch restores the snapshot and starts the old version`, async t => {
    const f = install(t);
    f.seed();
    f.snapshotTx();
    if (after) { f.movePayload(); fs.writeFileSync(path.join(f.root, 'current.json'), JSON.stringify({ format: 1, payload: f.newId })); }
    f.damage();
    const run = f.start();
    const port = await f.port(f.oldId);
    assert.equal(f.result().outcome, 'rolled-back');
    assert.equal(f.pointer(), f.oldId);
    assert.deepEqual(f.runs(), ['before', 'old']);
    assert.equal(f.tx(), false);
    assert.deepEqual(f.leases(), []);
    if (after) assert.deepEqual(f.retained().payloads, [f.newId]);
    await f.post(port, '/test/quit');
    assert.equal(await run, 0);
  });
}

test('crash in the middle of a restore: the next launch finishes it from the snapshot', async t => {
  const f = install(t);
  f.seed();
  f.snapshotTx();
  f.movePayload();
  fs.writeFileSync(path.join(f.root, 'current.json'), JSON.stringify({ format: 1, payload: f.oldId }));
  // Half done: a staged copy beside the target, the candidate's WAL, a torn DB.
  fs.writeFileSync(f.db + '.restore-tmp', 'partial');
  fs.writeFileSync(f.db + '-wal', 'candidate wal');
  fs.writeFileSync(f.db, 'torn');
  const run = f.start();
  const port = await f.port(f.oldId);
  assert.equal(f.result().outcome, 'rolled-back');
  assert.deepEqual(f.runs(), ['before', 'old']);
  assert.equal(fs.existsSync(f.db + '.restore-tmp'), false);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('crash between commit and release: the new version is started under a fresh lease and never restored', async t => {
  const f = install(t);
  f.seed();
  f.snapshotTx('committed');
  f.movePayload();
  fs.writeFileSync(path.join(f.root, 'current.json'), JSON.stringify({ format: 1, payload: f.newId }));
  f.damage(); // the committed version's data
  const run = f.start();
  const port = await f.port(f.newId);
  const r = await f.waitResult();
  assert.equal(r.outcome, 'installed');
  assert.equal(f.pointer(), f.newId);
  assert.deepEqual(f.runs(), ['before', 'new', 'new']);
  assert.equal(f.tx(), false);
  assert.deepEqual(f.leases(), []);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('crash between commit and release, engine now down: the committed version still starts, not restored', async t => {
  const f = install(t, { engine: { ready: false, failed: true, noModels: false, status: 'ace-server not found at X:\nowhere.exe' } });
  f.seed();
  f.snapshotTx('committed');
  f.movePayload();
  fs.writeFileSync(path.join(f.root, 'current.json'), JSON.stringify({ format: 1, payload: f.newId }));
  const run = f.start();
  const port = await f.port(f.newId);
  const r = await f.waitResult();
  assert.equal(r.outcome, 'installed', r.message);
  assert.equal(f.pointer(), f.newId);
  assert.equal(f.tx(), false);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('a failed restore leaves the lease and record, starts nothing and says recovery is required', async t => {
  const f = install(t);
  f.seed();
  const tx = f.snapshotTx();
  fs.writeFileSync(path.join(tx.snapshot.dir, fs.readdirSync(tx.snapshot.dir).find(n => n.endsWith('hotstep.db'))), 'damaged snapshot');
  let err;
  await assert.rejects(f.start(), e => (err = e) instanceof RecoveryRequired);
  assert.equal(f.result().outcome, 'recovery-required');
  assert.equal(f.tx(), true);
  assert.equal(f.leases().length, 2);
  assert.equal(fs.existsSync(path.join(f.root, 'app', f.oldId, 'port')), false, 'no server started');
  // The window gives the first step and the full URL of the user page, which ships in the repo.
  const text = failureText(err);
  assert.match(text, /^\[HOT-Step\] Recovery required: An update was interrupted\. Restoring the data snapshot failed: snapshot file .*hotstep\.db is damaged/);
  assert.match(text, /Do not delete or edit anything in the updater folder/);
  assert.match(text, /start HOT-Step again/);
  assert.ok(text.endsWith(`[HOT-Step] ${RECOVERY_DOC}`));
  assert.equal(RECOVERY_DOC, 'https://github.com/scragnog/HOT-Step-CPP/blob/master/docs/user/update-recovery.md');
  assert.ok(fs.existsSync(path.join(here, '..', '..', 'docs', 'user', 'update-recovery.md')));
  // The first step is safe: starting again retries from the snapshot and keeps the same state.
  await assert.rejects(f.start(), RecoveryRequired);
  assert.equal(f.tx(), true);
  assert.equal(f.leases().length, 2);
  assert.ok(fs.existsSync(tx.snapshot.dir));
});

test('a second launcher on the same root is refused', async t => {
  const f = install(t);
  const run = f.start();
  const port = await f.port(f.oldId);
  await assert.rejects(f.start(), AlreadyRunning);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('Restart in the app relaunches through the launcher, with no restart marker left behind', async t => {
  const f = install(t);
  const run = f.start();
  await f.post(await f.port(f.oldId), '/test/quit', 'restart');
  await quit(f, f.oldId, run);
  assert.deepEqual(f.runs(), ['old', 'old']);
  assert.equal(fs.existsSync(path.join(f.profile, 'restart-requested')), false);
});

test('a lock on the old payload delays only its cleanup, never the activation', { skip: process.platform !== 'win32' }, async t => {
  const f = install(t);
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  const port = await f.port(f.newId);
  assert.equal((await f.waitResult()).outcome, 'installed');
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
  // Hold a file in the old payload open with no sharing, the way antivirus does.
  const locked = path.join(f.root, 'app', f.oldId, 'build-info.json');
  const holder = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
    `$f = [IO.File]::Open($env:LOCKED, 'Open', 'Read', 'None'); [Console]::Out.WriteLine('locked'); Start-Sleep -Seconds 20; $f.Close()`],
  { env: { ...process.env, LOCKED: locked }, stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true });
  t.after(() => holder.kill());
  await new Promise(r => holder.stdout.once('data', r));
  const run2 = f.start();
  const port2 = await f.port(f.newId);
  for (const end = Date.now() + 10_000; Date.now() < end && !f.logs.some(l => /in use/.test(l)); await sleep(50));
  assert.ok(f.logs.some(l => /in use/.test(l)), f.logs.join('\n'));
  assert.deepEqual(f.retained().payloads, [f.oldId], 'kept for a later start');
  assert.equal(f.pointer(), f.newId);
  await f.post(port2, '/test/quit');
  assert.equal(await run2, 0);
});

for (const [name, write] of [
  ['pointer', f => fs.writeFileSync(path.join(f.root, 'current.json'), '{"format":1,"payload":"../escape"}')],
  ['pointer to a missing payload', f => fs.writeFileSync(path.join(f.root, 'current.json'), '{"format":1,"payload":"v9.9.9-none"}')],
  ['transaction record', f => fs.writeFileSync(path.join(f.root, 'updater', 'transaction.json'), '{"format":1,"state":"pending"}')],
  ['transaction record (null)', f => fs.writeFileSync(path.join(f.root, 'updater', 'transaction.json'), 'null')],
  ['lease', f => fs.writeFileSync(path.join(f.root, 'updater', 'maintenance-lease.json'), '[]')],
]) test(`a malformed ${name} fails closed: nothing starts and the DB is untouched`, async t => {
  const f = install(t);
  f.seed();
  write(f);
  await assert.rejects(f.start(), RecoveryRequired);
  assert.equal(fs.existsSync(path.join(f.root, 'app', f.oldId, 'port')), false);
  assert.deepEqual(f.runs(), ['before']);
});

test('a stale preparation lock and a lease with no transaction are cleared before the server starts', async t => {
  const f = install(t);
  fs.writeFileSync(path.join(f.root, 'updater', 'prepare.lock'), JSON.stringify({ pid: process.pid, candidateId: 'x' }));
  const lease = { format: 1, id: 'orphan', tokenSha256: sha('x'), installRoot: fs.realpathSync.native(f.root), dataRoot: fs.realpathSync.native(f.data), pid: 1, createdAt: '' };
  fs.writeFileSync(path.join(f.root, 'updater', 'maintenance-lease.json'), JSON.stringify(lease));
  fs.writeFileSync(path.join(f.data, '.hot-step-maintenance.json'), JSON.stringify(lease));
  const run = f.start();
  const port = await f.port(f.oldId);
  assert.equal(fs.existsSync(path.join(f.root, 'updater', 'prepare.lock')), false);
  assert.deepEqual(f.leases(), []);
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('a first start records the data folder beside the install, never inside app/', async t => {
  const f = install(t);
  const ptr = path.join(f.profile, 'data-location.json');
  fs.rmSync(ptr);
  fs.rmSync(f.data, { recursive: true });
  fs.mkdirSync(f.data);
  fs.writeFileSync(path.join(f.data, 'someone-elses.txt'), '');
  await assert.rejects(f.start(), /not a HOT-Step data folder/);
  assert.equal(fs.existsSync(ptr), false);
  fs.rmSync(f.data, { recursive: true });
  const run = f.start();
  const port = await f.port(f.oldId);
  assert.equal(JSON.parse(fs.readFileSync(ptr, 'utf8')).root, path.join(path.dirname(f.root), 'HOT-Step-Data'));
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
});

test('parseDotenv reads settings/.env exactly as the server\'s dotenv does', () => {
  const dotenv = createRequire(import.meta.url)(path.join(repo, 'server', 'node_modules', 'dotenv'));
  const src = [
    'A=C:\\x\\models', 'B="C:\\y z\\models"', "C='rel/dir'", 'D=`C:\\back tick`', 'E=C:\\e # comment', 'export F=f',
    'G = spaced ', 'H:colon', '# comment', 'I=', 'J="multi\\nline"', 'K=a#b', 'L="quoted # not a comment"', 'M="unclosed', 'N.dot=n',
  ].join('\r\n');
  assert.deepEqual(parseDotenv(src), dotenv.parse(src));
});

function settings(f, text) {
  fs.mkdirSync(path.join(f.data, 'settings'), { recursive: true });
  fs.writeFileSync(path.join(f.data, 'settings', '.env'), text);
}
for (const [name, prepare, snapshots] of [
  ['the next start never answers health', f => fs.writeFileSync(path.join(f.root, 'app', f.newId, 'stub.json'), JSON.stringify({ version: 'new', health503: true })), 1],
  // Forward slashes: inside double quotes dotenv turns \r and \n into line breaks, as the server reads it.
  ['a setting points into the old payload', f => settings(f, `ACESTEPCPP_MODELS="${path.join(f.root, 'app', f.oldId, 'models').replaceAll('\\', '/')}"\n`), 0],
  // config.ts writes backtick quoting and resolves relative values against the data folder.
  ['a backtick-quoted setting points into it', f => settings(f, `ACESTEPCPP_MODELS=\`${path.join(f.root, 'app', f.oldId, 'models')}\`\n`), 0],
  ['a setting relative to the data folder points into it', f => settings(f, `ACESTEPCPP_MODELS=${path.relative(f.data, path.join(f.root, 'app', f.oldId, 'models'))}\n`), 0],
  ['an unquoted setting with a comment points into it', f => settings(f, `ACESTEPCPP_MODELS=${path.join(f.root, 'app', f.oldId, 'models')} # moved\n`), 0],
  ['the settings file cannot be read', f => fs.mkdirSync(path.join(f.data, 'settings', '.env'), { recursive: true }), 0],
]) test(`retained folders are kept when ${name}`, async t => {
  const f = install(t);
  const run = f.start();
  await f.apply(await f.port(f.oldId));
  const port = await f.port(f.newId);
  assert.equal((await f.waitResult()).outcome, 'installed');
  await f.post(port, '/test/quit');
  assert.equal(await run, 0);
  prepare(f);
  const run2 = f.start();
  const port2 = await f.port(f.newId);
  await sleep(timing.healthMs + 500); // past the health deadline and any cleanup
  assert.ok(fs.existsSync(path.join(f.root, 'app', f.oldId)), f.logs.join('\n'));
  assert.deepEqual(f.retained().payloads, [f.oldId]);
  assert.equal(f.retained().snapshots.length, snapshots, 'a setting keeps the payload, not the snapshot');
  await f.post(port2, '/test/quit');
  assert.equal(await run2, 0);
});

// ── Packaging ────────────────────────────────────────────────────────

function flatPackage(dir, protocol) {
  const files = {
    'VERSION': '2.0.0', 'build-info.json': JSON.stringify({ ...NEW, updaterProtocol: protocol }), 'HOT-Step.bat': 'flat', 'README.txt': 'x',
    'runtime/node.exe': 'node', 'server/server.mjs': 's', 'server/updater-tool.mjs': 't', 'engine/ace-server.exe': 'e', 'ui/dist/index.html': 'ui',
  };
  for (const [rel, text] of Object.entries(files)) { fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true }); fs.writeFileSync(path.join(dir, rel), text); }
  return files;
}
const listTree = dir => fs.readdirSync(dir, { recursive: true }).map(p => p.replaceAll('\\', '/')).sort();

test('protocol 0 packages stay flat and unchanged', () => {
  const dir = fs.mkdtempSync(path.join(base, 'pkg0-'));
  flatPackage(dir, 0);
  const before = listTree(dir);
  assert.deepEqual(packageLayout(dir), { layout: 'flat', id: null });
  assert.deepEqual(listTree(dir), before);
  assert.deepEqual(checkPackageBuildInfo(dir, { os: 'windows', variant: 'cpu' }), []);
});

test('protocol 1 packages get the launcher root, one payload and a manifest of every shipped file', () => {
  const dir = fs.mkdtempSync(path.join(base, 'pkg1-'));
  const files = flatPackage(dir, 1);
  const id = payloadIdFor(NEW);
  assert.deepEqual(packageLayout(dir), { layout: 'versioned', id });
  assert.deepEqual(fs.readdirSync(dir).sort(), ['HOT-Step.bat', 'README.txt', 'app', 'current.json', 'launcher.mjs', 'runtime']);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dir, 'current.json'), 'utf8')), { format: 1, payload: id });
  assert.match(fs.readFileSync(path.join(dir, 'HOT-Step.bat'), 'utf8'), /runtime\\node\.exe" "%~dp0launcher\.mjs"/);
  const payload = path.join(dir, 'app', id);
  assert.equal(fs.existsSync(path.join(payload, 'HOT-Step.bat')), false, 'only the launcher starts a payload');
  const manifest = JSON.parse(fs.readFileSync(path.join(payload, MANIFEST_FILE), 'utf8'));
  delete files['HOT-Step.bat'];
  assert.deepEqual(manifest.files, Object.fromEntries(Object.entries(files).map(([k, v]) => [k, sha(v)]).sort()));
  assert.deepEqual(checkPackageBuildInfo(dir, { os: 'windows', variant: 'cpu' }), []);
  fs.writeFileSync(path.join(dir, 'stray.txt'), '');
  assert.match(checkPackageBuildInfo(dir, { os: 'windows', variant: 'cpu' }).join('\n'), /unexpected stray\.txt/);
  assert.throws(() => packageLayout(dir), /already laid out/);
});
