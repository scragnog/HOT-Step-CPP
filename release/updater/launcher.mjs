// launcher.mjs — the stable launcher of a Windows portable install (updater protocol 1).
//
// It sits at the install root beside HOT-Step.bat, runtime/ (the Node that runs
// it), current.json, updater/ and the app/<id>/ payloads, and an update never
// replaces it. It is the only thing that starts a server on this root:
//   1. a second launcher on the same root is refused (a named pipe the OS
//      frees when this process dies, so there is no stale lock or PID);
//   2. an interrupted update is settled before anything opens SQLite;
//   3. the payload current.json names is started and supervised (restart
//      marker, Quit, crash).
// When that server hands over a prepared update (IPC 'apply'), the launcher
// acknowledges, waits for that exact process to report a clean stop and exit,
// snapshots the databases with the old payload's own tool, writes the pending
// record, switches the pointer and starts the candidate under the maintenance
// lease. Its attestation passes within 300 s: committed, gate released, UI
// opened. The engine must have answered: /health, or an empty models folder
// reported by the binary itself. It fails: candidate stopped, pointer back,
// snapshot restored, old version started. Plain Node with no dependencies; see docs/dev/releasing.md.

import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const LAUNCHER_PROTOCOL = 1;
/** Tests shorten these. healthMs covers a real engine start (first load of a new
 *  binary past the antivirus, CUDA kernel JIT); an engine that reports it cannot
 *  start ends the wait at once, so only a hung one costs the whole bound. */
export const timing = { retryDelaysMs: [5_000, 10_000], healthMs: 300_000, pollMs: 1_000, ackMs: 15_000, stopMs: 90_000, toolMs: 900_000 };

const PAYLOAD_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const PREPARED_ID = /^prepare-[A-Za-z0-9]{6}$/;
const TOKEN_HEADER = 'x-hotstep-maintenance-token';
const DATA_LEASE = '.hot-step-maintenance.json';
/** Locked (antivirus, indexer) or full targets get bounded retries. */
const RETRYABLE = new Set(['EPERM', 'EBUSY', 'EACCES', 'ENOSPC', 'ENOTEMPTY']);

/** Nothing may start until a person fixes what the message names. */
export class RecoveryRequired extends Error {}
export class AlreadyRunning extends Error {}

export const validPayloadId = id => typeof id === 'string' && PAYLOAD_ID.test(id) && !id.includes('..');
/** The app/<id> name for a build: tag (or "master"), short commit, exact variant. */
export const payloadIdFor = info => `${info.tag ?? 'master'}-${info.sha.slice(0, 12)}-${info.variant}`;
const sha256 = s => crypto.createHash('sha256').update(s).digest('hex');
const sleep = ms => new Promise(r => setTimeout(r, ms).unref());
const inside = (parent, child) => {
  const r = path.relative(path.resolve(parent), path.resolve(child));
  return !!r && r !== '..' && !r.startsWith(`..${path.sep}`) && !path.isAbsolute(r);
};
const real = p => { try { return fs.realpathSync.native(path.resolve(p)); } catch { return path.resolve(p); } };
const sameDir = (a, b) => typeof a === 'string' && typeof b === 'string'
  && (process.platform === 'win32' ? real(a).toLowerCase() === real(b).toLowerCase() : real(a) === real(b));
const isDir = p => { try { return fs.statSync(p).isDirectory(); } catch { return false; } };

export function layout(root) {
  const updater = path.join(root, 'updater');
  return {
    root, app: path.join(root, 'app'), pointer: path.join(root, 'current.json'), updater,
    tx: path.join(updater, 'transaction.json'), lease: path.join(updater, 'maintenance-lease.json'),
    result: path.join(updater, 'last-result.json'), retained: path.join(updater, 'retained.json'),
    snapshots: path.join(updater, 'snapshots'), prepareLock: path.join(updater, 'prepare.lock'),
  };
}

/** Three attempts, waiting 5 s then 10 s, for locked or full targets only. */
export async function retry(fn) {
  for (let i = 0; ; i++) {
    try { return await fn(); } catch (e) {
      if (i >= timing.retryDelaysMs.length || !RETRYABLE.has(e?.code)) throw e;
      await sleep(timing.retryDelaysMs[i]);
    }
  }
}

/** undefined only when the file is missing; anything unreadable throws. */
function readJson(file) {
  let raw;
  try { raw = fs.readFileSync(file, 'utf8'); } catch (e) { if (e.code === 'ENOENT') return undefined; throw e; }
  try { return JSON.parse(raw); } catch { throw new Error(`${file} is not valid JSON`); }
}

/** Written beside the target and renamed over it, so a reader sees old or new, never half. */
function writeJsonAtomic(file, data, compact = false) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, (compact ? JSON.stringify(data) : JSON.stringify(data, null, 2)) + '\n');
  fs.renameSync(tmp, file);
}

// ── Pointer ──────────────────────────────────────────────────────────

/** The payload current.json selects: one validated directory name under app/. */
export function readPointer(root) {
  const L = layout(root);
  let p;
  try { p = readJson(L.pointer); } catch (e) { throw new RecoveryRequired(e.message); }
  if (p === undefined) throw new RecoveryRequired(`${L.pointer} is missing`);
  if (!p || typeof p !== 'object' || p.format !== 1 || Object.keys(p).length !== 2 || !validPayloadId(p.payload)) {
    throw new RecoveryRequired(`${L.pointer} is not a valid version pointer`);
  }
  if (!isDir(path.join(L.app, p.payload))) throw new RecoveryRequired(`${L.pointer} names ${p.payload}, which is not in ${L.app}`);
  return p.payload;
}

export const writePointer = (root, id) => {
  if (!validPayloadId(id)) throw new Error(`refusing to point at ${id}`);
  return retry(() => writeJsonAtomic(layout(root).pointer, { format: 1, payload: id }, true));
};

// ── Transaction record (format shared with server/src/services/updates/maintenance.ts) ──

export function readTx(L) {
  let t;
  try { t = readJson(L.tx); } catch (e) { throw new RecoveryRequired(e.message); }
  if (t === undefined) return null;
  const ok = t && t.format === 1 && typeof t.id === 'string' && (t.state === 'pending' || t.state === 'committed')
    && typeof t.leaseId === 'string' && validPayloadId(t.oldPayload) && validPayloadId(t.newPayload)
    && typeof t.dataRoot === 'string' && path.isAbsolute(t.dataRoot)
    && typeof t.snapshot?.dir === 'string' && inside(L.snapshots, t.snapshot.dir)
    && /^[0-9a-f]{64}$/.test(t.snapshot?.manifestSha256 ?? '');
  if (!ok) throw new RecoveryRequired(`${L.tx} is not a valid update record`);
  return t;
}
const writeTx = (L, t) => writeJsonAtomic(L.tx, t);
const clearTx = L => fs.rmSync(L.tx, { force: true });

// ── Lease (format shared with maintenance.ts) ────────────────────────

function newLease(root, dataRoot) {
  const token = crypto.randomBytes(32).toString('hex');
  return {
    token,
    lease: { format: 1, id: crypto.randomUUID(), tokenSha256: sha256(token), installRoot: real(root), dataRoot: real(dataRoot), pid: process.pid, createdAt: new Date().toISOString() },
  };
}

function writeLease(L, lease) {
  writeJsonAtomic(L.lease, lease);
  writeJsonAtomic(path.join(lease.dataRoot, DATA_LEASE), lease);
}

/** Remove this root's lease: the data folder's copy first, ours last. A lease
 *  another install put in the data folder is left alone. */
function dropLeases(L, dataRoot) {
  if (dataRoot) {
    const f = path.join(dataRoot, DATA_LEASE);
    let l;
    try { l = readJson(f); } catch (e) { throw new RecoveryRequired(`${f}: ${e.message}`); }
    if (l !== undefined && (!l || sameDir(l.installRoot, L.root))) fs.rmSync(f, { force: true });
  }
  fs.rmSync(L.lease, { force: true });
}

/** A lease with no transaction: a server opened one and died before handing
 *  over. With this launcher's lock held no server runs here, so it is stale. */
function dropStaleLease(L) {
  let l;
  try { l = readJson(L.lease); } catch (e) { throw new RecoveryRequired(`${L.lease}: ${e.message}`); }
  if (l === undefined) return;
  if (!l || typeof l.dataRoot !== 'string') throw new RecoveryRequired(`${L.lease} is not a maintenance lease`);
  dropLeases(L, l.dataRoot);
}

// ── Profile and data folder ──────────────────────────────────────────

/** Same rules as server/src/dataroot/paths.ts profileDir and release/profile-path.mjs. */
export function profileDir(env = process.env, platform = process.platform, home = os.homedir()) {
  if (env.HOT_STEP_PROFILE_DIR) return path.resolve(env.HOT_STEP_PROFILE_DIR);
  if (platform === 'win32') return path.join(env.LOCALAPPDATA || path.join(home, 'AppData', 'Local'), 'HOT-Step');
  if (platform === 'darwin') return path.join(home, 'Library', 'Application Support', 'HOT-Step');
  return path.join(env.XDG_CONFIG_HOME || path.join(home, '.config'), 'hot-step');
}

function usableDefault(dir) {
  if (!fs.existsSync(dir)) { fs.mkdirSync(dir, { recursive: true }); return dir; }
  if (!fs.readdirSync(dir).length || fs.existsSync(path.join(dir, 'hot-step-data.json')) || fs.existsSync(path.join(dir, 'library', 'hotstep.db'))) return dir;
  throw new RecoveryRequired(`${dir} exists but is not a HOT-Step data folder. Rename it, or set HOT_STEP_DATA_ROOT, then start HOT-Step again.`);
}

/** Env that makes every payload resolve the same profile and data folder.
 *  The profile is always pinned. The data folder is pinned only where the
 *  server would otherwise derive it from the payload folder (a saved folder
 *  that is missing); a first start records <parent of install>/HOT-Step-Data
 *  in the profile pointer, so Settings can still move it later. */
export function dataEnv(root, env, log) {
  const prof = profileDir(env);
  const out = { HOT_STEP_PROFILE_DIR: prof };
  if (env.HOT_STEP_DATA_ROOT) return out;
  const ptrFile = path.join(prof, 'data-location.json');
  let ptr;
  try { ptr = readJson(ptrFile); } catch { return out; } // the server explains a broken pointer
  const fallback = path.join(path.dirname(path.resolve(root)), 'HOT-Step-Data');
  if (ptr === undefined) {
    writeJsonAtomic(ptrFile, { version: 1, root: usableDefault(fallback), pending: null });
    return out;
  }
  if (ptr?.pending || (typeof ptr?.root === 'string' && isDir(ptr.root))) return out;
  log(`[Launcher] Saved data folder ${ptr?.root} is missing or cannot be accessed. Using ${fallback} for this run; the saved location is unchanged.`);
  return { ...out, HOT_STEP_DATA_ROOT: usableDefault(fallback) };
}

// ── Result and retained files ────────────────────────────────────────

/** The durable outcome GET /api/updates/status reports. */
function writeResult(ctx, outcome, rec, message) {
  writeJsonAtomic(ctx.L.result, { format: 1, outcome, from: rec.oldPayload ?? null, to: rec.newPayload ?? null, at: new Date().toISOString(), message: message ?? null });
  ctx.log(`[Launcher] Update ${outcome}${message ? `: ${message}` : ''}`);
}

function readRetained(L) {
  try {
    const r = readJson(L.retained);
    if (r?.format === 1 && ['payloads', 'snapshots', 'staging'].every(k => Array.isArray(r[k]))) return r;
  } catch { /* forget it: a lost entry only leaves an old folder behind */ }
  return { format: 1, payloads: [], snapshots: [], staging: [] };
}

/** Record updater-owned folders to delete after a later good start. */
function retain(ctx, add) {
  const r = readRetained(ctx.L);
  for (const k of Object.keys(add)) r[k] = [...new Set([...r[k], ...add[k]])];
  writeJsonAtomic(ctx.L.retained, r);
}

/** dotenv 16 parse(), which the server's config.ts uses for settings/.env:
 *  same line rule, quote stripping (', " and `) and "\n" expansion. */
const DOTENV_LINE = /(?:^|^)\s*(?:export\s+)?([\w.-]+)(?:\s*=\s*?|:\s+?)(\s*'(?:\\'|[^'])*'|\s*"(?:\\"|[^"])*"|\s*`(?:\\`|[^`])*`|[^#\r\n]+)?\s*(?:#.*)?(?:$|$)/mg;
export function parseDotenv(src) {
  const out = {};
  const lines = src.replace(/\r\n?/mg, '\n');
  for (const m of lines.matchAll(DOTENV_LINE)) {
    let value = (m[2] || '').trim();
    const quote = value[0];
    value = value.replace(/^(['"`])([\s\S]*)\1$/mg, '$2');
    if (quote === '"') value = value.replace(/\\n/g, '\n').replace(/\\r/g, '\r');
    out[m[1]] = value;
  }
  return out;
}

/** Every path the server could take from a setting: process env and the data
 *  folder's settings/.env, relative values resolved against the data folder
 *  as config.ts does. null when they cannot be read safely: then no retained
 *  payload is deleted. */
function settingPaths(ctx) {
  let dataRoot = ctx.dataEnv?.HOT_STEP_DATA_ROOT || ctx.env.HOT_STEP_DATA_ROOT;
  if (!dataRoot) {
    try { dataRoot = readJson(path.join(ctx.dataEnv.HOT_STEP_PROFILE_DIR, 'data-location.json'))?.root; } catch { return null; }
  }
  if (typeof dataRoot !== 'string' || !path.isAbsolute(dataRoot)) return null;
  let file = {};
  try { file = parseDotenv(fs.readFileSync(path.join(dataRoot, 'settings', '.env'), 'utf8')); }
  catch (e) { if (e.code !== 'ENOENT') return null; }
  return [...Object.values(ctx.env), ...Object.values(file)]
    .filter(v => typeof v === 'string' && v.trim())
    .map(v => path.resolve(dataRoot, v));
}

/** Delete retained folders. Only recorded names in updater-owned places, never
 *  the current payload or one a setting points into; a locked or referenced
 *  folder stays recorded for a later start. */
async function cleanRetained(ctx) {
  const L = ctx.L;
  if (fs.existsSync(L.tx)) return;
  const r = readRetained(L);
  if (!r.payloads.length && !r.snapshots.length && !r.staging.length) return;
  const current = readPointer(L.root);
  const referenced = settingPaths(ctx);
  const keep = { format: 1, payloads: [], snapshots: [], staging: [] };
  const remove = async (kind, value, target) => {
    try { await retry(() => fs.rmSync(target, { recursive: true, force: true })); }
    catch (e) { keep[kind].push(value); ctx.log(`[Launcher] ${target} is in use (${e.code ?? e.message}); it will be removed after a later start.`); }
  };
  for (const id of r.payloads) {
    if (!validPayloadId(id) || id === current) continue;
    const dir = path.join(L.app, id);
    if (!referenced) { keep.payloads.push(id); ctx.log(`[Launcher] Keeping ${dir}: the settings could not be read to check it is unused.`); continue; }
    const used = referenced.find(p => sameDir(p, dir) || inside(real(dir), real(p)));
    if (used) { keep.payloads.push(id); ctx.log(`[Launcher] Keeping ${dir}: a setting points into it (${used}).`); continue; }
    await remove('payloads', id, dir);
  }
  for (const dir of r.snapshots) if (inside(L.snapshots, dir)) await remove('snapshots', dir, path.resolve(dir));
  for (const id of r.staging) if (PREPARED_ID.test(id)) await remove('staging', id, path.join(L.updater, id));
  if (keep.payloads.length + keep.snapshots.length + keep.staging.length) writeJsonAtomic(L.retained, keep);
  else fs.rmSync(L.retained, { force: true });
}

// ── Processes ────────────────────────────────────────────────────────

/** Start a payload's server. The ChildProcess handle is the only way the
 *  launcher waits for or stops it: never a name, never a bare PID. */
function startServer(ctx, payload, extraEnv = {}) {
  const dir = path.join(ctx.L.app, payload);
  const env = {
    ...ctx.env, ...ctx.dataEnv,
    HOT_STEP_ROOT: dir, HOT_STEP_INSTALL_ROOT: ctx.root, HOT_STEP_UPDATER_DIR: ctx.L.updater,
    HOT_STEP_LAUNCHER_PROTOCOL: String(LAUNCHER_PROTOCOL), ...extraEnv,
  };
  if (!extraEnv.HOT_STEP_MAINTENANCE_TOKEN) delete env.HOT_STEP_MAINTENANCE_TOKEN;
  ctx.log(`[Launcher] Starting ${payload}...`);
  const child = spawn(ctx.nodeFor(payload), [path.join(dir, 'server', 'server.mjs')], { cwd: dir, env, stdio: [...ctx.stdio, 'ipc'] });
  const s = { payload, child, exit: null, queue: [], waiters: new Set() };
  // 'close' comes after the IPC channel drains, so a final message is never lost behind 'exit'.
  s.exited = new Promise(resolve => {
    const done = exit => { if (!s.exit) { s.exit = exit; for (const w of s.waiters) w(null); resolve(exit); } };
    child.once('close', code => done({ code }));
    child.once('error', error => done({ code: -1, error }));
  });
  child.on('message', m => {
    if (!m || typeof m.type !== 'string') return;
    for (const w of s.waiters) if (w(m)) return;
    s.queue.push(m);
  });
  ctx.current = s;
  return s;
}

/** The next message of `type`, or null when the server exits or `ms` passes. */
function nextMessage(s, type, ms = Infinity) {
  const i = s.queue.findIndex(m => m.type === type);
  if (i >= 0) return Promise.resolve(s.queue.splice(i, 1)[0]);
  if (s.exit) return Promise.resolve(null);
  return new Promise(resolve => {
    let timer;
    const w = m => {
      if (m && m.type !== type) return false;
      s.waiters.delete(w); clearTimeout(timer); resolve(m); return true;
    };
    s.waiters.add(w);
    if (Number.isFinite(ms)) timer = setTimeout(() => { s.waiters.delete(w); resolve(null); }, ms).unref();
  });
}

const send = (s, m) => { try { s.child.send(m); } catch { /* exiting */ } };

/** Ask a server to stop (updater shutdown path), then end the exact process if it hangs. */
async function stopServer(s) {
  if (s.exit) return;
  send(s, { type: 'stop' });
  if (await Promise.race([s.exited, sleep(timing.stopMs).then(() => null)])) return;
  s.child.kill();
  await s.exited;
}

/** Where the server listens, from its own 'listening' message. */
async function address(s, ms) {
  if (s.address) return s.address;
  const m = await nextMessage(s, 'listening', ms);
  if (!m || !Number.isInteger(m.port)) return null;
  const host = !m.host || m.host === '0.0.0.0' || m.host === '::' ? '127.0.0.1' : m.host;
  s.address = { host, port: m.port, url: `http://${host.includes(':') ? `[${host}]` : host}:${m.port}` };
  return s.address;
}

async function getJson(url, headers = {}, method = 'GET') {
  const r = await fetch(url, { method, headers, signal: AbortSignal.timeout(5_000) });
  return { ok: r.ok, status: r.status, body: await r.json().catch(() => null) };
}

/** Why the candidate's attestation does not match, or null when it does.
 *  `engine: false` skips the engine (recovery of a committed update). */
export function attestProblem(s, want, { engine = true } = {}) {
  if (!s || typeof s !== 'object') return 'no status';
  if (s.closed !== true || s.leaseId !== want.leaseId) return 'it is not running under this update\'s lease';
  const b = s.build;
  if (!b || b.sha !== want.sha || b.tag !== want.tag || b.variant !== want.variant || b.updaterProtocol !== want.updaterProtocol) return 'it reports a different build';
  if (s.db?.ok !== true) return `its database did not open (${(s.db?.errors ?? []).join('; ') || 'not finished'})`;
  if (s.ui !== true) return 'its UI files are missing';
  // No models yet still counts: the binary ran and said so, and the user must be able to update before downloading one.
  if (engine && s.engine?.ready !== true && s.engine?.noModels !== true) return `its engine is not ready (${s.engine?.status ?? 'unknown'})`;
  return null;
}

/** Wait up to timing.healthMs for /api/maintenance/status to attest exactly `want`. */
async function attest(s, token, want, opts = {}) {
  const deadline = Date.now() + timing.healthMs;
  const a = await Promise.race([address(s, timing.healthMs), s.exited.then(() => null)]);
  if (!a) return s.exit ? `it exited (code ${s.exit.code})` : 'it never started listening';
  let last = 'no answer';
  while (Date.now() < deadline) {
    if (s.exit) return `it exited (code ${s.exit.code})`;
    try {
      const r = await getJson(`${a.url}/api/maintenance/status`, { [TOKEN_HEADER]: token });
      last = r.ok ? attestProblem(r.body, want, opts) : `HTTP ${r.status}`;
      if (!last) return null;
      if (opts.engine !== false && r.body?.engine?.failed === true) return `its engine cannot start (${r.body.engine.status ?? 'unknown'})`;
    } catch (e) { last = e.message; }
    await sleep(timing.pollMs);
  }
  return `no healthy answer within ${Math.round(timing.healthMs / 1000)} s (${last})`;
}

async function releaseGate(s, token) {
  for (let i = 0; i < 3; i++) {
    try { if ((await getJson(`${s.address.url}/api/maintenance/release`, { [TOKEN_HEADER]: token }, 'POST')).ok) return true; } catch { /* retry */ }
    await sleep(timing.pollMs);
  }
  return false;
}

function readBuild(ctx, dir) {
  const info = readJson(path.join(dir, 'build-info.json'));
  if (!info || typeof info.sha !== 'string' || !/^[0-9a-f]{40}$/.test(info.sha)) throw new Error(`${dir} has no valid build-info.json`);
  return { tag: info.tag ?? null, sha: info.sha, variant: info.variant, updaterProtocol: info.updaterProtocol };
}

/** Run the payload's own snapshot/restore tool (its Node and its SQLite build). */
function runTool(ctx, payload, args) {
  const dir = path.join(ctx.L.app, payload);
  return new Promise((resolve, reject) => {
    const p = spawn(ctx.nodeFor(payload), [path.join(dir, 'server', 'updater-tool.mjs'), ...args], { cwd: dir, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    let out = '', err = '';
    p.stdout.on('data', d => { out += d; });
    p.stderr.on('data', d => { err += d; });
    const timer = setTimeout(() => p.kill(), timing.toolMs);
    p.once('error', e => { clearTimeout(timer); reject(e); });
    p.once('close', code => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(err.trim().split('\n').pop() || `the updater tool exited with code ${code}`)); return; }
      try { resolve(JSON.parse(out.trim().split('\n').pop())); } catch { reject(new Error('the updater tool printed no result')); }
    });
  });
}

function defaultOpenBrowser(ctx, payload, url) {
  const dir = path.join(ctx.L.app, payload);
  try {
    spawn(ctx.nodeFor(payload), [path.join(dir, 'open-browser-if-needed.mjs'), `${url}/`], { cwd: dir, detached: true, stdio: 'ignore', windowsHide: true })
      .on('error', () => {}).unref();
  } catch { /* the console already printed the address */ }
}

// ── Update transaction ───────────────────────────────────────────────

/** Validate the running server's handover before acknowledging it. */
function checkApply(ctx, s, m) {
  const L = ctx.L;
  if (!PREPARED_ID.test(m.preparedId ?? '')) throw new Error('unknown prepared update');
  if (!validPayloadId(m.newPayload) || m.oldPayload !== s.payload || readPointer(ctx.root) !== s.payload || m.newPayload === s.payload) throw new Error('the update does not match the running version');
  const staged = path.join(L.updater, m.preparedId, 'payload', 'app', m.newPayload);
  if (!isDir(staged)) throw new Error('the prepared payload is missing');
  const want = readBuild(ctx, staged);
  if (want.updaterProtocol !== LAUNCHER_PROTOCOL) throw new Error(`the update needs launcher protocol ${want.updaterProtocol}; install it manually`);
  if (payloadIdFor(want) !== m.newPayload) throw new Error('the prepared payload is named for another build');
  for (const f of [path.join('server', 'server.mjs'), path.join('server', 'updater-tool.mjs'), path.join('runtime', 'node.exe')]) {
    if (!fs.existsSync(path.join(staged, f))) throw new Error(`the prepared payload has no ${f}`);
  }
  let lease;
  try { lease = readJson(L.lease); } catch { lease = null; }
  if (!lease || lease.id !== m.leaseId || typeof m.token !== 'string' || lease.tokenSha256 !== sha256(m.token)
      || typeof m.dataRoot !== 'string' || !sameDir(lease.dataRoot, m.dataRoot)) throw new Error('the maintenance lease does not match');
  const set = m.snapshot;
  const paths = [...(set?.dbs ?? []), ...(set?.files ?? [])];
  if (!Array.isArray(set?.dbs) || !Array.isArray(set?.files) || !set.dbs.length
      || paths.some(p => typeof p !== 'string' || !path.isAbsolute(p) || inside(ctx.root, p))) throw new Error('the snapshot list is invalid');
  const target = path.join(L.app, m.newPayload);
  if (fs.existsSync(target) && !readRetained(L).payloads.includes(m.newPayload)) throw new Error(`${target} already exists`);
  return { ...m, staged, want, snapshot: { dbs: set.dbs, files: set.files } };
}

/** The old server handed over: install, or restart it if anything is not proven. */
async function handleApply(ctx, s, m) {
  let req;
  try { req = checkApply(ctx, s, m); } catch (e) {
    ctx.log(`[Launcher] Update refused: ${e.message}`);
    send(s, { type: 'apply-ack', ok: false, error: e.message });
    return s;
  }
  send(s, { type: 'apply-ack', ok: true });
  ctx.log(`[Launcher] Installing ${req.newPayload}; waiting for ${s.payload} to stop...`);
  const stopped = await nextMessage(s, 'stopped', timing.stopMs);
  const exit = await Promise.race([s.exited, sleep(timing.stopMs).then(() => null)]);
  if (!stopped?.clean || !exit || exit.code !== 0) {
    if (!exit) { s.child.kill(); await s.exited; }
    dropLeases(ctx.L, req.dataRoot);
    const why = stopped?.problems?.length ? stopped.problems.join('; ') : exit ? `exit code ${exit.code}` : 'it did not exit';
    writeResult(ctx, 'failed', req, `The running version did not stop cleanly (${why}), so the update was not installed.`);
    return startServer(ctx, req.oldPayload);
  }
  return install(ctx, req);
}

async function install(ctx, req) {
  const L = ctx.L;
  await ctx.cleaning;
  const id = crypto.randomUUID();
  const tx = {
    format: 1, id, state: 'pending', leaseId: req.leaseId, oldPayload: req.oldPayload, newPayload: req.newPayload,
    dataRoot: req.dataRoot, snapshot: { dir: path.join(L.snapshots, id), manifestSha256: '' },
    createdAt: new Date().toISOString(), committedAt: null,
  };
  try {
    const target = path.join(L.app, req.newPayload);
    if (fs.existsSync(target)) await retry(() => fs.rmSync(target, { recursive: true, force: true })); // a retained earlier attempt
    tx.snapshot.manifestSha256 = (await runTool(ctx, req.oldPayload, ['snapshot', tx.snapshot.dir, JSON.stringify(req.snapshot)])).manifestSha256;
  } catch (e) {
    dropLeases(L, req.dataRoot);
    writeResult(ctx, 'failed', req, `The update could not start: ${e.message}. Nothing was changed.`);
    return startServer(ctx, req.oldPayload);
  }
  writeTx(L, tx); // from here an interruption rolls back
  try {
    await retry(() => fs.renameSync(req.staged, path.join(L.app, req.newPayload)));
    retain(ctx, { staging: [req.preparedId] });
    await writePointer(ctx.root, req.newPayload);
  } catch (e) {
    return rollback(ctx, tx, `The new version could not be activated: ${e.message}`);
  }
  const c = startServer(ctx, req.newPayload, { HOT_STEP_MAINTENANCE_TOKEN: req.token, HOT_STEP_DATA_ROOT: req.dataRoot });
  const problem = await attest(c, req.token, { ...req.want, leaseId: tx.leaseId });
  if (problem) {
    await stopServer(c);
    return rollback(ctx, tx, `The new version failed its health check: ${problem}`);
  }
  tx.state = 'committed';
  tx.committedAt = new Date().toISOString();
  writeTx(L, tx);
  if (!await releaseGate(c, req.token)) {
    await stopServer(c);
    return finishCommitted(ctx, tx);
  }
  finalize(ctx, tx);
  ctx.openBrowser(ctx, c.payload, c.address.url);
  return c;
}

/** Committed and released: the old payload and the snapshot stay until a later good start. */
function finalize(ctx, tx) {
  retain(ctx, { payloads: [tx.oldPayload], snapshots: [tx.snapshot.dir] });
  clearTx(ctx.L);
  dropLeases(ctx.L, tx.dataRoot);
  writeResult(ctx, 'installed', tx, null);
}

/** Before commit: point back, restore the snapshot, start the old version. If
 *  the restore fails the lease and record stay and nothing starts. */
async function rollback(ctx, tx, why) {
  ctx.log(`[Launcher] ${why}. Going back to ${tx.oldPayload}...`);
  try {
    await writePointer(ctx.root, tx.oldPayload);
    await runTool(ctx, tx.oldPayload, ['restore', tx.snapshot.dir, tx.snapshot.manifestSha256]);
  } catch (e) {
    writeResult(ctx, 'recovery-required', tx, `${why}. Restoring the data snapshot failed: ${e.message}. Nothing was started; the snapshot is kept in ${tx.snapshot.dir}.`);
    throw new RecoveryRequired(`${why}. Restoring the data snapshot failed: ${e.message}`);
  }
  retain(ctx, { payloads: [tx.newPayload], snapshots: [tx.snapshot.dir] });
  clearTx(ctx.L);
  dropLeases(ctx.L, tx.dataRoot);
  writeResult(ctx, 'rolled-back', tx, why);
  return startServer(ctx, tx.oldPayload);
}

/** After commit the new version stays and its data is never restored. Start it
 *  under a fresh lease, attest, release, then finish what the crash left. */
async function finishCommitted(ctx, tx) {
  await writePointer(ctx.root, tx.newPayload);
  const { lease, token } = newLease(ctx.root, tx.dataRoot);
  writeLease(ctx.L, lease);
  tx.leaseId = lease.id;
  writeTx(ctx.L, tx);
  const c = startServer(ctx, tx.newPayload, { HOT_STEP_MAINTENANCE_TOKEN: token, HOT_STEP_DATA_ROOT: tx.dataRoot });
  // The engine already answered before the commit; a broken one now must not
  // keep a working server and database from starting. The app shows its status.
  let problem = await attest(c, token, { ...readBuild(ctx, path.join(ctx.L.app, tx.newPayload)), leaseId: lease.id }, { engine: false });
  if (!problem && !await releaseGate(c, token)) problem = 'its maintenance gate did not open';
  if (problem) {
    await stopServer(c);
    const msg = `The new version was installed but does not start: ${problem}. Its data is already in the new format, so it was not rolled back.`;
    writeResult(ctx, 'recovery-required', tx, msg);
    throw new RecoveryRequired(msg);
  }
  finalize(ctx, tx);
  ctx.openBrowser(ctx, c.payload, c.address.url);
  return c;
}

// ── Main ─────────────────────────────────────────────────────────────

/** One launcher per install root: a named pipe the OS removes when we die. */
function lockRoot(root) {
  const key = sha256(real(root).toLowerCase()).slice(0, 32);
  const name = process.platform === 'win32' ? `\\\\.\\pipe\\hot-step-launcher-${key}` : path.join(os.tmpdir(), `hot-step-launcher-${key}.sock`);
  return new Promise((resolve, reject) => {
    const srv = net.createServer(c => c.destroy());
    srv.once('error', e => reject(e.code === 'EADDRINUSE' ? new AlreadyRunning(`HOT-Step is already running from ${root}.`) : e));
    srv.listen(name, () => resolve(srv));
  });
}

/** Once a server answers: open the browser (first start only) and, after a
 *  plain start or restart, delete what earlier updates retained. A server an
 *  update just started cleans nothing, so the old payload outlives this run. */
function afterStart(ctx, s, open, clean) {
  ctx.cleaning = (async () => {
    const a = await address(s, timing.healthMs);
    if (!a) return;
    let healthy = false;
    for (const deadline = Date.now() + timing.healthMs; !healthy && Date.now() < deadline && !s.exit;) {
      try { healthy = (await getJson(`${a.url}/api/health`)).ok; } catch { /* starting */ }
      if (!healthy) await sleep(timing.pollMs);
    }
    // No proven good start: the previous payload and its snapshot stay.
    if (!healthy || s.exit) return;
    if (open) ctx.openBrowser(ctx, s.payload, a.url);
    if (clean) await cleanRetained(ctx);
  })().catch(e => ctx.log(`[Launcher] Cleanup skipped: ${e.message}`));
}

/** Run the launcher until the server stops for good. Returns the exit code. */
export async function launch(opts = {}) {
  const root = path.resolve(opts.root ?? path.dirname(fileURLToPath(import.meta.url)));
  const ctx = {
    root, L: layout(root), env: opts.env ?? process.env, log: opts.log ?? (m => console.log(m)),
    nodeFor: opts.nodeFor ?? (id => path.join(root, 'app', id, 'runtime', 'node.exe')),
    openBrowser: opts.openBrowser ?? defaultOpenBrowser, stdio: opts.stdio ?? ['inherit', 'inherit', 'inherit'],
    current: null, cleaning: Promise.resolve(),
  };
  const lock = await lockRoot(root);
  try {
    fs.mkdirSync(ctx.L.updater, { recursive: true });
    // No server runs on this root while we hold the lock, so a preparation lock
    // left by a crash (or a reused PID) is stale.
    fs.rmSync(ctx.L.prepareLock, { force: true });
    ctx.dataEnv = dataEnv(root, ctx.env, ctx.log);
    const tx = readTx(ctx.L);
    let s;
    let plain = false;
    if (tx?.state === 'pending') s = await rollback(ctx, tx, 'An update was interrupted');
    else if (tx?.state === 'committed') s = await finishCommitted(ctx, tx);
    else { dropStaleLease(ctx.L); s = startServer(ctx, readPointer(root)); plain = true; }
    let opened = !plain && !!tx && tx.state === 'committed';
    for (;;) {
      afterStart(ctx, s, !opened, plain);
      opened = true;
      const ev = await Promise.race([s.exited.then(() => null), nextMessage(s, 'apply')]);
      if (ev) { s = await handleApply(ctx, s, ev); plain = false; continue; }
      const marker = path.join(ctx.dataEnv.HOT_STEP_PROFILE_DIR, 'restart-requested');
      if (fs.existsSync(marker)) {
        fs.rmSync(marker, { force: true });
        ctx.log('[Launcher] Restarting...');
        s = startServer(ctx, readPointer(root));
        plain = true;
        continue;
      }
      await ctx.cleaning;
      return s.exit.code ?? 1;
    }
  } catch (e) {
    if (ctx.current && !ctx.current.exit) { ctx.current.child.kill(); await ctx.current.exited; }
    throw e;
  } finally {
    lock.close();
  }
}

export const RECOVERY_DOC = 'https://github.com/scragnog/HOT-Step-CPP/blob/master/docs/user/update-recovery.md';

/** What the window shows when the launcher stops. An update never replaces this
 *  file, so the text stays general and the steps live in RECOVERY_DOC. */
export function failureText(e) {
  const head = e instanceof RecoveryRequired ? 'Recovery required' : 'Launcher error';
  return [
    `[HOT-Step] ${head}: ${e.message}`,
    '[HOT-Step] Nothing was started and nothing was deleted. Do not delete or edit anything in the updater folder.',
    '[HOT-Step] First step: close this window and anything that may hold HOT-Step files open (backup or sync tools,',
    '[HOT-Step] an antivirus scan), check the disk has free space, then start HOT-Step again. Each start retries',
    '[HOT-Step] from the beginning. If it stops here again, follow the steps at',
    `[HOT-Step] ${RECOVERY_DOC}`,
  ].join('\n');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  // Ctrl+C reaches the server too; it shuts down and we return when it exits.
  process.on('SIGINT', () => {});
  launch().then(code => process.exit(code), e => {
    if (e instanceof AlreadyRunning) { console.error(`[HOT-Step] ${e.message} Use the window that is already open.`); process.exit(2); }
    console.error(`\n${failureText(e)}`);
    process.exit(1);
  });
}
