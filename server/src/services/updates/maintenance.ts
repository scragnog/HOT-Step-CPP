// maintenance.ts — The safety layer an update transaction stands on.
//
// Installs nothing. It holds four pieces the Windows launcher uses:
//   - the maintenance lease: one record written to the updater folder and the
//     data root, so no second process opens SQLite or runs bootstrap work;
//   - the gate: one early Express middleware that answers 503 to everything but
//     token-authenticated health and control while a transaction is open, plus
//     whenOpen(), which holds startup background work until the gate opens;
//   - the transaction record format (pending / committed);
//   - DB and settings snapshot / restore, validated before anything starts.
//
// With no lease on disk all of it is a no-op: the gate calls next() and
// whenOpen() runs its callback at once, so startup order and request handling
// stay as they were. This module imports no config (bootstrap.ts uses it before
// config exists); callers pass paths in.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';
import { Router, type NextFunction, type Request, type Response } from 'express';

export const UPDATER_DIR_ENV = 'HOT_STEP_UPDATER_DIR';
export const TOKEN_ENV = 'HOT_STEP_MAINTENANCE_TOKEN';
export const TOKEN_HEADER = 'x-hotstep-maintenance-token';
/** In the updater folder. */
export const LEASE_FILE = 'maintenance-lease.json';
export const TRANSACTION_FILE = 'transaction.json';
export const SNAPSHOTS_DIR = 'snapshots';
/** In the data root: the same lease, for a process that only knows the data root. */
export const DATA_LEASE_FILE = '.hot-step-maintenance.json';
/** Paths that pass a closed gate, with the token only. */
export const CONTROL_PATHS = ['/api/health', '/api/maintenance', '/api/shutdown'];

const sha256 = (data: string | Buffer) => crypto.createHash('sha256').update(data).digest('hex');
const sha256File = (file: string) => sha256(fs.readFileSync(file));
const fail = (message: string, status: number, extra: object = {}) => Object.assign(new Error(message), { status, ...extra });
const PAYLOAD_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function writeJsonAtomic(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n');
  fs.renameSync(tmp, file);
}

/** Create `file` with `data` only if it does not exist. The content is
 *  complete before the name appears (written aside, then hard-linked). */
function createJsonExclusive(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n');
  try { fs.linkSync(tmp, file); } finally { fs.rmSync(tmp, { force: true }); }
}

function readJson(file: string): unknown | null {
  let raw: string;
  try { raw = fs.readFileSync(file, 'utf-8'); } catch (e: any) { if (e.code === 'ENOENT') return null; throw e; }
  try { return JSON.parse(raw); } catch { throw new Error(`${file} is not valid JSON`); }
}

// ── Lease ────────────────────────────────────────────────────────────

export interface Lease {
  format: 1;
  id: string;
  /** sha256 of the token; the token itself is only ever in memory and env. */
  tokenSha256: string;
  installRoot: string;
  dataRoot: string;
  /** Diagnostic only. A dead PID never means the transaction finished. */
  pid: number;
  createdAt: string;
}

function parseLease(v: unknown, file: string): Lease {
  const l = v as Lease;
  if (!l || l.format !== 1 || typeof l.id !== 'string' || !/^[0-9a-f]{64}$/.test(l.tokenSha256 ?? '')
      || typeof l.installRoot !== 'string' || typeof l.dataRoot !== 'string') {
    throw new Error(`${file} is not a format 1 maintenance lease`);
  }
  return l;
}

export interface LeaseDirs { updaterDir?: string | null; dataRoot?: string | null }

const leaseFiles = (d: LeaseDirs) => [
  ...(d.updaterDir ? [path.join(d.updaterDir, LEASE_FILE)] : []),
  ...(d.dataRoot ? [path.join(d.dataRoot, DATA_LEASE_FILE)] : []),
];

/** Every lease present in these folders. Throws on an unreadable one. */
export function readLeases(d: LeaseDirs): Lease[] {
  const out: Lease[] = [];
  for (const f of leaseFiles(d)) { const v = readJson(f); if (v !== null) out.push(parseLease(v, f)); }
  return out;
}

const realDir = (p: string) => fs.realpathSync.native(path.resolve(p));

/** Take the lease in both places. Throws 409 when either is already held. */
export function acquireLease(o: { updaterDir: string; installRoot: string; dataRoot: string }): { lease: Lease; token: string } {
  const token = crypto.randomBytes(32).toString('hex');
  const lease: Lease = {
    format: 1, id: crypto.randomUUID(), tokenSha256: sha256(token),
    installRoot: realDir(o.installRoot), dataRoot: realDir(o.dataRoot), pid: process.pid, createdAt: new Date().toISOString(),
  };
  const [own, data] = leaseFiles(o);
  try { createJsonExclusive(own, lease); } catch (e: any) {
    if (e.code === 'EEXIST') throw fail(`An update is already in progress (${own})`, 409);
    throw e;
  }
  try { createJsonExclusive(data, lease); } catch (e: any) {
    fs.rmSync(own, { force: true });
    if (e.code === 'EEXIST') throw fail(`The data folder is held by another update (${data})`, 409);
    throw e;
  }
  return { lease, token };
}

/** Drop this lease from both places; a lease with another id is left alone. */
export function releaseLease(d: LeaseDirs, id: string): void {
  for (const f of leaseFiles(d).reverse()) {
    const v = readJson(f);
    if (v === null) continue;
    if (parseLease(v, f).id !== id) throw fail(`${f} belongs to another update`, 409);
    fs.rmSync(f, { force: true });
  }
}

const tokenMatches = (token: string | undefined | null, wantSha: string) =>
  !!token && crypto.timingSafeEqual(Buffer.from(sha256(token), 'hex'), Buffer.from(wantSha, 'hex'));

/** The lease this process was started under (its token is in env), or null
 *  when there is no lease. Throws when a lease exists that is not ours, or a
 *  transaction record exists without one: never open SQLite then. */
export function leaseForStartup(d: LeaseDirs, env: NodeJS.ProcessEnv = process.env): Lease | null {
  const leases = readLeases(d);
  const tx = d.updaterDir ? readTransaction(d.updaterDir) : null;
  if (!leases.length) {
    if (tx) throw new Error(`An unfinished update (${tx.id}) has no maintenance lease. Start HOT-Step from its launcher so it can recover.`);
    return null;
  }
  const lease = leases[0];
  if (leases.some(l => l.id !== lease.id) || !tokenMatches(env[TOKEN_ENV], lease.tokenSha256)) {
    throw new Error(`An update is being installed (started ${lease.createdAt}). Wait for it to finish; if nothing is running, start HOT-Step from its launcher so it can recover.`);
  }
  return lease;
}

// ── Transaction record ───────────────────────────────────────────────

export interface TransactionRecord {
  format: 1;
  id: string;
  /** Written before the pointer moves; committed after the candidate's health passes. */
  state: 'pending' | 'committed';
  leaseId: string;
  /** Directory names under app/, never paths. */
  oldPayload: string;
  newPayload: string;
  dataRoot: string;
  snapshot: { dir: string; manifestSha256: string };
  createdAt: string;
  committedAt: string | null;
}

function parseTransaction(v: unknown, file: string): TransactionRecord {
  const t = v as TransactionRecord;
  const ok = t && t.format === 1 && typeof t.id === 'string' && (t.state === 'pending' || t.state === 'committed')
    && typeof t.leaseId === 'string' && PAYLOAD_ID.test(t.oldPayload ?? '') && PAYLOAD_ID.test(t.newPayload ?? '')
    && !t.oldPayload.includes('..') && !t.newPayload.includes('..') && typeof t.dataRoot === 'string'
    && typeof t.snapshot?.dir === 'string' && /^[0-9a-f]{64}$/.test(t.snapshot?.manifestSha256 ?? '');
  if (!ok) throw new Error(`${file} is not a valid format 1 update transaction`);
  return t;
}

/** null when there is none; throws when one exists but is malformed (fail closed). */
export function readTransaction(updaterDir: string): TransactionRecord | null {
  const file = path.join(updaterDir, TRANSACTION_FILE);
  const v = readJson(file);
  return v === null ? null : parseTransaction(v, file);
}

export function writeTransaction(updaterDir: string, t: TransactionRecord): void {
  const file = path.join(updaterDir, TRANSACTION_FILE);
  writeJsonAtomic(file, parseTransaction(t, file));
}

export function clearTransaction(updaterDir: string): void {
  fs.rmSync(path.join(updaterDir, TRANSACTION_FILE), { force: true });
}

// ── Gate ─────────────────────────────────────────────────────────────

const MUTATING = (m: string) => m !== 'GET' && m !== 'HEAD' && m !== 'OPTIONS';

export function createGate() {
  let closed = false;
  let lease: Lease | null = null;
  let inflight = 0;
  const deferred: Array<() => void> = [];
  /** Update-only blockers (queued work a data-folder move may leave queued). */
  const extraBlockers: Array<() => string[]> = [];

  const authorised = (req: Request) => !!lease && tokenMatches(req.get(TOKEN_HEADER), lease.tokenSha256);

  function middleware(req: Request, res: Response, next: NextFunction): void {
    if (closed && !(CONTROL_PATHS.some(p => req.path === p || req.path.startsWith(p + '/')) && authorised(req))) {
      res.status(503).set('Retry-After', '5').json({
        error: 'HOT-Step is installing an update. It will be back shortly.',
        maintenance: { state: 'closed', leaseId: lease?.id ?? null },
      });
      return;
    }
    if (MUTATING(req.method)) {
      inflight++;
      let done = false;
      res.once('close', () => { if (!done) { done = true; inflight--; } });
    }
    next();
  }

  /** Run now when open (always, with no lease), else when open() is called. */
  function whenOpen(fn: () => void): void {
    if (closed) deferred.push(fn); else fn();
  }

  function close(l: Lease): void { closed = true; lease = l; }

  /** Reopen admission and run deferred startup work, in registration order. */
  function open(): void {
    closed = false;
    lease = null;
    for (const fn of deferred.splice(0)) {
      try { fn(); } catch (e) { console.error('[Maintenance] deferred startup work failed:', e); }
    }
  }

  async function drain(allow: number, timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (inflight > allow) {
      if (Date.now() > deadline) return false;
      await new Promise(r => setTimeout(r, 50));
    }
    return true;
  }

  /** Open an update transaction on a running server: preflight, take the
   *  lease, close admission, let in-flight writes finish, then refuse (and
   *  reopen) if any work runs or is queued. Never cancels a job. Call from
   *  inside a request handler when `fromRequest` (that request is excluded). */
  async function begin(o: {
    updaterDir: string; installRoot: string; dataRoot: string;
    preflight: () => string[]; blockers: () => string[];
    fromRequest?: boolean; drainMs?: number;
  }): Promise<{ lease: Lease; token: string }> {
    if (closed) throw fail('An update is already in progress', 409);
    const errors = o.preflight();
    if (errors.length) throw fail(errors.join('; '), 400, { errors });
    const taken = acquireLease(o);
    close(taken.lease);
    // ponytail: only non-GET requests are waited for; a GET that starts work is caught by the blocker check alone.
    const drained = await drain(o.fromRequest ? 1 : 0, o.drainMs ?? 10_000);
    const blockers = drained ? [...o.blockers(), ...extraBlockers.flatMap(f => f())] : ['requests still in flight'];
    if (blockers.length) {
      releaseLease(o, taken.lease.id);
      closed = false;
      lease = null;
      throw fail(`Wait for running work to finish: ${blockers.join('; ')}`, 409, { blockers });
    }
    return taken;
  }

  return {
    middleware, whenOpen, close, open, begin, authorised,
    addBlocker: (f: () => string[]) => { extraBlockers.push(f); },
    get state() { return { closed, leaseId: lease?.id ?? null, deferred: deferred.length }; },
  };
}

export type Gate = ReturnType<typeof createGate>;

/** The server's one gate (index.ts mounts it first; the apply route calls begin()). */
export const maintenance = createGate();

/** /api/maintenance: status (the launcher's health attestation) and release.
 *  Both need the token while a lease is held. */
export function maintenanceRouter(gate: Gate, attest: () => object): Router {
  const r = Router();
  const guard = (req: Request, res: Response) => {
    if (gate.state.leaseId && !gate.authorised(req)) { res.status(403).json({ error: 'maintenance token required' }); return false; }
    return true;
  };
  r.get('/status', (req, res) => {
    if (guard(req, res)) res.json({ ...gate.state, ...attest() });
  });
  r.post('/release', (req, res) => {
    if (!guard(req, res)) return;
    gate.open();
    res.json(gate.state);
  });
  return r;
}

// ── Snapshot and restore ─────────────────────────────────────────────

export interface SnapshotEntry {
  kind: 'db' | 'file';
  /** Absolute path the entry restores to. */
  target: string;
  /** false: the file was absent, so restore removes whatever the candidate made. */
  existed: boolean;
  name: string | null;
  sha256: string | null;
  bytes: number | null;
}

export interface SnapshotManifest { format: 1; createdAt: string; entries: SnapshotEntry[] }

export const MANIFEST_FILE = 'manifest.json';
const SIDECARS = ['-wal', '-shm', '-journal'];

const tablesOf = (db: Database.Database) =>
  (db.prepare(`SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name`).all() as Array<{ name: string }>).map(r => r.name).join(',');

function integrityOk(file: string): void {
  const db = new Database(file, { readonly: true, fileMustExist: true });
  try {
    const rows = db.pragma('integrity_check') as Array<{ integrity_check: string }>;
    if (rows.length !== 1 || rows[0].integrity_check !== 'ok') {
      throw new Error(`integrity_check failed for ${file}: ${rows.map(r => r.integrity_check).join('; ')}`);
    }
  } finally { db.close(); }
}

/** A standalone copy of a SQLite DB, WAL content included: SQLite's backup
 *  API reads through a connection, so committed WAL frames come with it and
 *  the source WAL is never discarded. The copy is switched to a rollback
 *  journal so it is one self-contained file, then integrity-checked. */
async function snapshotDb(source: string, dest: string): Promise<void> {
  const src = new Database(source, { fileMustExist: true });
  let want: string;
  try {
    src.pragma('busy_timeout = 5000');
    want = tablesOf(src);
    await src.backup(dest);
  } finally { src.close(); }
  const copy = new Database(dest);
  try {
    copy.pragma('journal_mode = DELETE');
    if (tablesOf(copy) !== want) throw new Error(`snapshot of ${source} has a different table list`);
  } finally { copy.close(); }
  integrityOk(dest);
}

/** Snapshot `dbs` and `files` into a new folder `dir` (it must not exist).
 *  Call after a clean shutdown. Throws, leaving nothing behind, on any failure. */
export async function createSnapshot(dir: string, set: { dbs: string[]; files: string[] }): Promise<{ manifest: SnapshotManifest; manifestSha256: string }> {
  if (fs.existsSync(dir)) throw new Error(`${dir} already exists`);
  fs.mkdirSync(dir, { recursive: true });
  try {
    const entries: SnapshotEntry[] = [];
    const all = [...set.dbs.map(p => ['db', p] as const), ...set.files.map(p => ['file', p] as const)];
    for (const [i, [kind, p]] of all.entries()) {
      const target = path.resolve(p);
      if (!fs.existsSync(target)) { entries.push({ kind, target, existed: false, name: null, sha256: null, bytes: null }); continue; }
      const name = `${i}-${path.basename(target)}`;
      const dest = path.join(dir, name);
      if (kind === 'db') await snapshotDb(target, dest); else fs.copyFileSync(target, dest);
      entries.push({ kind, target, existed: true, name, sha256: sha256File(dest), bytes: fs.statSync(dest).size });
    }
    const manifest: SnapshotManifest = { format: 1, createdAt: new Date().toISOString(), entries };
    writeJsonAtomic(path.join(dir, MANIFEST_FILE), manifest);
    return { manifest, manifestSha256: sha256File(path.join(dir, MANIFEST_FILE)) };
  } catch (e) {
    fs.rmSync(dir, { recursive: true, force: true });
    throw e;
  }
}

function readManifest(dir: string, wantSha?: string): SnapshotManifest {
  const file = path.join(dir, MANIFEST_FILE);
  if (wantSha && sha256File(file) !== wantSha) throw new Error(`${file} does not match the transaction record`);
  const m = readJson(file) as SnapshotManifest | null;
  if (!m || m.format !== 1 || !Array.isArray(m.entries)) throw new Error(`${file} is not a format 1 snapshot manifest`);
  for (const e of m.entries) {
    if ((e.kind !== 'db' && e.kind !== 'file') || !path.isAbsolute(e.target ?? '')
        || (e.existed && (typeof e.name !== 'string' || path.basename(e.name) !== e.name || !/^[0-9a-f]{64}$/.test(e.sha256 ?? '')))) {
      throw new Error(`${file} has an unreadable entry`);
    }
  }
  return m;
}

const LOCKED = new Set(['EPERM', 'EBUSY', 'EACCES']);
/** Bounded retries for antivirus and indexer locks: three attempts, 3 s at most. */
async function retry<T>(fn: () => T): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try { return fn(); } catch (e: any) {
      if (attempt >= 3 || !LOCKED.has(e.code)) throw e;
      await new Promise(r => setTimeout(r, 1000 * attempt));
    }
  }
}

/** Check that every target matches the snapshot. Throws on the first mismatch. */
export function verifyRestored(dir: string, manifestSha256?: string): void {
  for (const e of readManifest(dir, manifestSha256).entries) {
    if (!e.existed) {
      if (fs.existsSync(e.target)) throw new Error(`${e.target} should not exist after restore`);
      continue;
    }
    if (e.kind === 'db') {
      const extra = SIDECARS.filter(s => fs.existsSync(e.target + s));
      if (extra.length) throw new Error(`${e.target} still has ${extra.join(', ')} beside it`);
      integrityOk(e.target);
    }
    if (sha256File(e.target) !== e.sha256) throw new Error(`${e.target} does not match its snapshot`);
  }
}

/** Put every snapshotted DB, sidecar set and file back, as one set. Run only
 *  with every handle closed. The candidate's files are copied to
 *  <dir>/candidate/ first for diagnosis (once). Repeatable after an
 *  interruption: the snapshot is never modified and each run starts from it.
 *  Returns only when the whole set validates; nothing may start otherwise. */
export async function restoreSnapshot(dir: string, manifestSha256?: string): Promise<void> {
  const m = readManifest(dir, manifestSha256);
  for (const e of m.entries) {
    if (e.existed && sha256File(path.join(dir, e.name!)) !== e.sha256) throw new Error(`snapshot file ${e.name} is damaged; restore stopped before touching anything`);
  }

  const diag = path.join(dir, 'candidate');
  if (!fs.existsSync(path.join(diag, 'done'))) {
    fs.rmSync(diag, { recursive: true, force: true });
    fs.mkdirSync(diag, { recursive: true });
    for (const [i, e] of m.entries.entries()) {
      for (const s of e.kind === 'db' ? ['', ...SIDECARS] : ['']) {
        if (fs.existsSync(e.target + s)) await retry(() => fs.copyFileSync(e.target + s, path.join(diag, `${i}-${path.basename(e.target)}${s}`)));
      }
    }
    fs.writeFileSync(path.join(diag, 'done'), '');
  }

  // Stage every file beside its target first, so the swap below is renames only.
  const tmp = (e: SnapshotEntry) => `${e.target}.restore-tmp`;
  for (const e of m.entries.filter(x => x.existed)) {
    fs.mkdirSync(path.dirname(e.target), { recursive: true });
    await retry(() => fs.copyFileSync(path.join(dir, e.name!), tmp(e)));
    if (sha256File(tmp(e)) !== e.sha256) throw new Error(`staged copy of ${e.target} does not match its snapshot`);
  }
  for (const e of m.entries) {
    // Never pair an old DB with the candidate's WAL: sidecars go first.
    if (e.kind === 'db') for (const s of SIDECARS) await retry(() => fs.rmSync(e.target + s, { force: true }));
    if (e.existed) await retry(() => fs.renameSync(tmp(e), e.target));
    else await retry(() => fs.rmSync(e.target, { force: true }));
  }
  verifyRestored(dir, manifestSha256);
}
