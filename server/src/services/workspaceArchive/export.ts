// workspaceArchive/export.ts — Write a workspace archive (GH #228, slice A).
//
// The caller holds writes first (maintenance hold() in the app, no live server
// for the CLI). Then: verified DB snapshot (backupDatabase), credential rows
// removed with Secrets off, VACUUM so nothing lingers in free pages, external
// path scan; files streamed in store mode to <out>.zip.part with a sha256 each;
// settings/.env filtered; manifest.json last; rename to <out>.zip. Any
// failure, a blocker appearing, a file changing while read, or cancel leaves
// no zip, no .part and no staging folder. The source tree is only read.

import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Transform } from 'node:stream';
import archiver from 'archiver';
import Database from 'better-sqlite3';
import { parse as dotenvParse } from 'dotenv';
import { backupDatabase } from '../../db/backup.js';
import type { DataChildren } from '../../dataroot/locate.js';
import { isCredentialKey } from '../generation/requestCapture.js';
import { STALE_MS } from '../instance.js';
import { ENV_PATH_KEYS, samePath } from '../../dataroot/paths.js';
import { JSON_COLUMNS, PLAIN_COLUMNS } from '../../dataroot/transfer.js';
import { destinationErrors, ownerOf, planSources, takeInventory, type Inventory, type PlannedSource } from './inventory.js';
import {
  ARCHIVE_FORMAT, ARCHIVE_KIND, MANIFEST_NAME,
  type CategoryChoice, type ExternalRef, type Manifest, type ManifestFile,
} from './manifest.js';

export class ExportError extends Error {
  constructor(message: string, readonly status = 500, readonly extra: Record<string, unknown> = {}) { super(message); }
}
export class ExportCancelled extends ExportError {
  constructor() { super('Export cancelled', 499); }
}

export interface ExportProgress {
  phase: 'database' | 'files' | 'finishing';
  filesDone: number; filesTotal: number;
  bytesDone: number; bytesTotal: number;
}

export interface ExportOptions {
  children: DataChildren;
  dataRoot: string | null;
  /** Absolute path ending in .zip; it, <out>.part and <out>.staging must not exist. */
  outFile: string;
  choice: CategoryChoice;
  appVersion: string;
  /** Re-checked every `recheckMs` while the export runs; any line aborts it. */
  blockers?: () => string[];
  recheckMs?: number;
  signal?: AbortSignal;
  onProgress?: (p: ExportProgress) => void;
}

export interface ExportResult { outFile: string; bytes: number; manifest: Manifest }

/** .env keys that describe this machine, not the workspace: ports, hosts,
 *  CORS, GPU and worker selection, launcher variables. */
export function isMachineKey(key: string): boolean {
  return /(^|_)(PORT|HOST)$/.test(key) || key.startsWith('HOT_STEP_') || [
    'ALLOWED_HOSTS', 'CORS_ORIGINS', 'CUDA_VISIBLE_DEVICES', 'ACESTEPCPP_WORKER_GPUS',
    'TRAINING_WORKERS', 'TRAINING_WORKER_TOKEN', 'WORKER_TOKEN', 'DOWNLOAD_DIR',
  ].includes(key);
}

/** One KEY=value line that dotenv parses back to exactly `v`, or null when no
 *  quoting can (a value holding all three quote kinds). Single quotes and
 *  backticks expand nothing; double quotes turn \n and \r into line breaks. */
export function envLine(key: string, v: string): string | null {
  if (/^[^\s#'"`\\]*$/.test(v)) return `${key}=${v}`;
  const q = ["'", '`', '"'].find(c => !v.includes(c) && !(c === '"' && /\\[nr]/.test(v)));
  return q ? `${key}=${q}${v}${q}` : null;
}

/** The archive's settings/.env, rebuilt from parsed values: comments (inline
 *  ones too) never travel, since they can hold pasted keys. Without path keys,
 *  machine keys and, with secrets off, credential keys. Returns the dropped
 *  key names (never values). */
export function archiveEnvText(text: string, secrets: boolean): { text: string; dropped: string[] } {
  const keep: string[] = [];
  const dropped: string[] = [];
  for (const [key, value] of Object.entries(dotenvParse(text))) {
    const line = /^[A-Za-z_][A-Za-z0-9_]*$/.test(key)
      && !(ENV_PATH_KEYS.includes(key) || isMachineKey(key) || (!secrets && isCredentialKey(key))) ? envLine(key, value) : null;
    if (line === null) dropped.push(key); else keep.push(line);
  }
  return { text: keep.length ? keep.join('\n') + '\n' : '', dropped };
}

/** Test seam: runs after a file has been read into the zip, before it is re-checked. */
export const hooks = {
  afterRead: (_abs: string): void | Promise<void> => {},
  /** Runs once a source file's read stream exists. */
  opened: (_abs: string, _stream: fs.ReadStream): void => {},
};

export const DB_ENTRY = 'library/hotstep.db';
export const ENV_ENTRY = 'settings/.env';

/** Absolute paths in the DB's path columns that exist here but are not
 *  carried (outside every selected source). Existence separates a file from
 *  a URL such as /audio/<id>.wav, which is rooted on Windows too. */
export function scanExternals(db: Database.Database, sources: PlannedSource[]): ExternalRef[] {
  const counts = new Map<string, number>();
  const see = (s: string) => {
    if (s.length < 3 || !path.isAbsolute(s) || /^[a-z]+:\/\//i.test(s)) return;
    if (ownerOf(sources, s)?.selected) return;
    counts.set(s, (counts.get(s) ?? 0) + 1);
  };
  const walk = (x: unknown): void => {
    if (typeof x === 'string') see(x);
    else if (Array.isArray(x)) x.forEach(walk);
    else if (x && typeof x === 'object') Object.values(x).forEach(walk);
  };
  const cols = (t: string) => new Set((db.prepare(`PRAGMA table_info("${t}")`).all() as Array<{ name: string }>).map(c => c.name));
  for (const [table, wanted, json] of [
    ...Object.entries(PLAIN_COLUMNS).map(([t, c]) => [t, c, false] as const),
    ...Object.entries(JSON_COLUMNS).map(([t, c]) => [t, c, true] as const),
  ]) {
    const have = cols(table);
    for (const c of wanted.filter(c => have.has(c))) {
      for (const { v } of db.prepare(`SELECT "${c}" AS v FROM "${table}" WHERE "${c}" IS NOT NULL AND "${c}" != ''`).iterate() as Iterable<{ v: unknown }>) {
        if (typeof v !== 'string') continue;
        if (!json) { see(v); continue; }
        try { walk(JSON.parse(v)); } catch { /* not JSON */ }
      }
    }
  }
  return [...counts].filter(([p]) => fs.existsSync(p)).map(([p, refs]) => ({ path: p, refs })).sort((a, b) => a.path.localeCompare(b.path));
}

/** Snapshot the DB into `dest`, self-contained, credential rows removed with
 *  Secrets off, VACUUMed. Returns what the manifest records. */
async function snapshotDb(source: string, dest: string, secrets: boolean, sources: PlannedSource[]) {
  await backupDatabase(source, dest);
  const db = new Database(dest);
  try {
    db.pragma('journal_mode = DELETE');
    let removed = 0;
    const hasSettings = !!db.prepare(`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settings'`).get();
    if (!secrets && hasSettings) {
      const del = db.prepare('DELETE FROM settings WHERE key = ?');
      for (const { key } of db.prepare('SELECT key FROM settings').all() as Array<{ key: string }>) {
        if (isCredentialKey(key)) removed += del.run(key).changes;
      }
    }
    // Always: deleted rows (here or in the live DB) must not survive in free pages.
    db.exec('VACUUM');
    const userVersion = db.pragma('user_version', { simple: true }) as number;
    const externals = scanExternals(db, sources);
    return { userVersion, removed, externals };
  } finally { db.close(); }
}

/** Live HOT-Step servers registered in this DB other than `self`. */
export function otherLiveServers(dbPath: string, self: string | null): string[] {
  if (!fs.existsSync(dbPath)) return [];
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  try {
    let rows: Array<{ id: string; pid: number; host: string; copy: string; heartbeat_at: number }>;
    try { rows = db.prepare('SELECT id, pid, host, copy, heartbeat_at FROM server_instances WHERE stopped_at IS NULL').all() as typeof rows; }
    catch { return []; }   // no registry yet: nothing has run
    const alive = (pid: number) => { try { process.kill(pid, 0); return true; } catch (e: any) { return e.code === 'EPERM'; } };
    return rows
      .filter(r => r.id !== self && Date.now() - r.heartbeat_at <= STALE_MS && (r.host !== os.hostname() || alive(r.pid)))
      .map(r => `HOT-Step is running in another copy (${r.copy}, pid ${r.pid})`);
  } finally { db.close(); }
}

/** Why this export cannot start (empty when it can). */
export function exportPreflight(outFile: string, children: DataChildren, choice: CategoryChoice): string[] {
  if (!path.isAbsolute(outFile) || !/\.zip$/i.test(outFile)) return ['the archive must be an absolute path ending in .zip'];
  const errors = destinationErrors(path.dirname(outFile), planSources(children, choice));
  for (const p of [outFile, `${outFile}.part`, `${outFile}.staging`]) if (fs.existsSync(p)) errors.push(`${p} already exists`);
  return errors;
}

export async function exportWorkspace(o: ExportOptions): Promise<ExportResult> {
  const errors = exportPreflight(o.outFile, o.children, o.choice);
  if (errors.length) throw new ExportError(errors.join('; '), 400, { errors });
  const part = `${o.outFile}.part`;
  const staging = `${o.outFile}.staging`;

  const ac = new AbortController();
  const onCancel = () => ac.abort(new ExportCancelled());
  if (o.signal?.aborted) throw new ExportCancelled();
  o.signal?.addEventListener('abort', onCancel);
  const aborted = new Promise<never>((_, reject) => ac.signal.addEventListener('abort', () => reject(ac.signal.reason)));
  aborted.catch(() => {});
  const check = () => { if (ac.signal.aborted) throw ac.signal.reason; };
  const timer = o.blockers && setInterval(() => {
    let b: string[];
    // A throw here would be uncaught (timer callback): stop the export instead.
    try { b = o.blockers!(); } catch (e: any) { b = [`could not check for running work (${e.message})`]; }
    if (b.length) ac.abort(new ExportError(`Export stopped: work started while it ran (${b.join('; ')})`, 409, { blockers: b }));
  }, o.recheckMs ?? 3000);

  let archive: archiver.Archiver | null = null;
  let out: fs.WriteStream | null = null;
  /** The source file being read, so an abort can close it. */
  let reading: { rs: fs.ReadStream; hasher: Transform } | null = null;
  try {
    const inv: Inventory = takeInventory(o.children, o.choice);
    const hasDb = fs.existsSync(o.children.db);
    const progress: ExportProgress = {
      phase: 'database', filesDone: 0, filesTotal: inv.files.length + (hasDb ? 1 : 0),
      bytesDone: 0, bytesTotal: inv.files.reduce((s, f) => s + f.bytes, 0) + (hasDb ? fs.statSync(o.children.db).size : 0),
    };
    o.onProgress?.({ ...progress });

    fs.mkdirSync(staging);
    let db: Manifest['db'] = null;
    let externals: ExternalRef[] = [];
    const dbCopy = path.join(staging, 'hotstep.db');
    if (hasDb) {
      // Not raced against cancel: staging is removed only once SQLite has let go of it.
      const snap = await snapshotDb(o.children.db, dbCopy, o.choice.secrets, inv.sources);
      db = { name: DB_ENTRY, userVersion: snap.userVersion, credentialRowsRemoved: snap.removed };
      externals = snap.externals;
      progress.bytesTotal += fs.statSync(dbCopy).size - fs.statSync(o.children.db).size;
    }
    check();

    let env: Manifest['env'] = null;
    let envText: string | null = null;
    if (fs.existsSync(o.children.envFile)) {
      const r = archiveEnvText(fs.readFileSync(o.children.envFile, 'utf-8'), o.choice.secrets);
      env = { name: ENV_ENTRY, droppedKeys: r.dropped };
      envText = r.text;
    }
    const skipped = [...inv.skipped];
    // List what else sits in settings/ (.env backups hold old keys). Without a
    // data root the .env is the checkout's, whose folder is not ours to list.
    const envDir = path.dirname(o.children.envFile);
    if (o.dataRoot) try {
      for (const n of fs.readdirSync(envDir)) {
        if (samePath(path.join(envDir, n), o.children.envFile)) continue;
        skipped.push({ path: `settings/${n}`, reason: /\.(bak|before)/i.test(n) ? 'backup' : 'machine' });
      }
    } catch { /* no settings folder */ }

    // ── Zip ──
    progress.phase = 'files';
    out = fs.createWriteStream(part, { flags: 'wx' });
    archive = archiver('zip', { store: true });
    const closed = new Promise<void>((resolve, reject) => { out!.on('close', resolve); out!.on('error', reject); });
    closed.catch(() => {});
    out.on('error', e => ac.abort(e));
    archive.on('error', e => ac.abort(e));
    archive.on('warning', e => ac.abort(e));
    archive.pipe(out);
    let entryDone: (() => void) | null = null;
    archive.on('entry', () => { const r = entryDone; entryDone = null; r?.(); });
    const append = (src: NodeJS.ReadableStream | Buffer, name: string, date: Date) => {
      const done = new Promise<void>(resolve => { entryDone = resolve; });
      archive!.append(src as any, { name, date, store: true });
      return Promise.race([done, aborted]);
    };

    const files: ManifestFile[] = [];
    const addFile = async (abs: string, name: string) => {
      check();
      let before: fs.Stats;
      try { before = fs.statSync(abs); } catch { throw new ExportError(`${abs} disappeared while the archive was written`, 409); }
      const hash = crypto.createHash('sha256');
      let n = 0;
      const hasher = new Transform({ transform(chunk, _e, cb) { hash.update(chunk); n += chunk.length; cb(null, chunk); } });
      const rs = fs.createReadStream(abs);
      reading = { rs, hasher };
      rs.on('error', e => ac.abort(e));
      hooks.opened(abs, rs);
      await append(rs.pipe(hasher), name, before.mtime);
      reading = null;
      await hooks.afterRead(abs);
      let after: fs.Stats | null = null;
      try { after = fs.statSync(abs); } catch { /* removed */ }
      if (!after || after.size !== before.size || after.mtimeMs !== before.mtimeMs || n !== before.size) {
        throw new ExportError(`${abs} changed while it was being archived; stop whatever is writing it and export again`, 409);
      }
      files.push({ name, bytes: n, sha256: hash.digest('hex') });
      progress.filesDone++; progress.bytesDone += n;
      o.onProgress?.({ ...progress });
    };

    if (hasDb) await addFile(dbCopy, DB_ENTRY);
    for (const f of inv.files) await addFile(f.abs, f.name);
    check();
    progress.phase = 'finishing';
    o.onProgress?.({ ...progress });
    if (envText !== null) {
      const buf = Buffer.from(envText, 'utf-8');
      await append(buf, ENV_ENTRY, new Date());
      files.push({ name: ENV_ENTRY, bytes: buf.length, sha256: crypto.createHash('sha256').update(buf).digest('hex') });
    }

    const manifest: Manifest = {
      kind: ARCHIVE_KIND, format: ARCHIVE_FORMAT, createdAt: new Date().toISOString(),
      appVersion: o.appVersion, platform: process.platform, dataRoot: o.dataRoot,
      categories: { library: true, ...o.choice },
      sources: inv.sources.filter(s => s.selected).map(s => ({ category: s.category, prefix: s.prefix, path: s.dir, files: 0, bytes: 0 })),
      files, db, env, skipped, skippedNotListed: inv.skippedNotListed, externals,
    };
    // Each file counts toward the longest source prefix it sits under.
    for (const f of files) {
      const s = manifest.sources.filter(x => f.name.startsWith(x.prefix + '/')).sort((a, b) => b.prefix.length - a.prefix.length)[0];
      if (s) { s.files++; s.bytes += f.bytes; }
    }
    await append(Buffer.from(JSON.stringify(manifest, null, 1)), MANIFEST_NAME, new Date());
    check();
    await Promise.race([archive.finalize(), aborted]);
    await Promise.race([closed, aborted]);
    const fd = fs.openSync(part, 'r+');
    try { fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
    check();
    if (fs.existsSync(o.outFile)) throw new ExportError(`${o.outFile} appeared while the archive was written`, 409);
    fs.renameSync(part, o.outFile);
    return { outFile: o.outFile, bytes: fs.statSync(o.outFile).size, manifest };
  } catch (e) {
    // The first failure wins: tearing the streams down raises errors of its own.
    if (!ac.signal.aborted) ac.abort(e);
    const reason = ac.signal.reason;
    archive?.abort();
    // Close the source file, the hasher and the part file before anything is
    // removed or the caller releases its hold.
    const shut = (s: NodeJS.ReadableStream & { closed: boolean; destroy(): void } | null) =>
      !s || s.closed ? Promise.resolve() : new Promise<void>(r => { s.once('close', () => r()); s.destroy(); });
    const src = reading as { rs: fs.ReadStream; hasher: Transform } | null;   // set inside addFile; TS cannot see it
    await Promise.all([shut(src?.rs ?? null), shut(src?.hasher ?? null), shut(out as any)]);
    fs.rmSync(part, { force: true });
    throw reason;
  } finally {
    if (timer) clearInterval(timer);
    o.signal?.removeEventListener('abort', onCancel);
    fs.rmSync(staging, { recursive: true, force: true });
  }
}
