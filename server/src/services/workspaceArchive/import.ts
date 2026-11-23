// workspaceArchive/import.ts — Read a workspace archive into a new data folder (GH #228, slice B).
//
// Import never writes into the live data folder. Flow: check the target (new
// or empty, outside the app, the current root and every folder this machine
// still uses) → read the central directory and manifest.json, check every
// name (relative, under a carried category, no links, no case clashes, the
// zip and manifest list the same files) → extract → every sha256 and
// integrity_check → schema check against this machine's DB → rewrite stored
// paths with one pair per carried category → settings/.env from this
// machine's file with the archive's values on top → import-report.json →
// schedule('repoint'). Any failure removes what it wrote into the target.
// Windows archives on other platforms (and the reverse) are refused in v1:
// stored paths cannot be rewritten between the two path styles.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import Database from 'better-sqlite3';
import { parse as dotenvParse } from 'dotenv';
import yauzl from 'yauzl';
import type { DataChildren } from '../../dataroot/locate.js';
import { dataLayout, ENV_PATH_KEYS, initLayout, isEmptyDir, isInside, writeJsonAtomic, type PathPair } from '../../dataroot/paths.js';
import { canonical, rewriteDb, rewriteJsonFiles } from '../../dataroot/transfer.js';
import { removeSync } from '../fsRemove.js';
import { isCredentialKey } from '../generation/requestCapture.js';
import { DB_ENTRY, ENV_ENTRY, envLine, isMachineKey } from './export.js';
import { freeBytes, LIBRARY_NAMES, TRAINING_NAMES } from './inventory.js';
import { ARCHIVE_FORMAT, ARCHIVE_KIND, MANIFEST_NAME, type CategoryId, type Manifest, type ManifestFile } from './manifest.js';

export class ImportError extends Error {
  constructor(message: string, readonly status = 400, readonly extra: Record<string, unknown> = {}) { super(message); }
}
export class ImportCancelled extends ImportError {
  constructor() { super('Import cancelled', 499); }
}

export const REPORT_NAME = 'import-report.json';

/** This machine, as the import sees it. */
export interface ImportHere {
  /** The install folder; the target must lie outside it. */
  appRoot: string;
  dataRoot: string | null;
  /** Current locations: overlap checks, this machine's .env and DB, and the
   *  path keys of categories the archive does not carry. */
  children: DataChildren;
  /** Effective values of path keys whose default follows the data root but
   *  that no category carries (WHISPER_EXE, TENSORRT_LIBS, WHISPER_MODELS_DIR). */
  paths: Partial<Record<string, string>>;
}

export interface ImportProgress {
  phase: 'checking' | 'extracting' | 'rewriting' | 'waiting' | 'finishing';
  filesDone: number; filesTotal: number;
  bytesDone: number; bytesTotal: number;
  /** Set while waiting for running work to finish before the switch. */
  blockers?: string[];
}

export interface ImportOptions {
  zipFile: string;
  target: string;
  here: ImportHere;
  /** Extra reasons the target cannot become the data folder (the app's repoint checks). */
  targetErrors?: (target: string) => string[];
  /** Running work; the switch waits until this is empty. */
  blockers?: () => string[];
  /** Last step, e.g. scheduleOperation('repoint') and a restart. A throw removes the folder. */
  schedule?: (target: string) => void;
  signal?: AbortSignal;
  onProgress?: (p: ImportProgress) => void;
  waitMs?: number;
}

export interface ImportReport {
  importedAt: string;
  zipFile: string;
  target: string;
  archive: Pick<Manifest, 'createdAt' | 'appVersion' | 'platform' | 'dataRoot' | 'categories'>;
  files: number;
  bytes: number;
  integrity: 'ok' | 'no database';
  pathsRewritten: { db: number; jsonFiles: number; pairs: PathPair[] };
  credentialRowsCopied: number;
  env: { fromArchive: string[]; pointedHere: Record<string, string>; dropped: string[] };
  /** Paths the archive's DB names outside what it carried, left as they are. */
  externals: Array<{ path: string; refs: number; existsHere: boolean }>;
  skipped: number;
  scheduled: boolean;
}

// ── Archive reading ──────────────────────────────────────────────────

const MANIFEST_CAP = 512 * 1024 * 1024;
const WIN_BAD = /[<>:"|?*\u0000-\u001f]/;
const WIN_RESERVED = /^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i;
/** The prefixes planSources() writes. */
const SOURCE_PREFIXES = new Set(['library', 'library/training', 'adapters', 'models', 'models/muscriptor']);
const pathStyle =(p: string) => (p === 'win32' ? 'win32' : 'posix');

/** Why one entry name cannot be extracted (null when it can). yauzl itself
 *  refuses absolute names, backslashes and `..` before this runs. */
export function nameError(name: string, platform: NodeJS.Platform = process.platform): string | null {
  if (name === MANIFEST_NAME || name === ENV_ENTRY) return null;
  if (!/^(library|adapters|models)\/./.test(name)) return `${name}: not under library/, adapters/ or models/`;
  // The library takes only what the exporter writes: the one DB snapshot and
  // its allowlisted folders, so no WAL, journal or second SQLite file rides in.
  const [top, a, b] = name.split('/');
  if (top === 'library' && name !== DB_ENTRY && !(a === 'training' ? b !== undefined && TRAINING_NAMES.has(b) : LIBRARY_NAMES.has(a))) {
    return `${name}: not a library entry HOT-Step exports`;
  }
  for (const seg of name.split('/')) {
    if (!seg || seg === '.' || seg === '..') return `${name}: empty, . or .. path segment`;
    if (platform === 'win32' && (WIN_BAD.test(seg) || WIN_RESERVED.test(seg) || /[. ]$/.test(seg))) return `${name}: not a valid Windows file name`;
  }
  return null;
}

const isLink = (e: yauzl.Entry) => (e.versionMadeBy >> 8) === 3 && ((e.externalFileAttributes >>> 16) & 0o170000) === 0o120000;

export interface OpenedArchive {
  zip: yauzl.ZipFile;
  manifest: Manifest;
  entries: Map<string, yauzl.Entry>;
}

async function readAll(zip: yauzl.ZipFile, e: yauzl.Entry, cap: number): Promise<Buffer> {
  if (e.uncompressedSize > cap) throw new ImportError(`${e.fileName} is too large`);
  const chunks: Buffer[] = [];
  for await (const c of await zip.openReadStreamPromise(e)) chunks.push(c as Buffer);
  return Buffer.concat(chunks);
}

/** Open and check an archive: every name, the manifest, and that the zip and
 *  the manifest list the same files. Throws ImportError (400) listing what is
 *  wrong. The caller closes `zip`. */
export async function openArchive(zipFile: string): Promise<OpenedArchive> {
  let zip: yauzl.ZipFile;
  try { zip = await yauzl.openPromise(zipFile, { lazyEntries: true, autoClose: false, strictFileNames: true, validateEntrySizes: true }); }
  catch (e: any) { throw new ImportError(`${zipFile} is not a readable zip (${e.message})`); }
  try {
    const entries = new Map<string, yauzl.Entry>();
    const folded = new Map<string, string>();
    const errors: string[] = [];
    await new Promise<void>((resolve, reject) => {
      zip.on('entry', (e: yauzl.Entry) => {
        const n = e.fileName;
        const err = nameError(n) ?? (isLink(e) ? `${n}: a link` : e.isEncrypted() ? `${n}: encrypted` : null);
        const fold = n.normalize('NFC').toLowerCase();
        if (err) errors.push(err);
        else if (entries.has(n)) errors.push(`${n}: appears twice`);
        else if (folded.has(fold)) errors.push(`${n}: differs only in case from ${folded.get(fold)}`);
        else { entries.set(n, e); folded.set(fold, n); }
        zip.readEntry();
      });
      zip.once('end', () => resolve());
      zip.once('error', reject);
      zip.readEntry();
    }).catch((e: Error) => { throw new ImportError(`${zipFile} is not a valid workspace archive: ${e.message}`); });

    const me = entries.get(MANIFEST_NAME);
    if (!me) throw new ImportError('the archive has no manifest.json: it is unfinished or not a HOT-Step workspace archive', 400, { errors });
    let manifest: Manifest;
    try { manifest = JSON.parse((await readAll(zip, me, MANIFEST_CAP)).toString('utf-8')); }
    catch (e: any) { throw new ImportError(`manifest.json cannot be read (${e.message})`); }
    if (manifest?.kind !== ARCHIVE_KIND || !Array.isArray(manifest.files) || !Array.isArray(manifest.sources)) {
      throw new ImportError('manifest.json is not a HOT-Step workspace manifest');
    }
    if (manifest.format !== ARCHIVE_FORMAT) throw new ImportError(`archive format ${manifest.format} is not supported by this version (it reads ${ARCHIVE_FORMAT}); update HOT-Step`);
    if (pathStyle(manifest.platform) !== pathStyle(process.platform)) {
      throw new ImportError(`this archive was made on ${manifest.platform}; importing it on ${process.platform} is not supported yet (stored paths cannot be moved between Windows and macOS or Linux)`);
    }

    for (const s of manifest.sources) {
      if (!SOURCE_PREFIXES.has(s.prefix) || typeof s.path !== 'string' || !path.isAbsolute(s.path)) errors.push(`manifest source ${s.prefix} (${s.path}) is not one this app knows`);
    }
    if (entries.has(DB_ENTRY) !== !!manifest.db || (manifest.db && manifest.db.name !== DB_ENTRY)) {
      errors.push(`manifest.db does not match the zip (${entries.has(DB_ENTRY) ? 'the zip holds' : 'the zip lacks'} ${DB_ENTRY})`);
    }
    const prefixes = manifest.sources.map(s => s.prefix + '/');
    const listed = new Set<string>();
    for (const f of manifest.files as ManifestFile[]) {
      if (listed.has(f.name)) errors.push(`manifest lists ${f.name} twice`);
      listed.add(f.name);
      const e = entries.get(f.name);
      if (!e) errors.push(`${f.name}: in the manifest, missing from the zip`);
      else if (e.uncompressedSize !== f.bytes) errors.push(`${f.name}: ${e.uncompressedSize} bytes in the zip, ${f.bytes} in the manifest`);
      if (f.name !== ENV_ENTRY && !prefixes.some(p => f.name.startsWith(p))) errors.push(`${f.name}: not under a category the archive carries`);
      if (!/^[0-9a-f]{64}$/.test(String(f.sha256))) errors.push(`${f.name}: no sha256 in the manifest`);
    }
    for (const n of entries.keys()) if (n !== MANIFEST_NAME && !listed.has(n)) errors.push(`${n}: in the zip, not in the manifest`);
    if (errors.length) {
      throw new ImportError(`the archive failed its checks: ${errors.slice(0, 5).join('; ')}${errors.length > 5 ? `; and ${errors.length - 5} more` : ''}`, 400, { errors: errors.slice(0, 200) });
    }
    return { zip, manifest, entries };
  } catch (e) { zip.close(); throw e; }
}

// ── Target ───────────────────────────────────────────────────────────

/** Folders the new data folder must not overlap: the app, the current root,
 *  and every location this machine still uses. */
function guarded(here: ImportHere): Array<[string, string]> {
  const c = here.children;
  const out: Array<[string, string]> = [['the app folder', here.appRoot]];
  if (here.dataRoot) out.push(['the current data folder', here.dataRoot]);
  out.push(['the library', c.library], ['the training folder', c.training], ['the adapters folder', c.adapters],
    ['the models folder', c.models], ['the MuScriptor models folder', c.muscriptorModels]);
  return out;
}

/** Why `target` cannot take the import (empty when it can). */
export function targetErrors(target: string, here: ImportHere): string[] {
  if (!path.isAbsolute(target)) return ['the new data folder must be an absolute path'];
  if (fs.existsSync(target)) {
    if (!fs.statSync(target).isDirectory()) return [`${target} is not a folder`];
    if (!isEmptyDir(target)) return [`${target} is not empty; import only into a new or empty folder`];
  } else if (!fs.existsSync(path.dirname(target))) return [`${path.dirname(target)} does not exist`];
  const t = canonical(target);
  const errors: string[] = [];
  for (const [what, p] of guarded(here)) {
    const g = canonical(p);
    if (isInside(g, t)) errors.push(`${target} is inside ${what} (${p})`);
    else if (isInside(t, g)) errors.push(`${target} contains ${what} (${p})`);
  }
  return errors;
}

// ── Schema, credentials, settings ────────────────────────────────────

const columnsOf = (db: Database.Database) => {
  const out = new Map<string, Set<string>>();
  for (const { name } of db.prepare(`SELECT name FROM sqlite_master WHERE type = 'table'`).all() as Array<{ name: string }>) {
    out.set(name, new Set((db.prepare(`PRAGMA table_info("${name.replace(/"/g, '""')}")`).all() as Array<{ name: string }>).map(c => c.name)));
  }
  return out;
};

/** Why this app cannot open `archived` (empty when it can): a column this
 *  machine's DB lacks in a table both have, or a higher user_version. Tables
 *  only the archive has are fine: services create theirs on first use. An
 *  older archive is migrated by initDb() at the next start, like any upgrade. */
export function schemaErrors(archived: Database.Database, hereDb: string): string[] {
  if (!fs.existsSync(hereDb)) return [];
  const here = new Database(hereDb, { readonly: true, fileMustExist: true });
  try {
    const errors: string[] = [];
    const a = archived.pragma('user_version', { simple: true }) as number;
    const h = here.pragma('user_version', { simple: true }) as number;
    if (a > h) errors.push(`database version ${a} is newer than this app's ${h}`);
    const mine = columnsOf(here);
    for (const [t, cols] of columnsOf(archived)) {
      const have = mine.get(t);
      if (have) for (const c of cols) if (!have.has(c)) errors.push(`${t}.${c} is unknown to this app`);
    }
    return errors;
  } finally { here.close(); }
}

/** Copy this machine's credential rows the archive does not have, so an
 *  import never drops keys already entered here. */
export function copyCredentialRows(target: Database.Database, hereDb: string): number {
  if (!fs.existsSync(hereDb)) return 0;
  const hasSettings = (db: Database.Database) => !!db.prepare(`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settings'`).get();
  if (!hasSettings(target)) return 0;
  const here = new Database(hereDb, { readonly: true, fileMustExist: true });
  try {
    if (!hasSettings(here)) return 0;
    const ins = target.prepare('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)');
    let n = 0;
    target.transaction(() => {
      for (const r of here.prepare('SELECT key, value FROM settings').all() as Array<{ key: string; value: string }>) {
        if (isCredentialKey(r.key)) n += ins.run(r.key, r.value).changes;
      }
    })();
    return n;
  } finally { here.close(); }
}

/** Keys config.ts resolves against the data folder when relative. */
const DATA_RELATIVE_KEYS = new Set(['ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'DOWNLOAD_DIR', 'LYRICS_EXPORT_DIR', 'TENSORRT_LIBS',
  'TRAINING_DIR', 'WHISPER_EXE', 'WHISPER_MODELS_DIR', 'MUSCRIPTOR_MODELS_DIR']);

/** Path keys that follow a carried category to the new layout's default. */
const CATEGORY_KEYS: Record<Exclude<CategoryId, 'secrets' | 'library'>, string[]> = {
  training: ['TRAINING_DIR'],
  adapters: ['ACESTEPCPP_ADAPTERS'],
  models: ['ACESTEPCPP_MODELS', 'MUSCRIPTOR_MODELS_DIR', 'WHISPER_MODELS_DIR'],
};

/** The new settings/.env. This machine's file first, the archive's values on
 *  top (never its path or machine keys, nor credentials unless the archive
 *  carried secrets). Path keys: DATA_DIR and those of carried categories take
 *  the new layout's default; those of categories left behind, and tools no
 *  category carries, point at this machine's current folders when they exist;
 *  other path keys keep this machine's value, made absolute against
 *  `hereBase` (this machine's data folder) where config would resolve it there. */
export function mergeEnvText(hereText: string, archiveText: string, carried: Manifest['categories'], here: Record<string, string | undefined>, hereBase: string): {
  text: string; fromArchive: string[]; pointedHere: Record<string, string>; dropped: string[];
} {
  const out = new Map<string, string>();
  const fromArchive: string[] = [];
  const pointedHere: Record<string, string> = {};
  const toLayout = new Set(['DATA_DIR']);
  for (const [cat, keys] of Object.entries(CATEGORY_KEYS)) if (carried[cat as CategoryId]) keys.forEach(k => toLayout.add(k));

  // A relative value resolves against the data folder (config.ts resolveDataPath):
  // made absolute against this machine's, it keeps pointing where it points now.
  for (const [k, v] of Object.entries(dotenvParse(hereText))) {
    if (!toLayout.has(k)) out.set(k, DATA_RELATIVE_KEYS.has(k) && v && !path.isAbsolute(v) ? path.resolve(hereBase, v) : v);
  }
  for (const [k, v] of Object.entries(dotenvParse(archiveText))) {
    if (ENV_PATH_KEYS.includes(k) || isMachineKey(k) || (!carried.secrets && isCredentialKey(k))) continue;
    out.set(k, v);
    fromArchive.push(k);
  }
  for (const [k, p] of Object.entries(here)) {
    if (toLayout.has(k) || !p || !fs.existsSync(p)) continue;
    out.set(k, p);
    pointedHere[k] = p;
  }
  const lines: string[] = [];
  const dropped: string[] = [];
  for (const [k, v] of out) {
    const line = /^[A-Za-z_][A-Za-z0-9_]*$/.test(k) ? envLine(k, v) : null;
    if (line === null) dropped.push(k); else lines.push(line);
  }
  return { text: lines.length ? lines.join('\n') + '\n' : '', fromArchive, pointedHere, dropped };
}

/** One {from, to} pair per carried source, most specific first. A training
 *  folder that sat in the library but was not carried maps to itself, so the
 *  library pair leaves its paths alone. */
export function rewritePairs(m: Manifest, target: string): PathPair[] {
  const pairs: PathPair[] = m.sources.map(s => ({ from: s.path, to: path.join(target, ...s.prefix.split('/')) }));
  const lib = m.sources.find(s => s.category === 'library');
  if (lib && !m.categories.training) { const t = path.join(lib.path, 'training'); pairs.push({ from: t, to: t }); }
  return pairs.sort((a, b) => b.from.length - a.from.length);
}

// ── Import ───────────────────────────────────────────────────────────

/** Read the archive and check the target without writing anything. */
export async function importPreview(zipFile: string, target: string | null, here: ImportHere, extra?: (t: string) => string[]) {
  const { zip, manifest } = await openArchive(zipFile);
  zip.close();
  const bytes = manifest.files.reduce((s, f) => s + f.bytes, 0);
  let dest: { path: string; freeBytes: number | null; errors: string[] } | null = null;
  if (target) {
    const errors = targetErrors(target, here);
    if (!errors.length && extra) errors.push(...extra(target));
    const free = path.isAbsolute(target) ? freeBytes(fs.existsSync(target) ? target : path.dirname(target)) : null;
    if (free !== null && free < bytes) errors.push(`not enough free space: ${(bytes / 1e9).toFixed(1)} GB needed, ${(free / 1e9).toFixed(1)} GB free`);
    dest = { path: target, freeBytes: free, errors };
  }
  return {
    archive: {
      createdAt: manifest.createdAt, appVersion: manifest.appVersion, platform: manifest.platform,
      dataRoot: manifest.dataRoot, categories: manifest.categories,
    },
    sources: manifest.sources.map(s => ({ category: s.category, prefix: s.prefix, files: s.files, bytes: s.bytes })),
    files: manifest.files.length, bytes,
    externals: manifest.externals.map(x => ({ ...x, existsHere: fs.existsSync(x.path) })),
    skipped: manifest.skipped.length + manifest.skippedNotListed,
    target: dest,
  };
}
export type ImportPreview = Awaited<ReturnType<typeof importPreview>>;

/** Test seam: runs after each extracted file is closed. */
export const hooks = { afterFile: (_abs: string): void | Promise<void> => {} };

export async function importWorkspace(o: ImportOptions): Promise<ImportReport> {
  const target = path.resolve(o.target);
  const errorsNow = () => {
    const e = targetErrors(o.target, o.here);
    if (!e.length && o.targetErrors) e.push(...o.targetErrors(target));
    return e;
  };
  let pre = errorsNow();
  if (pre.length) throw new ImportError(pre.join('; '), 400, { errors: pre });
  const check = () => { if (o.signal?.aborted) throw new ImportCancelled(); };
  check();

  const { zip, manifest, entries } = await openArchive(o.zipFile);
  const bytesTotal = manifest.files.reduce((s, f) => s + f.bytes, 0);
  const progress: ImportProgress = { phase: 'checking', filesDone: 0, filesTotal: manifest.files.length, bytesDone: 0, bytesTotal };
  const tell = () => o.onProgress?.({ ...progress });
  tell();

  // Everything is built in a staging folder beside the target that only this
  // import creates, then renamed into place once the target is re-checked.
  // Cleanup removes only the staging folder, or after the rename only the
  // entries it moved in. Stored paths name the target, not the staging folder.
  const stage = path.join(path.dirname(target), `.${path.basename(target)}.hs-import-${crypto.randomBytes(4).toString('hex')}`);
  const inStage = (name: string) => path.join(stage, ...name.split('/'));
  let staged = false;
  let moved: string[] | null = null;
  let createdTarget = false;
  let restoreEmpty = false;
  try {
    pre = errorsNow();   // the target may have changed while the archive was read
    if (pre.length) throw new ImportError(pre.join('; '), 409, { errors: pre });
    const free = freeBytes(path.dirname(target));
    if (free !== null && free < bytesTotal) throw new ImportError(`not enough free space at ${target}: ${(bytesTotal / 1e9).toFixed(1)} GB needed, ${(free / 1e9).toFixed(1)} GB free`);
    fs.mkdirSync(stage);
    staged = true;

    // ── Extract and verify ──
    progress.phase = 'extracting'; tell();
    let archiveEnv = '';
    let current: { abort(e: Error): void } | null = null;
    const onAbort = () => current?.abort(new ImportCancelled());
    o.signal?.addEventListener('abort', onAbort);
    try {
      for (const f of manifest.files) {
        check();
        const e = entries.get(f.name)!;
        const hash = crypto.createHash('sha256');
        let n = 0;
        const hasher = new Transform({ transform(chunk, _e, cb) { hash.update(chunk); n += chunk.length; cb(null, chunk); } });
        const abs = f.name === ENV_ENTRY ? null : inStage(f.name);
        if (abs && !isInside(stage, abs)) throw new ImportError(`${f.name} leads outside the new folder`);
        if (abs) fs.mkdirSync(path.dirname(abs), { recursive: true });
        const ac = new AbortController();
        current = ac;
        const rs = await zip.openReadStreamPromise(e);
        if (!abs) {
          const chunks: Buffer[] = [];
          await pipeline(rs, hasher, async function* (src) { for await (const c of src) chunks.push(c); }, { signal: ac.signal });
          archiveEnv = Buffer.concat(chunks).toString('utf-8');
        } else {
          await pipeline(rs, hasher, fs.createWriteStream(abs, { flags: 'wx' }), { signal: ac.signal });
        }
        current = null;
        if (n !== f.bytes || hash.digest('hex') !== f.sha256) throw new ImportError(`${f.name} does not match its hash in the manifest; the archive is damaged`);
        if (abs) await hooks.afterFile(abs);
        progress.filesDone++; progress.bytesDone += n; tell();
      }
    } catch (e: any) {
      if (o.signal?.aborted) throw new ImportCancelled();
      throw e instanceof ImportError ? e : new ImportError(`extracting failed: ${e.message}`, 500);
    } finally {
      o.signal?.removeEventListener('abort', onAbort);
    }
    zip.close();

    // ── Database: integrity, schema, credentials ──
    // openArchive guarantees: manifest.db is set exactly when the zip holds DB_ENTRY,
    // and no other SQLite file (WAL, journal, a second DB) can be in the library.
    progress.phase = 'rewriting'; tell();
    const dbFile = inStage(DB_ENTRY);
    const hasDb = !!manifest.db;
    let credentialRowsCopied = 0;
    if (hasDb) {
      const db = new Database(dbFile);
      try {
        const integrity = db.pragma('integrity_check') as Array<{ integrity_check: string }>;
        if (integrity.length !== 1 || integrity[0].integrity_check !== 'ok') {
          throw new ImportError(`the archive's database fails integrity_check: ${integrity.slice(0, 3).map(r => r.integrity_check).join('; ')}`);
        }
        const schema = schemaErrors(db, o.here.children.db);
        if (schema.length) throw new ImportError(`the archive comes from a newer HOT-Step (${schema.slice(0, 3).join('; ')}); update this app first`, 400, { errors: schema });
        credentialRowsCopied = copyCredentialRows(db, o.here.children.db);
      } finally { db.close(); }
    }
    check();

    // ── Paths ──
    const pairs = rewritePairs(manifest, target);
    let dbRewritten = 0;
    if (hasDb) {
      dbRewritten = rewriteDb(dbFile, pairs, 'import', () => {});
      // rewriteDb keeps a copy of what it changed; here that is the archive itself.
      for (const ext of ['', '-wal', '-shm']) removeSync(path.join(stage, 'library', `hotstep.pre-transfer-import.db${ext}`), { force: true });
    }
    let jsonFiles = 0;
    const roots = [...new Set(pairs.map(p => p.to).filter(t => isInside(target, t)))];
    for (const r of roots.filter(r => !roots.some(o => o !== r && isInside(o, r)))) {
      if (!isInside(path.join(target, 'models'), r)) jsonFiles += rewriteJsonFiles(path.join(stage, path.relative(target, r)), pairs);
    }
    check();

    // ── Settings ──
    progress.phase = 'finishing'; tell();
    const l = dataLayout(stage);
    const c = o.here.children;
    const hereText = fs.existsSync(c.envFile) ? fs.readFileSync(c.envFile, 'utf-8') : '';
    const herePaths: Record<string, string | undefined> = {
      TRAINING_DIR: c.training, ACESTEPCPP_ADAPTERS: c.adapters, ACESTEPCPP_MODELS: c.models,
      MUSCRIPTOR_MODELS_DIR: c.muscriptorModels, ...o.here.paths,
    };
    const env = mergeEnvText(hereText, archiveEnv, manifest.categories, herePaths, o.here.dataRoot ?? path.join(o.here.appRoot, 'server'));
    initLayout(l);
    if (env.text) fs.writeFileSync(l.envPath, env.text, { encoding: 'utf-8', flag: 'wx' });

    const report: ImportReport = {
      importedAt: new Date().toISOString(), zipFile: o.zipFile, target,
      archive: { createdAt: manifest.createdAt, appVersion: manifest.appVersion, platform: manifest.platform, dataRoot: manifest.dataRoot, categories: manifest.categories },
      files: manifest.files.length, bytes: bytesTotal, integrity: hasDb ? 'ok' : 'no database',
      pathsRewritten: { db: dbRewritten, jsonFiles, pairs },
      credentialRowsCopied,
      env: { fromArchive: env.fromArchive, pointedHere: env.pointedHere, dropped: env.dropped },
      externals: manifest.externals.map(x => ({ ...x, existsHere: fs.existsSync(x.path) })),
      skipped: manifest.skipped.length + manifest.skippedNotListed,
      scheduled: !!o.schedule,
    };
    writeJsonAtomic(path.join(stage, REPORT_NAME), report);

    // ── Wait for running work, then move into place ──
    if (o.schedule) {
      for (let b = o.blockers?.() ?? []; b.length; b = o.blockers!()) {
        progress.phase = 'waiting'; progress.blockers = b; tell();
        await new Promise(r => setTimeout(r, o.waitMs ?? 5000));
        check();
      }
      progress.phase = 'finishing'; delete progress.blockers; tell();
    }
    check();
    pre = errorsNow();
    if (pre.length) throw new ImportError(`the new folder was checked again before the switch: ${pre.join('; ')}`, 409, { errors: pre });
    const names = fs.readdirSync(stage);
    createdTarget = !fs.existsSync(target);
    // rmdir refuses a folder that gained an entry; rename refuses a target that reappeared.
    if (!createdTarget) { fs.rmdirSync(target); restoreEmpty = true; }
    fs.renameSync(stage, target);
    staged = false;
    moved = names;

    o.schedule?.(target);
    return report;
  } catch (e) {
    zip.close();
    // Retries: Windows can hold a just-destroyed write stream's file for a moment.
    const rm = (p: string) => removeSync(p, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    try {
      if (staged) rm(stage);
      if (moved) {
        for (const n of moved) rm(path.join(target, n));
        if (createdTarget && isEmptyDir(target)) fs.rmdirSync(target);
      }
      if (restoreEmpty && !fs.existsSync(target)) fs.mkdirSync(target);
    } catch (c: any) {
      if (e instanceof Error) e.message += `; the half-written folder ${staged ? stage : target} could not be removed (${c.message}): delete it by hand`;
    }
    throw e;
  }
}
