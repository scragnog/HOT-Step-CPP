// dataroot/paths.ts — Data-root layout, profile pointer and legacy-install
// inspection. A release (HOT_STEP_ROOT set) uses data-location.json; a dev
// checkout uses its own data-location.dev-<hash>.json and only once Settings
// has scheduled something into it.
//
// Pure helpers only: nothing here runs at import. bootstrap.ts decides which
// root is active, transfer.ts moves data, pointer.ts is the Settings API.

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { parse as dotenvParse } from 'dotenv';

const BIN_EXT = process.platform === 'win32' ? '.exe' : '';
const CASE_INSENSITIVE = process.platform === 'win32' || process.platform === 'darwin';

/** Names the release ships in server/data. A stock release's .env pointed
 *  DATA_DIR at that same folder, so migration must leave these behind.
 *  ponytail: fixed list; a new bundled resource in server/src/data must be added here. */
export const BUNDLED_DATA_NAMES = ['assistant-knowledge.md', 'mm3-corpus.json', 'mm3-dit-trt.fp8.json', 'mm3-reg-corpus', 'model-registry.json'];

export const DEFAULT_DATA_FOLDER = 'HOT-Step-Data';

// ── Path boundaries ──────────────────────────────────────────────────

function norm(p: string): string {
  const r = path.resolve(p);
  return CASE_INSENSITIVE ? r.toLowerCase() : r;
}

/** True when `child` is `parent` or lies under it, at a path boundary. */
export function isInside(parent: string, child: string): boolean {
  const rel = path.relative(norm(parent), norm(child));
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

export function samePath(a: string, b: string): boolean {
  return norm(a) === norm(b);
}

export interface PathPair { from: string; to: string }

/** Map an absolute path inside one of `pairs[].from` to the same place under
 *  `to`. Returns null for anything else (relative, URL, outside). */
export function rewritePath(value: string, pairs: PathPair[]): string | null {
  if (typeof value !== 'string' || value.length < 3 || !path.isAbsolute(value) || /^[a-z]+:\/\//i.test(value)) return null;
  for (const { from, to } of pairs) {
    if (!isInside(from, value)) continue;
    const rel = path.relative(norm(from), norm(value));
    // Rebuild from the original spelling so case survives on Windows.
    const tail = rel === '' ? '' : path.resolve(value).slice(path.resolve(value).length - rel.length);
    return tail ? path.join(to, tail) : to;
  }
  return null;
}

// ── Layout ───────────────────────────────────────────────────────────

export interface DataLayout {
  root: string;
  library: string;
  models: string;
  adapters: string;
  settingsDir: string;
  envPath: string;
  logs: string;
  tools: string;
  whisperDir: string;
  runtimeLibDir: string;
  plugins: string;
  tmp: string;
  cache: string;
  marker: string;
}

export function dataLayout(root: string): DataLayout {
  const r = path.resolve(root);
  return {
    root: r,
    library: path.join(r, 'library'),
    models: path.join(r, 'models'),
    adapters: path.join(r, 'adapters'),
    settingsDir: path.join(r, 'settings'),
    envPath: path.join(r, 'settings', '.env'),
    logs: path.join(r, 'logs'),
    tools: path.join(r, 'tools'),
    whisperDir: path.join(r, 'tools', 'whisper'),
    runtimeLibDir: path.join(r, 'tools', 'runtime'),
    plugins: path.join(r, 'plugins'),
    tmp: path.join(r, 'tmp'),
    cache: path.join(r, 'cache'),
    marker: path.join(r, 'hot-step-data.json'),
  };
}

/** Create the folders and marker of a layout (idempotent). */
export function initLayout(l: DataLayout): void {
  for (const d of [l.library, l.models, l.adapters, l.settingsDir, l.logs, l.whisperDir, l.runtimeLibDir, l.plugins, l.tmp, l.cache]) {
    fs.mkdirSync(d, { recursive: true });
  }
  if (!fs.existsSync(l.marker)) writeJsonAtomic(l.marker, { version: 1, createdAt: new Date().toISOString() });
}

/** A folder Repoint may accept: it has the marker or an existing library DB. */
export function isDataLayout(root: string): boolean {
  const l = dataLayout(root);
  return fs.existsSync(l.marker) || fs.existsSync(path.join(l.library, 'hotstep.db'));
}

export function isEmptyDir(dir: string): boolean {
  try { return fs.readdirSync(dir).length === 0; } catch { return !fs.existsSync(dir); }
}

/** Defaults a release .env line can be dropped in favour of. */
export function layoutDefaults(l: DataLayout): Record<string, string> {
  return {
    DATA_DIR: l.library,
    ACESTEPCPP_MODELS: l.models,
    ACESTEPCPP_ADAPTERS: l.adapters,
    TRAINING_DIR: path.join(l.library, 'training'),
    LYRICS_EXPORT_DIR: path.join(l.library, 'lyrics'),
    MUSCRIPTOR_MODELS_DIR: path.join(l.models, 'muscriptor'),
    TENSORRT_LIBS: l.runtimeLibDir,
    WHISPER_EXE: path.join(l.whisperDir, `whisper-cli${BIN_EXT}`),
    WHISPER_MODELS_DIR: path.join(l.models, 'whisper'),
  };
}

/** .env keys that hold a path. */
export const ENV_PATH_KEYS = [
  'DATA_DIR', 'ACESTEPCPP_MODELS', 'ACESTEPCPP_ADAPTERS', 'TRAINING_DIR', 'LYRICS_EXPORT_DIR',
  'MUSCRIPTOR_MODELS_DIR', 'TENSORRT_LIBS', 'WHISPER_EXE', 'WHISPER_MODELS_DIR',
  'ACESTEPCPP_EXE', 'ESSENTIA_BIN', 'ACESTEPCPP_NOISE_PROFILE', 'ACESTEPCPP_DRAFT_LM',
];
/** Keys that point at app binaries: an old install's value must not follow the data. */
const APP_BINARY_KEYS = new Set(['ACESTEPCPP_EXE', 'ESSENTIA_BIN', 'ACESTEPCPP_NOISE_PROFILE']);

// ── Profile pointer ──────────────────────────────────────────────────

export function profileDir(env: NodeJS.ProcessEnv = process.env, platform: string = process.platform, home: string = os.homedir()): string {
  if (env.HOT_STEP_PROFILE_DIR) return path.resolve(env.HOT_STEP_PROFILE_DIR);
  if (platform === 'win32') return path.join(env.LOCALAPPDATA || path.join(home, 'AppData', 'Local'), 'HOT-Step');
  if (platform === 'darwin') return path.join(home, 'Library', 'Application Support', 'HOT-Step');
  return path.join(env.XDG_CONFIG_HOME || path.join(home, '.config'), 'hot-step');
}

export const POINTER_FILE = 'data-location.json';

/** A dev checkout's own pointer, keyed to the checkout folder so it never
 *  shares the release pointer or another checkout's. */
export function devPointerFile(prof: string, checkout: string): string {
  return path.join(prof, `data-location.dev-${crypto.createHash('sha256').update(norm(checkout)).digest('hex').slice(0, 12)}.json`);
}

/** Roots (and pending destinations) named by every pointer in `prof` except
 *  `except`, so one install never picks a folder another one uses. */
export function otherPointerRoots(prof: string, except: string): string[] {
  const out: string[] = [];
  let names: string[] = [];
  try { names = fs.readdirSync(prof); } catch { return out; }
  for (const n of names.filter(n => /^data-location(\.dev-[0-9a-f]+)?\.json$/.test(n))) {
    const f = path.join(prof, n);
    if (samePath(f, except)) continue;
    try {
      const p = readPointer(f);
      if (p?.root) out.push(p.root);
      if (p?.pending && p.pending.kind !== 'import') out.push(p.pending.destination);
    } catch { /* unreadable: its own install reports it */ }
  }
  return out;
}
export const LOCK_FILE = 'data-transfer.lock';
export const RESTART_MARKER_FILE = 'restart-requested';

export type OperationKind = 'move' | 'repoint' | 'create' | 'import' | 'migrate';
export type MappingState = 'pending' | 'moved' | 'copied' | 'cleaned';

export interface Mapping {
  name: string;
  from: string;
  to: string;
  /** Top-level names in `from` that stay behind (bundled resources). */
  exclude?: string[];
  state: MappingState;
  /** Journaled before the first file is touched. Only a rename removes its
   *  source (once every entry has moved), so only a rename may finish with
   *  the source gone. */
  method?: 'rename' | 'copy';
}

export interface PendingOperation {
  id: string;
  kind: OperationKind;
  /** Current root (move), old app folder (migrate/import), or null. */
  source: string | null;
  destination: string;
  createdAt: string;
  mappings: Mapping[];
  /** Source .env to carry into settings/.env (migrate/import). */
  envSource: string | null;
  /** Base for relative values in envSource. */
  envBase: string | null;
  /** DB to rewrite once the data has moved. */
  dbPath: string | null;
  /** Extra single files to copy (import: plugins, runtime DLLs). */
  copies: PathPair[];
  rewritten: boolean;
  /** settings/.env is final for this operation (the step is not repeated). */
  envDone?: boolean;
  /** Import only: a blank library (fresh install, no songs) renamed aside first. */
  setAside?: string[];
  setAsideDone?: boolean;
  /** The source is this dev checkout, which stays the app: its .env binary
   *  paths are kept and its real sources are checked against git again
   *  before anything moves. */
  devCheckout?: string;
  /** Set by hand (see the error page) to undo a same-volume move. */
  abandon?: boolean;
  error: string | null;
}

export interface Pointer {
  version: 1;
  root: string | null;
  pending: PendingOperation | null;
  /** Dev pointers only: the checkout this file belongs to (for people reading it). */
  checkout?: string;
}

export function writeJsonAtomic(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + '\n', 'utf-8');
  fs.renameSync(tmp, file);
}

/** null when there is no pointer; throws when it exists but is not usable. */
export function readPointer(file: string): Pointer | null {
  if (!fs.existsSync(file)) return null;
  const raw = fs.readFileSync(file, 'utf-8');
  let p: any;
  try { p = JSON.parse(raw); } catch { throw new Error(`${file} is not valid JSON`); }
  if (!p || p.version !== 1 || (p.root !== null && typeof p.root !== 'string') || (p.pending !== null && typeof p.pending !== 'object')) {
    throw new Error(`${file} is not a version 1 data-location file`);
  }
  if (p.pending && (!Array.isArray(p.pending.mappings) || typeof p.pending.destination !== 'string')) {
    throw new Error(`${file} has an unreadable pending operation`);
  }
  return { version: 1, root: p.root ?? null, pending: p.pending ?? null, ...(typeof p.checkout === 'string' ? { checkout: p.checkout } : {}) };
}

export function writePointer(file: string, p: Pointer): void {
  writeJsonAtomic(file, p);
}

export function newPending(kind: OperationKind, source: string | null, destination: string, extra: Partial<PendingOperation> = {}): PendingOperation {
  return {
    id: crypto.randomUUID(), kind, source, destination: path.resolve(destination), createdAt: new Date().toISOString(),
    mappings: [], envSource: null, envBase: null, dbPath: null, copies: [], rewritten: false, error: null, ...extra,
  };
}

// ── Legacy install inspection ────────────────────────────────────────

export interface LegacyInstall {
  appRoot: string;
  /** What relative .env values resolve against. */
  base: string;
  /** A dev checkout: server/data is its library, not bundled resources. */
  dev?: boolean;
  envFile: string | null;
  env: Record<string, string>;
  /** Effective DATA_DIR of that install. */
  dataDir: string;
  models: string;
  adapters: string;
  logs: string;
  whisperDir: string;
}

/** Read an install's .env and resolve its paths the way the old bundle did:
 *  DATA_DIR against <app> (bundle __dirname is <app>/server, plus '..'),
 *  everything else against <app> (the launcher's usual cwd). */
export function inspectLegacy(appRoot: string): LegacyInstall {
  const app = path.resolve(appRoot);
  const envFile = path.join(app, '.env');
  const env = fs.existsSync(envFile) ? dotenvParse(fs.readFileSync(envFile, 'utf-8')) : {};
  const at = (v: string | undefined, d: string) => (v ? path.resolve(app, v) : d);
  return {
    appRoot: app,
    base: app,
    envFile: fs.existsSync(envFile) ? envFile : null,
    env,
    dataDir: path.resolve(app, env.DATA_DIR || './data'),
    models: at(env.ACESTEPCPP_MODELS, path.join(app, 'models')),
    adapters: at(env.ACESTEPCPP_ADAPTERS, path.join(app, 'adapters')),
    logs: path.join(app, 'logs'),
    whisperDir: env.WHISPER_EXE ? path.dirname(path.resolve(app, env.WHISPER_EXE)) : path.join(app, 'tools', 'whisper'),
  };
}

/** A dev checkout's paths, resolved the way config.ts does without a data
 *  folder: DATA_DIR against server/, other relative values against the
 *  server's cwd (server/, where dev.bat starts it), logs and Whisper under
 *  the checkout. */
export function inspectDevLegacy(checkout: string): LegacyInstall {
  const app = path.resolve(checkout);
  const base = path.join(app, 'server');
  const envFile = path.join(app, '.env');
  const env = fs.existsSync(envFile) ? dotenvParse(fs.readFileSync(envFile, 'utf-8')) : {};
  const at = (v: string | undefined, d: string) => (v ? path.resolve(base, v) : d);
  return {
    appRoot: app,
    base,
    dev: true,
    envFile: fs.existsSync(envFile) ? envFile : null,
    env,
    dataDir: path.resolve(base, env.DATA_DIR || './data'),
    models: at(env.ACESTEPCPP_MODELS, path.join(app, 'models')),
    adapters: at(env.ACESTEPCPP_ADAPTERS, path.join(app, 'adapters')),
    logs: path.join(app, 'logs'),
    whisperDir: env.WHISPER_EXE ? path.dirname(path.resolve(base, env.WHISPER_EXE)) : path.join(app, 'tools', 'whisper'),
  };
}

function hasContent(dir: string, exclude: string[] = []): boolean {
  try {
    const skip = new Set(exclude.map(n => n.toLowerCase()));
    return fs.readdirSync(dir).some(n => !skip.has(n.toLowerCase()));
  } catch { return false; }
}

/** Mappings that carry `legacy`'s data into `destRoot`. Only folders inside the
 *  old app move; folders it already kept elsewhere stay external. A mapping
 *  nested in another is dropped: the outer move carries it. */
export function legacyMappings(legacy: LegacyInstall, destRoot: string): Mapping[] {
  const l = dataLayout(destRoot);
  const bundled = !legacy.dev && samePath(legacy.dataDir, path.join(legacy.appRoot, 'server', 'data')) ? BUNDLED_DATA_NAMES : undefined;
  const candidates: Mapping[] = [
    { name: 'library', from: legacy.dataDir, to: l.library, exclude: bundled, state: 'pending' },
    { name: 'models', from: legacy.models, to: l.models, state: 'pending' },
    { name: 'adapters', from: legacy.adapters, to: l.adapters, state: 'pending' },
    { name: 'logs', from: legacy.logs, to: l.logs, state: 'pending' },
    { name: 'whisper', from: legacy.whisperDir, to: l.whisperDir, state: 'pending' },
  ].filter(m => isInside(legacy.appRoot, m.from) && !samePath(legacy.appRoot, m.from) && hasContent(m.from, m.exclude)) as Mapping[];
  return candidates.filter(m => !candidates.some(o => o !== m && isInside(o.from, m.from) && !samePath(o.from, m.from)));
}

/** True when an old install in `appRoot` has data worth migrating. */
export function hasLegacyData(appRoot: string): boolean {
  const legacy = inspectLegacy(appRoot);
  return legacyMappings(legacy, path.join(path.dirname(legacy.appRoot), DEFAULT_DATA_FOLDER)).length > 0;
}

/** Turn an old install's .env into the new settings/.env text. Relative values
 *  become absolute against `base`; app-binary keys from another app are
 *  dropped; values that land on the new layout's default are dropped. */
export function migrateEnvText(text: string, base: string, pairs: PathPair[], l: DataLayout, dropAppBinariesUnder: string | null): string {
  const defaults = layoutDefaults(l);
  const out: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)$/);
    if (!m || !ENV_PATH_KEYS.includes(m[1])) { out.push(line); continue; }
    const key = m[1];
    const raw = m[2].trim().replace(/^(['"])(.*)\1$/, '$2');
    if (!raw) { out.push(line); continue; }
    const abs = path.resolve(base, raw);
    if (APP_BINARY_KEYS.has(key) && dropAppBinariesUnder && isInside(dropAppBinariesUnder, abs)) continue;
    const mapped = rewritePath(abs, pairs) ?? abs;
    if (defaults[key] && samePath(defaults[key], mapped)) continue;
    out.push(`${key}=${mapped}`);
  }
  return out.join('\n');
}

/** Rewrite path values already in a release .env (a move between roots). */
export function rewriteEnvText(text: string, pairs: PathPair[], l: DataLayout): string {
  return migrateEnvText(text, l.root, pairs, l, null);
}
