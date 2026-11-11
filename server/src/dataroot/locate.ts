// dataroot/locate.ts — Where is the data folder? Read-only, safe to import anywhere.
//
// Nothing here writes, creates a folder, applies a pending move or opens a DB.
// bootstrap.ts uses pointerLocation() to find the pointer before its side
// effects; agents and tools/mcp-lyricstudio use locateData() to read the
// effective root and children without a running app (server/scripts/data-paths.ts).

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { parse as dotenvParse } from 'dotenv';
import {
  DEFAULT_DATA_FOLDER, POINTER_FILE, dataLayout, devPointerFile, profileDir, readPointer,
  type PendingOperation, type Pointer,
} from './paths.js';

/** server/src/dataroot -> checkout root. */
export const CHECKOUT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

export interface PointerLocation {
  mode: 'dev' | 'release';
  /** HOT_STEP_ROOT in a release, the checkout in dev. */
  appRoot: string;
  /** The folder a user unzipped: HOT_STEP_INSTALL_ROOT under the stable launcher
   *  (appRoot is then its app/<id> payload), else appRoot. The default data
   *  folder sits beside it and an old version's in-app data sits inside it. */
  installRoot: string;
  profileDir: string;
  pointerPath: string;
}

/** Which pointer file this process reads (no I/O). */
export function pointerLocation(env: NodeJS.ProcessEnv = process.env, checkoutRoot: string = CHECKOUT_ROOT): PointerLocation {
  const prof = profileDir(env);
  if (!env.HOT_STEP_ROOT) {
    const checkout = path.resolve(checkoutRoot);
    return { mode: 'dev', appRoot: checkout, installRoot: checkout, profileDir: prof, pointerPath: devPointerFile(prof, checkout) };
  }
  const appRoot = path.resolve(env.HOT_STEP_ROOT);
  const installRoot = env.HOT_STEP_INSTALL_ROOT ? path.resolve(env.HOT_STEP_INSTALL_ROOT) : appRoot;
  return { mode: 'release', appRoot, installRoot, profileDir: prof, pointerPath: path.join(prof, POINTER_FILE) };
}

export function accessibleRoot(root: string): boolean {
  try {
    if (!fs.statSync(root).isDirectory()) return false;
    fs.accessSync(root, fs.constants.R_OK | fs.constants.W_OK);
    return true;
  } catch { return false; }
}

/** Shared boot/inspection selection after pending work. Never creates the fallback. */
export function selectPointerRoot(loc: Pick<PointerLocation, 'mode' | 'installRoot'>, savedRoot: string): { root: string | null; staleRoot: string | null } {
  if (accessibleRoot(savedRoot)) return { root: path.resolve(savedRoot), staleRoot: null };
  return {
    root: loc.mode === 'release' ? path.join(path.dirname(loc.installRoot), DEFAULT_DATA_FOLDER) : null,
    staleRoot: savedRoot,
  };
}

export interface DataChildren {
  library: string;
  db: string;
  audio: string;
  logs: string;
  models: string;
  adapters: string;
  training: string;
  /** MuScriptor working data (<library>/muscriptor). */
  muscriptor: string;
  /** Where MuScriptor downloads go: MUSCRIPTOR_MODELS_DIR, else <models>/muscriptor. */
  muscriptorModels: string;
  /** Pre-#228 <library>/models/muscriptor, still read when complete; null under an override. */
  muscriptorLegacyModels: string | null;
  plugins: string | null;
  envFile: string;
}

export interface DataReport extends PointerLocation {
  /** The data folder, or null: dev checkout with no folder (legacy checkout paths) or a release before first start. */
  root: string | null;
  /** What chose the root. */
  source: 'HOT_STEP_DATA_ROOT' | 'pointer' | 'fallback' | 'checkout' | 'none';
  /** A scheduled or half-finished move/import/migration. While set, do not open the DB. */
  pendingMove: Pick<PendingOperation, 'kind' | 'source' | 'destination' | 'error'> | null;
  /** The pointer exists but cannot be read. */
  pointerError: string | null;
  /** Children with .env path overrides applied; null when nothing is resolvable yet. */
  children: DataChildren | null;
  /** True when a child differs from the plain <root>/... layout because of an override. */
  overrides: string[];
}

function envFileValues(file: string): Record<string, string> {
  try { return fs.existsSync(file) ? dotenvParse(fs.readFileSync(file, 'utf-8')) : {}; } catch { return {}; }
}

/** Resolve the effective paths for a root the way config.ts does:
 *  process.env beats the .env file; relative values resolve against `base`. */
function children(root: string | null, checkout: string, env: NodeJS.ProcessEnv): { c: DataChildren; overrides: string[] } {
  const overrides: string[] = [];
  const l = root ? dataLayout(root) : null;
  const envFile = l ? l.envPath : path.join(checkout, '.env');
  const file = envFileValues(envFile);
  // config.ts: dotenv never overrides a variable the process already has, even an empty one.
  const val = (k: string) => (k in env && env[k] !== undefined ? env[k]! : file[k]) ?? '';
  const base = root ?? path.join(checkout, 'server');
  const pick = (key: string, fallback: string, rel = base): string => {
    const v = val(key);
    if (!v) return fallback;
    const abs = path.resolve(rel, v);
    if (abs !== path.resolve(fallback)) overrides.push(key);
    return abs;
  };
  const library = pick('DATA_DIR', l ? l.library : path.resolve(base, 'data'));
  const models = pick('ACESTEPCPP_MODELS', l ? l.models : path.join(checkout, 'models'), root ?? path.join(checkout, 'server'));
  const adapters = pick('ACESTEPCPP_ADAPTERS', l ? l.adapters : path.join(checkout, 'adapters'), root ?? path.join(checkout, 'server'));
  return {
    overrides,
    c: {
      library, db: path.join(library, 'hotstep.db'), audio: path.join(library, 'audio'),
      logs: l ? l.logs : path.join(checkout, 'logs'),
      models, adapters,
      training: pick('TRAINING_DIR', path.join(library, 'training')),
      muscriptor: path.join(library, 'muscriptor'),
      muscriptorModels: pick('MUSCRIPTOR_MODELS_DIR', path.join(models, 'muscriptor')),
      muscriptorLegacyModels: val('MUSCRIPTOR_MODELS_DIR') ? null : path.join(library, 'models', 'muscriptor'),
      plugins: l ? l.plugins : null,
      envFile,
    },
  };
}

/** Report the active data root and its children. Read-only. */
export function locateData(opts: { env?: NodeJS.ProcessEnv; checkoutRoot?: string } = {}): DataReport {
  const env = opts.env ?? process.env;
  const loc = pointerLocation(env, opts.checkoutRoot);
  let ptr: Pointer | null = null;
  let pointerError: string | null = null;
  try { ptr = readPointer(loc.pointerPath); } catch (e: any) { pointerError = e.message; }

  let root: string | null = null;
  let source: DataReport['source'] = 'none';
  if (loc.mode === 'release' && env.HOT_STEP_DATA_ROOT) { root = path.resolve(env.HOT_STEP_DATA_ROOT); source = 'HOT_STEP_DATA_ROOT'; }
  else if (ptr?.root) {
    // A transfer must finish before its effective root can be known.
    const selected = ptr.pending ? { root: path.resolve(ptr.root), staleRoot: null } : selectPointerRoot(loc, ptr.root);
    root = selected.root;
    source = selected.staleRoot ? 'fallback' : 'pointer';
  }
  else if (loc.mode === 'dev') source = 'checkout';

  const pendingMove = ptr?.pending
    ? { kind: ptr.pending.kind, source: ptr.pending.source, destination: ptr.pending.destination, error: ptr.pending.error }
    : null;
  const resolvable = root !== null || loc.mode === 'dev';
  const r = resolvable ? children(root, loc.appRoot, env) : null;
  return { ...loc, root, source, pendingMove, pointerError, children: r?.c ?? null, overrides: r?.overrides ?? [] };
}
