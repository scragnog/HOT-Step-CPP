// dataroot/bootstrap.ts — Decide the data root before anything else loads.
//
// config.ts imports RUNTIME from here, so this module is evaluated before any
// module that opens SQLite, log files or the engine. In a release it picks the
// root (HOT_STEP_DATA_ROOT, profile pointer, this install's old layout, then
// <parent of install>/HOT-Step-Data), runs a pending move/import/migration, and on
// transfer or pointer failure serves one plain page and never lets the app start.
// An unavailable saved root falls back without changing the pointer.
// A dev checkout (no HOT_STEP_ROOT) keeps its checkout paths until Settings
// writes its own dev pointer; from then on it works like a release, except
// that nothing is migrated automatically and restart keeps the dev marker.

import { removeSync } from '../services/fsRemove.js';
import fs from 'fs';
import http from 'http';
import net from 'net';
import path from 'path';
import { parse as dotenvParse } from 'dotenv';
import {
  DEFAULT_DATA_FOLDER, LOCK_FILE, POINTER_FILE, RESTART_MARKER_FILE,
  dataLayout, initLayout, isDataLayout, isEmptyDir, legacyMappings, inspectLegacy,
  readPointer, writePointer,
  type Pointer,
} from './paths.js';
import { planOperation, runPending, sharedRoot } from './transfer.js';
import { CHECKOUT_ROOT, accessibleRoot, pointerLocation, selectPointerRoot } from './locate.js';
import { UPDATER_DIR_ENV, leaseForStartup } from '../services/updates/maintenance.js';

export interface Runtime {
  mode: 'dev' | 'release';
  /** The unpacked app folder (HOT_STEP_ROOT), or the dev checkout. */
  appRoot: string | null;
  dataRoot: string | null;
  profileDir: string | null;
  pointerPath: string | null;
  /** Set when HOT_STEP_DATA_ROOT chose the root. */
  override: string | null;
  envPath: string | null;
  logsDir: string | null;
  restartMarkerPath: string | null;
  libraryDir: string | null;
  modelsDir: string | null;
  adaptersDir: string | null;
  whisperDir: string | null;
  runtimeLibDir: string | null;
  userPluginsDir: string | null;
}

/** Dev with no data folder: every path null, so config.ts keeps the checkout
 *  defaults. With one: the release layout, but runtime DLLs stay beside the
 *  dev engine build and restart keeps the checkout's marker for the dev wrapper. */
function dev(checkout: string, prof: string, ptrFile: string, root: string | null): Runtime {
  const base: Runtime = {
    mode: 'dev', appRoot: checkout, dataRoot: null, profileDir: prof, pointerPath: ptrFile, override: null, envPath: null,
    logsDir: null, restartMarkerPath: null, libraryDir: null, modelsDir: null, adaptersDir: null, whisperDir: null,
    runtimeLibDir: null, userPluginsDir: null,
  };
  if (!root) return base;
  const l = dataLayout(root);
  return {
    ...base, dataRoot: l.root, envPath: l.envPath, logsDir: l.logs, libraryDir: l.library, modelsDir: l.models,
    adaptersDir: l.adapters, whisperDir: l.whisperDir, userPluginsDir: l.plugins,
  };
}

/** `lease`: an update holds this install or data folder; the process exits
 *  rather than serve the error page on a port the update will need. */
export type Resolution = { ok: true; runtime: Runtime } | { ok: false; error: string; pointerPath: string | null; lease?: true };

/** Another process's maintenance lease on the updater folder or `root`.
 *  Checked before any SQLite or bootstrap work; a no-op with no lease on disk. */
function leaseRefusal(env: NodeJS.ProcessEnv, root: string | null, pointerPath: string | null): Resolution | null {
  try { leaseForStartup({ updaterDir: env[UPDATER_DIR_ENV] || null, dataRoot: root }, env); return null; }
  catch (e: any) { return { ok: false, error: String(e?.message ?? e), pointerPath, lease: true }; }
}

function release(appRoot: string, root: string, prof: string, override: string | null): Runtime {
  const l = dataLayout(root);
  return {
    mode: 'release', appRoot, dataRoot: l.root, profileDir: prof, pointerPath: path.join(prof, POINTER_FILE), override,
    envPath: l.envPath, logsDir: l.logs, restartMarkerPath: path.join(prof, RESTART_MARKER_FILE), libraryDir: l.library,
    modelsDir: l.models, adaptersDir: l.adapters, whisperDir: l.whisperDir, runtimeLibDir: l.runtimeLibDir, userPluginsDir: l.plugins,
  };
}

/** Exclusive lock beside the pointer; a lock whose PID is gone is cleared. */
function withLock<T>(prof: string, fn: () => T): T {
  const lock = path.join(prof, LOCK_FILE);
  fs.mkdirSync(prof, { recursive: true });
  for (let attempt = 0; ; attempt++) {
    try {
      fs.writeFileSync(lock, String(process.pid), { flag: 'wx' });
      break;
    } catch (e: any) {
      if (e.code !== 'EEXIST' || attempt > 0) throw new Error(`another HOT-Step is moving data (lock ${lock})`);
      const pid = parseInt(fs.readFileSync(lock, 'utf-8'), 10);
      let alive = false;
      try { if (pid) { process.kill(pid, 0); alive = true; } } catch (k: any) { alive = k.code === 'EPERM'; }
      if (alive) throw new Error(`another HOT-Step (PID ${pid}) is moving data`);
      removeSync(lock, { force: true });
    }
  }
  try { return fn(); } finally { removeSync(lock, { force: true }); }
}

function portInUse(port: number, host = '127.0.0.1'): Promise<boolean> {
  return new Promise(resolve => {
    const s = net.connect({ port, host });
    s.setTimeout(500);
    s.once('connect', () => { s.destroy(); resolve(true); });
    s.once('timeout', () => { s.destroy(); resolve(false); });
    s.once('error', () => resolve(false));
  });
}

/** The engine port of whichever .env the old or current layout uses. */
function enginePort(envFile: string | null): number {
  try { if (envFile && fs.existsSync(envFile)) return parseInt(dotenvParse(fs.readFileSync(envFile, 'utf-8')).ACESTEPCPP_PORT || '8085', 10); } catch { /* default */ }
  return 8085;
}

export interface ResolveOptions {
  env?: NodeJS.ProcessEnv;
  log?: (msg: string) => void;
  /** Skip the "engine still running" probe (tests). */
  skipEngineProbe?: boolean;
  /** Dev checkout root (tests); defaults to the checkout this file is in. */
  checkoutRoot?: string;
}

/** Run ptr.pending (if any) under the transfer lock. Returns an error or null. */
async function runPendingAtBoot(ptrFile: string, ptr: Pointer, appRoot: string, prof: string, opts: ResolveOptions, log: (m: string) => void): Promise<string | null> {
  const op = ptr.pending;
  if (!op) return null;
  const shared = op.abandon ? null : sharedRoot(prof, ptrFile, op.destination);
  if (shared) return `${op.destination} is the same folder as ${shared}, which another HOT-Step install uses. Cancel this change in ${ptrFile}, then restart.`;
  if (!opts.skipEngineProbe) {
    const envFile = op.envSource ?? (ptr.root ? dataLayout(ptr.root).envPath : null);
    const port = enginePort(envFile);
    if (await portInUse(port)) return `Something is still listening on the engine port ${port}. Close every HOT-Step window, then restart.`;
  }
  try {
    withLock(prof, () => runPending(ptrFile, ptr, appRoot, log));
  } catch (e: any) {
    return `Moving data to ${op.destination} stopped: ${e.message}. Nothing was overwritten; restart to try again.`;
  }
  return null;
}

/** Dev: no pointer (or one left with neither root nor work) keeps the
 *  checkout paths. Nothing is migrated unless Settings scheduled it. */
async function resolveDev(opts: ResolveOptions, env: NodeJS.ProcessEnv, log: (m: string) => void): Promise<Resolution> {
  const { appRoot: checkout, profileDir: prof, pointerPath: ptrFile } = pointerLocation(env, opts.checkoutRoot ?? CHECKOUT_ROOT);
  const fail = (error: string): Resolution => ({ ok: false, error, pointerPath: ptrFile });
  try {
    let ptr: Pointer | null;
    try { ptr = readPointer(ptrFile); } catch (e: any) { return fail(`${e.message}. Fix or delete it, then restart.`); }
    if (!ptr || (!ptr.root && !ptr.pending)) return { ok: true, runtime: dev(checkout, prof, ptrFile, null) };
    const held = leaseRefusal(env, ptr.root, ptrFile);
    if (held) return held;
    const err = await runPendingAtBoot(ptrFile, ptr, checkout, prof, opts, log);
    if (err) return fail(err);
    if (!ptr.root) return { ok: true, runtime: dev(checkout, prof, ptrFile, null) }; // abandoned before any root
    const selected = selectPointerRoot({ mode: 'dev', installRoot: checkout }, ptr.root);
    if (selected.staleRoot) {
      log(`[Data] WARNING: Saved data folder ${ptr.root} is missing or cannot be accessed. Using this checkout's default folders in ${checkout}. Saved location unchanged; reconnect the drive and restart to use it again.`);
      return { ok: true, runtime: dev(checkout, prof, ptrFile, null) };
    }
    const sharedWith = sharedRoot(prof, ptrFile, ptr.root);
    if (sharedWith) return fail(`The data folder ${ptr.root} is the same folder as ${sharedWith}, which another HOT-Step install uses. Point this one elsewhere in ${ptrFile}, then restart.`);
    initLayout(dataLayout(ptr.root));
    log(`[Data] Dev data folder: ${ptr.root}`);
    return { ok: true, runtime: dev(checkout, prof, ptrFile, ptr.root) };
  } catch (e: any) {
    return fail(String(e?.message ?? e));
  }
}

/** Pick the data root and run whatever transfer is pending. */
export async function resolveRuntime(opts: ResolveOptions = {}): Promise<Resolution> {
  const env = opts.env ?? process.env;
  const log = opts.log ?? (m => console.log(m));
  if (!env.HOT_STEP_ROOT) return resolveDev(opts, env, log);

  const { appRoot, installRoot, profileDir: prof, pointerPath: ptrFile } = pointerLocation(env);
  const fail = (error: string): Resolution => ({ ok: false, error, pointerPath: ptrFile });

  try {
    const held = leaseRefusal(env, null, ptrFile);
    if (held) return held;
    if (env.HOT_STEP_DATA_ROOT) {
      const root = path.resolve(env.HOT_STEP_DATA_ROOT);
      if (!fs.existsSync(root)) return fail(`HOT_STEP_DATA_ROOT is ${root}, which does not exist.`);
      const heldRoot = leaseRefusal(env, root, ptrFile);
      if (heldRoot) return heldRoot;
      initLayout(dataLayout(root));
      log(`[Data] Using HOT_STEP_DATA_ROOT: ${root}`);
      return { ok: true, runtime: release(appRoot, root, prof, root) };
    }

    let ptr: Pointer | null;
    try { ptr = readPointer(ptrFile); } catch (e: any) { return fail(`${e.message}. Fix or delete it, then restart.`); }

    if (!ptr) {
      // First run of the new layout on this machine. Under the stable launcher
      // the old version's data is in the install folder, not in app/<id>: a
      // versioned package unzipped over an old one leaves it there.
      const sibling = path.join(path.dirname(installRoot), DEFAULT_DATA_FOLDER);
      if (sharedRoot(prof, ptrFile, sibling)) {
        return fail(`${sibling} is a development checkout's data folder. Set HOT_STEP_DATA_ROOT, or move that checkout's data elsewhere, then restart.`);
      }
      const legacy = inspectLegacy(installRoot);
      const hasLegacy = legacyMappings(legacy, sibling).length > 0 || !!legacy.envFile;
      if (hasLegacy) {
        if (!isEmptyDir(sibling)) {
          return fail(`This app folder still holds data from an older version, and ${sibling} already exists with other data. `
            + `Move one of them aside, then restart.`);
        }
        const { op, errors } = planOperation('migrate', sibling, { appRoot: installRoot, currentRoot: null, currentDb: null });
        if (errors.length) return fail(`Could not plan moving this install's data to ${sibling}: ${errors.join('; ')}`);
        ptr = { version: 1, root: null, pending: op };
        writePointer(ptrFile, ptr);
        log(`[Data] Moving this install's data to ${sibling}`);
      } else if (fs.existsSync(sibling) && !isEmptyDir(sibling) && !isDataLayout(sibling)) {
        return fail(`${sibling} exists but is not a HOT-Step data folder. Rename it, or set HOT_STEP_DATA_ROOT, then restart.`);
      } else {
        initLayout(dataLayout(sibling));
        writePointer(ptrFile, { version: 1, root: sibling, pending: null });
        log(`[Data] Data folder: ${sibling}`);
        return { ok: true, runtime: release(appRoot, sibling, prof, null) };
      }
    }

    const heldRoot = leaseRefusal(env, ptr.root, ptrFile);
    if (heldRoot) return heldRoot;
    const err = await runPendingAtBoot(ptrFile, ptr, appRoot, prof, opts, log);
    if (err) return fail(err);

    let root = ptr.root;
    if (!root) return fail(`${ptrFile} names no data folder.`);
    const selected = selectPointerRoot({ mode: 'release', installRoot }, root);
    const staleRoot = selected.staleRoot;
    if (staleRoot) {
      root = selected.root!;
      if (fs.existsSync(root) && !isEmptyDir(root) && !isDataLayout(root)) {
        return fail(`${root} exists but is not a HOT-Step data folder. Rename it, or set HOT_STEP_DATA_ROOT, then restart.`);
      }
      if (fs.existsSync(root) && !accessibleRoot(root)) return fail(`The default data folder ${root} cannot be accessed.`);
    }
    const sharedWith = sharedRoot(prof, ptrFile, root);
    if (sharedWith) return fail(`The data folder ${root} is the same folder as ${sharedWith}, which another HOT-Step install uses. Point this one elsewhere in ${ptrFile}, then restart.`);
    initLayout(dataLayout(root));
    if (staleRoot) log(`[Data] WARNING: Saved data folder ${staleRoot} is missing or cannot be accessed. Using default data folder ${root}. Saved location unchanged; reconnect the drive and restart to use it again.`);
    return { ok: true, runtime: release(appRoot, root, prof, null) };
  } catch (e: any) {
    return fail(String(e?.message ?? e));
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
}

/** The only thing a broken data location serves. Never returns. */
async function serveErrorPage(error: string, pointerPath: string | null): Promise<never> {
  const port = parseInt(process.env.SERVER_PORT || '3001', 10);
  const host = process.env.SERVER_HOST || '0.0.0.0';
  console.error(`\n[Data] HOT-Step cannot start: ${error}\n`);
  const body = `<!doctype html><meta charset="utf-8"><title>HOT-Step: data folder problem</title>
<body style="font-family:sans-serif;max-width:46em;margin:3em auto;line-height:1.5">
<h1>HOT-Step cannot open its data folder</h1><p>${escapeHtml(error)}</p>
${pointerPath ? `<p>The data location is recorded in <code>${escapeHtml(pointerPath)}</code>. To undo an unfinished move on the same drive, set <code>"abandon": true</code> inside its <code>"pending"</code> entry and restart.</p>` : ''}
<p>Nothing has been deleted. Close this window and restart HOT-Step after fixing the problem.</p></body>`;
  http.createServer((_req, res) => {
    res.writeHead(503, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(body);
  }).listen(port, host);
  return new Promise<never>(() => { /* hold module evaluation forever */ });
}

const resolution = await resolveRuntime();
if (!resolution.ok && resolution.lease) {
  console.error(`\n[Data] HOT-Step cannot start: ${resolution.error}\n`);
  process.exit(1);
}
if (!resolution.ok) await serveErrorPage(resolution.error, resolution.pointerPath);

export const RUNTIME: Runtime = (resolution as { ok: true; runtime: Runtime }).runtime;
