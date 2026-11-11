// dataroot/pointer.ts — What Settings uses to show and change the data folder.
//
// Nothing here moves data. scheduleOperation() records a pending operation in
// the profile pointer; the caller then restarts the app gracefully and
// bootstrap.ts carries it out before the DB opens.

import fs from 'fs';
import path from 'path';
import { RUNTIME } from './bootstrap.js';
import { ENV_PATH_KEYS, isInside, readPointer, writePointer, type OperationKind, type PendingOperation } from './paths.js';
import { planOperation, sharedRoot, sizeOperation } from './transfer.js';

export type ScheduleKind = Exclude<OperationKind, 'migrate'>;

// ── Busy checks ──────────────────────────────────────────────────────

const busyChecks = new Map<string, () => string[]>();

/** Register a source of "work still running" reasons. Each check returns one
 *  line per blocker (empty when idle). Re-registering a name replaces it. */
export function registerBusyCheck(name: string, check: () => string[]): void {
  busyChecks.set(name, check);
}

export function getBusyBlockers(): string[] {
  const out: string[] = [];
  for (const [name, check] of busyChecks) {
    try { out.push(...check()); } catch (e: any) { out.push(`${name}: could not check (${e.message})`); }
  }
  return out;
}

// ── Display ──────────────────────────────────────────────────────────

export interface DataLocation {
  mode: 'dev' | 'release';
  appRoot: string | null;
  dataRoot: string | null;
  override: string | null;
  pending: PendingOperation | null;
  /** Path settings in .env that point outside the data folder. */
  externals: Array<{ key: string; path: string }>;
}

function pointerPath(): string {
  if (!RUNTIME.pointerPath) throw Object.assign(new Error('The data folder cannot be changed in this app'), { status: 400 });
  return RUNTIME.pointerPath;
}

export function getDataLocation(env: NodeJS.ProcessEnv = process.env): DataLocation {
  const externals: DataLocation['externals'] = [];
  if (RUNTIME.dataRoot) {
    for (const key of ENV_PATH_KEYS) {
      const v = env[key];
      if (v && path.isAbsolute(v) && !isInside(RUNTIME.dataRoot, v) && !isInside(RUNTIME.appRoot!, v)) externals.push({ key, path: v });
    }
  }
  let pending: PendingOperation | null = null;
  try { pending = RUNTIME.pointerPath ? readPointer(RUNTIME.pointerPath)?.pending ?? null : null; } catch { /* shown by bootstrap */ }
  return { mode: RUNTIME.mode, appRoot: RUNTIME.appRoot, dataRoot: RUNTIME.dataRoot, override: RUNTIME.override, pending, externals };
}

// ── Update preflight ─────────────────────────────────────────────────

/** Why this install cannot open an update transaction (empty when it can).
 *  Paths are compared after links resolve. */
export function updatePreflight(env: NodeJS.ProcessEnv = process.env): string[] {
  const { appRoot, dataRoot, pending } = getDataLocation(env);
  if (RUNTIME.mode !== 'release' || !appRoot) return ['only a portable release installs updates'];
  if (!dataRoot) return ['this install has no data folder'];
  if (pending) return ['a data-folder change is scheduled; restart to finish it first'];
  let app: string, data: string;
  try { app = fs.realpathSync.native(appRoot); data = fs.realpathSync.native(dataRoot); } catch (e: any) { return [`cannot resolve the install or data folder: ${e.message}`]; }
  const errors: string[] = [];
  if (isInside(app, data) || isInside(data, app)) errors.push(`the data folder ${dataRoot} overlaps the app folder ${appRoot}`);
  for (const key of ENV_PATH_KEYS) {
    const v = env[key];
    if (v && path.isAbsolute(v) && isInside(app, v)) errors.push(`${key} points inside the app folder (${v}); move it to the data folder first`);
  }
  return errors;
}

// ── Preview / schedule / cancel ──────────────────────────────────────

export interface OperationPreview {
  ok: boolean;
  errors: string[];
  kind: ScheduleKind;
  /** move/create/repoint: the new data folder. import: the old app folder. */
  target: string;
  mappings: Array<{ name: string; from: string; to: string }>;
  bytes: number;
  crossVolume: boolean;
  needBytes: number;
  freeBytes: number | null;
}

/** A dev checkout with no data folder yet moves its checkout folders (the
 *  same plan a release runs for its own old layout, with dev paths). */
function plan(kind: ScheduleKind, target: string, currentDb: string | null) {
  if (!RUNTIME.appRoot) throw Object.assign(new Error('The data folder cannot be changed in this app'), { status: 400 });
  const dev = RUNTIME.mode === 'dev';
  const k: OperationKind = dev && kind === 'move' && !RUNTIME.dataRoot ? 'migrate' : kind;
  return planOperation(k, target, { appRoot: RUNTIME.appRoot, currentRoot: RUNTIME.dataRoot, currentDb, dev });
}

/** Validate an operation and size it. Writes nothing. `currentDb` is
 *  config.data.dbPath (passed in to keep this module free of config). */
export function previewOperation(kind: ScheduleKind, target: string, currentDb: string | null): OperationPreview {
  const { op, errors } = plan(kind, target, currentDb);
  if (RUNTIME.override) errors.push('HOT_STEP_DATA_ROOT is set for this process; clear it before changing the data folder');
  if (kind !== 'import' && RUNTIME.profileDir && RUNTIME.pointerPath) {
    const other = sharedRoot(RUNTIME.profileDir, RUNTIME.pointerPath, op.destination);
    if (other) errors.push(`${other} belongs to another HOT-Step install (${RUNTIME.mode === 'dev' ? 'the release app or another checkout' : 'a development checkout'}); each keeps its own data folder`);
  }
  const size = errors.length ? { bytes: 0, crossVolume: false, needBytes: 0, freeBytes: null } : sizeOperation(op);
  if (size.crossVolume && size.freeBytes !== null && size.freeBytes < size.needBytes) {
    errors.push(`not enough free space: ${(size.needBytes / 1e9).toFixed(1)} GB needed, ${(size.freeBytes / 1e9).toFixed(1)} GB free`);
  }
  return {
    ok: errors.length === 0, errors, kind, target,
    mappings: op.mappings.map(m => ({ name: m.name, from: m.from, to: m.to })), ...size,
  };
}

/** Record the operation for the next start. Refuses while work is running.
 *  The caller restarts the app afterwards. */
export function scheduleOperation(kind: ScheduleKind, target: string, currentDb: string | null): PendingOperation {
  const file = pointerPath();
  const blockers = getBusyBlockers();
  if (blockers.length) throw Object.assign(new Error(`Wait for running work to finish: ${blockers.join('; ')}`), { status: 409, blockers });
  const preview = previewOperation(kind, target, currentDb);
  if (!preview.ok) throw Object.assign(new Error(preview.errors.join('; ')), { status: 400, errors: preview.errors });
  const ptr = readPointer(file) ?? {
    version: 1 as const, root: RUNTIME.dataRoot, pending: null, ...(RUNTIME.mode === 'dev' ? { checkout: RUNTIME.appRoot! } : {}),
  };
  if (ptr.pending) throw Object.assign(new Error('Another data-folder change is already scheduled'), { status: 409 });
  const { op } = plan(kind, target, currentDb);
  ptr.pending = op;
  writePointer(file, ptr);
  return op;
}

/** Drop a scheduled operation that has not started moving anything. */
export function cancelPending(): boolean {
  const file = pointerPath();
  const ptr = readPointer(file);
  if (!ptr?.pending) return false;
  if (ptr.pending.mappings.some(m => m.state !== 'pending') || ptr.pending.setAsideDone) {
    throw Object.assign(new Error('This change has already started; restart to let it finish'), { status: 409 });
  }
  ptr.pending = null;
  writePointer(file, ptr);
  return true;
}

const FINISHED = new Set(['succeeded', 'done', 'failed', 'cancelled', 'interrupted']);

/** Busy while any job in an in-memory job map has not finished. */
export function registerJobsBusyCheck(name: string, jobs: { values(): Iterable<{ status: string }> }): void {
  registerBusyCheck(name, () => {
    let n = 0;
    for (const j of jobs.values()) if (!FINISHED.has(j.status)) n++;
    return n ? [`${n} ${name} job(s) running or queued`] : [];
  });
}

const running = new Map<string, number>();
registerBusyCheck('running-work', () => [...running].filter(([, n]) => n > 0).map(([name, n]) => `${n} ${name} job(s) still running`));

/** Count `work` as running until it settles, whatever happens to the job's
 *  map entry (deleted, cancelled). Returns `work` unchanged, so the caller's
 *  own error handling is unaffected and no rejection is left unhandled. */
export function trackBusy<T>(name: string, work: Promise<T>): Promise<T> {
  running.set(name, (running.get(name) ?? 0) + 1);
  const done = () => { running.set(name, running.get(name)! - 1); };
  work.then(done, done);
  return work;
}
