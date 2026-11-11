// apply.ts — POST /api/updates/apply: hand a prepared update to the stable
// Windows launcher (release/updater/launcher.mjs), which installs it.
//
// The server never touches the install itself. It revalidates the staged
// receipt, refuses a payload someone edited, opens the maintenance
// transaction (preflight, lease, gate closed, in-flight work drained, busy
// check; a refusal reopens the gate and cancels nothing), asks the launcher
// to take ownership and, once it acknowledges, stops through the updater
// shutdown path. Everything after that (snapshot, pointer, candidate health,
// rollback) is the launcher's, and it records the outcome in
// <updater>/last-result.json, which status() reports.
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import type { UpdateApplyResult, UpdateInstallStatus, UpdatesApplyResponse } from '../../contracts/updates.js';
import { PrepareError } from './download.js';
import { releaseLease, type Gate } from './maintenance.js';

export const MANIFEST_FILE = 'payload-manifest.json';
export const RESULT_FILE = 'last-result.json';

export interface ApplyDeps {
  /** Why this process cannot install updates at all, else null. */
  unsupported(): string | null;
  /** The staged update, checked again (throws PrepareError). */
  revalidate(): { id: string; payloadId: string; candidate: { id: string } };
  /** This payload: its folder and its name under app/. */
  payloadDir: string;
  installRoot: string;
  updaterDir: string;
  dataRoot: string;
  preflight(): string[];
  blockers(): string[];
  gate: Gate;
  /** Databases and files the launcher snapshots before switching. */
  snapshotSet(): { dbs: string[]; files: string[] };
  /** Send to the launcher and wait for a reply of type `reply` (null on timeout). */
  ask(msg: object, reply: string, ms: number): Promise<any>;
  /** Stop for the launcher; called once, after the 202 is on its way. */
  stop(): void;
}

function* walk(dir: string, rel = ''): Generator<string> {
  for (const e of fs.readdirSync(path.join(dir, rel), { withFileTypes: true })) {
    const r = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory() && !e.isSymbolicLink()) yield* walk(dir, r);
    else yield r;
  }
}

async function hashFile(file: string): Promise<string> {
  const h = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) h.update(chunk as Buffer);
  return h.digest('hex');
}

/** Files added to, changed in or missing from this payload since it was
 *  packaged (payload-manifest.json). An update would leave them behind in an
 *  old folder, so any of them blocks it. */
export async function payloadProblems(dir: string): Promise<string[]> {
  let manifest: { format?: number; files?: Record<string, string> };
  try { manifest = JSON.parse(await fsp.readFile(path.join(dir, MANIFEST_FILE), 'utf8')); }
  catch { return [`this version has no readable ${MANIFEST_FILE}; install the update manually`]; }
  if (manifest?.format !== 1 || !manifest.files || typeof manifest.files !== 'object') return [`${MANIFEST_FILE} is not format 1`];
  const want = new Map(Object.entries(manifest.files));
  const added: string[] = [], changed: string[] = [];
  for (const rel of walk(dir)) {
    if (rel === MANIFEST_FILE) continue;
    const sha = want.get(rel);
    if (sha === undefined) { added.push(rel); continue; }
    want.delete(rel);
    if (await hashFile(path.join(dir, ...rel.split('/'))) !== sha) changed.push(rel);
  }
  const out: string[] = [];
  const list = (label: string, xs: string[]) => {
    if (xs.length) out.push(`${label}: ${xs.slice(0, 5).join(', ')}${xs.length > 5 ? ` and ${xs.length - 5} more` : ''}`);
  };
  list('files were added to the app folder (move them to the data folder first)', added);
  list('shipped files were edited', changed);
  list('shipped files are missing', [...want.keys()]);
  return out;
}

export function readLastResult(updaterDir: string | null): UpdateApplyResult | null {
  if (!updaterDir) return null;
  try {
    const r = JSON.parse(fs.readFileSync(path.join(updaterDir, RESULT_FILE), 'utf8'));
    const ok = r?.format === 1 && ['installed', 'rolled-back', 'recovery-required', 'failed'].includes(r.outcome) && typeof r.at === 'string';
    return ok ? { outcome: r.outcome, from: r.from ?? null, to: r.to ?? null, at: r.at, message: r.message ?? null } : null;
  } catch { return null; }
}

/** IPC to the launcher that started this process. */
export function askLauncher(msg: object, reply: string, ms: number): Promise<any> {
  return new Promise(resolve => {
    if (typeof process.send !== 'function') { resolve(null); return; }
    const on = (m: any) => { if (m?.type === reply) { clearTimeout(timer); process.off('message', on); resolve(m); } };
    const timer = setTimeout(() => { process.off('message', on); resolve(null); }, ms);
    process.on('message', on);
    process.send(msg);
  });
}

export function createApplier(d: ApplyDeps) {
  let phase: UpdateInstallStatus['phase'] = 'idle';

  function status(): UpdateInstallStatus {
    const reason = d.unsupported();
    return { supported: !reason, reason, phase, lastResult: readLastResult(d.updaterDir || null) };
  }

  async function apply(candidateId: string): Promise<UpdatesApplyResponse> {
    if (phase !== 'idle') throw new PrepareError('An update is already being installed');
    const why = d.unsupported();
    if (why) throw new PrepareError(why);
    const prepared = d.revalidate();
    if (prepared.candidate.id !== candidateId) throw new PrepareError('That is not the update that was downloaded; check for updates again');
    const oldPayload = path.basename(d.payloadDir);
    const edits = await payloadProblems(d.payloadDir);
    const dirs = { updaterDir: d.updaterDir, installRoot: d.installRoot, dataRoot: d.dataRoot };
    const { lease, token } = await d.gate.begin({
      ...dirs, preflight: () => [...d.preflight(), ...edits], blockers: d.blockers, fromRequest: true,
    });
    const reopen = () => {
      try { releaseLease(dirs, lease.id); } catch (e) { console.error('[Updates] could not release the maintenance lease:', (e as Error).message); }
      d.gate.open();
    };
    let ack: any;
    try {
      ack = await d.ask({
        type: 'apply', preparedId: prepared.id, newPayload: prepared.payloadId, oldPayload,
        leaseId: lease.id, token, dataRoot: d.dataRoot, snapshot: d.snapshotSet(),
      }, 'apply-ack', 15_000);
    } catch (e) { reopen(); throw e; }
    if (!ack?.ok) {
      reopen();
      throw new PrepareError(`The launcher did not take the update: ${ack?.error ?? 'no answer'}`);
    }
    phase = 'stopping';
    console.log(`[Updates] The launcher is installing ${prepared.payloadId}; stopping ${oldPayload}.`);
    setTimeout(d.stop, 300);
    return { phase: 'stopping', from: oldPayload, to: prepared.payloadId };
  }

  return { apply, status };
}
