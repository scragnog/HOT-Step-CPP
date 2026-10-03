// Git-bundle updates for a worker. A worker accepts one update at a time and
// blocks new training writes from the reset until its launcher restarts it.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import type { NextFunction, Request, Response } from 'express';
import { PROJECT_ROOT, config } from '../../config.js';
import { listJobs } from './labelingQueue.js';
import { listBatches } from './yue2BatchRunner.js';
import { listPipelines } from './pipelineRunner.js';
import { requestWorkerRestart } from '../../routes/shutdown.js';
import { gpuLaneBusy, gpuLaneDepth } from '../generation/gpuLane.js';

const git = (args: string[], cwd = PROJECT_ROOT) =>
  execFileSync('git', args, { cwd, encoding: 'utf8', timeout: 15000, windowsHide: true }).trim();
export const startupCommit = (() => { try { return git(['rev-parse', 'HEAD']); } catch { return ''; } })();
export const currentCommit = () => git(['rev-parse', 'HEAD']);
export const dirtyCheckout = () => { try { return !!git(['status', '--porcelain', '--untracked-files=normal', '--ignore-submodules=none']); } catch { return false; } };

export function activeTraining() {
  if (gpuLaneBusy() || gpuLaneDepth() > 0) return { kind: 'generation', dataset: '', done: 0, total: 0, status: 'running' };
  const job = listJobs().find(j => j.status === 'running' || j.status === 'queued');
  if (job) return { kind: job.kind, dataset: job.datasetId, done: job.done, total: job.total, status: job.status };
  const batch = listBatches().find(b => b.status === 'running' || b.status === 'paused');
  if (batch) return { kind: 'yue2-batch', dataset: '', done: 0, total: 0, status: batch.status };
  const pipeline = listPipelines().find(p => p.status === 'running' || p.status === 'paused');
  if (pipeline) return { kind: 'pipeline', dataset: '', done: 0, total: 0, status: pipeline.status };
  return null;
}

let applying = false;
export function workerUpdateGate(req: Request, res: Response, next: NextFunction): void {
  if (applying && req.method !== 'GET' && req.method !== 'HEAD') {
    res.status(409).json({ error: 'Worker update is in progress' }); return;
  }
  next();
}

export function classifyCommit(worker: string, head = currentCommit()): { relation: 'current' | 'behind' | 'diverged'; commits: number } {
  if (worker === head) return { relation: 'current', commits: 0 };
  if (!/^[a-f0-9]{40}$/i.test(worker)) return { relation: 'diverged', commits: 0 };
  try {
    execFileSync('git', ['merge-base', '--is-ancestor', worker, head], { cwd: PROJECT_ROOT, stdio: 'ignore', timeout: 15000, windowsHide: true });
    return { relation: 'behind', commits: Number(git(['rev-list', '--count', `${worker}..${head}`])) };
  } catch { return { relation: 'diverged', commits: 0 }; }
}

export function changedSteps(files: string[]) {
  return {
    serverInstall: files.includes('server/package-lock.json'),
    uiInstall: files.includes('ui/package-lock.json'),
    engineBuild: files.some(f => f.startsWith('engine/')),
  };
}

export async function runUpdatePlan(files: string[], ops: {
  advance: () => boolean;
  idle: () => void;
  reset: () => Promise<void>;
  serverInstall: () => Promise<void>;
  uiInstall: () => Promise<void>;
  uiBuild: () => Promise<void>;
  engineBuild: () => Promise<void>;
  restart: () => void;
}): Promise<void> {
  if (!ops.advance()) throw new Error('Bundle does not advance the worker commit');
  ops.idle(); // The last check before a destructive reset.
  await ops.reset();
  const steps = changedSteps(files);
  if (steps.serverInstall) await ops.serverInstall();
  if (steps.uiInstall) await ops.uiInstall();
  await ops.uiBuild();
  if (steps.engineBuild) await ops.engineBuild();
  ops.restart();
}

function assertIdle(): void {
  const job = activeTraining();
  if (job) throw Object.assign(new Error(`Worker has an active or queued ${job.kind} job`), { status: 409 });
}

async function command(file: string, args: string[], cwd: string, log: (line: string) => void): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(file, args, { cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = '';
    const feed = (chunk: Buffer) => {
      tail += chunk.toString();
      const lines = tail.split(/\r?\n/); tail = lines.pop() ?? '';
      for (const line of lines) if (line.trim()) log(line);
    };
    child.stdout.on('data', feed); child.stderr.on('data', feed);
    child.on('error', reject);
    child.on('close', code => { if (tail.trim()) log(tail); code === 0 ? resolve() : reject(new Error(`${file} exited ${code}`)); });
  });
}

export async function receiveUpdate(body: NodeJS.ReadableStream, base: string, target: string, log: (line: string, phase?: string) => void): Promise<void> {
  if (applying) throw Object.assign(new Error('Worker update already running'), { status: 409 });
  assertIdle();
  if (!/^[a-f0-9]{40}$/i.test(base) || !/^[a-f0-9]{40}$/i.test(target)) throw Object.assign(new Error('Invalid commit'), { status: 400 });
  if (currentCommit() !== base) throw Object.assign(new Error('Worker commit changed; refresh status'), { status: 409 });
  applying = true;
  const bundle = path.join(os.tmpdir(), `hotstep-worker-${randomUUID()}.bundle`);
  let resetStarted = false;
  let bundlePresent = true;
  try {
    await pipeline(body, fs.createWriteStream(bundle));
    assertIdle();
    log('Verifying bundle', 'verifying');
    await command('git', ['bundle', 'verify', bundle], PROJECT_ROOT, line => log(line));
    await command('git', ['fetch', '--no-tags', bundle, 'master'], PROJECT_ROOT, line => log(line));
    if (git(['rev-parse', 'FETCH_HEAD']) !== target) throw new Error('Bundle target did not match requested commit');
    const files = git(['diff', '--name-only', `${base}..${target}`]).split(/\r?\n/).filter(Boolean).map(f => f.replaceAll('\\', '/'));
    await runUpdatePlan(files, {
      advance: () => classifyCommit(base, target).relation === 'behind' && currentCommit() === base,
      idle: assertIdle,
      reset: async () => {
        log(`Resetting to ${target.slice(0, 8)}; discarding worker checkout changes`, 'resetting');
        resetStarted = true;
        await command('git', ['-c', 'submodule.recurse=false', 'reset', '--hard', target], PROJECT_ROOT, line => log(line));
      },
      serverInstall: async () => { log('Installing server dependencies', 'server-install'); await command('cmd.exe', ['/d', '/c', 'npm ci'], path.join(PROJECT_ROOT, 'server'), line => log(line)); },
      uiInstall: async () => { log('Installing UI dependencies', 'ui-install'); await command('cmd.exe', ['/d', '/c', 'npm ci'], path.join(PROJECT_ROOT, 'ui'), line => log(line)); },
      uiBuild: async () => { log('Building UI', 'ui-build'); await command('cmd.exe', ['/d', '/c', 'npm run build'], path.join(PROJECT_ROOT, 'ui'), line => log(line)); },
      engineBuild: async () => { log('Building engine', 'engine-build'); await command('cmd.exe', ['/d', '/c', path.join(PROJECT_ROOT, 'engine', 'build.cmd')], path.join(PROJECT_ROOT, 'engine'), line => log(line)); },
      restart: () => { log('Build complete; requesting restart', 'restarting'); fs.unlinkSync(bundle); bundlePresent = false; requestWorkerRestart(); },
    });
  } finally {
    if (bundlePresent) try { fs.unlinkSync(bundle); } catch { /* temp bundle is disposable */ }
    if (!resetStarted) applying = false;
    // After reset, failed builds must remain unavailable for training until a
    // person repairs them; a successful build exits through requestWorkerRestart.
  }
}

export interface WorkerUpdateJob {
  id: string; worker: string; status: 'preparing' | 'uploading' | 'verifying' | 'resetting' | 'server-install' | 'ui-install' | 'ui-build' | 'engine-build' | 'restarting' | 'done' | 'failed' | 'cancelled';
  cancellable: boolean; lines: string[]; error?: string;
}
const updates = new Map<string, WorkerUpdateJob & { controller: AbortController }>();
export function getUpdate(name: string): WorkerUpdateJob | null {
  const j = [...updates.values()].reverse().find(j => j.worker === name);
  return j ? { id: j.id, worker: j.worker, status: j.status, cancellable: j.cancellable, lines: j.lines, error: j.error } : null;
}
export function cancelUpdate(name: string): boolean {
  const job = [...updates.values()].reverse().find(j => j.worker === name && j.status === 'preparing');
  if (!job) return false;
  job.controller.abort(); job.status = 'cancelled'; job.cancellable = false; return true;
}
export function startUpdate(name: string, url: string, token: string): WorkerUpdateJob {
  if ([...updates.values()].some(j => j.worker === name && !['done', 'failed', 'cancelled'].includes(j.status))) throw Object.assign(new Error('Update already running'), { status: 409 });
  const job = { id: randomUUID(), worker: name, status: 'preparing' as WorkerUpdateJob['status'], cancellable: true, lines: [] as string[], controller: new AbortController(), error: undefined as string | undefined };
  updates.set(job.id, job);
  void (async () => {
    let bundle = '';
    try {
      const response = await fetch(`${url}/api/training/worker/status`, { headers: token ? { 'x-hotstep-worker-token': token } : {}, signal: AbortSignal.timeout(8000) });
      const remote = await response.json() as { commit?: string; idle?: boolean; error?: string };
      if (!response.ok || !remote.commit) throw new Error(remote.error || `Worker status HTTP ${response.status}; this worker may need a one-time bootstrap`);
      if (!remote.idle) throw new Error('Worker has an active or queued job');
      const head = currentCommit();
      const relation = classifyCommit(remote.commit, head);
      if (relation.relation !== 'behind') throw new Error(relation.relation === 'diverged' ? 'Worker commit diverged from this machine' : 'Worker is already current');
      bundle = path.join(os.tmpdir(), `hotstep-send-${job.id}.bundle`);
      await command('git', ['bundle', 'create', bundle, `${remote.commit}..master`], PROJECT_ROOT, line => job.lines.push(line));
      if (job.controller.signal.aborted) return;
      job.status = 'uploading'; job.cancellable = false;
      const sent = await fetch(`${url}/api/training/worker/update?base=${remote.commit}&target=${head}`, {
        method: 'POST', headers: { 'content-type': 'application/octet-stream', ...(token ? { 'x-hotstep-worker-token': token } : {}) },
        body: Readable.toWeb(fs.createReadStream(bundle)) as ReadableStream, duplex: 'half',
      } as RequestInit);
      if (!sent.ok) {
        const error = await sent.json().catch(() => ({})) as { error?: string };
        throw new Error(error.error || `Worker update HTTP ${sent.status}`);
      }
      if (!sent.body) throw new Error('Worker closed update stream');
      let pending = '';
      for await (const chunk of Readable.fromWeb(sent.body as any)) {
        pending += chunk.toString();
        const rows = pending.split('\n'); pending = rows.pop() ?? '';
        for (const row of rows) {
          if (!row) continue;
          const event = JSON.parse(row) as { line?: string; phase?: WorkerUpdateJob['status']; error?: string };
          if (event.line) job.lines.push(event.line);
          if (event.phase) job.status = event.phase;
          if (event.error) throw new Error(event.error);
        }
      }
      if (pending.trim()) { const event = JSON.parse(pending) as { error?: string }; if (event.error) throw new Error(event.error); }
      job.status = 'done';
    } catch (err: any) {
      if (job.status !== 'cancelled') { job.error = err?.message || String(err); job.lines.push(job.error!); job.status = 'failed'; }
    } finally { if (bundle) try { fs.unlinkSync(bundle); } catch { /* disposable */ } }
  })();
  return getUpdate(name)!;
}
