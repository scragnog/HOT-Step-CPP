// Durable YuE2 preparation chains. The existing training routes remain the
// single stage validators and GPU admission point; this module owns ordering.
import fs from 'fs';
import path from 'path';
import { createHash, randomUUID } from 'crypto';
import { config } from '../../config.js';
import type { TrainingSnapshot } from '../../contracts/trainingOperation.js';
import type { Yue2PreparationPayload, Yue2PreparationStage, Yue2PreparationStageStatus,
  Yue2PreparationSummary } from '../../contracts/trainingPreparation.js';
import { trainingBaseDir } from './paths.js';
import { getDataset } from './datasetsRepo.js';
import { activeJobForDataset, cancelJob } from './labelingQueue.js';
import { hasActivePipeline, registerYue2PreparationActivity } from './pipelineRunner.js';
import { hasActiveBatch } from './yue2BatchRunner.js';
import { pushDataset, workerFetch } from './trainingWorkers.js';

type Snapshot = TrainingSnapshot<Yue2PreparationPayload>;
type Status = Yue2PreparationSummary['status'];
type StageStatus = Yue2PreparationStageStatus['status'];
type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface Yue2PreparationArtifacts {
  preprocess: { done: boolean; captionModeOk: boolean; captionsStale?: boolean };
  tokenize: { done: boolean };
  sheet: { done: boolean };
  align: { done: boolean; stemsReady: number };
  minted: { present: boolean };
  narCompleted: boolean;
  arCompleted: boolean;
}

export interface Yue2PreparationAdapter {
  datasetRevision(snapshot: Snapshot): Promise<string | null>;
  sourceRevision(snapshot: Snapshot): Promise<string>;
  artifacts(snapshot: Snapshot, workerUrl: string | null): Promise<Yue2PreparationArtifacts>;
  start(snapshot: Snapshot, stage: Yue2PreparationStage, body: Record<string, unknown>, workerUrl: string | null): Promise<string>;
  job(snapshot: Snapshot, id: string, workerUrl: string | null): Promise<{ status: JobState; error?: string | null } | null>;
  cancel(snapshot: Snapshot, id: string, workerUrl: string | null): Promise<void>;
  admission(snapshot: Snapshot, workerUrl: string | null): Promise<void>;
}

interface RecordFile {
  summary: Yue2PreparationSummary;
  snapshot: Snapshot;
  workerUrl: string | null;
  trainBodies: Partial<Record<Yue2PreparationStage, Record<string, unknown>>>;
  pauseRequested: boolean;
  cancelRequested: boolean;
}

const ORDER: readonly Yue2PreparationStage[] = ['latents', 'codes', 'sheet', 'stems', 'align', 'nar', 'ar', 'joint'];
const POLL_MS = 1500;
const terminal = (s: Status) => s === 'done' || s === 'failed' || s === 'cancelled' || s === 'interrupted';
const active = (s: Status) => !terminal(s);

export class Yue2PreparationConflict extends Error {
  constructor(message: string) { super(message); }
}

export class Yue2PreparationPipeline {
  private readonly records = new Map<string, RecordFile>();
  private readonly loops = new Set<string>();
  private unreadableRecord = false;

  constructor(private readonly dir: string, private readonly adapter: Yue2PreparationAdapter,
    private readonly schedule: (fn: () => void) => void = setImmediate,
    private readonly delay: (ms: number) => Promise<void> = ms => new Promise(resolve => setTimeout(resolve, ms))) {
    fs.mkdirSync(dir, { recursive: true });
    for (const name of fs.readdirSync(dir)) {
      if (!name.endsWith('.json')) continue;
      try {
        const record = JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8')) as RecordFile;
        if (!record.summary?.id || !record.snapshot?.operation) { this.unreadableRecord = true; continue; }
        if (active(record.summary.status)) {
          // A submitted stage may still have been running when Node died. Its
          // recorded jobId is retained for reconciliation, never replayed.
          record.summary.status = 'interrupted';
          record.summary.error = 'Server restarted during preparation; inspect the recorded job before retrying.';
          for (const stage of record.summary.stages) if (stage.status === 'running') stage.status = 'interrupted';
          record.summary.updatedAt = Date.now();
          this.write(record);
        }
        this.records.set(record.summary.id, record);
      } catch { this.unreadableRecord = true; }
    }
  }

  private write(record: RecordFile): void {
    record.summary.updatedAt = Date.now();
    const file = path.join(this.dir, `${record.summary.id}.json`);
    const temp = `${file}.${randomUUID()}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(record, null, 2));
    fs.renameSync(temp, file);
  }

  get(id: string): Yue2PreparationSummary | null {
    return this.records.get(id)?.summary ?? null;
  }

  async currentJob(id: string): Promise<unknown | null> {
    const r = this.records.get(id);
    if (!r) return null;
    const stage = [...r.summary.stages].reverse().find(s => s.jobId);
    if (!stage?.jobId) return null;
    return this.adapter.job(r.snapshot, stage.jobId, r.workerUrl);
  }

  list(datasetId?: string): Yue2PreparationSummary[] {
    return [...this.records.values()].map(r => r.summary)
      .filter(s => !datasetId || s.datasetId === datasetId)
      .sort((a, b) => b.createdAt - a.createdAt);
  }

  byKey(kind: string, key: string): RecordFile | undefined {
    return [...this.records.values()].find(r => r.snapshot.operation.kind === kind
      && r.snapshot.operation.idempotencyKey === key);
  }

  async start(snapshot: Snapshot, trainBodies: Record<string, Record<string, unknown>>, workerUrl: string | null = null): Promise<Yue2PreparationSummary> {
    if (this.unreadableRecord) throw new Yue2PreparationConflict('A preparation record is unreadable; reconcile it before starting another');
    const trains = snapshot.payload.stages.filter(s => s === 'nar' || s === 'ar' || s === 'joint');
    if (snapshot.payload.mode === 'prepare-only' && trains.length) {
      throw new Yue2PreparationConflict('Preparation only cannot start a trainer');
    }
    if (snapshot.payload.mode === 'train-after-preparation' && !trains.length) {
      throw new Yue2PreparationConflict('Train mode requires a training stage');
    }
    if (snapshot.worker.kind === 'remote' && !workerUrl) throw new Yue2PreparationConflict('Remote worker URL was not captured');
    const { kind, idempotencyKey } = snapshot.operation;
    const old = this.byKey(kind, idempotencyKey);
    if (old) {
      if (JSON.stringify(old.snapshot) !== JSON.stringify(snapshot)) throw new Yue2PreparationConflict('Idempotency key belongs to a different preparation command');
      return old.summary;
    }
    const datasetId = snapshot.dataset?.id;
    if (!datasetId) throw new Yue2PreparationConflict('Preparation requires a dataset revision');
    if (snapshot.dataset!.revision !== await this.adapter.datasetRevision(snapshot)) {
      throw new Yue2PreparationConflict('Dataset changed; reload preparation');
    }
    const source = snapshot.sources.find(s => s.kind === 'dataset-sources' && s.id === datasetId);
    if (!source || snapshot.sources.length !== 1) throw new Yue2PreparationConflict('Preparation requires the source revision');
    if (source.revision !== await this.adapter.sourceRevision(snapshot)) throw new Yue2PreparationConflict('Dataset sources changed; reload preparation');
    if (this.list(datasetId).some(s => active(s.status)) || hasActivePipeline() || hasActiveBatch() || activeJobForDataset(datasetId)) {
      throw new Yue2PreparationConflict('A pipeline or job is already active for this dataset');
    }
    await this.adapter.admission(snapshot, workerUrl);
    if (snapshot.dataset!.revision !== await this.adapter.datasetRevision(snapshot)) {
      throw new Yue2PreparationConflict('Dataset changed while accepting preparation');
    }
    // admission may await a worker; another client can win in the meantime.
    const claimed = this.byKey(kind, idempotencyKey);
    if (claimed) {
      if (JSON.stringify(claimed.snapshot) !== JSON.stringify(snapshot)) throw new Yue2PreparationConflict('Idempotency key belongs to a different preparation command');
      return claimed.summary;
    }
    if (this.list(datasetId).some(s => active(s.status)) || hasActivePipeline() || hasActiveBatch() || activeJobForDataset(datasetId)) {
      throw new Yue2PreparationConflict('A pipeline or job is already active for this dataset');
    }
    const stages = ORDER.filter(s => snapshot.payload.stages.includes(s));
    const now = Date.now();
    const record: RecordFile = {
      summary: { id: randomUUID(), status: 'running', datasetId, worker: snapshot.worker,
        stages: stages.map(stage => ({ stage, status: 'pending', jobId: null, error: null })),
        createdAt: now, updatedAt: now, error: null },
      snapshot: structuredClone(snapshot), workerUrl, trainBodies: structuredClone(trainBodies),
      pauseRequested: false, cancelRequested: false,
    };
    this.write(record); // no stage may start unless its accepted command is durable
    this.records.set(record.summary.id, record);
    this.schedule(() => { void this.run(record); });
    return record.summary;
  }

  pause(id: string): Yue2PreparationSummary | null {
    const r = this.records.get(id);
    if (!r) return null;
    if (active(r.summary.status) && !r.cancelRequested) {
      r.pauseRequested = true;
      r.summary.status = 'pausing';
      this.write(r);
    }
    return r.summary;
  }

  resume(id: string): Yue2PreparationSummary | null {
    const r = this.records.get(id);
    if (!r) return null;
    if (r.summary.status === 'paused' || r.summary.status === 'pausing') {
      r.pauseRequested = false;
      r.summary.status = 'running';
      this.write(r);
    }
    return r.summary;
  }

  async cancel(id: string): Promise<Yue2PreparationSummary | null> {
    const r = this.records.get(id);
    if (!r) return null;
    if (active(r.summary.status) && !r.cancelRequested) {
      r.cancelRequested = true;
      r.pauseRequested = false;
      r.summary.status = 'cancelling';
      this.write(r);
      const running = r.summary.stages.find(s => s.status === 'running' && s.jobId);
      if (running?.jobId) await this.adapter.cancel(r.snapshot, running.jobId, r.workerUrl);
    }
    return r.summary;
  }

  async retry(id: string): Promise<Yue2PreparationSummary | null> {
    if (this.unreadableRecord) throw new Yue2PreparationConflict('A preparation record is unreadable; reconcile it before retrying');
    const r = this.records.get(id);
    if (!r) return null;
    if (!['failed', 'interrupted'].includes(r.summary.status)) throw new Yue2PreparationConflict('Only failed or interrupted preparation can be retried');
    if (this.list(r.summary.datasetId).some(s => s.id !== id && active(s.status))
      || hasActivePipeline() || hasActiveBatch() || activeJobForDataset(r.summary.datasetId)) throw new Yue2PreparationConflict('Another pipeline or job is active');
    if (r.snapshot.sources[0]?.revision !== await this.adapter.sourceRevision(r.snapshot)) {
      throw new Yue2PreparationConflict('Dataset sources changed; reload preparation');
    }
    if (r.snapshot.dataset!.revision !== await this.adapter.datasetRevision(r.snapshot)) {
      throw new Yue2PreparationConflict('Dataset changed; reload preparation');
    }
    for (const stage of r.summary.stages) {
      if ((stage.status !== 'interrupted' && stage.status !== 'failed') || !stage.jobId) continue;
      const job = await this.adapter.job(r.snapshot, stage.jobId, r.workerUrl);
      if (job && (job.status === 'running' || job.status === 'queued')) {
        throw new Yue2PreparationConflict('The recorded job is still active; cancel or reconcile it first');
      }
      if (job?.status === 'done') { stage.status = 'done'; stage.error = null; }
    }
    await this.adapter.admission(r.snapshot, r.workerUrl);
    if (r.snapshot.sources[0]?.revision !== await this.adapter.sourceRevision(r.snapshot)) {
      throw new Yue2PreparationConflict('Dataset sources changed; reload preparation');
    }
    if (r.snapshot.dataset!.revision !== await this.adapter.datasetRevision(r.snapshot)) {
      throw new Yue2PreparationConflict('Dataset changed; reload preparation');
    }
    if (!['failed', 'interrupted'].includes(r.summary.status)
      || this.list(r.summary.datasetId).some(s => s.id !== id && active(s.status))
      || hasActivePipeline() || hasActiveBatch() || activeJobForDataset(r.summary.datasetId)) {
      throw new Yue2PreparationConflict('Another pipeline or job is active');
    }
    r.cancelRequested = false;
    r.pauseRequested = false;
    r.summary.status = 'running';
    r.summary.error = null;
    for (const stage of r.summary.stages) if (stage.status === 'failed' || stage.status === 'interrupted') {
      stage.status = 'pending'; stage.jobId = null; stage.error = null;
    }
    this.write(r);
    this.schedule(() => { void this.run(r); });
    return r.summary;
  }

  private async run(r: RecordFile): Promise<void> {
    if (this.loops.has(r.summary.id)) return;
    this.loops.add(r.summary.id);
    try {
      for (const stage of r.summary.stages) {
        if (stage.status === 'done' || stage.status === 'skipped') continue;
        if (r.cancelRequested) break;
        while (r.pauseRequested && !r.cancelRequested) {
          r.summary.status = 'paused'; this.write(r);
          await this.delay(POLL_MS);
        }
        if (r.cancelRequested) break;
        r.summary.status = 'running';
        if (r.snapshot.dataset!.revision !== await this.adapter.datasetRevision(r.snapshot)) throw new Error('Dataset changed during preparation');
        const source = r.snapshot.sources[0];
        if (source.revision !== await this.adapter.sourceRevision(r.snapshot)) throw new Error('Dataset sources changed during preparation');
        const artifacts = await this.adapter.artifacts(r.snapshot, r.workerUrl);
        if (r.cancelRequested) break;
        if (this.skip(stage.stage, artifacts)) {
          stage.status = 'skipped'; this.write(r); continue;
        }
        stage.status = 'running'; this.write(r);
        // Persist a submitted jobId immediately. If Node dies during the POST
        // before it can record the reply, restart marks the stage interrupted.
        stage.jobId = await this.adapter.start(r.snapshot, stage.stage, this.body(r, stage.stage, artifacts), r.workerUrl);
        this.write(r);
        for (;;) {
          if (r.cancelRequested) await this.adapter.cancel(r.snapshot, stage.jobId, r.workerUrl);
          const job = await this.adapter.job(r.snapshot, stage.jobId, r.workerUrl);
          if (!job) throw new Error(`Submitted ${stage.stage} job ${stage.jobId} is missing`);
          if (job.status === 'done') { stage.status = 'done'; break; }
          if (job.status === 'failed' || job.status === 'cancelled') {
            stage.status = job.status;
            stage.error = job.error || `${stage.stage} ${job.status}`;
            break;
          }
          await this.delay(POLL_MS);
        }
        this.write(r);
        if (stage.status !== 'done') break;
      }
      if (r.cancelRequested) {
        for (const stage of r.summary.stages) if (stage.status === 'pending') stage.status = 'cancelled';
        r.summary.status = 'cancelled';
      } else if (r.summary.stages.some(s => s.status === 'failed' || s.status === 'cancelled')) {
        r.summary.status = 'failed';
        r.summary.error = r.summary.stages.find(s => s.error)?.error ?? 'A stage failed';
      } else r.summary.status = 'done';
    } catch (err) {
      const stage = r.summary.stages.find(s => s.status === 'running');
      if (stage) { stage.status = 'failed'; stage.error = err instanceof Error ? err.message : String(err); }
      r.summary.status = 'failed';
      r.summary.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.write(r);
      this.loops.delete(r.summary.id);
    }
  }

  private skip(stage: Yue2PreparationStage, a: Yue2PreparationArtifacts): boolean {
    switch (stage) {
      case 'latents': return a.preprocess.done && a.preprocess.captionModeOk && !a.preprocess.captionsStale;
      case 'codes': return a.tokenize.done;
      case 'sheet': return a.sheet.done;
      case 'stems': return a.align.stemsReady > 0;
      case 'align': return a.align.done;
      case 'nar': return a.narCompleted;
      case 'ar': return a.arCompleted;
      case 'joint': return false;
    }
  }

  private body(r: RecordFile, stage: Yue2PreparationStage, a: Yue2PreparationArtifacts): Record<string, unknown> {
    if (stage === 'latents') return { captionMode: 'yue2' };
    const form = stage === 'nar' || stage === 'ar' ? r.snapshot.payload.recipes[stage] : undefined;
    const formTrigger = form && Object.hasOwn(form.overrides, 'trigger') ? form.overrides.trigger : undefined;
    const trigger = typeof formTrigger === 'string' ? formTrigger.trim() : r.snapshot.payload.trigger.trim();
    if (stage === 'nar') return { ...r.trainBodies.nar, ...(trigger
      ? { trigger } : { allowNoTrigger: true }) };
    if (stage === 'ar') return { ...r.trainBodies.ar, ...(trigger
      ? { trigger } : { allowNoTrigger: true }),
      ...(a.minted.present ? {} : { allowNoMinted: true }) };
    if (stage === 'joint') return { ...r.trainBodies.joint };
    return {};
  }
}

const ROUTE: Record<Yue2PreparationStage, string> = {
  latents: 'yue2-preprocess', codes: 'yue2-tokenize', sheet: 'yue2-sheet', stems: 'yue2-stems',
  align: 'yue2-align', nar: 'yue2-train', ar: 'yue2-ar-train', joint: 'yue2-joint-train',
};

export const defaultYue2PreparationAdapter: Yue2PreparationAdapter = {
  async datasetRevision(snapshot) { return getDataset(snapshot.dataset!.id)?.updatedAt ?? null; },
  async sourceRevision(snapshot) {
    const ds = getDataset(snapshot.dataset!.id);
    if (!ds) throw new Error('Dataset disappeared');
    return sourceRevision(ds.sourceDir);
  },
  async admission(snapshot, workerUrl) {
    if (snapshot.worker.kind === 'remote') {
      const { getWorker, workerStatus } = await import('./trainingWorkers.js');
      const worker = getWorker(snapshot.worker.name);
      if (!worker || worker.url !== workerUrl) throw new Error(`Training worker ${snapshot.worker.name} was replaced or removed`);
      const status = await workerStatus(worker);
      if (!status.online || !status.versionMatch) throw new Error(`Training worker ${worker.name} is unavailable or has a different version`);
      const ds = getDataset(snapshot.dataset!.id);
      if (!ds) throw new Error('Dataset disappeared');
      await pushDataset(worker, ds);
    }
  },
  async artifacts(snapshot, workerUrl) {
    const id = encodeURIComponent(snapshot.dataset!.id);
    const base = `/api/training/datasets/${id}`;
    const ar = await request(snapshot, `${base}/yue2-ar`, {}, workerUrl) as any;
    const nar = await request(snapshot, `${base}/yue2-runs`, {}, workerUrl) as any;
    const arRuns = await request(snapshot, `${base}/yue2-ar-runs`, {}, workerUrl) as any;
    return { preprocess: ar.stages.preprocess, tokenize: ar.stages.tokenize,
      sheet: ar.stages.sheet, align: ar.stages.align, minted: ar.minted,
      narCompleted: nar.runs?.[0]?.outcome === 'completed',
      arCompleted: arRuns.runs?.[0]?.outcome === 'completed' };
  },
  async start(snapshot, stage, body, workerUrl) {
    const id = encodeURIComponent(snapshot.dataset!.id);
    const result = await request(snapshot, `/api/training/datasets/${id}/${ROUTE[stage]}`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    }, workerUrl) as { jobId?: string };
    if (!result.jobId) throw new Error(`${stage} returned no jobId`);
    return result.jobId;
  },
  async job(snapshot, id, workerUrl) {
    try { return await request(snapshot, `/api/training/jobs/${encodeURIComponent(id)}`, {}, workerUrl) as { status: JobState; error?: string | null }; }
    catch (err: any) { if (err?.status === 404) return null; throw err; }
  },
  async cancel(snapshot, id, workerUrl) {
    if (snapshot.worker.kind === 'local') { cancelJob(id); return; }
    await request(snapshot, `/api/training/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }, workerUrl);
  },
};

/** Signature of the exact source tree used by preparation. Generated training
 *  caches live outside the source folder, so this stays stable across stages. */
export function sourceRevision(root: string): string {
  const hash = createHash('sha256');
  const walk = (dir: string, relative: string): void => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const rel = path.join(relative, entry.name);
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(file, rel);
      else if (entry.isFile()) {
        const stat = fs.statSync(file);
        hash.update(`${rel.replaceAll('\\', '/')}:${stat.size}:${stat.mtimeMs}\n`);
      }
    }
  };
  walk(root, '');
  return hash.digest('hex');
}

async function request(snapshot: Snapshot, url: string, init: RequestInit = {}, workerUrl: string | null = null): Promise<unknown> {
  let response: Response;
  if (snapshot.worker.kind === 'remote') {
    const { getWorker } = await import('./trainingWorkers.js');
    const worker = getWorker(snapshot.worker.name);
    if (!worker || worker.url !== workerUrl) throw new Error(`Training worker ${snapshot.worker.name} was replaced or removed`);
    response = await workerFetch(worker, url, init);
  } else response = await fetch(`http://127.0.0.1:${config.server.port}${url}`, init);
  const body = await response.json().catch(() => ({})) as { error?: string };
  if (!response.ok) throw Object.assign(new Error(body.error || `HTTP ${response.status}`), { status: response.status });
  return body;
}

let instance: Yue2PreparationPipeline | undefined;
export function getYue2PreparationPipeline(): Yue2PreparationPipeline {
  if (!instance) {
    instance = new Yue2PreparationPipeline(path.join(trainingBaseDir, 'yue2-preparation'), defaultYue2PreparationAdapter);
    registerYue2PreparationActivity(() => instance!.list().some(s => active(s.status)));
  }
  return instance;
}
