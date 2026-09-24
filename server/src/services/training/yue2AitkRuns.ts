// Durable catalogue for native AITK YuE2 joint runs. The native trainer owns
// its output directory; this index only records paths after that directory has
// appeared, so a failed launch cannot reserve or overwrite a user directory.

import fs from 'fs';
import path from 'path';
import { trainingBaseDir } from './paths.js';
import { runStamp } from './adapterLayout.js';

export function yue2JointOutputDirectory(adaptersRoot: string, trigger: string, when = new Date()): string {
  const name = trigger.trim().replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^[. ]+|[. ]+$/g, '').slice(0, 120) || 'dataset';
  return path.join(adaptersRoot, 'yue2-joint-adapters', `${name}_${runStamp(when)}`);
}

const INDEX = path.join(trainingBaseDir, 'yue2-aitk-runs.json');
const MAX_RECORDS = 256;
const MAX_INDEX_BYTES = 4 * 1024 * 1024;

export interface Yue2AitkCheckpointRecord {
  step: number;
  dir: string;
  loss?: number;
  /** From the checkpoint's meters.json: the planner's KL stop reading (or
   *  its 20-step mean), the decoder's reconstruction meter, frozen state. */
  kl?: number;
  recon?: number;
  frozen?: boolean;
  /** Written at a KL rung (--kl-checkpoint-every), not a routine save. */
  rung?: boolean;
  adapterPath?: string;
  optimizerPath?: string;
  arPath?: string;
  narPath?: string;
}

export interface Yue2AitkRunRecord {
  version: 1;
  jobId: string;
  datasetId: string;
  datasetSlug: string;
  method: 'aitk';
  output: string;
  options: Record<string, unknown>;
  status: 'running' | 'done' | 'failed' | 'cancelled';
  createdAt: number;
  updatedAt: number;
  error?: string;
  checkpoints: Yue2AitkCheckpointRecord[];
}

function readIndex(): Yue2AitkRunRecord[] {
  try {
    const stat = fs.statSync(INDEX);
    if (!stat.isFile() || stat.size > MAX_INDEX_BYTES) return [];
    const value = JSON.parse(fs.readFileSync(INDEX, 'utf8')) as unknown;
    if (!Array.isArray(value)) return [];
    return value.filter(isRunRecord);
  } catch { return []; }
}

function isRunRecord(value: unknown): value is Yue2AitkRunRecord {
  if (!value || typeof value !== 'object') return false;
  const r = value as Yue2AitkRunRecord;
  return r.version === 1 && typeof r.jobId === 'string' && r.jobId.length <= 128
    && typeof r.datasetId === 'string' && typeof r.datasetSlug === 'string'
    && r.method === 'aitk' && typeof r.output === 'string' && r.output.length <= 32768
    && (r.status === 'running' || r.status === 'done' || r.status === 'failed' || r.status === 'cancelled')
    && Number.isFinite(r.createdAt) && Number.isFinite(r.updatedAt)
    && Array.isArray(r.checkpoints) && r.checkpoints.length <= 1024
    && r.checkpoints.every(c => !!c && Number.isInteger(c.step) && c.step >= 0
      && typeof c.dir === 'string' && c.dir.length <= 32768
      && (c.loss === undefined || (typeof c.loss === 'number' && Number.isFinite(c.loss)))
      && ['adapterPath', 'optimizerPath', 'arPath', 'narPath'].every(k => {
        const v = c[k as keyof Yue2AitkCheckpointRecord];
        return v === undefined || (typeof v === 'string' && v.length <= 32768);
      }));
}

function writeIndex(records: Yue2AitkRunRecord[]): void {
  fs.mkdirSync(path.dirname(INDEX), { recursive: true });
  const tmp = `${INDEX}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(records.slice(-MAX_RECORDS), null, 2), 'utf8');
  fs.renameSync(tmp, INDEX);
}

export function aitkRunIndexPath(): string { return INDEX; }

export function checkpointRecords(output: string): Yue2AitkCheckpointRecord[] {
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(output, { withFileTypes: true }); } catch { return []; }
  const rows: Yue2AitkCheckpointRecord[] = [];
  const dirs = [output];
  const segments = entries.find(e => e.isDirectory() && e.name === 'segments');
  if (segments) {
    try { for (const e of fs.readdirSync(path.join(output, 'segments'), { withFileTypes: true })) {
      if (e.isDirectory() && /^segment-\d{6}$/.test(e.name)) dirs.push(path.join(output, 'segments', e.name));
    } } catch { /* incomplete catalogue is handled by the caller */ }
  }
  for (const base of dirs) {
   const losses = new Map<number, number>();
   try {
     const log = path.join(base, 'train.jsonl');
     if (fs.statSync(log).size <= 8 * 1024 * 1024) {
       for (const line of fs.readFileSync(log, 'utf8').split(/\r?\n/)) {
         try {
           const event = JSON.parse(line) as Record<string, unknown>;
           if (event.stage !== 'joint' || !Number.isInteger(event.step)) continue;
           const { ar_ce, ar_kl, nar_mse, cursor_ce, cursor_weight } = event;
           if (![ar_ce, ar_kl, nar_mse].every(v => typeof v === 'number' && Number.isFinite(v))) continue;
           const cursor = typeof cursor_ce === 'number' && Number.isFinite(cursor_ce)
             && typeof cursor_weight === 'number' && Number.isFinite(cursor_weight)
             ? cursor_ce * cursor_weight : 0;
           losses.set(event.step as number, (ar_ce as number) + 0.2 * (ar_kl as number) + (nar_mse as number) + cursor);
         } catch { /* an incomplete log line does not invalidate other steps */ }
       }
     }
   } catch { /* an unfinished segment may not have its JSONL log yet */ }
   let local: fs.Dirent[];
   try { local = fs.readdirSync(base, { withFileTypes: true }); } catch { continue; }
   for (const e of local) {
    const match = e.isDirectory() ? /^checkpoint-step(\d+)$/.exec(e.name) : null;
    if (!match) continue;
    const dir = path.join(base, e.name);
    const file = (name: string): string | undefined => {
      const candidate = path.join(dir, name);
      return fs.existsSync(candidate) && fs.statSync(candidate).isFile() ? candidate : undefined;
    };
    let meters: Record<string, unknown> = {};
    try { meters = JSON.parse(fs.readFileSync(path.join(dir, 'meters.json'), 'utf8')) as Record<string, unknown>; } catch { /* older checkpoints have none */ }
    const num = (v: unknown) => typeof v === 'number' && Number.isFinite(v) ? v : undefined;
    const kl = num(meters.kl_reading) ?? num(meters.ar_kl_mean20);
    rows.push({
      step: Number(match[1]), dir,
      ...(losses.has(Number(match[1])) ? { loss: losses.get(Number(match[1])) } : {}),
      ...(kl !== undefined ? { kl } : {}),
      ...(num(meters.nar_recon) !== undefined ? { recon: num(meters.nar_recon) } : {}),
      ...(meters.planner_frozen === true ? { frozen: true } : {}),
      ...(meters.kl_rung === true ? { rung: true } : {}),
      adapterPath: file('adapter.safetensors'), optimizerPath: file('optimizer.resume'),
      arPath: file('native-ar.safetensors'), narPath: file('native-nar.safetensors'),
    });
   }
  }
  return rows.sort((a, b) => b.step - a.step);
}

export function recordYue2AitkRun(record: Yue2AitkRunRecord): void {
  try {
    const prior = readIndex().filter(r => r.jobId !== record.jobId);
    writeIndex([...prior, { ...record, checkpoints: checkpointRecords(record.output) }]);
  } catch { /* a catalogue failure must never change the training result */ }
}

export function listYue2AitkRuns(datasetId: string, datasetSlug?: string): Yue2AitkRunRecord[] {
  return readIndex().filter(r => r.datasetId === datasetId || (!!datasetSlug && r.datasetSlug === datasetSlug))
    .map(r => ({ ...r, checkpoints: checkpointRecords(r.output) }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

/** Every indexed joint run, newest first, with checkpoint files re-scanned.
 * The generation picker has no dataset filter, so it must use the same durable
 * index and checkpoint scanner as Training Studio rather than walking a second
 * directory tree. */
export function listAllYue2AitkRuns(): Yue2AitkRunRecord[] {
  return readIndex()
    .map(r => ({ ...r, checkpoints: checkpointRecords(r.output) }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

/** Resolve only paths present in the durable joint catalogue. This is shared by
 * the model picker, prompt builder and caption-source route. */
export function jointRunForAdapter(adapterPath: string): Yue2AitkRunRecord | undefined {
  if (!adapterPath) return undefined;
  const wanted = path.resolve(adapterPath).toLowerCase();
  return listAllYue2AitkRuns().find(run => run.checkpoints.some(checkpoint =>
    [checkpoint.arPath, checkpoint.narPath].some(ref => ref && path.resolve(ref).toLowerCase() === wanted)));
}
