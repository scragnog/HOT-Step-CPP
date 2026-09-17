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
    rows.push({
      step: Number(match[1]), dir,
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
