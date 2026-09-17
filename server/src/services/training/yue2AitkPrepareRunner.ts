// CPU-only bridge from the existing HOT-Step YuE2 cache stages to the native
// AITK schema-1 dataset. This is deliberately a separate job kind: it must not
// stop ace-server, claim a GPU, or fall through to a Legacy training stage.

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import readline from 'readline';
import { aceTrainExe } from './aceTrain.js';
import { emitProgress, finishJob, isCancelled, pushEvent, type TrainingJob } from './labelingQueue.js';
import { log } from './yue2TrainRunner.js';

export type AitkPrepareModelName = 'vae' | 'semantic' | 'sheetsage';

export interface ResolvedYue2AitkPrepareOptions {
  legacyManifest: string;
  checkpoint: string;
  tokenizer: string;
  output: string;
  models: Record<AitkPrepareModelName, string>;
}

/** Parse the CLI-shaped repeatable `--model name=path` request field. */
export function parseYue2AitkModelArgs(value: unknown): Record<AitkPrepareModelName, string> | null {
  const entries = Array.isArray(value) ? value : (typeof value === 'string' ? [value] : []);
  const out: Partial<Record<AitkPrepareModelName, string>> = {};
  for (const item of entries) {
    if (typeof item !== 'string') return null;
    const split = item.indexOf('=');
    if (split <= 0) return null;
    const name = item.slice(0, split).trim() as AitkPrepareModelName;
    const file = item.slice(split + 1).trim();
    if (!MODEL_NAMES.includes(name) || !file || out[name]) return null;
    out[name] = file;
  }
  return MODEL_NAMES.every(name => !!out[name]) ? out as Record<AitkPrepareModelName, string> : null;
}

export function parseYue2AitkModels(value: unknown, repeatable: unknown): Record<AitkPrepareModelName, string> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const object = value as Record<string, unknown>;
    const out: Partial<Record<AitkPrepareModelName, string>> = {};
    for (const name of MODEL_NAMES) {
      if (typeof object[name] !== 'string' || !(object[name] as string).trim()) return null;
      out[name] = (object[name] as string).trim();
    }
    if (Object.keys(object).some(name => !MODEL_NAMES.includes(name as AitkPrepareModelName))) return null;
    return out as Record<AitkPrepareModelName, string>;
  }
  return parseYue2AitkModelArgs(repeatable ?? value);
}

const MODEL_NAMES: readonly AitkPrepareModelName[] = ['vae', 'semantic', 'sheetsage'];
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_OUTPUT_MANIFEST_BYTES = 16 * 1024 * 1024;

function regularFile(file: string, label: string, maxBytes?: number): string | null {
  if (!file) return `${label} is required`;
  try {
    const st = fs.statSync(file);
    if (!st.isFile()) return `${label} is not a regular file: ${file}`;
    if (st.size <= 0) return `${label} is empty: ${file}`;
    if (maxBytes !== undefined && st.size > maxBytes) return `${label} exceeds ${maxBytes} bytes: ${file}`;
  } catch { return `${label} is missing: ${file}`; }
  return null;
}

function fileOrDirectory(file: string, label: string): string | null {
  if (!file) return `${label} is required`;
  try {
    const st = fs.statSync(file);
    if (!st.isFile() && !st.isDirectory()) return `${label} is not a file or directory: ${file}`;
  } catch { return `${label} is missing: ${file}`; }
  return null;
}

export function buildYue2AitkPrepareArgs(o: ResolvedYue2AitkPrepareOptions): string[] {
  const args = [
    'yue2-prepare-aitk', '--legacy-manifest', o.legacyManifest,
    '--checkpoint', o.checkpoint, '--tokenizer', o.tokenizer, '--output', o.output,
  ];
  for (const name of MODEL_NAMES) args.push('--model', `${name}=${o.models[name]}`);
  return args;
}

export function validateYue2AitkPrepareOptions(o: ResolvedYue2AitkPrepareOptions): string | null {
  const manifestError = regularFile(o.legacyManifest, 'legacy manifest', MAX_MANIFEST_BYTES);
  if (manifestError) return manifestError;
  const checkpointError = regularFile(o.checkpoint, 'raw ConvRot checkpoint');
  if (checkpointError) return checkpointError;
  const tokenizerError = fileOrDirectory(o.tokenizer, 'tokenizer');
  if (tokenizerError) return tokenizerError;
  if (!o.output) return 'output directory is required';
  if (fs.existsSync(o.output)) return `output directory already exists; choose a new directory: ${o.output}`;
  if (!fs.existsSync(path.dirname(o.output)) || !fs.statSync(path.dirname(o.output)).isDirectory()) return `output parent directory is missing: ${path.dirname(o.output)}`;
  const names = Object.keys(o.models).sort();
  if (names.length !== MODEL_NAMES.length || names.some(name => !MODEL_NAMES.includes(name as AitkPrepareModelName))) {
    return 'models must contain exactly vae, semantic, and sheetsage';
  }
  for (const name of MODEL_NAMES) {
    const error = regularFile(o.models[name], `${name} model`);
    if (error) return error;
  }
  return null;
}

function parseProgress(job: TrainingJob, line: string): void {
  try {
    const event = JSON.parse(line) as Record<string, unknown>;
    if (typeof event.stage === 'string') job.phase = event.stage;
    if (typeof event.done === 'number' && Number.isFinite(event.done)) job.done = Math.max(0, Math.min(1, Math.trunc(event.done)));
    if (typeof event.total === 'number' && Number.isFinite(event.total)) job.total = Math.max(1, Math.trunc(event.total));
    emitProgress(job);
  } catch {
    log(job, 'info', line);
  }
}

export async function runYue2AitkPrepareJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2AitkPrepareOptions | undefined;
  const validation = opts ? validateYue2AitkPrepareOptions(opts) : 'job is missing AITK preparation options';
  if (validation) { finishJob(job, 'failed', validation); return; }
  const o = opts!;
  const exe = aceTrainExe();
  if (!exe) { finishJob(job, 'failed', 'ace-train is not in this build — rebuild the engine'); return; }

  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'preparing';
  job.total = 1;
  emitProgress(job);
  const args = buildYue2AitkPrepareArgs(o);
  pushEvent(job, { type: 'log', level: 'info', message: `Starting CPU-only AITK preparation: ${exe} ${args[0]}`, ts: Date.now() });

  const child = spawn(exe, args, {
    windowsHide: true,
    env: { ...process.env, CUDA_VISIBLE_DEVICES: '' },
  });
  job.child = child;
  const tail: string[] = [];
  const handle = (line: string): void => {
    const trimmed = line.trim();
    if (!trimmed) return;
    tail.push(trimmed);
    if (tail.length > 20) tail.shift();
    parseProgress(job, trimmed);
  };
  const rlErr = readline.createInterface({ input: child.stderr! });
  const rlOut = readline.createInterface({ input: child.stdout! });
  rlErr.on('line', handle); rlOut.on('line', handle);
  try {
    const code = await new Promise<number | null>((resolve, reject) => {
      child.on('error', reject);
      child.on('close', (value, signal) => resolve(signal ? null : value));
    });
    if (isCancelled(job)) return;
    if (code !== 0) throw new Error(`yue2-prepare-aitk exited with code ${code === null ? 'null (killed)' : code}${tail.length ? `: ${tail.slice(-3).join(' | ')}` : ''}`);
    const manifest = path.join(o.output, 'dataset.json');
    const error = regularFile(manifest, 'prepared dataset manifest', MAX_OUTPUT_MANIFEST_BYTES);
    if (error) throw new Error(`AITK preparation finished without a valid output: ${error}`);
    const value = JSON.parse(fs.readFileSync(manifest, 'utf8')) as Record<string, unknown>;
    if (value.schema_version !== 1 || !Array.isArray(value.items)) throw new Error('AITK preparation wrote an invalid schema-1 dataset manifest');
    job.done = 1; job.phase = 'done'; emitProgress(job); finishJob(job, 'done');
  } catch (err: unknown) {
    if (!isCancelled(job)) finishJob(job, 'failed', err instanceof Error ? err.message : String(err));
  } finally {
    try { rlErr.close(); } catch { /* already closed */ }
    try { rlOut.close(); } catch { /* already closed */ }
    job.child = undefined;
  }
}
