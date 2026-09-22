import fs from 'fs';
import path from 'path';
import { createHash, randomUUID } from 'crypto';
import { isCancelled, emitProgress, type TrainingJob } from './labelingQueue.js';
import { log } from './yue2TrainRunner.js';
import { refreshYue2ManifestCaptions, runYue2AitkPrepareJob, type ResolvedYue2AitkPrepareOptions } from './yue2AitkPrepareRunner.js';

function stamp(file: string): unknown {
  const st = fs.statSync(file);
  if (!st.isFile()) throw new Error(`Preparation input is not a file: ${file}`);
  return [path.resolve(file), st.size, st.mtimeMs, st.ctimeMs];
}

/** Hash the small manifest; use filesystem identity for large immutable weights/caches. */
export function preparationFingerprint(o: ResolvedYue2AitkPrepareOptions): string {
  if (fs.statSync(o.legacyManifest).size > 16 * 1024 * 1024) throw new Error('Legacy manifest exceeds 16 MiB');
  const text = fs.readFileSync(o.legacyManifest, 'utf8');
  const legacy = JSON.parse(text);
  const files = [o.checkpoint, ...Object.values(o.models)];
  if (fs.statSync(o.tokenizer).isDirectory()) files.push(path.join(o.tokenizer, 'vocab.json'), path.join(o.tokenizer, 'merges.txt'));
  else files.push(o.tokenizer);
  for (const source of legacy.sources ?? []) {
    for (const key of ['latents', 'codec_ids', ...(o.lyricTiming === false ? [] : ['cursor_words'])]) {
      if (typeof source[key] === 'string' && source[key]) files.push(path.resolve(path.dirname(o.legacyManifest), source[key]));
    }
  }
  return createHash('sha256').update(JSON.stringify({ version: 2, text, timing: o.lyricTiming !== false, trigger: o.trigger || '',
    files: [...new Set(files)].sort().map(stamp) })).digest('hex');
}

function outputFingerprint(manifest: string): string {
  if (fs.statSync(manifest).size > 16 * 1024 * 1024) throw new Error('Prepared manifest exceeds 16 MiB');
  const text = fs.readFileSync(manifest, 'utf8');
  const data = JSON.parse(text);
  if (data.schema_version !== 1 || data.recipe_version !== 'aitk-yue2-2026-09-16' || !Array.isArray(data.items) || !data.items.length) {
    throw new Error('Invalid prepared dataset');
  }
  const payloads = data.items.map((item: { latent_file: string }) => stamp(path.resolve(path.dirname(manifest), item.latent_file)));
  return createHash('sha256').update(text).update(JSON.stringify(payloads)).digest('hex');
}

/** One server-owned job covers preparation and training, including cancellation. */
export async function ensureYue2PreparedDataset(job: TrainingJob, options: ResolvedYue2AitkPrepareOptions,
  prepare = runYue2AitkPrepareJob): Promise<string | undefined> {
  if (isCancelled(job)) return;
  job.status = 'running'; job.startedAt ??= Date.now();
  job.phase = 'preparing'; job.done = 0; job.total = 1; emitProgress(job);
  // Before the fingerprint: a refresh rewrites the manifest, which is exactly
  // what must invalidate a prepared dataset built from the stale captions.
  await refreshYue2ManifestCaptions(job, options.legacyManifest);
  const fingerprint = preparationFingerprint(options);
  const index = path.join(path.dirname(options.legacyManifest), 'aitk-auto-prepare-v1.json');
  try {
    const cached = JSON.parse(fs.readFileSync(index, 'utf8'));
    if (cached.fingerprint === fingerprint && cached.outputFingerprint === outputFingerprint(cached.manifest)) {
      log(job, 'info', 'Reusing prepared YuE2 dataset; source caches and settings are unchanged.');
      return cached.manifest;
    }
  } catch { /* Missing, changed or incomplete cache: prepare a fresh immutable dataset. */ }
  const output = path.join(path.dirname(options.legacyManifest), `aitk-prepared-${randomUUID()}`);
  log(job, 'info', 'Preparing YuE2 dataset automatically before training.');
  await prepare(job, { ...options, output });
  if (isCancelled(job) || ['failed'].includes(job.status)) return;
  if (preparationFingerprint(options) !== fingerprint) throw new Error('Dataset changed during preparation; start training again.');
  const manifest = path.join(output, 'dataset.json');
  const record = { fingerprint, manifest, outputFingerprint: outputFingerprint(manifest) };
  const temp = `${index}.${randomUUID()}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(record));
  fs.renameSync(temp, index);
  return manifest;
}
