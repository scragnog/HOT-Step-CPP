#!/usr/bin/env npx tsx
/**
 * lm-train-arm.ts — train one planner-LM adapter "arm" through the Node server's
 * train-lm job queue (docs/plans/lm-attr-probe/HANDOFF.md, the caption-dropout /
 * prior-preservation follow-up).
 *
 * Unlike lm-adapter-eval.ts and lm-plan-render.ts, which talk to the C++ engine
 * directly via aceClient, training is owned by the NODE SERVER
 * (POST /api/training/datasets/:id/train-lm — see server/src/routes/training.ts):
 * it stops ace-server for the run and restarts it afterward, so this script talks
 * HTTP to the server (default http://localhost:3001), not the engine.
 *
 *   npx tsx scripts/lm-train-arm.ts --dataset <slug> --arm <name> --opts '<json>'
 *                                   [--variant <key>] [--lm-size 4B] [--ledger <path>]
 *                                   [--adapter-name <name>] [--dry-run]
 *
 *   --dataset <slug>     training dataset slug (tensors/<slug>) — required
 *   --arm <name>         arbitrary arm label (e.g. ctrl, pp, cd, both) — required,
 *                        only used to name the adapter and the ledger key
 *   --opts <json>        JSON object merged into the train-lm POST body ON TOP of
 *                        the server's own defaults (default '{}') — e.g.
 *                        '{"regEvery":3,"captionDropout":0.3}'
 *   --variant <key>      tensor preprocess variant (default: server picks newest)
 *   --lm-size <size>     0.6B | 1.7B | 4B (default 4B, matches the server default)
 *   --adapter-name <n>   adapterName sent to the server (default "<slug>-<arm>")
 *   --ledger <path>      JSON ledger file (default docs/plans/lm-attr-probe/
 *                        train-arm-ledger.json) — resumable: an arm already
 *                        recorded there with a run dir that still has weights on
 *                        disk is skipped without hitting the server
 *   --server <url>       Node server base URL (default http://localhost:3001)
 *   --dry-run            print the resolved dataset id, POST URL and body, and the
 *                        ledger key, then exit — no HTTP calls
 *
 * Prints the resulting adapter run dir on stdout as the LAST line on success, so
 * a caller can capture it with `$out = ... | Select-Object -Last 1` etc. Also
 * records it in the ledger for run_experiment.ps1's own resume check.
 *
 * The engine is stopped by the server for the whole training run and restarted
 * once it finishes (success, failure or cancel) — polling here only hits the
 * Node server's job-status endpoint, never the engine, so it keeps working while
 * ace-server is down.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_LEDGER = path.join(REPO_ROOT, 'docs', 'plans', 'lm-attr-probe', 'train-arm-ledger.json');
const POLL_MS = 5_000;
// Generous ceiling: a staged 2.0->1.5 4B LoKR run over ~20 songs is well under
// this on the hardware this has been measured on; a wedged job should still be
// caught well short of a day.
const JOB_DEADLINE_MS = 20 * 60 * 60_000;

function die(msg: string): never {
  console.error(`\nERROR: ${msg}`);
  process.exit(1);
}

function parseArgs(argv: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith('--')) { out.set(a.slice(2), next); i++; }
    else out.set(a.slice(2), 'true');
  }
  return out;
}

interface Ledger {
  [key: string]: {
    dataset: string; arm: string; adapterName: string; lmSize: string;
    opts: Record<string, unknown>; jobId: string; runDir: string;
    trainedAt: string; finalLoss: number; targetLoss: number; recordedAt: string;
  };
}

function readLedger(ledgerPath: string): Ledger {
  if (!fs.existsSync(ledgerPath)) return {};
  try {
    return JSON.parse(fs.readFileSync(ledgerPath, 'utf-8')) as Ledger;
  } catch {
    die(`${ledgerPath} exists but is not valid JSON — fix or remove it`);
  }
}

function writeLedger(ledgerPath: string, ledger: Ledger): void {
  fs.mkdirSync(path.dirname(ledgerPath), { recursive: true });
  const tmp = `${ledgerPath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(ledger, null, 2));
  fs.renameSync(tmp, ledgerPath);
}

function hasWeights(dir: string): boolean {
  return fs.existsSync(path.join(dir, 'adapter_model.safetensors'))
    || fs.existsSync(path.join(dir, 'lokr_weights.safetensors'));
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  let body: unknown;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text }; }
  if (!res.ok) {
    const msg = (body as { error?: string })?.error ?? text ?? `HTTP ${res.status}`;
    throw new Error(`${init?.method ?? 'GET'} ${url} -> ${res.status}: ${msg}`);
  }
  return body as T;
}

interface DatasetRow { id: string; slug: string }
interface JobSummary { id: string; status: string; phase: string; done: number; total: number; error: string | null }
interface TrainLmStatusRes {
  adapterDir: string; adapterExists: boolean; trainedAt: string; finalLoss: number; targetLoss: number;
}

async function resolveDatasetId(server: string, slug: string): Promise<string> {
  const { datasets } = await fetchJson<{ datasets: DatasetRow[] }>(`${server}/api/training/datasets`);
  const ds = datasets.find(d => d.slug === slug);
  if (!ds) die(`no training dataset with slug "${slug}" — check /api/training/datasets`);
  return ds.id;
}

async function pollJob(server: string, jobId: string): Promise<JobSummary> {
  const deadline = Date.now() + JOB_DEADLINE_MS;
  let lastPrint = 0;
  let consecutiveErrors = 0;
  for (;;) {
    // A transient fetch failure (the engine restarting inside the job, a brief
    // server hiccup) must not kill a multi-hour experiment: the 2026-09-03
    // overnight run lost eight arms to exactly that. Retry for ~5 minutes.
    let job: JobSummary;
    try {
      job = await fetchJson<JobSummary>(`${server}/api/training/jobs/${jobId}`);
      consecutiveErrors = 0;
    } catch (e) {
      consecutiveErrors++;
      if (consecutiveErrors >= 60) die(`job ${jobId}: ${consecutiveErrors} consecutive poll failures — ${(e as Error).message}`);
      if (consecutiveErrors === 1 || consecutiveErrors % 12 === 0) {
        process.stdout.write(`  [job ${jobId}] poll error (${consecutiveErrors}): ${(e as Error).message} — retrying\n`);
      }
      await new Promise(r => setTimeout(r, POLL_MS));
      continue;
    }
    if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') return job;
    if (Date.now() - lastPrint > 30_000) {
      process.stdout.write(`  [job ${jobId}] ${job.status} phase=${job.phase} ${job.done}/${job.total || '?'}\n`);
      lastPrint = Date.now();
    }
    if (Date.now() > deadline) die(`job ${jobId} did not finish within ${JOB_DEADLINE_MS / 3_600_000}h`);
    await new Promise(r => setTimeout(r, POLL_MS));
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const slug = args.get('dataset') ?? '';
  const arm = args.get('arm') ?? '';
  if (!slug || !arm) die('needs --dataset <slug> and --arm <name>');

  let opts: Record<string, unknown> = {};
  // --opts-b64 exists because PowerShell strips the inner quotes of a JSON object passed as a
  // native-command argument ({"regEvery":3} arrives as {regEvery:3}); the .ps1 runner uses it.
  const optsB64 = args.get('opts-b64');
  const optsRaw = optsB64 ? Buffer.from(optsB64, 'base64').toString('utf-8') : args.get('opts');
  if (optsRaw) {
    try { opts = JSON.parse(optsRaw) as Record<string, unknown>; } catch { die(`--opts is not valid JSON: ${optsRaw}`); }
  }
  const lmSize = args.get('lm-size') || '4B';
  const variant = args.get('variant') || '';
  const adapterName = args.get('adapter-name') || `${slug}-${arm}`;
  const server = (args.get('server') || 'http://localhost:3001').replace(/\/$/, '');
  const ledgerPath = path.resolve(args.get('ledger') || DEFAULT_LEDGER);
  const dryRun = args.has('dry-run');
  const ledgerKey = `${slug}|${arm}`;

  const body: Record<string, unknown> = { ...opts, adapterName, lmSize };
  if (variant) body.variantKey = variant;

  const ledger = readLedger(ledgerPath);
  const existing = ledger[ledgerKey];
  if (existing && existing.runDir && hasWeights(existing.runDir)) {
    console.log(`[skip] ${ledgerKey}: already trained -> ${existing.runDir}`);
    console.log(existing.runDir);
    return;
  }

  if (dryRun) {
    console.log(`[dry-run] resolve dataset id for slug "${slug}" via GET ${server}/api/training/datasets`);
    console.log(`[dry-run] POST ${server}/api/training/datasets/<id>/train-lm`);
    console.log(`[dry-run] body: ${JSON.stringify(body)}`);
    console.log(`[dry-run] poll GET ${server}/api/training/jobs/<jobId> every ${POLL_MS}ms until done/failed/cancelled`);
    console.log(`[dry-run] on completion: GET ${server}/api/training/datasets/<id>/train-lm`
      + `?adapterName=${encodeURIComponent(adapterName)}&lmSize=${lmSize}`);
    console.log(`[dry-run] ledger key: ${ledgerKey} -> ${ledgerPath}`);
    return;
  }

  const datasetId = await resolveDatasetId(server, slug);
  console.log(`\nTraining arm "${arm}" for ${slug} (dataset id ${datasetId})`);
  console.log(`  adapterName: ${adapterName}  lmSize: ${lmSize}${variant ? `  variant: ${variant}` : ''}`);
  console.log(`  opts: ${JSON.stringify(opts)}`);

  const { jobId } = await fetchJson<{ jobId: string }>(
    `${server}/api/training/datasets/${datasetId}/train-lm`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
  );
  console.log(`  job queued: ${jobId} — the app's engine will stop for the duration and restart after`);

  const job = await pollJob(server, jobId);
  if (job.status !== 'done') {
    die(`job ${jobId} ended ${job.status}${job.error ? `: ${job.error}` : ''}`);
  }

  // adapterDir is computed server-side at request time (timestamped run dir) and
  // never echoed by the job-status endpoint — read it back the same way the
  // Training Studio UI does, off disk, by (adapterName, lmSize).
  const status = await fetchJson<TrainLmStatusRes>(
    `${server}/api/training/datasets/${datasetId}/train-lm`
    + `?adapterName=${encodeURIComponent(adapterName)}&lmSize=${lmSize}`,
  );
  if (!status.adapterExists || !status.adapterDir) {
    die(`job ${jobId} reported done but no adapter weights were found for ${adapterName} (${lmSize}) — ${status.adapterDir}`);
  }

  ledger[ledgerKey] = {
    dataset: slug, arm, adapterName, lmSize, opts, jobId,
    runDir: status.adapterDir, trainedAt: status.trainedAt,
    finalLoss: status.finalLoss, targetLoss: status.targetLoss,
    recordedAt: new Date().toISOString(),
  };
  writeLedger(ledgerPath, ledger);

  console.log(`  done: finalLoss=${status.finalLoss} targetLoss=${status.targetLoss} trainedAt=${status.trainedAt}`);
  console.log(`  ledger updated: ${ledgerPath}`);
  console.log(status.adapterDir);
}

main().catch(err => die(err instanceof Error ? err.message : String(err)));
