#!/usr/bin/env npx tsx
/**
 * lm-plan-render.ts — render planner-LM plans through the BASE DiT so their audio
 * attributes can be measured (tools/lm-attr-probe, Stream A-4).
 *
 * Input: a runs.json written by `lm-adapter-eval.ts generate` (base vs adapter
 * plans on the artist's own captions, identical seeds). For every plan, and for
 * the song's ground-truth codes ("gt" side), /synth is called CONDITIONED ON
 * THOSE CODES with the same caption/lyrics/seed, NO DiT adapter, fixed steps.
 * The DiT is therefore a constant decoder shared by all three sides; the only
 * thing that differs between sides is the plan.
 *
 *   npx tsx scripts/lm-plan-render.ts --runs <dir with runs.json> [--out <dir>] [--steps 8]
 *                                     [--sides gt,base,adapter] [--samples N] [--synth-model <name>]
 *
 * Output: <out>/<side>_<stem>_s<seed>.wav (16-bit) + <out>/renders.json manifest.
 * IDEMPOTENT: existing wavs are skipped. Needs the app running (dev.bat).
 */
import fs from 'fs';
import path from 'path';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';

const JOB_DEADLINE_MS = 10 * 60 * 1000;

interface EvalRow {
  id: string; file: string; caption: string; lyrics: string; bpm: number; duration: number;
  durUsed: number; gtCodes: number[];
}
interface EvalGen { rowId: string; side: 'base' | 'adapter'; seed: number; codes: number[] }
interface EvalRuns {
  dataset: string; variant: string; adapterPath: string; lmModel: string;
  params: { seeds: number; seedBase: number; maxDuration: number };
  rows: EvalRow[]; gens: EvalGen[];
}

function die(msg: string): never { console.error(msg); process.exit(1); }

function parseArgs(argv: string[]): Map<string, string> {
  const m = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) { m.set(a.slice(2), next); i++; } else m.set(a.slice(2), '1');
    }
  }
  return m;
}

async function awaitJob(jobId: string, what: string): Promise<void> {
  const deadline = Date.now() + JOB_DEADLINE_MS;
  for (;;) {
    if (Date.now() > deadline) { await aceClient.cancelJob(jobId).catch(() => {}); throw new Error(`${what} timed out`); }
    const st = await aceClient.pollJob(jobId);
    if (st.status === 'done') return;
    if (st.status === 'failed') throw new Error(`${what} failed — see ace_engine.log`);
    if (st.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 300));
  }
}

async function synthCodes(row: EvalRow, codes: number[], seed: number, steps: number, synthModel: string): Promise<Buffer> {
  const durSec = Math.max(10, Math.floor(codes.length / 5));
  const req: AceRequest = {
    caption: row.caption,
    lyrics: row.lyrics,
    duration: durSec,
    ...(row.bpm > 0 ? { bpm: row.bpm } : {}),
    seed,
    audio_codes: codes.slice(0, durSec * 5).join(','),
    inference_steps: steps,
    task_type: 'text2music',
    ...(synthModel ? { synth_model: synthModel } : {}),
    // deliberately NO adapter / adapter_scale: the DiT must be the bare base
  };
  const jobId = await aceClient.submitSynth(req, 'wav16');
  await awaitJob(jobId, `synth (${row.file}, seed ${seed})`);
  const res = await aceClient.getJobResult(jobId);
  if (!res.ok) throw new Error(`synth result fetch failed (${res.status})`);
  return Buffer.from(await res.arrayBuffer());
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const runsDir = args.get('runs') ?? '';
  if (!runsDir) die('--runs <dir with runs.json> is required');
  const runsPath = path.join(runsDir, 'runs.json');
  if (!fs.existsSync(runsPath)) die(`no runs.json in ${runsDir}`);
  const runs = JSON.parse(fs.readFileSync(runsPath, 'utf-8')) as EvalRuns;
  const outDir = path.resolve(args.get('out') ?? path.join(runsDir, '..', 'renders'));
  fs.mkdirSync(outDir, { recursive: true });
  const steps = Math.max(1, Math.trunc(Number(args.get('steps')) || 8));
  const sides = (args.get('sides') ?? 'gt,base,adapter').split(',').filter(Boolean);
  const synthModel = args.get('synth-model') ?? '';
  const nSamples = Math.trunc(Number(args.get('samples')) || 0);

  const rows = new Map(runs.rows.map(r => [r.id, r]));
  let rowIds = runs.rows.map(r => r.id);
  if (nSamples > 0 && nSamples < rowIds.length) {
    const step = rowIds.length / nSamples;
    rowIds = Array.from({ length: nSamples }, (_, i) => rowIds[Math.floor(i * step)]);
  }
  const keep = new Set(rowIds);
  const seedBase = runs.params.seedBase;

  type Work = { side: string; row: EvalRow; seed: number; codes: number[] };
  const work: Work[] = [];
  if (sides.includes('gt')) {
    for (const id of rowIds) {
      const r = rows.get(id)!;
      work.push({ side: 'gt', row: r, seed: seedBase, codes: r.gtCodes.slice(0, r.durUsed * 5) });
    }
  }
  for (const g of runs.gens) {
    if (!sides.includes(g.side) || !keep.has(g.rowId)) continue;
    work.push({ side: g.side, row: rows.get(g.rowId)!, seed: g.seed, codes: g.codes });
  }
  const stemOf = (f: string) => path.basename(f).replace(/\.safetensors$/i, '').replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 80);
  const manifestPath = path.join(outDir, 'renders.json');
  const manifest: Array<{ side: string; rowId: string; file: string; seed: number; wav: string; nCodes: number }> =
    fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) : [];
  const have = new Set(manifest.map(m => m.wav));

  console.log(`${runs.dataset}: ${work.length} renders (${sides.join('/')}), steps ${steps}, DiT ${synthModel || 'engine default'}, no DiT adapter`);
  const t0 = Date.now();
  let done = 0, skipped = 0, failed = 0;
  for (const w of work) {
    const wav = `${w.side}_${stemOf(w.row.file)}_s${w.seed}.wav`;
    const wavPath = path.join(outDir, wav);
    if (fs.existsSync(wavPath) && have.has(wav)) { skipped++; continue; }
    const t1 = Date.now();
    process.stdout.write(`  ${w.side.padEnd(7)} ${w.row.file.slice(0, 44).padEnd(46)} seed ${w.seed} (${w.codes.length} codes) … `);
    try {
      const bytes = await synthCodes(w.row, w.codes, w.seed, steps, synthModel);
      fs.writeFileSync(wavPath, bytes);
      manifest.push({ side: w.side, rowId: w.row.id, file: w.row.file, seed: w.seed, wav, nCodes: w.codes.length });
      have.add(wav);
      fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1));
      done++;
      console.log(`${((Date.now() - t1) / 1000).toFixed(1)}s`);
    } catch (e) {
      failed++;
      console.log(`FAILED: ${(e as Error).message}`);
    }
  }
  console.log(`done: ${done} rendered, ${skipped} skipped, ${failed} failed in ${((Date.now() - t0) / 1000).toFixed(0)}s → ${outDir}`);
}

main().catch(e => die((e as Error).stack ?? String(e)));
