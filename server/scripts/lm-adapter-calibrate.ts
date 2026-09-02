#!/usr/bin/env npx tsx
/**
 * lm-adapter-calibrate.ts — find and bake a planner-LM adapter's optimum
 * (2026-08-09).
 *
 * The eval tool (lm-adapter-eval.ts) showed each adapter has a single-peaked,
 * per-artist optimum in strength: too weak leaves base behavior, too strong
 * overshoots the artist's distribution (and at 2.0x sometimes breaks planning
 * outright — degenerate 1-token generations). This tool automates finding the
 * peak and shipping it:
 *
 *   sweep  candidates = scale grid x adapter snapshots (run dir + milestones/)
 *          -> one eval per candidate, base generations shared from a cache
 *   pick   best candidate by distribution distance, under hard guards
 *   bake   multiply the chosen scale INTO the adapter tensors (kron/LoRA
 *          deltas are linear in scale, so this is exact) -> a new run dir;
 *          the shipped adapter then runs at slider 1.0 = its optimum
 *   auto   sweep -> pick -> bake
 *
 * Run from server/ with the app up:
 *   npx tsx scripts/lm-adapter-calibrate.ts auto --dataset slipknot_allhope --adapter slipknot_allhope
 *   npx tsx scripts/lm-adapter-calibrate.ts sweep --dataset abba --adapter abba --scales 0.5,0.75,1,1.25,1.5
 *   npx tsx scripts/lm-adapter-calibrate.ts pick --dataset abba
 *   npx tsx scripts/lm-adapter-calibrate.ts bake --adapter <run dir> --scale 0.75
 *
 * Selection objective (pick): minimize marginalMean.adapter + transitionMean.adapter
 * (dense, low-variance; transitions included because marginals alone rewarded
 * ABBA's 1.5x run while its frame-to-frame dynamics degraded). Hard guards:
 *   - memorization.maxSec.adapter <= 5 s (style, not replay)
 *   - mean generated length >= 60% of the base side's (degenerate-gen guard)
 * Candidates failing a guard are excluded and reported, never silently kept.
 *
 * IDEMPOTENT: sweep skips candidates whose results.json already exists; bake
 * refuses an adapter that already carries hot_step_baked_scale.
 */
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const SERVER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
import { tensorsRoot } from '../src/services/training/aceTrain.js';
import { newestVariantKey } from '../src/services/training/trainLmStatus.js';
import { hasWeights, latestRunDir, lmAdapterRoots, runStamp } from '../src/services/training/adapterLayout.js';
import { planWorstWindow } from '../src/services/generation/planGuard.js';

const GUARD_MEM_SEC = 5;
const GUARD_LEN_RATIO = 0.6;
// Loop gate (2026-09-02, docs/plans/lm-attr-probe/) — see lm-adapter-rollout.ts
// for the measured separation this threshold sits inside.
const GUARD_LOOP_SHARE = 0.15;

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

function resolveAdapterDir(arg: string): string {
  const asPath = path.resolve(arg);
  if (fs.existsSync(asPath) && fs.statSync(asPath).isDirectory()) {
    if (hasWeights(asPath)) return asPath;
    const run = latestRunDir(asPath);
    if (run) return run;
    die(`${asPath} holds no adapter weights`);
  }
  const lc = arg.toLowerCase();
  for (const root of lmAdapterRoots()) {
    try {
      for (const e of fs.readdirSync(root.dir, { withFileTypes: true })) {
        if (e.isDirectory() && e.name.toLowerCase() === lc) {
          const run = latestRunDir(path.join(root.dir, e.name));
          if (run) return run;
        }
      }
    } catch { /* root absent */ }
  }
  die(`no adapter found for "${arg}"`);
}

// ── candidate discovery ───────────────────────────────────────────────────

interface Candidate {
  key: string;          // e.g. "run@1.0", "loss_1.8@0.75"
  adapterDir: string;
  scale: number;
  outDir: string;
}

function candidatesFor(evalRoot: string, adapterRun: string, scales: number[]): Candidate[] {
  const snapshots: Array<{ label: string; dir: string }> = [{ label: 'run', dir: adapterRun }];
  // Stage exports (2026-09-02): the default target-loss ladder writes each
  // non-final leg at <run>/stage1, <run>/stage2 — include them as candidates
  // if they carry weights (a single-leg chain writes neither, a no-op here).
  for (const stageName of ['stage1', 'stage2']) {
    const stageDir = path.join(adapterRun, stageName);
    if (fs.existsSync(stageDir) && hasWeights(stageDir)) snapshots.push({ label: stageName, dir: stageDir });
  }
  const msRoot = path.join(adapterRun, 'milestones');
  if (fs.existsSync(msRoot)) {
    for (const e of fs.readdirSync(msRoot, { withFileTypes: true })) {
      if (e.isDirectory() && hasWeights(path.join(msRoot, e.name))) {
        snapshots.push({ label: e.name, dir: path.join(msRoot, e.name) });
      }
    }
  }
  const out: Candidate[] = [];
  for (const snap of snapshots) {
    for (const s of scales) {
      out.push({
        key: `${snap.label}@${s}`,
        adapterDir: snap.dir,
        scale: s,
        outDir: path.join(evalRoot, `calib-${snap.label}-s${String(s).replace('.', '_')}`),
      });
    }
  }
  return out;
}

// ── sweep ─────────────────────────────────────────────────────────────────

interface BaseCache { runsPath: string; runs: any }

/** The newest full eval run for this dataset (base gens present) — the shared
 *  base-side cache every candidate reuses. */
function findBaseCache(evalRoot: string): BaseCache | null {
  if (!fs.existsSync(evalRoot)) return null;
  const dirs = fs.readdirSync(evalRoot, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => path.join(evalRoot, e.name))
    .filter(d => fs.existsSync(path.join(d, 'runs.json')))
    .sort((a, b) => fs.statSync(path.join(b, 'runs.json')).mtimeMs - fs.statSync(path.join(a, 'runs.json')).mtimeMs);
  for (const d of dirs) {
    try {
      const runs = JSON.parse(fs.readFileSync(path.join(d, 'runs.json'), 'utf-8'));
      if (Array.isArray(runs.gens) && runs.gens.some((g: any) => g.side === 'base')) {
        return { runsPath: path.join(d, 'runs.json'), runs };
      }
    } catch { /* skip torn files */ }
  }
  return null;
}

function seedCandidate(cache: BaseCache, cand: Candidate): void {
  const runs = JSON.parse(JSON.stringify(cache.runs));
  runs.gens = runs.gens.filter((g: any) => g.side === 'base');
  runs.adapterPath = cand.adapterDir;
  runs.adapterScale = cand.scale;
  fs.mkdirSync(cand.outDir, { recursive: true });
  fs.writeFileSync(path.join(cand.outDir, 'runs.json'), JSON.stringify(runs));
}

function cmdSweep(args: Map<string, string>): void {
  const dataset = args.get('dataset') ?? '';
  const adapterArg = args.get('adapter') ?? dataset;
  if (!dataset) die('sweep needs --dataset');
  const variant = args.get('variant') || newestVariantKey(dataset);
  const evalRoot = path.join(tensorsRoot(dataset), variant, 'lm-eval');
  const adapterRun = resolveAdapterDir(adapterArg);
  const scales = (args.get('scales') ?? '0.5,0.75,1,1.25,1.5').split(',').map(Number).filter(Number.isFinite);
  if (!scales.length) die('--scales parsed to nothing');

  const cands = candidatesFor(evalRoot, adapterRun, scales);
  const cache = findBaseCache(evalRoot);
  console.log(`\nCalibration sweep — ${dataset} (${variant})`);
  console.log(`  adapter:    ${adapterRun}`);
  console.log(`  candidates: ${cands.length} (${scales.length} scales x ${cands.length / scales.length} snapshots)`);
  console.log(`  base cache: ${cache ? cache.runsPath : 'NONE — first candidate pays for base generations too'}\n`);

  for (const cand of cands) {
    if (fs.existsSync(path.join(cand.outDir, 'results.json'))) {
      console.log(`  ${cand.key}: already evaluated — skipped`);
      continue;
    }
    if (cache && !fs.existsSync(path.join(cand.outDir, 'runs.json'))) seedCandidate(cache, cand);
    console.log(`  ${cand.key}: evaluating…`);
    const r = spawnSync('npx', [
      'tsx', 'scripts/lm-adapter-eval.ts', 'generate',
      '--dataset', dataset, '--variant', variant,
      '--adapter', cand.adapterDir, '--scale', String(cand.scale),
      '--out', cand.outDir,
    ], { stdio: 'inherit', shell: process.platform === 'win32', cwd: SERVER_DIR });
    if (r.status !== 0) console.log(`  ${cand.key}: eval FAILED (exit ${r.status}) — continuing with the rest`);
  }
  console.log('\nsweep complete — run `pick --dataset ' + dataset + '` next');
}

// ── pick ──────────────────────────────────────────────────────────────────

interface Scored {
  key: string; dir: string; adapterDir: string; scale: number;
  score: number; marginal: number; transition: number; unigram: number;
  memSec: number; lenRatio: number;
  /** Mean worst-40s-window repeat fraction (planGuard.planWorstWindow) over
   *  this candidate's adapter-side generated plans — the loop gate. */
  loopShare: number;
  excluded: string;
}

function meanGenLen(runs: any, side: string): number {
  const gens = runs.gens.filter((g: any) => g.side === side && g.codes.length > 0);
  if (!gens.length) return 0;
  return gens.reduce((a: number, g: any) => a + g.codes.length, 0) / gens.length;
}

function scoreCandidates(evalRoot: string): Scored[] {
  if (!fs.existsSync(evalRoot)) return [];
  const out: Scored[] = [];
  for (const e of fs.readdirSync(evalRoot, { withFileTypes: true })) {
    if (!e.isDirectory() || !e.name.startsWith('calib-')) continue;
    const dir = path.join(evalRoot, e.name);
    try {
      const results = JSON.parse(fs.readFileSync(path.join(dir, 'results.json'), 'utf-8'));
      const runs = JSON.parse(fs.readFileSync(path.join(dir, 'runs.json'), 'utf-8'));
      const memSec = results.memorization?.maxSec?.adapter ?? 99;
      const baseLen = meanGenLen(runs, 'base');
      const lenRatio = baseLen > 0 ? meanGenLen(runs, 'adapter') / baseLen : 0;
      // Loop gate (2026-09-02): `codes` on each gen is the array
      // lm-adapter-eval.ts wrote, not a CSV string — planGuard's functions
      // take the CSV the engine itself emits, so join it.
      const adapterGens = (runs.gens as any[]).filter((g: any) => g.side === 'adapter' && g.codes?.length > 0);
      const loopShare = adapterGens.length
        ? adapterGens.reduce((s: number, g: any) => s + planWorstWindow(g.codes.join(',')).worstFrac, 0) / adapterGens.length
        : 0;
      let excluded = '';
      if (memSec > GUARD_MEM_SEC) excluded = `memorization ${memSec.toFixed(1)}s > ${GUARD_MEM_SEC}s`;
      else if (lenRatio < GUARD_LEN_RATIO) excluded = `degenerate generations (len ratio ${lenRatio.toFixed(2)} < ${GUARD_LEN_RATIO})`;
      else if (loopShare > GUARD_LOOP_SHARE) excluded = `looping (loop share ${loopShare.toFixed(2)} > ${GUARD_LOOP_SHARE})`;
      out.push({
        key: e.name.replace(/^calib-/, ''),
        dir,
        adapterDir: runs.adapterPath,
        scale: runs.adapterScale,
        marginal: results.marginalMean.adapter,
        transition: results.transitionMean.adapter,
        unigram: results.unigram.adapter,
        score: results.marginalMean.adapter + results.transitionMean.adapter,
        memSec, lenRatio, loopShare, excluded,
      });
    } catch { /* candidate without results yet */ }
  }
  return out.sort((a, b) => a.score - b.score);
}

function cmdPick(args: Map<string, string>): Scored | null {
  const dataset = args.get('dataset') ?? '';
  if (!dataset) die('pick needs --dataset');
  const variant = args.get('variant') || newestVariantKey(dataset);
  const evalRoot = path.join(tensorsRoot(dataset), variant, 'lm-eval');
  const scored = scoreCandidates(evalRoot);
  if (!scored.length) die(`no calib-* results under ${evalRoot} — run sweep first`);

  console.log(`\nCandidates for ${dataset} (score = marginal+transition JS to artist, lower = closer):\n`);
  console.log('  candidate            score    marginal  transit.  unigram  memMax  lenRatio  loopShare');
  for (const s of scored) {
    const flag = s.excluded ? `  EXCLUDED: ${s.excluded}` : '';
    console.log(`  ${s.key.padEnd(20)} ${s.score.toFixed(4)}  ${s.marginal.toFixed(4)}    ${s.transition.toFixed(4)}   ${s.unigram.toFixed(3)}    ${s.memSec.toFixed(1)}s   ${s.lenRatio.toFixed(2)}      ${s.loopShare.toFixed(2)}${flag}`);
  }
  const eligible = scored.filter(s => !s.excluded);
  if (!eligible.length) die('every candidate failed a guard — the adapter needs retraining, not calibration');
  const best = eligible[0];
  console.log(`\n  WINNER: ${best.key} (adapter ${best.adapterDir} at scale ${best.scale})`);

  const calibration = {
    version: 1,
    pickedAt: new Date().toISOString(),
    dataset,
    variant,
    winner: { key: best.key, adapterDir: best.adapterDir, scale: best.scale, score: best.score },
    guards: { memMaxSec: GUARD_MEM_SEC, lenRatioMin: GUARD_LEN_RATIO },
    candidates: scored,
  };
  const calPath = path.join(evalRoot, 'calibration.json');
  fs.writeFileSync(calPath, JSON.stringify(calibration, null, 2));
  console.log(`  written: ${calPath}`);
  return best;
}

// ── bake ──────────────────────────────────────────────────────────────────
//
// safetensors surgery: header is <8-byte LE length><JSON><data>. We multiply
// the per-stem "delta carrier" tensors by the scale — lokr_w1 for LoKr (the
// kron is bilinear, so scaling one factor scales the delta exactly) and
// lora_B for PEFT (B starts at zero; the delta B·A is linear in B) — and add
// hot_step_baked_scale to __metadata__. Everything else is byte-preserved.

function bf16ToF32(u16: number): number {
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = u16 << 16;
  return new Float32Array(buf)[0];
}

function f32ToBf16(f: number): number {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = f;
  const u = new Uint32Array(buf)[0];
  // round-to-nearest-even on the dropped 16 bits
  const lsb = (u >> 16) & 1;
  const rounded = (u + 0x7fff + lsb) >>> 16;
  return rounded & 0xffff;
}

function bakeFile(srcFile: string, dstFile: string, scale: number): { scaled: number } {
  const raw = fs.readFileSync(srcFile);
  const headerLen = Number(raw.readBigUInt64LE(0));
  const header = JSON.parse(raw.subarray(8, 8 + headerLen).toString('utf-8'));
  const data = raw.subarray(8 + headerLen);

  const meta = header.__metadata__ ?? {};
  if (meta.hot_step_baked_scale) {
    die(`${srcFile} already carries hot_step_baked_scale=${meta.hot_step_baked_scale} — refusing to re-bake`);
  }

  const isLokr = Object.keys(header).some(k => k.endsWith('.lokr_w1'));
  const targetSuffix = isLokr ? '.lokr_w1' : '.lora_B.weight';

  let scaled = 0;
  for (const [name, info] of Object.entries<any>(header)) {
    if (name === '__metadata__' || !name.endsWith(targetSuffix)) continue;
    const [start, end] = info.data_offsets;
    if (info.dtype === 'F32') {
      const f = new Float32Array(data.buffer, data.byteOffset + start, (end - start) / 4);
      for (let i = 0; i < f.length; i++) f[i] *= scale;
    } else if (info.dtype === 'BF16') {
      const u = new Uint16Array(data.buffer, data.byteOffset + start, (end - start) / 2);
      for (let i = 0; i < u.length; i++) u[i] = f32ToBf16(bf16ToF32(u[i]) * scale);
    } else {
      die(`${name} has unsupported dtype ${info.dtype}`);
    }
    scaled++;
  }
  if (!scaled) die(`no ${targetSuffix} tensors found in ${srcFile} — not an LM adapter?`);

  meta.hot_step_baked_scale = String(scale);
  meta.hot_step_baked_from = path.basename(path.dirname(srcFile));
  header.__metadata__ = meta;

  // Re-serialize: metadata changed, so the header length changed; data offsets
  // are relative to the data section and stay valid.
  const newHeader = Buffer.from(JSON.stringify(header), 'utf-8');
  const out = Buffer.alloc(8 + newHeader.length + data.length);
  out.writeBigUInt64LE(BigInt(newHeader.length), 0);
  newHeader.copy(out, 8);
  data.copy(out, 8 + newHeader.length);
  fs.writeFileSync(dstFile, out);
  return { scaled };
}

function cmdBake(args: Map<string, string>, pre?: Scored): string {
  const adapterDir = pre ? pre.adapterDir : resolveAdapterDir(args.get('adapter') ?? '');
  const scale = pre ? pre.scale : Number(args.get('scale'));
  if (!Number.isFinite(scale) || scale <= 0) die('bake needs --scale > 0');
  if (scale === 1.0) {
    console.log('\n  scale 1.0 — nothing to bake; the adapter is already at its optimum');
    return adapterDir;
  }

  // Destination: a sibling stamped run dir so the artist's "newest run wins"
  // resolution picks the calibrated adapter up automatically.
  const artistDir = path.dirname(adapterDir.includes(`${path.sep}milestones${path.sep}`)
    ? path.dirname(path.dirname(adapterDir))
    : adapterDir);
  const dstDir = path.join(artistDir, `${runStamp()}-calibrated`);
  fs.mkdirSync(dstDir, { recursive: true });

  let scaledTotal = 0;
  for (const f of fs.readdirSync(adapterDir)) {
    const src = path.join(adapterDir, f);
    if (!fs.statSync(src).isFile()) continue;
    if (f.endsWith('.safetensors')) {
      scaledTotal += bakeFile(src, path.join(dstDir, f), scale).scaled;
    } else {
      fs.copyFileSync(src, path.join(dstDir, f));
    }
  }
  console.log(`\n  baked scale ${scale} into ${scaledTotal} tensors`);
  console.log(`  calibrated adapter: ${dstDir}`);
  console.log('  (newest-run resolution now serves this adapter; slider 1.0 = calibrated optimum)');
  return dstDir;
}

// ── main ──────────────────────────────────────────────────────────────────

// Exported for lm-adapter-rollout.ts (the batch orchestrator): bakeFile is the
// safetensors surgery, resolveAdapterDir the artist-name resolution.
export { bakeFile, resolveAdapterDir };

// CLI main — guarded so importing this module never runs a command.
const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const cmd = process.argv[2];
  const args = parseArgs(process.argv.slice(3));
  if (cmd === 'sweep') {
    cmdSweep(args);
  } else if (cmd === 'pick') {
    cmdPick(args);
  } else if (cmd === 'bake') {
    cmdBake(args);
  } else if (cmd === 'auto') {
    cmdSweep(args);
    const best = cmdPick(args);
    if (best) cmdBake(args, best);
  } else {
    console.log('usage: npx tsx scripts/lm-adapter-calibrate.ts <sweep|pick|bake|auto> [options]');
    console.log('  sweep --dataset <slug> [--adapter <path|artist>] [--scales 0.5,0.75,1,1.25,1.5] [--variant <key>]');
    console.log('  pick  --dataset <slug> [--variant <key>]');
    console.log('  bake  --adapter <run dir> --scale <s>   (or via auto, uses the pick winner)');
    console.log('  auto  sweep -> pick -> bake');
    process.exit(cmd ? 1 : 0);
  }
}
