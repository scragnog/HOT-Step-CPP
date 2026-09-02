#!/usr/bin/env npx tsx
/**
 * lm-adapter-rollout.ts — batch resume + calibrate + bake + preset repoint for
 * the whole planner-adapter corpus (2026-08-09).
 *
 * Per artist (lean protocol, ~10 min):
 *   1. resume-train the newest non-calibrated adapter deeper
 *      (--init-adapter, target 1.5, milestones on)
 *   2. eval candidates {old, new(+distinct milestones)} x {0.75, 1.0} —
 *      1 seed, 60 s generations, <=12 songs; base side generated once and
 *      shared across candidates
 *   3. pick by marginal+transition JS under guards (memorization <=5 s,
 *      length ratio >=0.6)
 *   4. REPOINT ONLY ON A STRICT WIN: if no new candidate beats the old
 *      adapter's best, the old adapter keeps its preset (logged 'kept-old')
 *   5. bake the winning scale when != 1.0; write hot_step_eval.json into the
 *      served dir (the score the UI shows) and the old dir (for comparison)
 *
 * Scores from this lean protocol are comparable to each other but NOT to the
 * rich 2-seed/120 s sweeps — the sidecar records the protocol for that reason.
 *
 * IDEMPOTENT: an artist whose adapter tree already carries a hot_step_eval.json
 * (or a *-calibrated run) is skipped; re-running processes only the remainder.
 *
 *   npx tsx scripts/lm-adapter-rollout.ts run [--limit N] [--artists a,b,c]
 *       [--scales 0.75,1] [--seeds 1] [--duration 60] [--samples 12]
 *       [--target-loss 1.5] [--dry-run]
 *   npx tsx scripts/lm-adapter-rollout.ts repoint [--apply]
 *   npx tsx scripts/lm-adapter-rollout.ts report
 */
import { spawnSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { aceClient } from '../src/services/aceClient.js';
import { getModelSnapshot, pickLmFor, refreshModelSnapshot, tensorsRoot } from '../src/services/training/aceTrain.js';
import { newestVariantKey } from '../src/services/training/trainLmStatus.js';
import { hasWeights, lmAdapterRoots, runStamp } from '../src/services/training/adapterLayout.js';
import { bakeFile } from './lm-adapter-calibrate.js';
import { planWorstWindow } from '../src/services/generation/planGuard.js';

const SERVER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require2 = createRequire(path.join(SERVER_DIR, 'package.json'));

const GUARD_MEM_SEC = 5;
const GUARD_LEN_RATIO = 0.6;
// Loop gate (2026-09-02, docs/plans/lm-attr-probe/): planWorstWindow's
// worst-40s-window repeat fraction separates healthy plans (<=30% on the
// probe corpus, chorus-heavy adapters included) from stuck-loop plans
// (measured 53-99%). 0.15 sits well inside the healthy band — a candidate
// whose adapter-side plans average above it is looping often enough that
// shipping it would ship the loop, not the artist.
const GUARD_LOOP_SHARE = 0.15;
const SIDECAR = 'hot_step_eval.json';

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

function sha1File(p: string): string {
  return crypto.createHash('sha1').update(fs.readFileSync(p)).digest('hex');
}

function weightsFileIn(dir: string): string {
  for (const f of ['lokr_weights.safetensors', 'adapter_model.safetensors']) {
    const p = path.join(dir, f);
    if (fs.existsSync(p)) return p;
  }
  return '';
}

// ── artist discovery ──────────────────────────────────────────────────────

interface Artist {
  name: string;
  dir: string;             // artist dir under an lm-* root
  sourceRun: string;       // newest non-calibrated run with weights
  variant: string;
  tensorsDir: string;
}

/** Newest stamped run that is NOT a -calibrated bake (those are outputs, not
 *  training states) — the resume source. */
function newestTrainedRun(artistDir: string): string {
  try {
    const runs = fs.readdirSync(artistDir, { withFileTypes: true })
      .filter(e => e.isDirectory() && !e.name.startsWith('.') && !/-calibrated$/i.test(e.name)
        && hasWeights(path.join(artistDir, e.name)))
      .map(e => e.name)
      .sort();
    if (runs.length) return path.join(artistDir, runs[runs.length - 1]);
  } catch { /* no dir */ }
  return hasWeights(artistDir) ? artistDir : '';
}

/** An artist counts as processed when any of its run dirs carries the eval
 *  sidecar (winner or kept-old both write one). */
function isProcessed(artistDir: string): boolean {
  try {
    for (const e of fs.readdirSync(artistDir, { withFileTypes: true })) {
      if (e.isDirectory() && fs.existsSync(path.join(artistDir, e.name, SIDECAR))) return true;
    }
  } catch { /* fall through */ }
  return false;
}

function discoverArtists(filter: string[], limit: number): { artists: Artist[]; skipped: string[] } {
  const artists: Artist[] = [];
  const skipped: string[] = [];
  const wanted = new Set(filter.map(s => s.toLowerCase()));
  for (const root of lmAdapterRoots()) {
    let entries: fs.Dirent[] = [];
    try { entries = fs.readdirSync(root.dir, { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      if (!e.isDirectory() || e.name.startsWith('.')) continue;
      if (wanted.size && !wanted.has(e.name.toLowerCase())) continue;
      const dir = path.join(root.dir, e.name);
      if (isProcessed(dir)) { skipped.push(`${e.name}: already processed (sidecar present)`); continue; }
      const sourceRun = newestTrainedRun(dir);
      if (!sourceRun) { skipped.push(`${e.name}: no adapter weights`); continue; }
      const variant = newestVariantKey(e.name);
      const tensorsDir = variant ? path.join(tensorsRoot(e.name), variant) : '';
      if (!variant || !fs.existsSync(path.join(tensorsDir, 'lm_codes.jsonl'))) {
        skipped.push(`${e.name}: no preprocessed dataset (lm_codes.jsonl) — dataset slug must match the adapter folder name`);
        continue;
      }
      artists.push({ name: e.name, dir, sourceRun, variant, tensorsDir });
      if (limit > 0 && artists.length >= limit) return { artists, skipped };
    }
  }
  return { artists, skipped };
}

// ── training (resume) ─────────────────────────────────────────────────────

function aceTrainExePath(): string {
  const p = path.resolve(SERVER_DIR, '..', 'engine', 'build', 'Release',
    process.platform === 'win32' ? 'ace-train.exe' : 'ace-train');
  if (!fs.existsSync(p)) die(`ace-train not found at ${p}`);
  return p;
}

interface SourceCfg { optimizer: string; muonLrScale: number; lr: number; gradAccum: number; seed: number }

function readSourceCfg(runDir: string): SourceCfg {
  const dflt: SourceCfg = { optimizer: 'muon', muonLrScale: 20, lr: 1e-4, gradAccum: 2, seed: 42 };
  try {
    const j = JSON.parse(fs.readFileSync(path.join(runDir, 'lm_train_log.json'), 'utf-8'));
    const c = j.config ?? {};
    return {
      optimizer: typeof c.optimizer === 'string' ? c.optimizer : dflt.optimizer,
      muonLrScale: Number.isFinite(c.muon_lr_scale) ? c.muon_lr_scale : dflt.muonLrScale,
      lr: Number.isFinite(c.lr) ? c.lr : dflt.lr,
      gradAccum: Number.isFinite(c.grad_accum) ? c.grad_accum : dflt.gradAccum,
      seed: Number.isFinite(c.seed) ? c.seed : dflt.seed,
    };
  } catch { return dflt; }
}

function resumeTrain(a: Artist, lmModel: string, targetLoss: number): { runDir: string; ok: boolean; err: string } {
  const runDir = path.join(a.dir, runStamp());
  const cfg = readSourceCfg(a.sourceRun);
  const args = [
    'train-lm', '--stages', 'train,export',
    '--init-adapter', a.sourceRun,
    '--tensors', a.tensorsDir,
    '--codes', path.join(a.tensorsDir, 'lm_codes.jsonl'),
    '--models', path.resolve(SERVER_DIR, '..', 'models'),
    '--lm', lmModel, '--lm-size', '4B', '--max-len', '8192',
    '--out', runDir,
    '--target-loss', String(targetLoss), '--epochs', '100',
    '--milestone-step', '0.1', '--milestone-keep', '12',
    '--optimizer', cfg.optimizer, '--muon-lr-scale', String(cfg.muonLrScale),
    '--lr', String(cfg.lr), '--grad-accum', String(cfg.gradAccum), '--seed', String(cfg.seed),
  ];
  const r = spawnSync(aceTrainExePath(), args, { encoding: 'utf-8', timeout: 45 * 60_000 });
  if (r.status !== 0) {
    const tail = (r.stderr ?? '').split('\n').filter(l => l.trim()).slice(-4).join(' | ');
    return { runDir, ok: false, err: `ace-train exit ${r.status}: ${tail}` };
  }
  return { runDir, ok: hasWeights(runDir), err: hasWeights(runDir) ? '' : 'run dir has no weights after training' };
}

// ── eval candidates ───────────────────────────────────────────────────────

interface Candidate { label: string; adapterDir: string; scale: number; outDir: string }

interface CandScore {
  label: string; adapterDir: string; scale: number;
  score: number; marginal: number; transition: number; unigram: number; unigramDelta: number;
  dimsWon: number; transWon: number; memSec: number; lenRatio: number; verdict: string;
  /** Mean worst-40s-window repeat fraction (planGuard.planWorstWindow) over
   *  this candidate's adapter-side generated plans. 0 when there were none. */
  loopShare: number;
  excluded: string;
}

function evalCandidate(a: Artist, cand: Candidate, baseCacheRuns: string, lean: { seeds: number; duration: number; samples: number }): boolean {
  if (fs.existsSync(path.join(cand.outDir, 'results.json'))) return true;
  if (baseCacheRuns && !fs.existsSync(path.join(cand.outDir, 'runs.json'))) {
    const runs = JSON.parse(fs.readFileSync(baseCacheRuns, 'utf-8'));
    runs.gens = runs.gens.filter((g: any) => g.side === 'base');
    runs.adapterPath = cand.adapterDir;
    runs.adapterScale = cand.scale;
    fs.mkdirSync(cand.outDir, { recursive: true });
    fs.writeFileSync(path.join(cand.outDir, 'runs.json'), JSON.stringify(runs));
  }
  const r = spawnSync('npx', [
    'tsx', 'scripts/lm-adapter-eval.ts', 'generate',
    '--dataset', a.name, '--variant', a.variant,
    '--adapter', cand.adapterDir, '--scale', String(cand.scale),
    '--seeds', String(lean.seeds), '--max-duration', String(lean.duration), '--samples', String(lean.samples),
    '--out', cand.outDir,
  ], { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf-8', shell: process.platform === 'win32', cwd: SERVER_DIR, timeout: 30 * 60_000 });
  if (r.status !== 0) {
    const tail = ((r.stdout ?? '') + (r.stderr ?? '')).split('\n').filter(l => l.trim()).slice(-3).join(' | ');
    console.log(`      eval ${cand.label} FAILED: ${tail}`);
    return false;
  }
  return fs.existsSync(path.join(cand.outDir, 'results.json'));
}

function scoreOf(cand: Candidate): CandScore | null {
  try {
    const results = JSON.parse(fs.readFileSync(path.join(cand.outDir, 'results.json'), 'utf-8'));
    const runs = JSON.parse(fs.readFileSync(path.join(cand.outDir, 'runs.json'), 'utf-8'));
    const lens = (side: string) => {
      const g = runs.gens.filter((x: any) => x.side === side && x.codes.length > 0);
      return g.length ? g.reduce((s: number, x: any) => s + x.codes.length, 0) / g.length : 0;
    };
    const baseLen = lens('base');
    const lenRatio = baseLen > 0 ? lens('adapter') / baseLen : 0;
    const memSec = results.memorization?.maxSec?.adapter ?? 99;
    // Loop gate (2026-09-02): worst-40s-window repeat fraction over every
    // adapter-side plan this candidate generated, averaged. `codes` here is
    // the array lm-adapter-eval.ts wrote (see GenResult), not a CSV string —
    // planGuard's functions take the CSV the engine itself emits, so join it.
    const adapterGens = (runs.gens as any[]).filter((x: any) => x.side === 'adapter' && x.codes?.length > 0);
    const loopShare = adapterGens.length
      ? adapterGens.reduce((s: number, g: any) => s + planWorstWindow(g.codes.join(',')).worstFrac, 0) / adapterGens.length
      : 0;
    let excluded = '';
    if (memSec > GUARD_MEM_SEC) excluded = `memorization ${memSec.toFixed(1)}s`;
    else if (lenRatio < GUARD_LEN_RATIO) excluded = `degenerate (lenRatio ${lenRatio.toFixed(2)})`;
    else if (loopShare > GUARD_LOOP_SHARE) excluded = `looping (loopShare ${loopShare.toFixed(2)})`;
    return {
      label: cand.label, adapterDir: cand.adapterDir, scale: cand.scale,
      score: results.marginalMean.adapter + results.transitionMean.adapter,
      marginal: results.marginalMean.adapter,
      transition: results.transitionMean.adapter,
      unigram: results.unigram.adapter,
      unigramDelta: results.unigram.base - results.unigram.adapter,
      dimsWon: results.verdict.dimsWon, transWon: results.verdict.transWon,
      memSec, lenRatio, loopShare, verdict: results.verdict.status, excluded,
    };
  } catch { return null; }
}

// ── sidecar ───────────────────────────────────────────────────────────────

function writeSidecar(dir: string, a: Artist, s: CandScore, extra: Record<string, unknown>, lean: { seeds: number; duration: number; samples: number }): void {
  const sidecar = {
    version: 1,
    evaluatedAt: new Date().toISOString(),
    dataset: a.name,
    variant: a.variant,
    /** marginal+transition JS to the artist's ground truth — LOWER = closer. */
    score: Number(s.score.toFixed(4)),
    marginal: Number(s.marginal.toFixed(4)),
    transition: Number(s.transition.toFixed(4)),
    unigram: Number(s.unigram.toFixed(4)),
    unigramDelta: Number(s.unigramDelta.toFixed(4)),
    dimsWon: s.dimsWon, transWon: s.transWon,
    memMaxSec: s.memSec,
    /** Mean worst-40s-window repeat fraction over this candidate's
     *  adapter-side plans (planGuard.planWorstWindow) — the loop gate.
     *  Candidates above 0.15 are excluded from picking, see GUARD_LOOP_SHARE. */
    loopShare: Number(s.loopShare.toFixed(4)),
    verdict: s.verdict,
    /** Runtime scale this score was measured at (1.0 for baked adapters). */
    measuredAtScale: s.scale,
    protocol: { seeds: lean.seeds, maxDuration: lean.duration, samples: lean.samples },
    ...extra,
  };
  fs.writeFileSync(path.join(dir, SIDECAR), JSON.stringify(sidecar, null, 2));
}

// ── the per-artist pipeline ───────────────────────────────────────────────

interface Outcome {
  artist: string;
  status: 'repointed' | 'kept-old' | 'baked-old' | 'failed' | 'skipped';
  servedDir: string;
  oldScore: number | null;
  newScore: number | null;
  winner: string;
  detail: string;
  minutes: number;
}

async function processArtist(a: Artist, lmModel: string, scales: number[], targetLoss: number,
                             lean: { seeds: number; duration: number; samples: number }): Promise<Outcome> {
  const t0 = Date.now();

  // 1. resume-train deeper
  console.log(`   training: resume ${path.basename(a.sourceRun)} -> target ${targetLoss}`);
  await aceClient.restoreEvictPolicy().catch(() => { /* engine may be idle already */ });
  const tr = resumeTrain(a, lmModel, targetLoss);
  if (!tr.ok) {
    return { artist: a.name, status: 'failed', servedDir: '', oldScore: null, newScore: null,
             winner: '', detail: `training: ${tr.err}`, minutes: Number(((Date.now() - t0) / 60_000).toFixed(1)) };
  }
  return calibrateArtist(a, tr.runDir, scales, lean, t0);
}

/** Steps 2-5 without the training: eval {old, newRun(+distinct milestones)} x
 *  scales, pick under guards, bake, sidecars. Shared by processArtist and the
 *  `calibrate` subcommand the Studio's lm-calibrate job invokes. */
async function calibrateArtist(a: Artist, newRunDir: string, scales: number[],
                               lean: { seeds: number; duration: number; samples: number },
                               t0 = Date.now(), tag = 'rollout'): Promise<Outcome> {
  const evalRoot = path.join(a.tensorsDir, 'lm-eval');
  const out = (status: Outcome['status'], servedDir: string, oldS: CandScore | null, newS: CandScore | null,
               winner: string, detail: string): Outcome => ({
    artist: a.name, status, servedDir,
    oldScore: oldS ? Number(oldS.score.toFixed(4)) : null,
    newScore: newS ? Number(newS.score.toFixed(4)) : null,
    winner, detail, minutes: Number(((Date.now() - t0) / 60_000).toFixed(1)),
  });
  const tr = { runDir: newRunDir };

  // 2. candidates: old + new final + distinct milestones, x scales
  const newWeights = weightsFileIn(tr.runDir);
  const finalHash = newWeights ? sha1File(newWeights) : '';
  // No old snapshot on a first-ever training (sourceRun empty or the new run
  // itself) — the new adapter then only has to beat the guards, not a rival.
  const snapshots: Array<{ label: string; dir: string }> = [];
  if (a.sourceRun && path.resolve(a.sourceRun) !== path.resolve(tr.runDir)) {
    snapshots.push({ label: 'old', dir: a.sourceRun });
  }
  snapshots.push({ label: 'new', dir: tr.runDir });
  // Stage exports (2026-09-02): the default target-loss ladder (2.0 -> 1.5 ->
  // final) writes each non-final leg at <run>/stage1, <run>/stage2 — earlier,
  // possibly-less-overfit snapshots that the sweep would otherwise never see.
  // Include any that carry weights and aren't already a snapshot (a
  // single-leg chain writes no stage dirs at all, so this is a no-op there).
  for (const stageName of ['stage1', 'stage2']) {
    const stageDir = path.join(tr.runDir, stageName);
    if (fs.existsSync(stageDir) && hasWeights(stageDir)
        && !snapshots.some(sn => path.resolve(sn.dir) === path.resolve(stageDir))) {
      snapshots.push({ label: stageName, dir: stageDir });
    }
  }
  const msRoot = path.join(tr.runDir, 'milestones');
  if (fs.existsSync(msRoot)) {
    for (const e of fs.readdirSync(msRoot, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      const w = weightsFileIn(path.join(msRoot, e.name));
      if (w && sha1File(w) !== finalHash) snapshots.push({ label: e.name, dir: path.join(msRoot, e.name) });
    }
  }
  const cands: Candidate[] = [];
  for (const snap of snapshots) {
    for (const s of scales) {
      cands.push({
        label: `${snap.label}@${s}`, adapterDir: snap.dir, scale: s,
        outDir: path.join(evalRoot, `${tag}-${snap.label}-s${String(s).replace('.', '_')}`),
      });
    }
  }

  // 3. evals — first candidate pays for the base side, the rest reuse it
  let baseCache = '';
  const scores: CandScore[] = [];
  for (const cand of cands) {
    console.log(`   eval ${cand.label}`);
    if (!evalCandidate(a, cand, baseCache, lean)) continue;
    if (!baseCache) baseCache = path.join(cand.outDir, 'runs.json');
    const s = scoreOf(cand);
    if (s) scores.push(s);
  }
  const eligible = scores.filter(s => !s.excluded);
  const oldBest = eligible.filter(s => s.label.startsWith('old')).sort((x, y) => x.score - y.score)[0] ?? null;
  const newBest = eligible.filter(s => !s.label.startsWith('old')).sort((x, y) => x.score - y.score)[0] ?? null;
  for (const s of scores) {
    console.log(`      ${s.label.padEnd(14)} score ${s.score.toFixed(4)}  loopShare ${s.loopShare.toFixed(2)}`
      + `${s.excluded ? `  EXCLUDED: ${s.excluded}` : ''}`);
  }
  if (!oldBest && !newBest) return out('failed', '', null, null, '', 'every candidate failed eval or guards');

  // 4. decide + 5. bake/sidecar. Strict-win rule: repoint only when a NEW
  // snapshot beats the old adapter's best score outright.
  const win = (s: CandScore): string => {
    if (s.scale === 1.0) return s.adapterDir;
    const artistDir = a.dir;
    const dstDir = path.join(artistDir, `${runStamp()}-calibrated`);
    fs.mkdirSync(dstDir, { recursive: true });
    let scaled = 0;
    for (const f of fs.readdirSync(s.adapterDir)) {
      const src = path.join(s.adapterDir, f);
      if (!fs.statSync(src).isFile()) continue;
      if (f.endsWith('.safetensors')) scaled += bakeFile(src, path.join(dstDir, f), s.scale).scaled;
      else fs.copyFileSync(src, path.join(dstDir, f));
    }
    console.log(`   baked ${s.label} (${scaled} tensors x ${s.scale}) -> ${path.basename(dstDir)}`);
    return dstDir;
  };

  if (oldBest) {
    writeSidecar(a.sourceRun, a, oldBest,
      { role: 'previous', bestScale: oldBest.scale, note: 'pre-rollout adapter; score is its best candidate' }, lean);
  }

  if (newBest && (!oldBest || newBest.score < oldBest.score)) {
    const servedDir = win(newBest);
    writeSidecar(servedDir, a, newBest, {
      role: 'served', winner: newBest.label,
      bakedScale: newBest.scale !== 1.0 ? newBest.scale : undefined,
      resumedFrom: a.sourceRun, trainedRun: tr.runDir,
    }, lean);
    return out('repointed', servedDir, oldBest, newBest, newBest.label, 'new adapter wins — repoint');
  }

  // old wins: still bake its better scale when that is a strict improvement
  // over old@1.0, else keep everything as-is.
  const old1 = scores.find(s => s.label === 'old@1');
  if (oldBest && oldBest.scale !== 1.0 && old1 && oldBest.score < old1.score) {
    const servedDir = win(oldBest);
    writeSidecar(servedDir, a, oldBest, { role: 'served', winner: oldBest.label, bakedScale: oldBest.scale, resumedFrom: a.sourceRun }, lean);
    return out('baked-old', servedDir, oldBest, newBest, oldBest.label, 'old adapter at rescaled optimum — repoint to bake');
  }
  return out('kept-old', a.sourceRun, oldBest, newBest, oldBest?.label ?? '',
    'no new candidate beat the old adapter — preset untouched');
}

// ── run ───────────────────────────────────────────────────────────────────

async function cmdRun(args: Map<string, string>): Promise<void> {
  const limit = Math.trunc(Number(args.get('limit')) || 0);
  const filter = (args.get('artists') ?? '').split(',').map(s => s.trim()).filter(Boolean);
  const scales = (args.get('scales') ?? '0.75,1').split(',').map(Number).filter(Number.isFinite);
  const lean = {
    seeds: Math.max(1, Math.trunc(Number(args.get('seeds')) || 1)),
    duration: Math.max(30, Math.trunc(Number(args.get('duration')) || 60)),
    samples: Math.max(4, Math.trunc(Number(args.get('samples')) || 12)),
  };
  const targetLoss = Number(args.get('target-loss')) || 1.5;

  const { artists, skipped } = discoverArtists(filter, limit);
  console.log(`\nLM adapter rollout — ${artists.length} artist(s) to process, ${skipped.length} skipped`);
  for (const s of skipped.slice(0, 8)) console.log(`  skip: ${s}`);
  if (skipped.length > 8) console.log(`  … and ${skipped.length - 8} more skips`);
  if (args.get('dry-run')) {
    for (const a of artists) console.log(`  would process: ${a.name} (resume ${path.basename(a.sourceRun)})`);
    return;
  }
  if (!artists.length) { console.log('nothing to do'); return; }

  await refreshModelSnapshot();
  const lmModel = pickLmFor('4B', getModelSnapshot().lm);
  if (!lmModel) die('no 4B LM base found — is the app running?');

  const logPath = path.join(SERVER_DIR, 'data', 'training', `lm-rollout-${runStamp()}.jsonl`);
  const outcomes: Outcome[] = [];
  for (let i = 0; i < artists.length; i++) {
    const a = artists[i];
    console.log(`\n[${i + 1}/${artists.length}] ${a.name}`);
    let o: Outcome;
    try {
      o = await processArtist(a, lmModel, scales, targetLoss, lean);
    } catch (err) {
      o = { artist: a.name, status: 'failed', servedDir: '', oldScore: null, newScore: null,
            winner: '', detail: err instanceof Error ? err.message : String(err), minutes: 0 };
    }
    outcomes.push(o);
    fs.appendFileSync(logPath, JSON.stringify(o) + '\n');
    console.log(`   -> ${o.status} (${o.minutes} min)${o.detail ? ` — ${o.detail}` : ''}`);
  }

  const by = (s: Outcome['status']) => outcomes.filter(o => o.status === s).length;
  console.log(`\n── Rollout summary ──────────────────────────────────`);
  console.log(`  repointed ${by('repointed')} · baked-old ${by('baked-old')} · kept-old ${by('kept-old')} · failed ${by('failed')}`);
  console.log(`  log: ${logPath}`);
  console.log(`\n  next: npx tsx scripts/lm-adapter-rollout.ts repoint          (dry run)`);
  console.log(`        npx tsx scripts/lm-adapter-rollout.ts repoint --apply`);
}

// ── calibrate (single artist, no training — the Studio's post-train job) ──

/**
 * Calibrate ONE artist around an existing freshly-trained run. Invoked by the
 * server's lm-calibrate job (calibrateRunner.ts) after every train-lm job.
 * Emits a final machine-readable line:  CALIBRATE_RESULT {json Outcome}
 * The caller owns preset repointing — this command never touches the DB.
 */
async function cmdCalibrate(args: Map<string, string>): Promise<void> {
  const dataset = args.get('dataset') ?? '';
  const newRun = path.resolve(args.get('new-run') ?? '');
  if (!dataset || !newRun) die('calibrate needs --dataset <slug> and --new-run <dir>');
  if (!hasWeights(newRun)) die(`${newRun} holds no adapter weights`);
  const variant = args.get('variant') || newestVariantKey(dataset);
  const tensorsDir = variant ? path.join(tensorsRoot(dataset), variant) : '';
  if (!variant || !fs.existsSync(path.join(tensorsDir, 'lm_codes.jsonl'))) {
    die(`dataset "${dataset}" has no extracted lm_codes.jsonl — calibration needs the ground-truth codes`);
  }
  const scales = (args.get('scales') ?? '0.75,1').split(',').map(Number).filter(Number.isFinite);
  const lean = {
    seeds: Math.max(1, Math.trunc(Number(args.get('seeds')) || 1)),
    duration: Math.max(30, Math.trunc(Number(args.get('duration')) || 60)),
    samples: Math.max(4, Math.trunc(Number(args.get('samples')) || 12)),
  };
  // Old rival: explicit --old-run, else the newest trained run that is not the
  // new one (and not a -calibrated bake). Empty = first-ever training.
  const artistDir = path.dirname(newRun);
  let oldRun = path.resolve(args.get('old-run') ?? '') || '';
  if (!args.get('old-run')) {
    oldRun = '';
    try {
      const runs = fs.readdirSync(artistDir, { withFileTypes: true })
        .filter(e => e.isDirectory() && !/-calibrated$/i.test(e.name)
          && hasWeights(path.join(artistDir, e.name))
          && path.resolve(path.join(artistDir, e.name)) !== newRun)
        .map(e => e.name)
        .sort();
      if (runs.length) oldRun = path.join(artistDir, runs[runs.length - 1]);
    } catch { /* no siblings */ }
  }

  const a: Artist = { name: dataset, dir: artistDir, sourceRun: oldRun, variant, tensorsDir };
  const outcome = await calibrateArtist(a, newRun, scales, lean, Date.now(), args.get('tag') || 'rollout');
  console.log(`CALIBRATE_RESULT ${JSON.stringify(outcome)}`);
  if (outcome.status === 'failed') process.exit(1);
}

// ── repoint (album presets) ───────────────────────────────────────────────

function cmdRepoint(args: Map<string, string>): void {
  const apply = args.get('apply') === 'true';
  // Served dirs = every run dir whose sidecar says role 'served'.
  const served = new Map<string, string>();   // artist -> dir
  for (const root of lmAdapterRoots()) {
    let entries: fs.Dirent[] = [];
    try { entries = fs.readdirSync(root.dir, { withFileTypes: true }); } catch { continue; }
    for (const artist of entries) {
      if (!artist.isDirectory()) continue;
      const adir = path.join(root.dir, artist.name);
      let best: { dir: string; at: string } | null = null;
      for (const run of fs.readdirSync(adir, { withFileTypes: true })) {
        if (!run.isDirectory()) continue;
        const scPath = path.join(adir, run.name, SIDECAR);
        if (!fs.existsSync(scPath)) continue;
        try {
          const sc = JSON.parse(fs.readFileSync(scPath, 'utf-8'));
          if (sc.role !== 'served') continue;
          if (!best || String(sc.evaluatedAt) > best.at) best = { dir: path.join(adir, run.name), at: String(sc.evaluatedAt) };
        } catch { /* ignore torn sidecars */ }
      }
      if (best) served.set(artist.name.toLowerCase(), best.dir);
    }
  }
  console.log(`\n${served.size} artist(s) with a served adapter`);

  const Database = require2('better-sqlite3');
  const dbPath = path.join(SERVER_DIR, 'data', 'hotstep.db');
  const db = new Database(dbPath);
  const rows = db.prepare(
    `SELECT ap.lyrics_set_id AS id, ls.album, ap.lm_adapter_path
     FROM album_presets ap LEFT JOIN lyrics_sets ls ON ls.id = ap.lyrics_set_id
     WHERE ap.lm_adapter_path IS NOT NULL AND ap.lm_adapter_path != ''`).all() as
    Array<{ id: number; album: string | null; lm_adapter_path: string }>;

  const updates: Array<{ id: number; album: string; from: string; to: string }> = [];
  for (const r of rows) {
    // The artist is the path segment under the lm-* root: …\lm-4b\<artist>\…
    const m = /[\\/]lm-(?:4b|17b|06b|)[\\/]([^\\/]+)/i.exec(r.lm_adapter_path)
      ?? /[\\/]lm[\\/]([^\\/]+)/i.exec(r.lm_adapter_path);
    if (!m) continue;
    const artist = m[1].toLowerCase().replace(/-(0\.6b|1\.7b|4b)$/i, '');
    const dest = served.get(artist);
    if (!dest) continue;
    if (path.resolve(r.lm_adapter_path).toLowerCase() === path.resolve(dest).toLowerCase()) continue;
    updates.push({ id: r.id, album: r.album ?? `(set ${r.id})`, from: r.lm_adapter_path, to: dest });
  }

  console.log(`${rows.length} preset(s) with an LM adapter, ${updates.length} to repoint (${apply ? 'APPLY' : 'DRY RUN'})\n`);
  for (const u of updates.slice(0, 10)) console.log(`  ${u.album}\n    ${u.from}\n    -> ${u.to}`);
  if (updates.length > 10) console.log(`  … and ${updates.length - 10} more, same shape`);
  if (!apply) { console.log('\nDRY RUN — re-run with --apply to write.'); db.close(); return; }

  const stampNow = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
  const bak = dbPath.replace(/\.db$/i, `_backup_lmrollout_${stampNow}.db`);
  db.backup(bak).then(() => {
    console.log(`\nbacked up to ${path.basename(bak)}`);
    const upd = db.prepare('UPDATE album_presets SET lm_adapter_path = ? WHERE lyrics_set_id = ?');
    let n = 0;
    for (const u of updates) { upd.run(u.to, u.id); n++; }
    db.close();
    console.log(`updated ${n} preset(s).`);
  }).catch((e: unknown) => die(`backup failed: ${e instanceof Error ? e.message : String(e)}`));
}

// ── report ────────────────────────────────────────────────────────────────

function cmdReport(): void {
  const dir = path.join(SERVER_DIR, 'data', 'training');
  const logs = fs.readdirSync(dir).filter(f => /^lm-rollout-.*\.jsonl$/.test(f)).sort();
  if (!logs.length) die('no rollout logs yet');
  const latest = new Map<string, Outcome>();
  for (const f of logs) {
    for (const line of fs.readFileSync(path.join(dir, f), 'utf-8').split('\n')) {
      if (!line.trim()) continue;
      try { const o = JSON.parse(line) as Outcome; latest.set(o.artist, o); } catch { /* torn */ }
    }
  }
  const all = [...latest.values()];
  const improved = all.filter(o => o.oldScore !== null && o.newScore !== null && o.status === 'repointed');
  const pct = improved.map(o => (1 - (o.newScore as number) / (o.oldScore as number)) * 100);
  console.log(`\n${all.length} artist(s) processed across ${logs.length} log(s):`);
  for (const s of ['repointed', 'baked-old', 'kept-old', 'failed'] as const) {
    const n = all.filter(o => o.status === s).length;
    if (n) console.log(`  ${s}: ${n}`);
  }
  if (pct.length) {
    pct.sort((a, b) => a - b);
    console.log(`  median improvement (repointed): ${pct[Math.floor(pct.length / 2)].toFixed(0)}% closer`);
  }
  for (const o of all.filter(x => x.status === 'failed')) console.log(`  FAILED ${o.artist}: ${o.detail}`);
}

// ── main ──────────────────────────────────────────────────────────────────

const cmd = process.argv[2];
const args = parseArgs(process.argv.slice(3));
if (cmd === 'run') {
  cmdRun(args).catch(err => die(err instanceof Error ? err.message : String(err)));
} else if (cmd === 'calibrate') {
  cmdCalibrate(args).catch(err => die(err instanceof Error ? err.message : String(err)));
} else if (cmd === 'repoint') {
  cmdRepoint(args);
} else if (cmd === 'report') {
  cmdReport();
} else {
  console.log('usage: npx tsx scripts/lm-adapter-rollout.ts <run|calibrate|repoint|report> [options]');
  console.log('  run       [--limit N] [--artists a,b] [--scales 0.75,1] [--seeds 1] [--duration 60] [--samples 12] [--target-loss 1.5] [--dry-run]');
  console.log('  calibrate --dataset <slug> --new-run <dir> [--old-run <dir>] [lean args]   (no training, no DB writes)');
  console.log('  repoint   [--apply]     update album_presets.lm_adapter_path to served adapters (dry-run default)');
  console.log('  report                  summarize all rollout logs');
  process.exit(cmd ? 1 : 0);
}
