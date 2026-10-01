// Dataset-Calibrated Training, step 1: profile every album that has ear-scored
// YuE2 rungs, join the profiles with those scores, and rank which measures
// track how well (and how fast) an album trains.
//
//   npx tsx server/scripts/dataset-profile-report.ts [--rebuild] [--all] [--calc <calc.py>] [--out <dir>]
//
// --rebuild   re-measure albums that already have a dataset-profile.json
// --all       profile every dataset with a source folder, scored or not
// --calc  also run a third-party settings calculator (its analyze_dataset_dsp
//             + calculate_yue2_comfy_config) per album and set its suggested
//             steps/lr beside the ear results; cached as calc-suggestion.json
// --out       report folder (default docs/plans/dataset-calibration, local-only)
//
// Read-only against the database and the datasets; writes only profiles,
// calculator caches and the report.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import Database from 'better-sqlite3';
import { config } from '../src/config.js';
import { datasetDir } from '../src/services/training/paths.js';
import { buildDatasetProfile, readDatasetProfile, saveDatasetProfile, trainLogArchiveDir, calibrateYue2Length, type DatasetProfile } from '../src/services/training/datasetProfile.js';
import { listYue2AitkRuns } from '../src/services/training/yue2AitkRuns.js';

const arg = (k: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined; };
const rebuild = process.argv.includes('--rebuild');
const all = process.argv.includes('--all');
const calcScript = arg('--calc');
const outDir = path.resolve(arg('--out') ?? path.join(import.meta.dirname, '..', '..', 'docs', 'plans', 'dataset-calibration'));

const db = new Database(config.data.dbPath, { readonly: true, fileMustExist: true });
const sourceOf = new Map((db.prepare('SELECT slug, source_dir FROM training_datasets').all() as Array<{ slug: string; source_dir: string }>).map(r => [r.slug, r.source_dir]));
const rows = db.prepare('SELECT dataset_slug, refine_run, checkpoint_dir, step, likeness, corruption, settings FROM yue2_rung_scores WHERE likeness IS NOT NULL ORDER BY refine_run, step').all() as Array<{
  dataset_slug: string; refine_run: string; checkpoint_dir: string; step: number; likeness: number; corruption: number | null; settings: string }>;

const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b), mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// ── Runs and their ear targets ──────────────────────────────────────────────
type Recipe = 'base-matched' | 'tuned (Prodigy)' | 'lr-leak (invalid)' | 'other';
function recipeOf(settings: string): Recipe {
  try {
    const s = JSON.parse(settings);
    // Recorded since 2026-09-29; older rows are told apart by optimizer and rate.
    if (s.method === 'base-matched') return s.lr === 0.0001 || s.lr === undefined ? 'base-matched' : 'lr-leak (invalid)';
    if (s.optimizer === 'prodigy') return 'tuned (Prodigy)';
    if (s.optimizer === 'adamw-lm') return s.lr === 0.0001 ? 'base-matched' : 'lr-leak (invalid)';
  } catch { /* fall through */ }
  return 'other';
}
interface Run { run: string; slug: string; recipe: Recipe; rungs: Array<{ step: number; likeness: number; corruption: number | null }>; segment: string }
const runs = new Map<string, Run>();
for (const r of rows) {
  let run = runs.get(r.refine_run);
  if (!run) runs.set(r.refine_run, run = { run: r.refine_run, slug: r.dataset_slug, recipe: recipeOf(r.settings), rungs: [], segment: path.dirname(r.checkpoint_dir) });
  run.rungs.push({ step: r.step, likeness: r.likeness, corruption: r.corruption });
}
// The album verdict per run (1-5); the table exists from 2026-09-29 on.
// Directions as numbers: under -1, right 0, over +1 (columns from 2026-09-30).
const albumScore = new Map<string, { score: number | null; instruments: number | null; vocals: number | null }>();
const dir = (v: string | null) => v === 'under' ? -1 : v === 'right' ? 0 : v === 'over' ? 1 : null;
try {
  for (const a of db.prepare('SELECT * FROM yue2_album_scores').all() as Array<{ refine_run: string; score: number | null; instruments?: string | null; vocals?: string | null }>)
    albumScore.set(a.refine_run, { score: a.score, instruments: dir(a.instruments ?? null), vocals: dir(a.vocals ?? null) });
} catch { /* older database: no album scores yet */ }

// Overall = the ladder scoreboard's base: likeness and (6 - corruption), averaged.
const overall = (g: { likeness: number; corruption: number | null }) => g.corruption == null ? g.likeness : (g.likeness + 6 - g.corruption) / 2;

/** First later rung (by step) whose overall, and every rung after it, stayed
 *  >= 1 below the run's peak overall — the point training turned, not a dip. */
function turnStepOf(r: Run['rungs']): number | null {
  let peakIdx = 0;
  for (let i = 1; i < r.length; i++) if (overall(r[i]) > overall(r[peakIdx])) peakIdx = i;
  const threshold = overall(r[peakIdx]) - 1;
  let idx: number | null = null;
  for (let i = r.length - 1; i > peakIdx; i--) {
    if (overall(r[i]) <= threshold) idx = i;
    else break;
  }
  return idx !== null ? r[idx].step : null;
}

function targets(run: Run): Record<string, number | null> {
  const r = run.rungs;
  const best = r.reduce((a, b) => overall(b) > overall(a) ? b : a);
  const peakLik = Math.max(...r.map(g => g.likeness));
  const steps = [...new Set(r.map(g => g.step))].sort((a, b) => a - b);
  let rungEvery: number | null = null;
  for (let i = 1; i < steps.length; i++) rungEvery = rungEvery === null ? steps[i] - steps[i - 1] : Math.min(rungEvery, steps[i] - steps[i - 1]);
  const arrive = r.find(g => g.likeness >= 4 && (g.corruption == null || g.corruption <= 2));
  return {
    rungs: r.length,
    albumScore: albumScore.get(run.run)?.score ?? null,
    instrumentsDir: albumScore.get(run.run)?.instruments ?? null,
    vocalsDir: albumScore.get(run.run)?.vocals ?? null,
    maxStep: Math.max(...r.map(g => g.step)),
    peakLikeness: peakLik,
    peakOverall: overall(best),
    bestStep: best.step,
    stepToLikeness4: r.find(g => g.likeness >= 4)?.step ?? null,
    corruptionAtBest: best.corruption,
    arriveStep: arrive?.step ?? null,
    lastScoredStep: r[r.length - 1].step,
    rungEvery,
    turnStep: turnStepOf(r),
  };
}

// ── Train log: resolve, tolerate bad lines, classify coverage ──────────────
interface JointRecord { step: number; ar_ce: number; nar_mse: number }
type LogStatus = 'usable' | 'missing' | 'malformed' | 'short';

/** Candidate train.jsonl locations, most-authoritative first: the run's own
 *  (possibly relocated) output folder from the durable run index, then the
 *  segment path this script already derives from the scored checkpoint_dir,
 *  then the copy archived with the dataset once a run folder is gone. */
function candidateLogPaths(run: Run): string[] {
  const candidates: string[] = [];
  const indexed = listYue2AitkRuns('', run.slug).find(r => r.jobId === run.run);
  if (indexed) {
    const segName = path.basename(run.segment);
    candidates.push(/^segment-\d{6}$/.test(segName) ? path.join(indexed.output, 'segments', segName, 'train.jsonl') : path.join(indexed.output, 'train.jsonl'));
  }
  candidates.push(path.join(run.segment, 'train.jsonl'));
  candidates.push(path.join(trainLogArchiveDir(run.slug), `${run.run}-${path.basename(run.segment)}.jsonl`));
  return candidates;
}

function loadTrainLog(run: Run): { status: LogStatus; records: JointRecord[] } {
  for (const p of candidateLogPaths(run)) {
    if (!fs.existsSync(p)) continue;
    let text: string;
    try { text = fs.readFileSync(p, 'utf8'); } catch { continue; }
    const records: JointRecord[] = [];
    for (const line of text.split(/\r?\n/)) {
      if (!line) continue;
      try {
        const event = JSON.parse(line) as Record<string, unknown>;
        if (event.stage !== 'joint' || !Number.isInteger(event.step)) continue;
        const { ar_ce, nar_mse } = event;
        if (typeof ar_ce !== 'number' || !Number.isFinite(ar_ce) || typeof nar_mse !== 'number' || !Number.isFinite(nar_mse)) continue;
        records.push({ step: event.step as number, ar_ce, nar_mse });
      } catch { /* an incomplete log line does not invalidate other steps */ }
    }
    if (records.length === 0) return { status: 'malformed', records: [] };
    if (records.length < 30) return { status: 'short', records };
    return { status: 'usable', records };
  }
  return { status: 'missing', records: [] };
}

/** Early-training facts from the run's train.jsonl when it survived and is usable. */
function lossFacts(log: { status: LogStatus; records: JointRecord[] }): Record<string, number | null> {
  if (log.status !== 'usable') return {};
  const recs = log.records;
  const first = recs.slice(0, 3), head = recs.slice(0, 10), tail = recs.slice(-20);
  return {
    earlyArCe: avg(first.map(x => x.ar_ce)), earlyNarMse: avg(first.map(x => x.nar_mse)),
    arDropPct: 100 * (1 - avg(tail.map(x => x.ar_ce)) / avg(head.map(x => x.ar_ce))),
    narDropPct: 100 * (1 - avg(tail.map(x => x.nar_mse)) / avg(head.map(x => x.nar_mse))),
  };
}

/** The Optimise phase's measurements, from the dataset folder's _hotstep-optimisation.json. */
function optimiseFacts(dir: string | undefined): Record<string, number | null> {
  if (!dir) return {};
  try {
    const bl = JSON.parse(fs.readFileSync(path.join(dir, '_hotstep-optimisation.json'), 'utf8')).baseLoss;
    if (!bl?.summary) return {};
    const ce = (bl.items ?? []).map((i: { arCe: number }) => i.arCe).filter(Number.isFinite) as number[];
    const m = ce.reduce((a, b) => a + b, 0) / (ce.length || 1);
    return { evalArCe: bl.summary.arCeMean, evalNarMse: bl.summary.narMseMean,
      evalArCeSpread: ce.length > 1 ? Math.sqrt(ce.reduce((a, b) => a + (b - m) ** 2, 0) / ce.length) : null };
  } catch { return {}; }
}

// ── Milestones: causal rolling-median smoothing, drop-from-early-mean ──────
function rollingMedian(xs: number[], window = 10): number[] {
  const out: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    const w = xs.slice(Math.max(0, i - window + 1), i + 1).slice().sort((a, b) => a - b);
    const mid = Math.floor(w.length / 2);
    out.push(w.length % 2 ? w[mid] : (w[mid - 1] + w[mid]) / 2);
  }
  return out;
}

const MILESTONE_PCTS = [50, 70, 90] as const;
/** First update (by step) where the smoothed series has covered pct% of the
 *  drop from the first-3-updates mean to the last-20-updates mean. */
function milestoneSteps(raw: number[], smoothed: number[], recs: JointRecord[]): Record<number, number | null> {
  const early = avg(raw.slice(0, 3)), final = avg(raw.slice(-20)), drop = early - final;
  const out: Record<number, number | null> = { 50: null, 70: null, 90: null };
  // Loss must actually have fallen: a flat or rising series is not progress
  // toward arrival, and must not report milestones by reversing the sense of "below".
  if (!Number.isFinite(drop) || drop <= 0) return out;
  for (const pct of MILESTONE_PCTS) {
    const target = early - drop * (pct / 100);
    const idx = smoothed.findIndex(v => v <= target);
    out[pct] = idx >= 0 ? recs[idx].step : null;
  }
  return out;
}
function milestones(log: { status: LogStatus; records: JointRecord[] }): Record<string, number | null> {
  if (log.status !== 'usable') return { nar50: null, nar70: null, nar90: null, ar50: null, ar70: null, ar90: null };
  const recs = log.records;
  const nar = recs.map(r => r.nar_mse), ar = recs.map(r => r.ar_ce);
  const narM = milestoneSteps(nar, rollingMedian(nar), recs), arM = milestoneSteps(ar, rollingMedian(ar), recs);
  return { nar50: narM[50], nar70: narM[70], nar90: narM[90], ar50: arM[50], ar70: arM[70], ar90: arM[90] };
}

// ── Early-only forecast: first M updates, nothing later ────────────────────
function linregSlope(ys: number[]): number {
  const n = ys.length, mx = (n - 1) / 2, my = avg(ys);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - mx) * (ys[i] - my); den += (i - mx) ** 2; }
  return den ? num / den : NaN;
}
function earlyForecastFeatures(log: { status: LogStatus; records: JointRecord[] }, M: number): Record<string, number | null> | null {
  if (log.status !== 'usable' || log.records.length < M) return null;
  const window = log.records.slice(0, M);
  const nar = window.map(r => r.nar_mse), ar = window.map(r => r.ar_ce);
  const narS = rollingMedian(nar), arS = rollingMedian(ar);
  const earlyNar = avg(nar.slice(0, 3)), earlyAr = avg(ar.slice(0, 3));
  return {
    slopeNar: linregSlope(narS), slopeAr: linregSlope(arS),
    dropFracNar: earlyNar ? (earlyNar - narS[narS.length - 1]) / earlyNar : null,
    dropFracAr: earlyAr ? (earlyAr - arS[arS.length - 1]) / earlyAr : null,
  };
}

/** Leave-one-album-out: fit `predict` on every other album's rows, score the
 *  held-out album, return the median of each album's mean absolute error. */
function looByAlbum<T extends { slug: string; target: number }>(items: T[], predict: (train: T[], row: T) => number): number | null {
  const albums = [...new Set(items.map(i => i.slug))];
  const errors: number[] = [];
  for (const heldOut of albums) {
    const train = items.filter(i => i.slug !== heldOut);
    const test = items.filter(i => i.slug === heldOut);
    if (!train.length || !test.length) continue;
    errors.push(avg(test.map(row => Math.abs(row.target - predict(train, row)))));
  }
  return errors.length ? median(errors) : null;
}
// ── Third-party calculator (optional) ───────────────────────────────────────
function thirdPartyCalc(slug: string, dir: string): Record<string, unknown> | null {
  if (!calcScript) return null;
  const cache = path.join(datasetDir(slug), 'calc-suggestion.json');
  if (!rebuild && fs.existsSync(cache)) { try { return JSON.parse(fs.readFileSync(cache, 'utf8')); } catch { /* re-run */ } }
  const py = [
    'import importlib.util, json, sys',
    `spec = importlib.util.spec_from_file_location("calc", ${JSON.stringify(path.resolve(calcScript))})`,
    'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)',
    `d = m.analyze_dataset_dsp(${JSON.stringify(dir)})`,
    'c = m.calculate_yue2_comfy_config(d["num_parent_tracks"], d["num_chunks"], d["total_duration_secs"], d["c_std"], d["spectral_flux"], d["zcr"], d["rms_std"], d["avg_caption_words"]) if d else None',
    'print(json.dumps({"dsp": d, "config": c}))',
  ].join('\n');
  const r = spawnSync('python', ['-c', py], { encoding: 'utf8', maxBuffer: 16 << 20, windowsHide: true });
  if (r.status !== 0) { console.warn(`  calculator failed on ${slug}: ${r.stderr.slice(-300)}`); return null; }
  const out = JSON.parse(r.stdout.trim().split('\n').pop()!);
  fs.writeFileSync(cache, JSON.stringify(out, null, 2));
  return out;
}

// ── Spearman ────────────────────────────────────────────────────────────────
function ranks(xs: number[]): number[] {
  const idx = xs.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
function spearman(a: number[], b: number[]): number {
  const ra = ranks(a), rb = ranks(b), n = a.length;
  const ma = (n + 1) / 2;
  let num = 0, da = 0, dbb = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - ma); da += (ra[i] - ma) ** 2; dbb += (rb[i] - ma) ** 2; }
  return da && dbb ? num / Math.sqrt(da * dbb) : NaN;
}

// Deterministic PRNG for the shuffle test (mulberry32) — no new dependency for 1000 reorderings.
function seededRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ── Main ────────────────────────────────────────────────────────────────────
const scoredSlugs = [...new Set([...runs.values()].map(r => r.slug))];
const slugs = all ? [...sourceOf.keys()] : scoredSlugs;
const profiles = new Map<string, DatasetProfile>();
const calcs = new Map<string, Record<string, unknown>>();
for (const slug of slugs) {
  const dir = sourceOf.get(slug);
  if (!dir || !fs.existsSync(dir)) { console.warn(`skip ${slug}: source folder missing (${dir ?? 'no dataset row'})`); continue; }
  let p = rebuild ? null : readDatasetProfile(slug);
  if (!p) {
    console.log(`profiling ${slug}`);
    try { p = await buildDatasetProfile(slug, dir, { log: s => console.log(s) }); saveDatasetProfile(p); }
    catch (e) { console.warn(`skip ${slug}: ${(e as Error).message}`); continue; }
  }
  profiles.set(slug, p);
  const c = thirdPartyCalc(slug, dir);
  if (c) calcs.set(slug, c);
}

interface Row {
  slug: string; run: string; recipe: Recipe; t: Record<string, number | null>; f: Record<string, number | null>; calc?: Record<string, unknown>;
  logStatus: LogStatus; m: Record<string, number | null>; forecast10: Record<string, number | null> | null; forecast20: Record<string, number | null> | null;
}
const table: Row[] = [];
for (const run of runs.values()) {
  const p = profiles.get(run.slug);
  if (!p) continue;
  const calc = calcs.get(run.slug) as { config?: Record<string, unknown> } | undefined;
  const log = loadTrainLog(run);
  const f: Record<string, number | null> = { ...p.album, ...lossFacts(log), ...optimiseFacts(sourceOf.get(run.slug)) };
  if (calc?.config) { f.calcSteps = Number(calc.config.steps); f.calcLr = Number(calc.config.learning_rate); f.calcRank = Number(calc.config.rank); }
  table.push({
    slug: run.slug, run: run.run, recipe: run.recipe, t: targets(run), f, calc: calc?.config,
    logStatus: log.status, m: milestones(log), forecast10: earlyForecastFeatures(log, 10), forecast20: earlyForecastFeatures(log, 20),
  });
}

const TARGETS = ['albumScore', 'instrumentsDir', 'vocalsDir', 'peakLikeness', 'peakOverall', 'bestStep', 'stepToLikeness4', 'corruptionAtBest'];
const MIN_N = 5;
const lines: string[] = [`# Dataset calibration report`, '', `Built ${new Date().toISOString()} from ${rows.length} scored rungs, ${runs.size} runs, ${profiles.size} profiled albums.`, ''];
lines.push('Spearman rank correlation between each album measure and each ear target, within one recipe.',
  `Only pairs with at least ${MIN_N} runs are ranked. Ladder targets (bestStep, stepToLikeness4) only use runs with 4+ scored rungs.`,
  'The listener\'s take-to-take scoring noise is about ±1 per criterion, so treat |rho| below ~0.5 at these sizes as noise.', '');
const json: Record<string, unknown> = { rows: table };
for (const recipe of ['base-matched', 'tuned (Prodigy)'] as Recipe[]) {
  const group = table.filter(r => r.recipe === recipe);
  lines.push(`## ${recipe}: ${group.length} runs, ${new Set(group.map(r => r.slug)).size} albums`, '');
  const coverage: Record<LogStatus, number> = { usable: 0, missing: 0, malformed: 0, short: 0 };
  for (const r of group) coverage[r.logStatus]++;
  lines.push(`Train log coverage: usable ${coverage.usable}, missing ${coverage.missing}, malformed ${coverage.malformed}, short ${coverage.short}.`, '');
  const feats = [...new Set(group.flatMap(r => Object.keys(r.f)))];
  const found: Array<{ feat: string; target: string; rho: number; n: number }> = [];
  for (const target of TARGETS) for (const feat of feats) {
    const pairs = group.filter(r => typeof r.f[feat] === 'number' && typeof r.t[target] === 'number'
      && (!['bestStep', 'stepToLikeness4'].includes(target) || (r.t.rungs as number) >= 4));
    if (pairs.length < MIN_N) continue;
    const rho = spearman(pairs.map(r => r.f[feat] as number), pairs.map(r => r.t[target] as number));
    if (Number.isFinite(rho)) found.push({ feat, target, rho, n: pairs.length });
  }
  found.sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho));
  json[recipe] = found;
  if (!found.length) { lines.push(`Not enough runs to rank anything (need ${MIN_N}).`, ''); }
  else {
    lines.push('| measure | target | rho | n |', '|---|---|---|---|');
    for (const x of found.slice(0, 25)) lines.push(`| ${x.feat} | ${x.target} | ${x.rho.toFixed(2)} | ${x.n} |`);
    lines.push('');
  }
  lines.push('| album | album score | rungs | peak likeness | peak overall | best step | first likeness 4 | songs | min | early AR CE (early training) | planner CE (Optimise) | decoder MSE (Optimise) | calc steps |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const r of group.sort((a, b) => a.slug.localeCompare(b.slug))) {
    const v = (x: unknown, d = 0) => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(d) : '';
    lines.push(`| ${r.slug} | ${v(r.t.albumScore)} | ${r.t.rungs} | ${v(r.t.peakLikeness)} | ${v(r.t.peakOverall, 1)} | ${v(r.t.bestStep)} | ${v(r.t.stepToLikeness4)} | ${r.f.songs} | ${v(r.f.totalMin)} | ${v(r.f.earlyArCe, 2)} | ${v(r.f.evalArCe, 3)} | ${v(r.f.evalNarMse, 3)} | ${v(r.f.calcSteps)} |`);
  }
  lines.push('');
}

// ── Milestone families: six measures, one family-wise shuffle test each ────
const MEASURES = ['nar50', 'nar70', 'nar90', 'ar50', 'ar70', 'ar90'] as const;
const qualifies = (r: Row) => r.recipe === 'base-matched' && r.logStatus === 'usable' && (r.t.rungs as number) >= 4;

function familyTest(targetKey: 'arriveStep' | 'turnStep'): { n: number; results: Array<{ measure: string; rho: number }>; p: number | null } {
  const pool = table.filter(r => qualifies(r) && typeof r.t[targetKey] === 'number' && MEASURES.every(m => typeof r.m[m] === 'number'));
  if (pool.length < MIN_N) return { n: pool.length, results: MEASURES.map(m => ({ measure: m, rho: NaN })), p: null };
  const rhoFor = (tgts: number[]) => MEASURES.map(m => spearman(pool.map(r => r.m[m] as number), tgts));
  const observed = rhoFor(pool.map(r => r.t[targetKey] as number));
  // A constant measure (or, in principle, a constant target) makes spearman return
  // NaN; Math.max propagates that NaN and every shuffle comparison then silently
  // fails, printing p = 0. Drop non-estimable measures from the max on both sides —
  // the set is fixed by which arrays are constant, which shuffling the target can't change.
  const estimable = observed.map(Number.isFinite);
  const maxAbsEstimable = (rhos: number[]): number => {
    const vals = rhos.filter((_, i) => estimable[i]).map(Math.abs);
    return vals.length ? Math.max(...vals) : NaN;
  };
  const observedMax = maxAbsEstimable(observed);
  let p: number | null = null;
  if (estimable.some(Boolean)) {
    const rng = seededRng(20261001);
    let hits = 0;
    const base = pool.map(r => r.t[targetKey] as number);
    for (let s = 0; s < 1000; s++) {
      const shuffled = base.slice();
      for (let i = shuffled.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }
      if (maxAbsEstimable(rhoFor(shuffled)) >= observedMax) hits++;
    }
    p = hits / 1000;
  }
  return { n: pool.length, results: MEASURES.map((m, i) => ({ measure: m, rho: observed[i] })), p };
}
function renderFamily(title: string, fam: ReturnType<typeof familyTest>): void {
  lines.push(`## ${title}`, '', `n = ${fam.n} base-matched runs with a usable log and 4+ scored rungs.`, '');
  if (fam.n < MIN_N) { lines.push(`Not enough runs to test (need ${MIN_N}).`, ''); return; }
  lines.push('| measure | rho |', '|---|---|');
  for (const x of fam.results) lines.push(`| ${x.measure} | ${Number.isFinite(x.rho) ? x.rho.toFixed(2) : ''} |`);
  lines.push('', fam.p === null
    ? 'Family-wise p: unavailable (every measure was constant in this pool).'
    : `Family-wise p (max |rho| over the six, 1000 seeded shuffles): ${fam.p.toFixed(3)}`, '');
}
const predeclared = familyTest('arriveStep');
renderFamily('Milestone correlations vs arriveStep (predeclared family)', predeclared);
json.milestonesVsArriveStep = predeclared;
lines.push('## Exploratory', '');
const exploratory = familyTest('turnStep');
renderFamily('Milestone correlations vs turnStep', exploratory);
json.milestonesVsTurnStep = exploratory;

// ── Early-only forecast: first 10 / first 20 updates, LOO by album ─────────
function forecastTable(M: 10 | 20): void {
  const forecastKey = M === 10 ? 'forecast10' as const : 'forecast20' as const;
  const pool = table.filter(r => qualifies(r) && typeof r.t.arriveStep === 'number' && typeof r.f.totalMin === 'number' && r[forecastKey]);
  lines.push(`### First ${M} updates`, '', `n = ${pool.length} base-matched runs with a usable log, 4+ scored rungs and ${M}+ joint updates.`, '');
  if (pool.length < 3) { lines.push('Not enough runs to fit anything.', ''); return; }
  const target = (r: Row) => r.t.arriveStep as number;
  const items = pool.map(r => ({ slug: r.slug, target: target(r), minutes: r.f.totalMin as number, feat: r[forecastKey]! }));
  const featureRows = (['slopeNar', 'slopeAr', 'dropFracNar', 'dropFracAr'] as const).map(key => {
    const withFeat = items.filter(i => typeof i.feat[key] === 'number' && Number.isFinite(i.feat[key] as number));
    const err = looLinear(withFeat.map(i => ({ slug: i.slug, target: i.target, feat: i.feat[key] as number })));
    return { name: key, err };
  });
  const baselineMedianErr = looByAlbum(items, (train) => median(train.map(i => i.target)) ?? NaN);
  const baselineCalibratedErr = looByAlbum(items, (train, row) => {
    const m = median(train.map(i => i.target)) ?? NaN;
    return m * calibrateYue2Length(row.minutes, 1, 1).factor;
  });
  lines.push('| feature | median abs step error (LOO by album) |', '|---|---|');
  for (const f of featureRows) lines.push(`| ${f.name} (first ${M}) | ${f.err === null ? '' : f.err.toFixed(0)} |`);
  lines.push(`| baseline: cohort median arriveStep | ${baselineMedianErr === null ? '' : baselineMedianErr.toFixed(0)} |`);
  lines.push(`| baseline: cohort median × calibrateYue2Length factor | ${baselineCalibratedErr === null ? '' : baselineCalibratedErr.toFixed(0)} |`, '');
}
function looLinear(items: Array<{ slug: string; target: number; feat: number }>): number | null {
  return looByAlbum(items, (train, row) => {
    const xs = train.map(i => i.feat), ys = train.map(i => i.target);
    const mx = avg(xs), my = avg(ys);
    let num = 0, den = 0;
    for (let i = 0; i < xs.length; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
    const slope = den ? num / den : 0, intercept = my - slope * mx;
    return intercept + slope * row.feat;
  });
}
lines.push('## Early-only forecast', '', 'Causal rolling-median smoothing, using only updates up to the window cutoff.', '');
forecastTable(10);
forecastTable(20);
json.forecast10 = table.filter(r => r.forecast10).map(r => ({ slug: r.slug, run: r.run, arriveStep: r.t.arriveStep, ...r.forecast10 }));
json.forecast20 = table.filter(r => r.forecast20).map(r => ({ slug: r.slug, run: r.run, arriveStep: r.t.arriveStep, ...r.forecast20 }));

fs.mkdirSync(outDir, { recursive: true });
const stamp = new Date().toISOString().slice(0, 10);
fs.writeFileSync(path.join(outDir, `report-${stamp}.md`), lines.join('\n'));
fs.writeFileSync(path.join(outDir, `report-${stamp}.json`), JSON.stringify(json, null, 2));
console.log(lines.join('\n'));
console.log(`\nwrote ${path.join(outDir, `report-${stamp}.md`)}`);
