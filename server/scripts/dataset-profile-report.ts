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
import { buildDatasetProfile, readDatasetProfile, saveDatasetProfile, type DatasetProfile } from '../src/services/training/datasetProfile.js';

const arg = (k: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined; };
const rebuild = process.argv.includes('--rebuild');
const all = process.argv.includes('--all');
const calcScript = arg('--calc');
const outDir = path.resolve(arg('--out') ?? path.join(process.cwd(), 'docs', 'plans', 'dataset-calibration'));

const db = new Database(config.data.dbPath, { readonly: true, fileMustExist: true });
const sourceOf = new Map((db.prepare('SELECT slug, source_dir FROM training_datasets').all() as Array<{ slug: string; source_dir: string }>).map(r => [r.slug, r.source_dir]));
const rows = db.prepare('SELECT dataset_slug, refine_run, checkpoint_dir, step, likeness, corruption, settings FROM yue2_rung_scores WHERE likeness IS NOT NULL ORDER BY refine_run, step').all() as Array<{
  dataset_slug: string; refine_run: string; checkpoint_dir: string; step: number; likeness: number; corruption: number | null; settings: string }>;

// ── Runs and their ear targets ──────────────────────────────────────────────
type Recipe = 'base-matched' | 'tuned (Prodigy)' | 'lr-leak (invalid)' | 'other';
function recipeOf(settings: string): Recipe {
  try {
    const s = JSON.parse(settings);
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

// Overall = the ladder scoreboard's base: likeness and (6 - corruption), averaged.
const overall = (g: { likeness: number; corruption: number | null }) => g.corruption == null ? g.likeness : (g.likeness + 6 - g.corruption) / 2;
function targets(run: Run): Record<string, number | null> {
  const r = run.rungs;
  const best = r.reduce((a, b) => overall(b) > overall(a) ? b : a);
  const peakLik = Math.max(...r.map(g => g.likeness));
  return {
    rungs: r.length,
    maxStep: Math.max(...r.map(g => g.step)),
    peakLikeness: peakLik,
    peakOverall: overall(best),
    bestStep: best.step,
    stepToLikeness4: r.find(g => g.likeness >= 4)?.step ?? null,
    corruptionAtBest: best.corruption,
  };
}

/** Loss facts from the run's train.jsonl when it survived: step-1..3 CE is the base model on this album. */
function lossFacts(run: Run): Record<string, number | null> {
  try {
    const recs = fs.readFileSync(path.join(run.segment, 'train.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(x => x.stage === 'joint');
    if (recs.length < 30) return {};
    const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const first = recs.slice(0, 3), head = recs.slice(0, 10), tail = recs.slice(-20);
    return {
      baseArCe: avg(first.map(x => x.ar_ce)), baseNarMse: avg(first.map(x => x.nar_mse)),
      arDropPct: 100 * (1 - avg(tail.map(x => x.ar_ce)) / avg(head.map(x => x.ar_ce))),
      narDropPct: 100 * (1 - avg(tail.map(x => x.nar_mse)) / avg(head.map(x => x.nar_mse))),
    };
  } catch { return {}; }
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

interface Row { slug: string; run: string; recipe: Recipe; t: Record<string, number | null>; f: Record<string, number | null>; calc?: Record<string, unknown> }
const table: Row[] = [];
for (const run of runs.values()) {
  const p = profiles.get(run.slug);
  if (!p) continue;
  const calc = calcs.get(run.slug) as { config?: Record<string, unknown> } | undefined;
  const f: Record<string, number | null> = { ...p.album, ...lossFacts(run) };
  if (calc?.config) { f.calcSteps = Number(calc.config.steps); f.calcLr = Number(calc.config.learning_rate); f.calcRank = Number(calc.config.rank); }
  table.push({ slug: run.slug, run: run.run, recipe: run.recipe, t: targets(run), f, calc: calc?.config });
}

const TARGETS = ['peakLikeness', 'peakOverall', 'bestStep', 'stepToLikeness4', 'corruptionAtBest'];
const MIN_N = 5;
const lines: string[] = [`# Dataset calibration report`, '', `Built ${new Date().toISOString()} from ${rows.length} scored rungs, ${runs.size} runs, ${profiles.size} profiled albums.`, ''];
lines.push('Spearman rank correlation between each album measure and each ear target, within one recipe.',
  `Only pairs with at least ${MIN_N} runs are ranked. Ladder targets (bestStep, stepToLikeness4) only use runs with 4+ scored rungs.`,
  'The listener\'s take-to-take scoring noise is about ±1 per criterion, so treat |rho| below ~0.5 at these sizes as noise.', '');
const json: Record<string, unknown> = { rows: table };
for (const recipe of ['base-matched', 'tuned (Prodigy)'] as Recipe[]) {
  const group = table.filter(r => r.recipe === recipe);
  lines.push(`## ${recipe}: ${group.length} runs, ${new Set(group.map(r => r.slug)).size} albums`, '');
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
  lines.push('| album | rungs | peak likeness | peak overall | best step | first likeness 4 | songs | min | base AR CE | calc steps |', '|---|---|---|---|---|---|---|---|---|---|');
  for (const r of group.sort((a, b) => a.slug.localeCompare(b.slug))) {
    const v = (x: unknown, d = 0) => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(d) : '';
    lines.push(`| ${r.slug} | ${r.t.rungs} | ${v(r.t.peakLikeness)} | ${v(r.t.peakOverall, 1)} | ${v(r.t.bestStep)} | ${v(r.t.stepToLikeness4)} | ${r.f.songs} | ${v(r.f.totalMin)} | ${v(r.f.baseArCe, 2)} | ${v(r.f.calcSteps)} |`);
  }
  lines.push('');
}
fs.mkdirSync(outDir, { recursive: true });
const stamp = new Date().toISOString().slice(0, 10);
fs.writeFileSync(path.join(outDir, `report-${stamp}.md`), lines.join('\n'));
fs.writeFileSync(path.join(outDir, `report-${stamp}.json`), JSON.stringify(json, null, 2));
console.log(lines.join('\n'));
console.log(`\nwrote ${path.join(outDir, `report-${stamp}.md`)}`);
