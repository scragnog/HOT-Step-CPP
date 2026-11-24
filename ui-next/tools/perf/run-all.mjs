// Re-measures everything the performance pass measured, for a before/after comparison: every action on three saved
// desks, every action on the full desk at 4x slower CPU, the load, idle and memory, and the motions' cost.
//
//   npm run perf -- [--label <name>] [--build <folder holding dist-file/ and dist-test/>] [--runs 3] [--quick]
//                   [--skip perf,throttle,load,idle,memory,motion,trace]
//
// With no --build it builds this checkout first (build:file and build:test) and measures that. Results go to
// tools/perf/results/<label>/ (perf.json, perf-throttle4.json, load.json, idle.json, memory*.json, motion.json,
// summary.md). Timings run at normal priority (priority changes the numbers): close what else runs, and compare two
// builds measured one after the other (`--build` the one kept from before). --quick: one run of each, fewer rounds.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), ui = path.resolve(here, '..', '..');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const quick = args.includes('--quick'), label = opt('label', new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-'));
const runs = +opt('runs', quick ? 1 : 3), skip = new Set((opt('skip', '') || '').split(',').filter(Boolean));
if (!args.includes('--trace')) skip.add('trace');
const out = path.join(here, 'results', label); fs.mkdirSync(out, { recursive: true });
let build = opt('build');
const node = (script, a, env = {}) => {
  console.log(`> node ${script} ${a.join(' ')}`);
  const r = spawnSync(process.execPath, [script, ...a], { cwd: here, stdio: 'inherit', env: { ...process.env, PERF_BUILD: build, ...env } });
  if (r.status) console.log(`  (exit ${r.status})`);
};
if (!build) {
  for (const s of ['build:file', 'build:test']) { const r = spawnSync('npm', ['run', s], { cwd: ui, stdio: 'inherit', shell: true }); if (r.status) process.exit(r.status); }
  build = ui;
}
build = path.resolve(build);
if (!['empty', 'typical', 'full'].every(d => fs.existsSync(path.join(here, 'desks', d + '.json')))) node('make-desks.cjs', []);
const load = spawn(process.execPath, ['machine-load.cjs', path.join(out, 'machine-load.jsonl'), '14400'], { cwd: here, stdio: 'ignore' });
try {
  if (!skip.has('perf')) node('perf.cjs', ['--runs', String(runs), '--out', path.join(out, 'perf.json')]);
  if (!skip.has('throttle')) node('perf.cjs', ['--runs', String(quick ? 1 : 2), '--throttle', '4', '--desks', 'full', '--out', path.join(out, 'perf-throttle4.json')]);
  if (!skip.has('load')) { node('load.cjs', ['--runs', quick ? '3' : '7', '--out', path.join(out, 'load.json')]); node('load.cjs', ['--runs', quick ? '1' : '3', '--desks', 'full', '--throttle', '4', '--out', path.join(out, 'load-throttle4.json')]); }
  if (!skip.has('idle')) node('idle.cjs', ['--desks', 'empty,typical,full', '--out', path.join(out, 'idle.json')]);
  if (!skip.has('memory')) { node('memory.cjs', ['--rounds', quick ? '5' : '15', '--out', path.join(out, 'memory.json')]); node('memory.cjs', ['--rounds', quick ? '8' : '30', '--no-generate', '--out', path.join(out, 'memory-nogen.json')]); }
  if (!skip.has('motion')) node(path.join(ui, 'tools', 'motion', 'cost.mjs'), ['--page', path.join(build, 'dist-test', 'index.html'), '--runs', String(runs), '--json', path.join(out, 'motion.json')]);
  if (!skip.has('trace')) node('trace.cjs', ['--desks', 'full', '--out', path.join(out, 'trace.json')]);
} finally { load.kill(); }
node('summarize.cjs', ['--dir', out]);
console.log(`results in ${out}`);
