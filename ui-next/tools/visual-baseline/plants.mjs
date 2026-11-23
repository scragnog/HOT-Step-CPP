// Plants each defect of plants/*.json in the test build, one at a time (vite.config.ts, HS_SWITCH, into dist-switch/),
// captures the slice's states that show it and compares them with the concept (with slice-accepted.json), and runs the
// parts gallery on it. A defect is caught when either fails. Prints caught or missed for each; exit 1 if one is missed.
// Needs the concept's capture first (vb.mjs capture concept-slice --states slice --scope slice --page <concept>).
//
//   node tools/visual-baseline/plants.mjs [--only <regex>] [--workers n]

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..');
const args = process.argv.slice(2), only = args.includes('--only') ? new RegExp(args[args.indexOf('--only') + 1]) : null, workers = args.includes('--workers') ? args[args.indexOf('--workers') + 1] : '10';
const plants = fs.readdirSync(path.join(here, 'plants')).filter(f => f.endsWith('.json')).flatMap(f => JSON.parse(fs.readFileSync(path.join(here, 'plants', f), 'utf8')).switches.map(s => s.feature)).filter(f => !only || only.test(f));
/* the states that show each kind of thing: the desk at every tier, values changed, a unit, the tier switch, the parts */
const STATES = 'slice-desk-ace-lab-light|slice-changed-ace-lab$|slice-changed-ace-studio$|slice-desk-ace-studio-light|slice-desk-ace-quick-light|slice-changed-ace-quick|slice-desk-sa3-lab-light|slice-hidden-lab-note|slice-parts-';
const node = (a, env = {}) => spawnSync(process.execPath, a, { cwd: repo, env: { ...process.env, ...env }, encoding: 'utf8' });
const vite = path.join(repo, 'node_modules', 'vite', 'bin', 'vite.js'), vb = path.join(here, 'vb.mjs'), page = path.join(repo, 'dist-switch', 'index.html');
/* the clean build's counts of each accepted difference on these states: a plant that changes one is seen */
const counts = log => [...log.matchAll(/^ {2}(?:ok {2}|FAIL) (\d+) of/gm)].map(m => m[1]).join(',');
node([vite, 'build', '--mode', 'test', '--logLevel', 'error']);
node([vb, 'capture', 'plant-clean', '--states', 'slice', '--scope', 'slice', '--plan-from', 'concept-slice', '--only', STATES, '--workers', workers, '--quiet']);
const clean = counts(node([vb, 'compare', 'concept-slice', 'plant-clean', '--scope', 'slice', '--only', STATES, '--accept', path.join(here, 'slice-accepted.json')]).stdout);
let missed = 0;
for (const f of [...new Set(plants)]) {
  const b = node([vite, 'build', '--mode', 'test', '--logLevel', 'error'], { HS_SWITCH: f });
  if (b.status !== 0) { console.log(`${f}: BUILD FAILED ${(b.stderr || b.stdout).split('\n').filter(l => /error|switch/i.test(l)).slice(0, 2).join(' / ')}`); missed++; continue; }
  node([vb, 'capture', 'plant-now', '--states', 'slice', '--scope', 'slice', '--plan-from', 'concept-slice', '--only', STATES, '--page', page, '--workers', workers, '--quiet']);
  const c = node([vb, 'compare', 'concept-slice', 'plant-now', '--scope', 'slice', '--only', STATES, '--accept', path.join(here, 'slice-accepted.json'), '--max', '3']);
  const g = node([path.join(here, 'gallery.mjs'), '--page', page]);
  /* the accepted counts are for the whole matrix: on these states a count may be lower, which is not a catch */
  const cl = c.stdout.split('\n'), diffs = cl.filter(l => /^ {2}slice|^ {2}shot/.test(l)), gfails = g.stdout.split('\n').filter(l => /^ {2}(light|dark)/.test(l));
  const moved = counts(c.stdout) !== clean;
  const caught = diffs.length > 0 || gfails.length > 0 || moved;
  if (!caught) missed++;
  console.log(`${caught ? 'caught' : 'MISSED'}  ${f}${diffs.length ? `  | compare: ${diffs[0].trim().slice(0, 150)}` : moved ? '  | compare: an accepted difference occurs a different number of times' : ''}${gfails.length ? `  | gallery: ${gfails.length} (${gfails[0].trim().slice(0, 110)})` : ''}`);
}
fs.rmSync(path.join(repo, 'dist-switch'), { recursive: true, force: true });
console.log(`\n${plants.length} planted defects: ${plants.length - missed} caught, ${missed} missed`);
process.exitCode = missed ? 1 : 0;
