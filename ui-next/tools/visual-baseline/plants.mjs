// Plants each defect of plants/*.json in the test build, one at a time (vite.config.ts, HS_SWITCH, into dist-switch/),
// captures the slice's states that show it and compares them with the slice's approved recording (approved/slice/),
// and runs the parts gallery on it. A defect is caught when either fails. Prints caught or missed for each; exit 1 if
// one is missed. First it checks that the clean test build matches the approved recording on those states, so an
// approval out of date cannot pass for a catch.
//
//   node tools/visual-baseline/plants.mjs [--only <regex>] [--workers n]

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..');
const args = process.argv.slice(2), only = args.includes('--only') ? new RegExp(args[args.indexOf('--only') + 1]) : null;
const workers = args.includes('--workers') ? args[args.indexOf('--workers') + 1] : String(Math.max(1, Math.floor(os.availableParallelism() / 2)));
const plants = fs.readdirSync(path.join(here, 'plants')).filter(f => f.endsWith('.json')).flatMap(f => JSON.parse(fs.readFileSync(path.join(here, 'plants', f), 'utf8')).switches.map(s => s.feature)).filter(f => !only || only.test(f));
/* the states that show each kind of thing: the desk at every tier, values changed, a unit, the tier switch, the parts */
const STATES = 'slice-desk-ace-lab-light|slice-changed-ace-lab$|slice-changed-ace-studio$|slice-desk-ace-studio-light|slice-desk-ace-quick-light|slice-changed-ace-quick|slice-desk-sa3-lab-light|slice-hidden-lab-note|slice-parts-';
const node = (a, env = {}) => spawnSync(process.execPath, a, { cwd: repo, env: { ...process.env, ...env }, encoding: 'utf8', maxBuffer: 1 << 26 });
const vite = path.join(repo, 'node_modules', 'vite', 'bin', 'vite.js'), vb = path.join(here, 'vb.mjs'), page = path.join(repo, 'dist-switch', 'index.html');
const approved = path.join(here, 'approved', 'slice');
if (!fs.existsSync(path.join(approved, 'meta.json'))) { console.log('no approved recording of the slice: npm run visual:approve -- --only "^slice$"'); process.exit(1); }
const capture = (name, pg) => node([vb, 'capture', name, '--scope', 'slice', '--plan-from', approved, '--only', STATES, '--page', pg, '--workers', workers, '--slim', '--quiet']);
const compare = name => node([vb, 'compare', approved, name, '--scope', 'slice', '--only', STATES, '--max', '3']).stdout;

/* the clean build first: it must match the approved recording, or every plant would look caught */
node([vite, 'build', '--mode', 'test', '--logLevel', 'error']);
capture('plant-clean', path.join(repo, 'dist-test', 'index.html'));
const clean = compare('plant-clean');
if (!/^RESULT: identical/m.test(clean)) { console.log('the clean test build differs from the approved slice recording on the planted states; run npm run visual:all first:\n' + clean.split('\n').filter(l => /^ {2}|^RESULT/.test(l)).slice(0, 6).join('\n')); process.exit(1); }
let missed = 0;
try {
  for (const f of [...new Set(plants)]) {
    const b = node([vite, 'build', '--mode', 'test', '--logLevel', 'error'], { HS_SWITCH: f });
    if (b.status !== 0) { console.log(`${f}: BUILD FAILED ${(b.stderr || b.stdout).split('\n').filter(l => /error|switch/i.test(l)).slice(0, 2).join(' / ')}`); missed++; continue; }
    capture('plant-now', page);
    const c = compare('plant-now'), g = node([path.join(here, 'gallery.mjs'), '--page', page]);
    const diffs = c.split('\n').filter(l => /^ {2}slice|^ {2}shot/.test(l)), gfails = g.stdout.split('\n').filter(l => /^ {2}(light|dark)/.test(l));
    const caught = diffs.length > 0 || gfails.length > 0;
    if (!caught) missed++;
    console.log(`${caught ? 'caught' : 'MISSED'}  ${f}${diffs.length ? `  | compare: ${diffs[0].trim().slice(0, 150)}` : ''}${gfails.length ? `  | gallery: ${gfails.length} (${gfails[0].trim().slice(0, 110)})` : ''}`);
  }
} finally { fs.rmSync(path.join(repo, 'dist-switch'), { recursive: true, force: true }); }
console.log(`\n${plants.length} planted defects: ${plants.length - missed} caught, ${missed} missed`);
process.exitCode = missed ? 1 : 0;
