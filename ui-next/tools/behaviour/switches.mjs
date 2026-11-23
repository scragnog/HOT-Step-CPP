// Proves the acceptance has teeth: for every slice feature listed in switches/*.json (one file per work package), the
// page is built with that feature switched off (one small change to its source, applied by vite.config.ts in the test
// build when HS_SWITCH names it, into dist-switch/), and the acceptance (run.mjs --slice) is run on it. Each feature
// switched off must make at least one scenario fail; the run fails if one does not, or if a switch no longer applies.
//
//   node tools/behaviour/switches.mjs [--only <regex>] [--workers n]
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const only = opt('only') ? new RegExp(opt('only')) : null;
const dir = path.join(here, 'switches');
const features = [...new Set(fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort().flatMap(f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')).switches.map(s => s.feature)))].filter(f => !only || only.test(f));
const vite = path.join(repo, 'node_modules', 'vite', 'bin', 'vite.js');
const rows = []; let bad = 0;
for (const f of features) {
  const t0 = Date.now();
  const b = spawnSync(process.execPath, [vite, 'build', '--mode', 'test', '--logLevel', 'error'], { cwd: repo, env: { ...process.env, HS_SWITCH: f }, encoding: 'utf8' });
  if (b.status !== 0) { rows.push([f, 'BUILD FAILED', (b.stderr || b.stdout).split('\n').filter(l => /error|switch/i.test(l)).slice(0, 2).join(' / ')]); bad++; process.stdout.write(`${f.padEnd(16)} ${rows[rows.length - 1].slice(1).join('  ')}\n`); continue; }
  const r = spawnSync(process.execPath, [path.join(here, 'run.mjs'), '--slice', '--page', path.join(repo, 'dist-switch', 'index.html'), '--workers', opt('workers', '10')], { cwd: repo, env: process.env, encoding: 'utf8' });
  const failed = [...new Set((r.stdout.match(/^FAIL (\S+)/gm) || []).map(l => l.slice(5)))];
  rows.push([f, failed.length ? `${failed.length} scenarios fail` : 'NOTHING FAILS', failed.slice(0, 4).join(', ') + (failed.length > 4 ? ', ...' : ''), ((Date.now() - t0) / 1000).toFixed(0) + ' s']);
  if (!failed.length) bad++;
  process.stdout.write(`${f.padEnd(16)} ${rows[rows.length - 1].slice(1).join('  ')}\n`);
}
fs.rmSync(path.join(repo, 'dist-switch'), { recursive: true, force: true });
console.log(`\n${features.length} features switched off one at a time: ${features.length - bad} caught by the acceptance, ${bad} not`);
process.exitCode = bad ? 1 : 0;
void execFileSync;
