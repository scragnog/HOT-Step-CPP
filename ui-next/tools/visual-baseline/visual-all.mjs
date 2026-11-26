// Every visual set, in one command: for each set a file of slice-states/ declares (its VISUAL_SET: the states it
// compares and their scope), captures the rebuild in those states and compares it with the set's approved recording
// (tools/visual-baseline/approved/<set>/, committed), and prints one line per set and a verdict. Exit 1 if any set
// differs. With --approve it records the current build as the approved recording of each set chosen instead.
// A file that adds a set (a VISUAL_SET) adds nothing here. See README.md.
//
//   npm run visual:all [-- options]                 compare every set with its approved recording
//   npm run visual:approve -- --only "^models$"     approve the current build for some sets (after a deliberate change)
//   node tools/visual-baseline/visual-all.mjs [--only <regex on set names>] [--approve] [--page <file>] [--workers n] [--list]
//
// --workers: pages at once for each capture (default: half the machine's threads, 6 on 12; lower it to leave more).
// --page: the build to check (default dist-test/index.html, built first if missing or older than the sources).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { STATES, SETS } from './slice-states.mjs';
import { ensureTestBuild, testPage } from '../test-build.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..'), OUT = path.join(here, 'out'), APPROVED = path.join(here, 'approved');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; }, flag = k => args.includes('--' + k);
const only = opt('only') ? new RegExp(opt('only')) : null, workers = opt('workers', String(Math.max(1, Math.floor(os.availableParallelism() / 2)))), approve = flag('approve');
const sec = ms => { const s = Math.round(ms / 1000); return s < 60 ? s + ' s' : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`; };

/* each set's states: a state belongs to the set whose regular expression matches its id; every state must be in one */
const sets = SETS.map(s => ({ ...s, re: new RegExp(s.only), ids: STATES.map(x => x.id).filter(id => new RegExp(s.only).test(id)) }));
const names = new Set(); for (const s of sets) { if (names.has(s.name)) throw new Error(`the set ${s.name} is declared twice (slice-states/${s.file})`); names.add(s.name); }
const orphans = STATES.filter(x => !sets.some(s => s.re.test(x.id))).map(x => x.id), twice = STATES.filter(x => sets.filter(s => s.re.test(x.id)).length > 1).map(x => x.id);
const chosen = sets.filter(s => !only || only.test(s.name));
const metaOf = dir => { try { return JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8')); } catch { return null; } };
if (flag('list')) {
  for (const s of sets) { const m = metaOf(path.join(APPROVED, s.name)); console.log(`${s.name.padEnd(16)} ${String(s.ids.length).padStart(4)} states  ${s.only}${s.scope ? '  scope ' + s.scope : ''}  approved ${m ? m.at.slice(0, 10) + (m.commit ? ' at ' + m.commit : '') : 'never'}  (slice-states/${s.file})`); }
  if (orphans.length) console.log(`${orphans.length} states in no set: ${orphans.join(', ')}`);
  process.exit(orphans.length ? 1 : 0);
}
/* a state no set compares is a hole, not a note: it fails at once, before anything is captured, naming what to do */
if (orphans.length) {
  console.log(`FAIL ${orphans.length} states belong to no visual set, so nothing compares them: ${orphans.slice(0, 12).join(', ')}${orphans.length > 12 ? ', ...' : ''}`);
  console.log(`     give the file that adds them its set (export const VISUAL_SET = { name, only, scope? }), or widen an existing set's \`only\` (README.md)`);
  process.exit(1);
}
if (twice.length) { console.log(`FAIL ${twice.length} states belong to two sets, compared twice: ${twice.slice(0, 12).join(', ')}`); process.exit(1); }
if (!chosen.length) throw new Error(`no set matches ${opt('only')}: ${sets.map(s => s.name).join(', ')}`);

const page = opt('page') ? path.resolve(opt('page')) : testPage;
if (!opt('page')) ensureTestBuild();
const logs = path.join(OUT, 'all-logs'); fs.mkdirSync(logs, { recursive: true });
const vb = (log, ...a) => { const fd = fs.openSync(path.join(logs, log), 'w'); const r = spawnSync(process.execPath, [path.join(here, 'vb.mjs'), ...a], { cwd: repo, stdio: ['ignore', fd, fd] }); fs.closeSync(fd); return { status: r.status, text: fs.readFileSync(path.join(logs, log), 'utf8') }; };
const commit = (() => { const r = spawnSync('git', ['rev-parse', '--short=10', 'HEAD'], { cwd: repo, encoding: 'utf8' }); return r.status === 0 ? r.stdout.trim() : ''; })();
const dirty = (() => { const r = spawnSync('git', ['status', '--porcelain', '--', 'src', 'index.html'], { cwd: repo, encoding: 'utf8' }); return r.status === 0 && !!r.stdout.trim(); })();

console.log(`visual sets: ${chosen.length} of ${sets.length} (${chosen.map(s => s.name).join(', ')}); ${approve ? 'approving' : 'against the approved recordings'}; ${path.relative(repo, page)}; ${workers} workers`);
const t0 = Date.now(), rows = [];
for (const s of chosen) {
  const t1 = Date.now(), adir = path.join(APPROVED, s.name), sc = s.scope ? ['--scope', s.scope] : [], common = ['--only', s.only, ...sc, '--workers', workers, '--page', page, '--slim'];
  let row;
  if (approve) {
    /* a fresh recording of the current build, its own plan of forced states, kept as the set's approved recording */
    const tmp = path.join(OUT, `approve-${s.name}`); fs.rmSync(tmp, { recursive: true, force: true });
    const r = vb(`${s.name}-approve.log`, 'capture', tmp, ...common);
    const setup = r.text.includes('setup notes');
    if (r.status !== 0 || !fs.existsSync(path.join(tmp, 'meta.json'))) row = [s.name, s.ids.length, 'CAPTURE FAILED', `${s.name}-approve.log`, []];
    else {
      const meta = metaOf(tmp); fs.writeFileSync(path.join(tmp, 'meta.json'), JSON.stringify({ set: s.name, at: meta.at, commit: commit + (dirty ? ' (with uncommitted changes)' : ''), states: meta.states }, null, 1));
      fs.rmSync(adir, { recursive: true, force: true }); fs.mkdirSync(APPROVED, { recursive: true }); fs.renameSync(tmp, adir);
      row = [s.name, s.ids.length, 'approved', `${s.name}-approve.log`, setup ? ['setup notes in the log: a state whose setup failed was approved as it is; read them'] : []];
    }
  } else {
    const m = metaOf(adir);
    if (!m) row = [s.name, s.ids.length, 'NO APPROVED RECORDING', '', [`approve it: npm run visual:approve -- --only "^${s.name}$"`]];
    else if (JSON.stringify(m.states) !== JSON.stringify(s.ids)) {
      const gone = m.states.filter(id => !s.ids.includes(id)), fresh = s.ids.filter(id => !m.states.includes(id));
      row = [s.name, s.ids.length, 'STATES CHANGED SINCE APPROVED', '', [`${fresh.length ? 'new: ' + fresh.slice(0, 6).join(', ') + ' ' : ''}${gone.length ? 'gone: ' + gone.slice(0, 6).join(', ') : ''}`, `approve the set once its new states look right: npm run visual:approve -- --only "^${s.name}$"`]];
    }
  }
  if (!row) {
    const now = path.join(OUT, `now-${s.name}`); fs.rmSync(now, { recursive: true, force: true });
    const r = vb(`${s.name}-capture.log`, 'capture', now, ...common, '--plan-from', adir);
    if (r.status !== 0) row = [s.name, s.ids.length, 'CAPTURE FAILED', `${s.name}-capture.log`, []];
    else {
      const c = vb(`${s.name}-compare.log`, 'compare', adir, now, ...sc, '--max', '12');
      const res = (c.text.match(/^RESULT: (.*)$/m) || [])[1], n = (c.text.match(/^compared (\d+) states/m) || [])[1];
      const verdict = res === 'identical' ? 'identical' : res ? `${parseInt(res, 10)} differences` : 'COMPARE FAILED';
      row = [s.name, n ?? s.ids.length, verdict, `${s.name}-compare.log`, res === 'identical' ? [] : c.text.split('\n').filter(l => /^ {2}\S/.test(l)).slice(0, 6)];
    }
  }
  row.push(sec(Date.now() - t1));
  rows.push(row);
  const [name, n, verdict, log, detail = [], took] = row, good = verdict === 'identical' || verdict === 'approved';
  console.log(`${good ? 'ok  ' : 'FAIL'} ${name.padEnd(16)} ${String(n).padStart(4)} states  ${verdict}  (${took})${!good && log ? '  -> ' + path.relative(repo, path.join(logs, log)) : ''}`);
  for (const l of detail) console.log('       ' + l.trim().slice(0, 220));
}
const bad = rows.filter(r => r[2] !== 'identical' && r[2] !== 'approved'), states = rows.reduce((a, r) => a + +r[1], 0);
if (approve) console.log(`\n${bad.length ? `RESULT: ${bad.length} of ${rows.length} sets not approved (${bad.map(r => r[0]).join(', ')})` : `RESULT: ${rows.length} sets approved (${states} states) into ${path.relative(repo, APPROVED)}: look at their screenshots, then commit them`}, in ${sec(Date.now() - t0)}`);
else console.log(`\n${bad.length ? `RESULT: ${bad.length} of ${rows.length} sets differ from their approved recordings (${bad.map(r => r[0]).join(', ')}); after a deliberate change, look at the differences, then npm run visual:approve -- --only "^(${bad.map(r => r[0]).join('|')})$"` : `RESULT: every set identical with its approved recording (${rows.length} sets, ${states} states)`}, in ${sec(Date.now() - t0)}`);
process.exitCode = bad.length ? 1 : 0;
