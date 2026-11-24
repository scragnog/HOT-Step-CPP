// Proves the acceptance has teeth: for every slice feature listed in switches/*.json (one file per work package), the
// page is built with that feature switched off (one small change to its source, applied by vite.config.ts in the test
// build when HS_SWITCH names it, into dist-switch/), and the acceptance's scenarios are run on it. Each feature switched
// off must make at least one scenario fail that passes with the feature on; the run fails if one does not, or if a
// switch no longer applies. See README.md, "Switches".
//
//   node tools/behaviour/switches.mjs [--only <regex>] [--package <regex>] [--workers n] [--scenarios <regex>]
//                                     [--whole] [--fresh] [--no-record]
//   node tools/behaviour/switches.mjs --applies [--only <regex>] [--package <regex>]
//
// A switch is tried against the scenarios that can catch it, in turn, stopping at the first that catches it:
//   1. recorded: the scenarios that caught it last time (caught/<package>.json, written by every run);
//   2. own: the rest of its own package's acceptance (acceptance/<package>.json);
//   3. rest: the rest of the whole acceptance.
// A scenario counts as catching a switch only if it fails with the feature off and passes with it on (each scenario
// that fails on the switched build is run once on the test build, dist-test/index.html, to know). A scenario waiting on a
// later package catches nothing. Running fewer scenarios can only miss a catch, which the next stage then looks for,
// so a switch is reported caught exactly when the whole acceptance would catch it.
// --whole runs the whole acceptance for every switch (the old way: every catching scenario is found, many times slower).
// --applies builds nothing: it checks that each switch's (and each planted defect's) text is found exactly once in
// its file, as the build applies it, which is what a merge most often breaks.
// --package <regex> takes the switches of the packages whose file name matches (e.g. "^(slice|5-models)$").
// --scenarios <regex> limits every stage to the acceptance's scenarios whose id matches.
// --fresh ignores the record (each switch starts at its own package's acceptance); --no-record leaves caught/*.json
// as they are.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ensureTestBuild, testPage } from '../test-build.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; }, flag = k => args.includes('--' + k);
const only = opt('only') ? new RegExp(opt('only')) : null, pkgRe = opt('package') ? new RegExp(opt('package')) : null;
const scRe = opt('scenarios') ? new RegExp(opt('scenarios')) : null, workers = opt('workers', '2');
const json = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const sec = ms => { const s = Math.round(ms / 1000); return s < 60 ? s + ' s' : s < 3600 ? `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s` : `${(s / 3600).toFixed(1)} h`; };

/* the switches, by package (the file's name), each feature once (a feature's changes may span several entries) */
const SW = path.join(here, 'switches'), ACC = path.join(here, 'acceptance'), CAUGHT = path.join(here, 'caught');
const pkgs = fs.readdirSync(SW).filter(f => f.endsWith('.json')).sort().map(f => ({ pkg: f.slice(0, -5), switches: json(path.join(SW, f)).switches }));
const owner = new Map(); for (const p of pkgs) for (const s of p.switches) if (!owner.has(s.feature)) owner.set(s.feature, p.pkg);
const features = [...owner].filter(([f, p]) => (!only || only.test(f)) && (!pkgRe || pkgRe.test(p))).map(([feature, pkg]) => ({ feature, pkg }));

/* --applies: each change of each feature found exactly once in its file, applied in order as vite.config.ts applies them */
if (flag('applies')) {
  const LF = '\n', CRLF = '\r\n', t0 = Date.now();
  const plants = fs.readdirSync(path.join(repo, 'tools', 'visual-baseline', 'plants')).filter(f => f.endsWith('.json')).flatMap(f => json(path.join(repo, 'tools', 'visual-baseline', 'plants', f)).switches.map(s => ({ ...s, pkg: 'plant ' + f.slice(0, -5) })));
  const all = [...pkgs.flatMap(p => p.switches.map(s => ({ ...s, pkg: p.pkg }))), ...plants];
  const names = [...new Set(all.map(s => s.feature))].filter(f => (!only || only.test(f)) && (!pkgRe || all.some(s => s.feature === f && pkgRe.test(s.pkg))));
  let bad = 0, changes = 0;
  for (const f of names) {
    const mine = all.filter(s => s.feature === f), text = {}, problems = [];
    if (new Set(mine.map(s => s.pkg)).size > 1) problems.push(`the feature is named in ${[...new Set(mine.map(s => s.pkg))].join(' and ')}: the build switches off all of them together`);
    for (const s of mine) {
      changes++;
      const file = path.join(repo, s.file);
      if (!(s.file in text)) { if (!fs.existsSync(file)) { problems.push(`${s.file}: no such file`); continue; } text[s.file] = fs.readFileSync(file, 'utf8'); }
      const code = text[s.file], find = code.includes(CRLF) ? s.find.split(LF).join(CRLF) : s.find, n = code.split(find).length - 1;
      if (n !== 1) { problems.push(`"${s.find.slice(0, 70)}" is found ${n} times in ${s.file}`); continue; }
      if (s.find === s.replace) problems.push(`its change in ${s.file} changes nothing`);
      text[s.file] = code.replace(find, () => s.replace);
    }
    if (problems.length) { bad++; console.log(`FAIL ${mine[0].pkg.padEnd(18)} ${f}: ${problems.join('; ')}`); }
  }
  console.log(`\n${names.length} switches and planted defects (${changes} changes): ${names.length - bad} apply exactly once, ${bad} do not, in ${sec(Date.now() - t0)}`);
  process.exit(bad ? 1 : 0);
}

/* the acceptance: every scenario id, by package; a waiting one catches nothing */
const accs = Object.fromEntries(fs.readdirSync(ACC).filter(f => f.endsWith('.json')).sort().map(f => [f.slice(0, -5), json(path.join(ACC, f)).scenarios]));
const WAITING = new Set(Object.values(accs).flat().filter(x => x.waiting).map(x => x.id));
const usable = ids => [...new Set(ids)].filter(id => !WAITING.has(id) && (!scRe || scRe.test(id)));
const ALL = usable(Object.values(accs).flat().map(x => x.id)), OWN = pkg => usable((accs[pkg] || []).map(x => x.id));
const recFile = pkg => path.join(CAUGHT, pkg + '.json');
const records = Object.fromEntries(pkgs.map(p => [p.pkg, fs.existsSync(recFile(p.pkg)) ? json(recFile(p.pkg)).caught : {}]));

/* runs these scenarios on a page; returns the ids that failed (with their failed checks) and passed. A scenario that
   did not run (the harness itself failed: the browser went away under load, say, or run.mjs did not report it) proves
   nothing yet: it is run once more, and only if it fails to run again does it count as failing (named in `broken`). The
   check on the test build then decides whether that failure is the switch's. */
const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function runOnce(ids, page) {
  const r = spawnSync(process.execPath, [path.join(here, 'run.mjs'), '--slice', '--page', page, '--workers', workers, '--only', '^(?:' + ids.map(esc).join('|') + ')$'], { cwd: repo, env: process.env, encoding: 'utf8', maxBuffer: 1 << 28 });
  const out = r.stdout || '', ids_ = re => [...new Set([...out.matchAll(re)].map(m => m[1]))];
  const checks = {}; for (const m of out.matchAll(/^ {2}FAIL (\S+): (.*)$/gm)) (checks[m[1]] ||= []).push(m[2]);
  const failed = ids_(/^FAIL (\S+)/gm), passed = ids_(/^ok +(\S+)/gm);
  const infra = c => c.startsWith('harness') || /Target page, context or browser has been closed|[Bb]rowser has (been closed|disconnected)|browserType\.launch/.test(c);
  const broken = [...ids.filter(id => !failed.includes(id) && !passed.includes(id)), ...failed.filter(id => (checks[id] || []).some(infra))];
  return { failed: failed.filter(id => !broken.includes(id)), passed, broken, checks, note: broken.length ? `exit ${r.status}: ${(r.stderr || out).split('\n').filter(Boolean).slice(-3).join(' / ').slice(0, 300)}` : '' };
}
function run(ids, page) {
  if (!ids.length) return { failed: [], passed: [], broken: [], checks: {} };
  const a = runOnce(ids, page); if (!a.broken.length) return a;
  const b = runOnce(a.broken, page);
  return { failed: [...a.failed, ...b.failed, ...b.broken], passed: [...a.passed, ...b.passed], broken: b.broken, retried: a.broken.length, checks: { ...a.checks, ...b.checks }, note: b.note };
}
/* each scenario's result with every feature on, run once when first needed */
const base = new Map();
const passesOnBase = ids => { const todo = ids.filter(id => !base.has(id)); const r = run(todo, testPage); for (const id of r.failed) base.set(id, false); for (const id of r.passed) base.set(id, true); return ids.filter(id => base.get(id)); };

const built = ensureTestBuild();
const vite = path.join(repo, 'node_modules', 'vite', 'bin', 'vite.js'), switched = path.join(repo, 'dist-switch', 'index.html');
const whole = flag('whole'), t0 = Date.now(), rows = [];
console.log(`${features.length} switches${pkgRe ? ` of the packages matching ${opt('package')}` : ''}, ${whole ? 'each against the whole acceptance' : 'each against the scenarios that caught it, then its own package\'s, then the rest'}; ${ALL.length} scenarios in the acceptance; ${workers} workers${built ? '; test page built' : ''}`);
for (const { feature, pkg } of features) {
  const t1 = Date.now(), row = { feature, pkg, by: [], stage: null, ran: 0, onBase: [], broken: [], checks: {} };
  const b = spawnSync(process.execPath, [vite, 'build', '--mode', 'test', '--logLevel', 'error'], { cwd: repo, env: { ...process.env, HS_SWITCH: feature }, encoding: 'utf8' });
  if (b.status !== 0) row.stage = 'BUILD FAILED', row.error = (b.stderr || b.stdout).split('\n').filter(l => /error|switch/i.test(l)).slice(0, 2).join(' / ');
  else {
    const rec = flag('fresh') ? [] : (records[pkg][feature] || []).filter(id => ALL.includes(id));
    const stages = whole ? [['whole', ALL]] : [['recorded', rec], ['own', OWN(pkg)], ['rest', ALL]];
    const tried = new Set();
    for (const [stage, list] of stages) {
      const ids = list.filter(id => !tried.has(id)); if (!ids.length) continue;
      ids.forEach(id => tried.add(id)); row.ran += ids.length;
      const r = run(ids, switched);
      row.broken.push(...r.broken); row.retried = (row.retried || 0) + (r.retried || 0); if (r.note) row.note = r.note;
      /* a scenario that fails with the feature on as well catches nothing */
      const good = passesOnBase(r.failed);
      row.onBase.push(...r.failed.filter(id => !good.includes(id)));
      if (good.length) { row.by = good; row.stage = stage; for (const id of good) row.checks[id] = r.checks[id]; break; }
    }
    if (!row.stage) row.stage = 'NOTHING FAILS';
  }
  row.took = Date.now() - t1; rows.push(row);
  const caught = row.by.length > 0;
  console.log(`${caught ? 'ok  ' : 'FAIL'} ${pkg.padEnd(15)} ${feature.padEnd(24)} ${caught ? `caught (${row.stage}, ${row.ran} run) by ${row.by.length}: ${row.by.slice(0, 3).join(', ')}${row.by.length > 3 ? ', ...' : ''}` : row.stage + (row.error ? ': ' + row.error : ` (${row.ran} scenarios run)`)}${row.onBase.length ? `  [failing with the feature on too, not counted: ${row.onBase.join(', ')}]` : ''}${row.retried ? `  [${row.retried} did not run and were run again${row.broken.length ? `; did not run twice, counted as failing: ${row.broken.slice(0, 3).join(', ')}${row.broken.length > 3 ? ', ...' : ''} (${row.note})` : ''}]` : ''}  ${sec(row.took)}`);
}
fs.rmSync(path.join(repo, 'dist-switch'), { recursive: true, force: true });

/* the record: for each switch the scenarios that caught it, kept per package beside its switches */
if (!flag('no-record') && !scRe) {
  fs.mkdirSync(CAUGHT, { recursive: true });
  for (const p of pkgs) {
    const mine = rows.filter(r => r.pkg === p.pkg && r.by.length); if (!mine.length) continue;
    /* a switch not run now keeps its entry; the file is written only when an entry changes */
    const old = records[p.pkg], caught = {};
    for (const s of p.switches) { if (caught[s.feature]) continue; const r = mine.find(x => x.feature === s.feature); if (r) caught[s.feature] = r.by.slice().sort(); else if (old[s.feature]) caught[s.feature] = old[s.feature]; }
    if (JSON.stringify(caught) === JSON.stringify(old)) continue;
    const about = `The scenarios that caught each switch of switches/${p.pkg}.json, as tools/behaviour/switches.mjs last found them (each passes with the feature on and fails with it off); it tries these first. Written by its runs; commit it with the package's work.`;
    fs.writeFileSync(recFile(p.pkg), '{\r\n "about": ' + JSON.stringify(about) + ',\r\n "caught": {\r\n' + Object.entries(caught).map(([f, by]) => '  ' + JSON.stringify(f) + ': ' + JSON.stringify(by)).join(',\r\n') + '\r\n }\r\n}\r\n');
  }
}
const out = path.join(here, 'out'); fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'switches-last.json'), JSON.stringify({ at: new Date().toISOString(), whole, rows }, null, 1));

const missed = rows.filter(r => !r.by.length), by = st => rows.filter(r => r.stage === st).length, runs = rows.reduce((a, r) => a + r.ran, 0);
console.log(`\n${rows.length} features switched off one at a time: ${rows.length - missed.length} caught by the acceptance, ${missed.length} not, in ${sec(Date.now() - t0)}`);
console.log(`  caught ${whole ? `by the whole acceptance ${by('whole')}` : `by the recorded scenarios ${by('recorded')}, by their own package's ${by('own')}, by the rest ${by('rest')}`}; ${runs} scenario runs on switched builds (${(runs / Math.max(1, rows.length)).toFixed(1)} a switch), ${base.size} on the test build`);
for (const r of missed) console.log(`  NOT CAUGHT ${r.pkg} / ${r.feature}: ${r.stage}${r.error ? ': ' + r.error : ''}`);
console.log(`  every switch's catching scenarios: tools/behaviour/out/switches-last.json${flag('no-record') || scRe ? '' : ' and caught/<package>.json'}`);
process.exitCode = missed.length ? 1 : 0;
