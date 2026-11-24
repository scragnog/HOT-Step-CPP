// Behaviour suite: drives hotstep-tiers.html through every handler (clicks, keys, drags, files, persistence, engines)
// and checks what each one does. See README.md in this folder.
//
//   node tools/behaviour/run.mjs [--only <regex>] [--workers n] [--page file.html] [--list]
//   node tools/behaviour/run.mjs --record <name> [...]     also record a trace: a digest of the page after every step
//   node tools/behaviour/run.mjs --compare <name> [...]    run, and compare the trace step by step with <name>'s
//   node tools/behaviour/run.mjs --dump ...                traces keep each top-level element's full markup (to diff)
//   node tools/behaviour/run.mjs --pending [...]           run only the pending scenarios (behaviours the page is known
//                                                          to get wrong): each "still failing" or "now passes"
//   node tools/behaviour/run.mjs --slice [...]             the rebuild's acceptance: every scenario listed in
//                                                          acceptance/*.json (one file per work package)
//
// In ui-next the page is the rebuild's test build (dist-test/index.html) unless --page names another.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { serve, runScenario } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const OUT = path.join(here, 'out');
const require = createRequire(import.meta.url);
const loadPW = () => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch (e) {} } throw new Error('Playwright not found: run npm install (devDependency), or set PLAYWRIGHT_MODULE to a playwright folder'); };
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const flag = k => args.includes('--' + k);

const files = fs.readdirSync(path.join(here, 'scenarios')).filter(f => f.endsWith('.mjs')).sort();
const all = [];
for (const f of files) { const mod = await import(pathToFileURL(path.join(here, 'scenarios', f)).href); for (const sc of mod.default) all.push({ ...sc, id: f.replace(/\.mjs$/, '') + '/' + sc.id }); }
const only = opt('only') ? new RegExp(opt('only')) : null;
/* --slice: the scenarios the acceptance files list (acceptance/*.json, one per work package), each with the reason it
   is there; a scenario the concept marks pending that the rebuild gets right is run as a normal one */
const ACC = path.join(here, 'acceptance');
const ENTRIES = flag('slice') ? fs.readdirSync(ACC).filter(f => f.endsWith('.json')).sort().flatMap(f => JSON.parse(fs.readFileSync(path.join(ACC, f), 'utf8')).scenarios) : null;
const SLICE = ENTRIES ? ENTRIES.map(x => x.id) : null;
/* an entry with { waiting: { package, because } } is right but cannot pass until that package lands (a placeholder it
   will replace changes what the scenario sees): it runs, is reported as waiting and not counted as a failure, and if it
   passes the run fails, so whoever lands the package moves it back to a normal entry (README, "Waiting scenarios") */
const WAITING = new Map((ENTRIES ?? []).filter(x => x.waiting).map(x => [x.id, x.waiting]));
if (SLICE) { const known = new Set(all.map(s => s.id)), bad = SLICE.filter(id => !known.has(id)); if (bad.length) throw new Error('acceptance lists scenarios that do not exist: ' + bad.join(', ')); }
/* every scenario of the concept's suite (the files numbered below 30) is in an acceptance list or excluded from one with
   its reason: a scenario nobody runs is a hole, not a choice (RA-06) */
if (SLICE && !only) {
  const listed = new Set(fs.readdirSync(ACC).filter(f => f.endsWith('.json')).flatMap(f => { const j = JSON.parse(fs.readFileSync(path.join(ACC, f), 'utf8')); return [...j.scenarios, ...(j.excluded ?? [])].map(x => x.id); }));
  const loose = all.filter(s => +s.id.split('-')[0] < 30 && !listed.has(s.id)).map(s => s.id);
  if (loose.length) throw new Error(`${loose.length} scenarios of the concept's suite are in no acceptance list and excluded from none (put each in one, or exclude it with its reason): ${loose.join(', ')}`);
}
/* a pending scenario ({ pending: { finding, right } }) states a behaviour the page is known to get wrong: a normal run
   leaves it out, --pending runs only those */
const pendingRun = flag('pending'), pendingAll = all.filter(s => s.pending);
const list = all.filter(s => (!only || only.test(s.id)) && (SLICE ? SLICE.includes(s.id) && !pendingRun : !!s.pending === pendingRun));
if (flag('list')) { list.forEach(s => console.log(s.id + (s.pending ? `  [pending ${s.pending.finding}]` : ''))); console.log(list.length + (pendingRun ? ' pending scenarios' : ' scenarios')); process.exit(0); }

const record = opt('record'), compareWith = opt('compare'), trace = !!(record || compareWith);
const gz = o => zlib.gzipSync(Buffer.from(JSON.stringify(o))), ungz = f => JSON.parse(zlib.gunzipSync(fs.readFileSync(f)).toString());
const tdir = name => path.join(OUT, name), tfile = (name, id) => path.join(tdir(name), id.replace(/[\\/]/g, '__') + '.json.gz');
const saveAs = record || (compareWith ? compareWith + '-now' : null);
if (saveAs) { fs.rmSync(tdir(saveAs), { recursive: true, force: true }); fs.mkdirSync(tdir(saveAs), { recursive: true }); }

const file = path.resolve(opt('page') || path.join(repo, 'dist-test', 'index.html'));
const server = await serve(file), pw = loadPW(), browser = await pw.chromium.launch();
const workers = +(opt('workers') || 4); let next = 0; const t0 = Date.now(), done = [];
await Promise.all(Array.from({ length: workers }, async () => {
  while (next < list.length) {
    const sc = list[next++];
    let r; try { r = await runScenario(browser, sc, server.url, { trace, dump: flag('dump') }); } catch (e) { r = { id: sc.id, results: [[false, 'harness', String(e.stack || e).split('\n')[0]]], trace: [] }; }
    if (saveAs) fs.writeFileSync(tfile(saveAs, sc.id), gz(r.trace));
    done.push(r); const bad = r.results.filter(x => !x[0]).length, w = WAITING.get(sc.id);
    if (w) process.stdout.write(bad ? `wait ${sc.id} (waiting on package ${w.package}: ${bad} of ${r.results.length} checks fail)\n` : `FAIL ${sc.id} (waiting on package ${w.package}, but it passes: move it back to a normal entry)\n`);
    else process.stdout.write(`${bad ? 'FAIL' : 'ok  '} ${sc.id} (${r.results.length} checks${bad ? `, ${bad} failed` : ''})\n`);
  }
}));
await browser.close(); server.close();

done.sort((a, b) => a.id.localeCompare(b.id));
const fails = done.flatMap(r => WAITING.has(r.id) ? (r.results.some(x => !x[0]) ? [] : [`${r.id}: waiting on package ${WAITING.get(r.id).package}, but it passes: move it back to a normal entry`]) : r.results.filter(x => !x[0]).map(x => `${r.id}: ${x[1]}${x[2] ? ' :: ' + x[2] : ''}`));
const waited = done.filter(r => WAITING.has(r.id) && r.results.some(x => !x[0])).sort((a, b) => a.id.localeCompare(b.id));
const checks = done.reduce((a, r) => a + r.results.length, 0);
if (pendingRun) {
  /* each pending scenario is expected to fail; one that passes is the signal to promote it (remove its pending field) */
  const now = done.filter(r => !r.results.some(x => !x[0]));
  console.log(`\n${done.length} pending scenarios: ${done.length - now.length} still failing, ${now.length} now pass, in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  for (const r of done) {
    const sc = list.find(s => s.id === r.id), bad = r.results.filter(x => !x[0]);
    console.log(`\n${bad.length ? 'still failing' : 'NOW PASSES'}  ${r.id}  [${sc.pending.finding}]\n  right: ${sc.pending.right}`);
    if (bad.length) bad.forEach(x => console.log(`  fails: ${x[1]}${x[2] ? ' :: ' + x[2] : ''}`));
    else console.log('  promote it: remove its pending field, so that every normal run keeps it passing');
  }
} else {
  console.log(`\n${done.length} scenarios, ${checks} checks, ${fails.length} failed, in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  fails.forEach(f => console.log('  FAIL ' + f));
  if (waited.length) { console.log(`${waited.length} scenarios waiting on a later package, not counted as failures:`); waited.forEach(r => { const w = WAITING.get(r.id); console.log(`  waiting on package ${w.package}: ${r.id}: ${w.because}`); r.results.filter(x => !x[0]).forEach(x => console.log(`    fails: ${x[1]}${x[2] ? ' :: ' + x[2] : ''}`)); }); }
  console.log(`${pendingAll.length} pending scenarios not run (behaviours known to be wrong; see README): --pending runs them`);
}

/* trace comparison: the same scenario on two builds must leave the page in the same state after every step */
const sortKeys = v => Array.isArray(v) ? v.map(sortKeys) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sortKeys(v[k])])) : v;
const canon = x => { if (x == null) return x; try { return JSON.stringify(sortKeys(JSON.parse(x))); } catch (e) { return x; } };
let tdiff = 0;
if (compareWith) {
  const lines = [];
  for (const r of done) {
    const f = tfile(compareWith, r.id); if (!fs.existsSync(f)) { lines.push(`${r.id}: no trace in ${compareWith}`); tdiff++; continue; }
    const A = ungz(f), B = r.trace;
    if (A.length !== B.length) { lines.push(`${r.id}: ${A.length} steps in ${compareWith}, ${B.length} now`); tdiff++; }
    for (let i = 0; i < Math.min(A.length, B.length); i++) {
      const a = A[i], b = B[i], d = [];
      if (a.label !== b.label) d.push(`label ${a.label} vs ${b.label}`);
      for (const k of ['S', 'act', 'title', 'root']) if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) d.push(k + (k === 'S' ? '' : ` ${JSON.stringify(a[k])} vs ${JSON.stringify(b[k])}`));
      if (JSON.stringify(a.scroll) !== JSON.stringify(b.scroll)) d.push(`scroll ${a.scroll} vs ${b.scroll}`);
      /* stored JSON is compared with its objects' keys sorted (the order a feature adds a key in is not behaviour); arrays keep their order */
      const lsk = new Set([...Object.keys(a.ls), ...Object.keys(b.ls)]); for (const k of lsk) if (canon(a.ls[k]) !== canon(b.ls[k])) d.push('localStorage ' + k);
      const ka = a.kids.map(x => x[0] + ':' + x[1]).join(' '), kb = b.kids.map(x => x[0] + ':' + x[1]).join(' ');
      if (ka !== kb) { const n = Math.max(a.kids.length, b.kids.length); for (let j = 0; j < n; j++) if (JSON.stringify(a.kids[j]?.slice(0, 2)) !== JSON.stringify(b.kids[j]?.slice(0, 2))) d.push(`markup of ${a.kids[j]?.[0] ?? '-'} / ${b.kids[j]?.[0] ?? '-'}`); }
      if (d.length) { lines.push(`${r.id} step ${i} (${a.label}): ${d.join('; ')}`); tdiff++; break; }
    }
  }
  console.log(`\ntrace compared with ${compareWith}: ${tdiff ? tdiff + ' scenarios differ' : 'identical'}`);
  lines.forEach(l => console.log('  ' + l));
}
/* coverage: every data-* key the delegated handlers read (d.key, 'key' in d), from the page's own script, against the
   keys carried by what the clicks landed on */
if (flag('coverage')) {
  const html = fs.readFileSync(file, 'utf8'), keys = new Set();
  for (const m of html.matchAll(/\bd\.([a-z][a-z0-9]*)\b/g)) keys.add(m[1]);
  for (const m of html.matchAll(/'([a-z0-9]+)' in (?:d|b\.dataset)\b/g)) keys.add(m[1]);
  /* a name counts as a hook when the page has it as a data- attribute, or tests for it with 'in' */
  for (const k of [...keys]) { const kebab = k.replace(/[A-Z]/g, c => '-' + c.toLowerCase()); if (!new RegExp('data-' + kebab + '(?![a-z-])').test(html) && !html.includes("'" + k + "' in d")) keys.delete(k); }
  const seen = Object.assign({}, ...done.map(r => r.cov || {}));
  const miss = [...keys].filter(k => !seen[k]).sort();
  console.log(`\nclick coverage: ${keys.size - miss.length} of ${keys.size} hooks clicked${miss.length ? '; never clicked: ' + miss.join(', ') : ''}`);
}
process.exitCode = fails.length || tdiff ? 1 : 0;
