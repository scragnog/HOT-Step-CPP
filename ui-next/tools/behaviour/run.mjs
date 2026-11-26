// Behaviour suite: drives the test build through its handlers (clicks, keys, drags, files, persistence, engines) and
// checks what each one does: the acceptance, every scenario that acceptance/*.json lists (one file per area of the
// page). See README.md in this folder.
//
//   node tools/behaviour/run.mjs [--only <regex>] [--workers n] [--page file.html] [--list]
//   node tools/behaviour/run.mjs --record <name> [...]     also record a trace: a digest of the page after every step
//   node tools/behaviour/run.mjs --compare <name> [...]    run, and compare the trace step by step with <name>'s
//   node tools/behaviour/run.mjs --dump ...                traces keep each top-level element's full markup (to diff)
//
// The page is the test build (dist-test/index.html) unless --page names another. (--slice, the acceptance's old name,
// is still taken and changes nothing.)
import fs from 'node:fs';
import os from 'node:os';
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
/* the acceptance: the scenarios the acceptance files list, each with the reason it is there */
const ACC = path.join(here, 'acceptance');
const ENTRIES = fs.readdirSync(ACC).filter(f => f.endsWith('.json')).sort().flatMap(f => JSON.parse(fs.readFileSync(path.join(ACC, f), 'utf8')).scenarios);
const LISTED = new Set(ENTRIES.map(x => x.id));
/* an entry with { waiting: { package, because } } is right but cannot pass until that work lands: it runs, is reported
   as waiting and not counted as a failure, and if it passes the run fails, so whoever lands the work moves it back to
   a normal entry (README, "Waiting scenarios") */
const WAITING = new Map(ENTRIES.filter(x => x.waiting).map(x => [x.id, { ...x.waiting, on: typeof x.waiting.package === 'number' ? 'package ' + x.waiting.package : x.waiting.package }]));
{ const known = new Set(all.map(s => s.id)), bad = [...LISTED].filter(id => !known.has(id)); if (bad.length) throw new Error('acceptance lists scenarios that do not exist: ' + bad.join(', ')); }
/* every scenario is in an acceptance list: a scenario nobody runs is a hole, not a choice */
const loose = all.filter(s => !LISTED.has(s.id)).map(s => s.id);
if (loose.length && !only) throw new Error(`${loose.length} scenarios are in no acceptance list (put each in the acceptance file of its area, with why it is there): ${loose.join(', ')}`);
const list = all.filter(s => !only || only.test(s.id));
if (flag('list')) { list.forEach(s => console.log(s.id + (WAITING.has(s.id) ? `  [waiting on ${WAITING.get(s.id).on}]` : ''))); console.log(list.length + ' scenarios'); process.exit(0); }

const record = opt('record'), compareWith = opt('compare'), trace = !!(record || compareWith);
const gz = o => zlib.gzipSync(Buffer.from(JSON.stringify(o))), ungz = f => JSON.parse(zlib.gunzipSync(fs.readFileSync(f)).toString());
const tdir = name => path.join(OUT, name), tfile = (name, id) => path.join(tdir(name), id.replace(/[\\/]/g, '__') + '.json.gz');
const saveAs = record || (compareWith ? compareWith + '-now' : null);
if (saveAs) { fs.rmSync(tdir(saveAs), { recursive: true, force: true }); fs.mkdirSync(tdir(saveAs), { recursive: true }); }

/* the longest scenarios first (their times from the last run, out/durations.json, git-ignored), so the run does not end
   waiting on one long scenario started last; a scenario with no time yet goes first */
const DUR = path.join(OUT, 'durations.json'), durations = (() => { try { return JSON.parse(fs.readFileSync(DUR, 'utf8')); } catch { return {}; } })();
const order = [...list].sort((a, b) => (durations[b.id] ?? Infinity) - (durations[a.id] ?? Infinity));
const file = path.resolve(opt('page') || path.join(repo, 'dist-test', 'index.html'));
const server = await serve(file), pw = loadPW(), browser = await pw.chromium.launch();
const workers = +(opt('workers') || Math.max(1, Math.floor(os.availableParallelism() / 2))); let next = 0; const t0 = Date.now(), done = [];
try {
  await Promise.all(Array.from({ length: workers }, async () => {
    while (next < order.length) {
      const sc = order[next++], t1 = Date.now();
      let r; try { r = await runScenario(browser, sc, server.url, { trace, dump: flag('dump') }); } catch (e) { r = { id: sc.id, results: [[false, 'harness', String(e.stack || e).split('\n')[0]]], trace: [] }; }
      if (saveAs) fs.writeFileSync(tfile(saveAs, sc.id), gz(r.trace));
      durations[sc.id] = Date.now() - t1;
      done.push(r); const bad = r.results.filter(x => !x[0]).length, w = WAITING.get(sc.id);
      if (w) process.stdout.write(bad ? `wait ${sc.id} (waiting on ${w.on}: ${bad} of ${r.results.length} checks fail)\n` : `FAIL ${sc.id} (waiting on ${w.on}, but it passes: move it back to a normal entry)\n`);
      else process.stdout.write(`${bad ? 'FAIL' : 'ok  '} ${sc.id} (${r.results.length} checks${bad ? `, ${bad} failed` : ''})\n`);
    }
  }));
} finally { await browser.close(); server.close(); }
try { fs.mkdirSync(OUT, { recursive: true }); fs.writeFileSync(DUR, JSON.stringify(durations)); } catch { /* only an order */ }

done.sort((a, b) => a.id.localeCompare(b.id));
const fails = done.flatMap(r => WAITING.has(r.id) ? (r.results.some(x => !x[0]) ? [] : [`${r.id}: waiting on ${WAITING.get(r.id).on}, but it passes: move it back to a normal entry`]) : r.results.filter(x => !x[0]).map(x => `${r.id}: ${x[1]}${x[2] ? ' :: ' + x[2] : ''}`));
const waited = done.filter(r => WAITING.has(r.id) && r.results.some(x => !x[0]));
const checks = done.reduce((a, r) => a + r.results.length, 0);
console.log(`\n${done.length} scenarios, ${checks} checks, ${fails.length} failed, in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
fails.forEach(f => console.log('  FAIL ' + f));
if (waited.length) { console.log(`${waited.length} scenarios waiting on later work, not counted as failures:`); waited.forEach(r => { const w = WAITING.get(r.id); console.log(`  waiting on ${w.on}: ${r.id}: ${w.because}`); r.results.filter(x => !x[0]).forEach(x => console.log(`    fails: ${x[1]}${x[2] ? ' :: ' + x[2] : ''}`)); }); }

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
process.exitCode = fails.length || tdiff ? 1 : 0;
