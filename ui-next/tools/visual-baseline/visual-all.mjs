// Every package's visual set, in one command: for each set a file of slice-states/ declares (its VISUAL_SET: the states it
// compares, their scope, its accepted differences), captures the concept page and the rebuild in those states, compares
// them with the set's accepted differences, and prints one line per set and a verdict. Exit 1 if any set differs.
// A package that adds a set (a file with a VISUAL_SET) adds nothing here. See README.md, "Every set at once".
//
//   npm run visual:all [-- options]
//   node tools/visual-baseline/visual-all.mjs [--only <regex on set names>] [--reuse-concept] [--concept <file>]
//                                             [--page <file>] [--workers n] [--list]
//
// --reuse-concept: a set's concept recording is kept (out/all-concept-<set>-<key>), keyed by the concept file's content
//   and by what decides the recording (the set, its states' files, vb.mjs, page-lib.js, Playwright's version); with this
//   option a kept recording of the same key is used instead of capturing the concept again.
// --concept: the concept page (default: $HOTSTEP_CONCEPT/hotstep-tiers.html, else HOT-Step-UI beside the repository).
// --workers: pages at once for each capture (default: half the machine's threads, 6 on 12; lower it to leave more).
// --page: the rebuild page to compare (default dist-test/index.html, built first if missing or older than the sources).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { STATES, SETS, STATE_FILE } from './slice-states.mjs';
import { ensureTestBuild, testPage } from '../test-build.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), repo = path.resolve(here, '..', '..'), OUT = path.join(here, 'out');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; }, flag = k => args.includes('--' + k);
const only = opt('only') ? new RegExp(opt('only')) : null, workers = opt('workers', String(Math.max(1, Math.floor(os.availableParallelism() / 2))));
const concept = path.resolve(opt('concept') || path.join(process.env.HOTSTEP_CONCEPT || path.join(repo, '..', '..', 'HOT-Step-UI'), 'hotstep-tiers.html'));
const sec = ms => { const s = Math.round(ms / 1000); return s < 60 ? s + ' s' : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`; };

/* each set's states: a state belongs to the set whose regular expression matches its id; every state must be in one */
const sets = SETS.map(s => ({ ...s, re: new RegExp(s.only), ids: STATES.map(x => x.id).filter(id => new RegExp(s.only).test(id)) }));
const names = new Set(); for (const s of sets) { if (names.has(s.name)) throw new Error(`the set ${s.name} is declared twice (slice-states/${s.file})`); names.add(s.name); if (!fs.existsSync(path.join(here, s.accept))) throw new Error(`set ${s.name}: no ${s.accept}`); }
const orphans = STATES.filter(x => !sets.some(s => s.re.test(x.id))).map(x => x.id), twice = STATES.filter(x => sets.filter(s => s.re.test(x.id)).length > 1).map(x => x.id);
const chosen = sets.filter(s => !only || only.test(s.name));
if (flag('list')) {
  for (const s of sets) console.log(`${s.name.padEnd(16)} ${String(s.ids.length).padStart(4)} states  ${s.only}${s.scope ? '  scope ' + s.scope : ''}  ${s.accept}  (slice-states/${s.file})`);
  if (orphans.length) console.log(`${orphans.length} states in no set: ${orphans.join(', ')}`);
  process.exit(orphans.length ? 1 : 0);
}
if (!chosen.length) throw new Error(`no set matches ${opt('only')}: ${sets.map(s => s.name).join(', ')}`);
if (!fs.existsSync(concept)) throw new Error(`no concept page at ${concept} (--concept <file>, or HOTSTEP_CONCEPT=<folder>)`);

/* what decides a concept recording besides the concept page: the set, its states' files, the recorder, the browser */
const require = createRequire(import.meta.url);
const pwVersion = (() => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return JSON.parse(fs.readFileSync(require.resolve(m + '/package.json'), 'utf8')).version; } catch (e) {} } return 'unknown'; })();
const sha = (...parts) => { const h = crypto.createHash('sha256'); for (const p of parts) h.update(p).update('\0'); return h.digest('hex').slice(0, 12); };
const read = f => fs.readFileSync(f);
const conceptHash = sha(read(concept));
const keyOf = s => sha(conceptHash, JSON.stringify([s.name, s.only, s.scope || '']), pwVersion, read(path.join(here, 'vb.mjs')), read(path.join(here, 'page-lib.js')), read(path.join(here, 'slice-states.mjs')),
  ...[...new Set(s.ids.map(id => STATE_FILE[id]))].sort().map(f => read(path.join(here, 'slice-states', f))));

const page = opt('page') ? path.resolve(opt('page')) : testPage;
if (!opt('page')) ensureTestBuild();
const logs = path.join(OUT, 'all-logs'); fs.mkdirSync(logs, { recursive: true });
const vb = (log, ...a) => { const fd = fs.openSync(path.join(logs, log), 'w'); const r = spawnSync(process.execPath, [path.join(here, 'vb.mjs'), ...a], { cwd: repo, stdio: ['ignore', fd, fd] }); fs.closeSync(fd); return { status: r.status, text: fs.readFileSync(path.join(logs, log), 'utf8') }; };
const complete = (dir, ids) => { try { const m = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8')); return JSON.stringify(m.states) === JSON.stringify(ids) && ids.every(id => fs.existsSync(path.join(dir, id + '.json.gz'))); } catch (e) { return false; } };

console.log(`visual sets: ${chosen.length} of ${sets.length} (${chosen.map(s => s.name).join(', ')}); concept ${path.basename(concept)} ${conceptHash}; rebuild ${path.relative(repo, page)}; ${workers} workers`);
if (orphans.length) console.log(`  ${orphans.length} states belong to no set, and are compared by none: ${orphans.slice(0, 8).join(', ')}${orphans.length > 8 ? ', ...' : ''}`);
if (twice.length) console.log(`  ${twice.length} states belong to two sets: ${twice.slice(0, 8).join(', ')}`);
const t0 = Date.now(), rows = [];
for (const s of chosen) {
  const t1 = Date.now(), key = keyOf(s), cname = `all-concept-${s.name}-${key}`, rname = `all-rebuild-${s.name}`, cdir = path.join(OUT, cname);
  const sc = s.scope ? ['--scope', s.scope] : [], common = ['--states', 'slice', '--only', s.only, ...sc, '--workers', workers];
  let reused = flag('reuse-concept') && complete(cdir, s.ids), row;
  if (!reused) {
    /* one kept recording per set: an older key's is removed */
    for (const d of fs.existsSync(OUT) ? fs.readdirSync(OUT) : []) if (d.startsWith(`all-concept-${s.name}-`)) fs.rmSync(path.join(OUT, d), { recursive: true, force: true });
    const c = vb(`${s.name}-concept.log`, 'capture', cname, ...common, '--page', concept);
    if (c.status !== 0 || !complete(cdir, s.ids)) row = [s.name, s.ids.length, 'CONCEPT CAPTURE FAILED', `${s.name}-concept.log`, []];
  }
  if (!row) {
    fs.rmSync(path.join(OUT, rname), { recursive: true, force: true });
    const r = vb(`${s.name}-rebuild.log`, 'capture', rname, ...common, '--page', page, '--plan-from', cname);
    if (r.status !== 0) row = [s.name, s.ids.length, 'REBUILD CAPTURE FAILED', `${s.name}-rebuild.log`, []];
  }
  if (!row) {
    const m = vb(`${s.name}-compare.log`, 'compare', cname, rname, ...sc, '--accept', path.join(here, s.accept), '--max', '12');
    const res = (m.text.match(/^RESULT: (.*)$/m) || [])[1], n = (m.text.match(/^compared (\d+) states/m) || [])[1];
    const verdict = res === 'identical' ? 'identical' : res ? `${parseInt(res, 10)} differences beyond the accepted ones` : 'COMPARE FAILED';
    row = [s.name, n ?? s.ids.length, verdict, `${s.name}-compare.log`, res === 'identical' ? [] : m.text.split('\n').filter(l => /^ {2}\S/.test(l) && !/^ {2}ok {2}/.test(l)).slice(0, 6)];
  }
  row.push(sec(Date.now() - t1), reused);
  rows.push(row);
  const [name, n, verdict, log, detail = [], took] = row;
  console.log(`${verdict === 'identical' ? 'ok  ' : 'FAIL'} ${name.padEnd(16)} ${String(n).padStart(4)} states  ${verdict}  (${reused ? 'concept reused, ' : ''}${took})${verdict === 'identical' ? '' : '  -> ' + path.relative(repo, path.join(logs, log))}`);
  for (const l of detail) console.log('       ' + l.trim().slice(0, 220));
}
const bad = rows.filter(r => r[2] !== 'identical');
console.log(`\n${bad.length ? `RESULT: ${bad.length} of ${rows.length} sets differ (${bad.map(r => r[0]).join(', ')})` : `RESULT: every set identical with its accepted differences (${rows.length} sets, ${rows.reduce((a, r) => a + +r[1], 0)} states)`}${orphans.length ? `; ${orphans.length} states in no set` : ''}, in ${sec(Date.now() - t0)}`);
process.exitCode = bad.length || orphans.length ? 1 : 0;
