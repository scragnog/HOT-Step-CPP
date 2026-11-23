// Visual baseline: records the computed style of every element (and its ::before, ::after, ::placeholder) in every state
// of states.mjs, plus forced :hover/:focus/:focus-visible/:focus-within/:active, plus screenshots; then compares
// two recordings. See README.md in this folder.
//
//   node tools/visual-baseline/vb.mjs capture <name> [--page file.html] [--only regex] [--workers n] [--plan-from name] [--no-shots]
//   node tools/visual-baseline/vb.mjs compare <a> <b> [--max n]
//   node tools/visual-baseline/vb.mjs check <base> [--only regex]      (capture "<base>-now" with base's plan, then compare)
//   node tools/visual-baseline/vb.mjs list
//
// In ui-next (changes from the concept's copy, docs/09-architecture.md "The test layer"): the page is the rebuild's test
// build (dist-test/index.html) unless --page names another (the concept's hotstep-tiers.html); --states slice runs the
// slice's matrix (slice-states.mjs) instead of the concept's; --scope <name> plans forced states and compares only
// inside the regions that scope names (slice-states.mjs SCOPES), the parts of the page the rebuild draws so far.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import http from 'node:http';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { STATES as CONCEPT_STATES } from './states.mjs';
import { STATES as SLICE_STATES, SCOPES } from './slice-states.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const OUT = path.join(here, 'out');
const require = createRequire(import.meta.url);
const loadPW = () => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch (e) {} } throw new Error('Playwright not found: run npm install (devDependency), or set PLAYWRIGHT_MODULE to a playwright folder'); };

const args = process.argv.slice(2), cmd = args[0];
const opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const flag = k => args.includes('--' + k);
const STATES = opt('states') === 'slice' ? SLICE_STATES : CONCEPT_STATES;
/* a key is in scope when it is one of the scope's roots or lies under one */
const SCOPE = opt('scope') ? SCOPES[opt('scope')] : null;
if (opt('scope') && !SCOPE) throw new Error(`no scope ${opt('scope')}: ${Object.keys(SCOPES).join(', ')}`);
const inScope = k => !SCOPE || SCOPE.some(r => k === r || k.startsWith(r + '>'));
const T0 = Date.UTC(2031, 0, 1, 12, 0, 0); // in the future, so pausing at it always moves the clock forward
const gz = o => zlib.gzipSync(Buffer.from(JSON.stringify(o)));
const ungz = f => JSON.parse(zlib.gunzipSync(fs.readFileSync(f)).toString());
const PAGE_LIB = fs.readFileSync(path.join(here, 'page-lib.js'), 'utf8');
/* deterministic Math.random for the page; smooth scrolling (a real-time animation) becomes instant */
const INIT = `(() => { let s = 0x2545F491; Math.random = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const inst = o => (o && typeof o === 'object' && o.behavior === 'smooth') ? { ...o, behavior: 'instant' } : o;
  for (const P of [Element.prototype, window]) for (const m of ['scrollTo', 'scrollBy', 'scroll']) { const f = P[m]; if (f) P[m] = function (a, b) { return typeof a === 'object' ? f.call(this, inst(a)) : f.call(this, a, b); }; }
  const siv = Element.prototype.scrollIntoView; Element.prototype.scrollIntoView = function (o) { return siv.call(this, inst(o)); }; })();`;

function customProps(html) { const set = new Set(); for (const m of html.matchAll(/--[a-zA-Z_][\w-]*/g)) set.add(m[0]); return [...set].filter(x => !/^--(check|only|page|workers|plan|max|no)/.test(x)); }

let classMap = null; // --classmap <json>: [[name, regex source]], for vb.nesting (analysis only)
async function runState(browser, st, pageUrl, custom, planIn, shotsDir, propsIn) {
  const ctx = await browser.newContext({ viewport: { width: st.w, height: st.h }, colorScheme: st.scheme, reducedMotion: st.reduced ? 'reduce' : 'no-preference', deviceScaleFactor: 1 });
  await ctx.route(u => !u.href.startsWith(pageUrl), r => r.abort());
  const page = await ctx.newPage();
  page.setDefaultTimeout(120000); /* fill, hover and the rest: a deadline for a broken state, never a pace */
  const errs = [], fails = [], tm = {}; let tl = Date.now(); const lap = k => { tm[k] = Date.now() - tl; tl = Date.now(); };
  page.on('pageerror', e => errs.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error' && !/net::ERR|Failed to load resource/.test(m.text())) errs.push('console: ' + m.text()); });
  await page.addInitScript(INIT);
  /* pauseAt installs the fake clock already paused: with install() first, real time would pass before the pause and
     shift performance.now() against the frame grid */
  await page.clock.pauseAt(T0);
  /* fake time for timers and requestAnimationFrame; a little real time for what the browser runs on its own frames
     (IntersectionObserver, ResizeObserver); then animations are finished */
  /* animations are finished before fake time runs, so code that measures the page in a frame callback (pkMore,
     placeDD, packDesk) measures the settled layout, not a frame of an animation */
  const fin = () => page.evaluate(() => window.__vb ? window.__vb.finish() : 0);
  /* real rendering updates, waited for rather than slept for: what the browser delivers on its own frames
     (ResizeObserver, IntersectionObserver, hover after layout) has run, however busy the machine is, before the next
     step of fake time and before a reading. Under load a fixed real-time wait could end before the frame came, and the
     page's reaction (a re-pack of the desk from its ResizeObserver) then landed before or after a reading by chance. */
  const frames = async () => { let t; try { await Promise.race([page.evaluate(() => window.__vb.frames(2)), new Promise((_, rej) => { t = setTimeout(() => rej(new Error('no rendering update within 60 s')), 60000); })]); } finally { clearTimeout(t); } };
  /* fake time always moves by the same amounts, so timers see the same clock in every run */
  const settle = async (ms = 600) => { const step = Math.max(50, Math.round(ms / 3)); for (let i = 0; i < 3; i++) { await fin(); await page.clock.runFor(step); await fin(); await frames(); } };
  const quiet = async () => { for (let i = 0; i < 2; i++) { await fin(); await frames(); await page.clock.runFor(100); } await fin(); await frames(); await fin(); };
  if (st.media) { const cdp0 = await ctx.newCDPSession(page); await cdp0.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: st.scheme }, { name: 'prefers-reduced-motion', value: st.reduced ? 'reduce' : 'no-preference' }, ...st.media] }); }
  await page.goto(pageUrl);
  await page.evaluate(PAGE_LIB);
  /* frame callbacks get the fake clock's time, like every other timer */
  await page.evaluate(() => { const raf = window.requestAnimationFrame; window.requestAnimationFrame = cb => raf(() => cb(performance.now())); });
  await settle(1500); lap('load');
  const h = {
    page,
    settle,
    js: (fn, arg) => page.evaluate(fn, arg),
    S: async (o, ms) => { await page.evaluate(o => { if (o.vals) { Object.assign(S.vals, o.vals); delete o.vals; } Object.assign(S, o); render(); }, o); await settle(ms); },
    /* the element is scrolled into view by the page itself (instantly, to the centre) before Playwright clicks it, so
       Playwright never scrolls, and where the page sits does not depend on how its scrolling went */
    /* the deadline is far beyond any normal click: on a busy machine a click took more than 5 s (the old deadline) and
       the state was then recorded without it. A click on an element that never appears costs the full 2 minutes. */
    click: async (sel, ms) => { try { await page.evaluate(sel => document.querySelector(sel)?.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' }), sel); await page.click(sel, { timeout: 120000, force: true }); } catch (e) { fails.push('click ' + sel + ': ' + String(e.message).split('\n')[0]); } await settle(ms); },
    press: async (k, ms) => { await page.keyboard.press(k); await settle(ms || 300); },
    expect: async (fn, arg) => { let ok = false; try { ok = await page.evaluate(fn, arg); } catch (e) {} if (!ok) fails.push('expect ' + String(fn).slice(0, 90)); },
  };
  try { await st.run(h); } catch (e) { fails.push('run: ' + String(e.message).split('\n')[0]); }
  if (!st.raw) await settle(600); lap('run');
  if (st.raw) await page.evaluate(() => window.__vb.finish()); else await quiet();
  if (propsIn) await page.evaluate(l => window.__vb.useProps(l), propsIn); else await page.evaluate(c => window.__vb.setProps(c), custom);
  const full = await page.evaluate(() => window.__vb.recordAll());
  const props = await page.evaluate(() => window.__vb.props);
  const nesting = classMap ? await page.evaluate(m => window.__vb.nesting(m), classMap) : null;
  const matches = await page.evaluate(() => window.__vb.matches()), classSets = await page.evaluate(() => window.__vb.classSets());
  lap('record');
  /* the screenshot shows the state that was recorded: the forced pass below runs fake time and moves the page on */
  if (st.shot && shotsDir) await page.screenshot({ path: path.join(shotsDir, st.id + '.png') });
  lap('shot');
  const rec ={ id: st.id, props, mode: full.mode, els: full.els, errs, fails, forced: {}, plan: null, tm, matches, classSets, nesting };
  if (st.pseudo) {
    const plan = planIn || { fprops: await page.evaluate(() => window.__vb.forcedPropList()), batches: (await page.evaluate(() => window.__vb.plan())).batches.map(b => b.map(({ els, force, region, sib }) => ({ els, force, region, sib })).filter(t => t.els.every(inScope))).filter(b => b.length) };
    rec.plan = plan;
    await page.evaluate(f => { window.__vb.fprops = f; }, plan.fprops);
    const fidx = plan.fprops.map(p => props.indexOf(p));
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('DOM.enable'); await cdp.send('CSS.enable'); await cdp.send('DOM.getDocument', { depth: -1 });
    /* absolute values of the unforced record, to store forced records as deltas */
    const base = new Map(); for (const [k, ps, d] of full.els) base.set(k + '|' + ps, d);
    for (const batch of plan.batches) {
      const bkey = JSON.stringify(batch.map(t => [t.els, t.force]));
      const allEls = batch.flatMap(t => t.els), allForce = batch.flatMap(t => t.force);
      const ok = await page.evaluate(keys => window.__vb.stash(keys), allEls);
      if (!ok.every(Boolean)) { rec.forced[bkey] = 'missing'; continue; }
      const ids = [];
      for (let i = 0; i < allEls.length; i++) {
        const { result } = await cdp.send('Runtime.evaluate', { expression: `window.__vb.stashed[${i}]` });
        const { nodeId } = await cdp.send('DOM.requestNode', { objectId: result.objectId });
        ids.push(nodeId); await cdp.send('CSS.forcePseudoState', { nodeId, forcedPseudoClasses: allForce[i] });
      }
      await quiet();
      const recs = await page.evaluate(list => window.__vb.recordRegion(list), batch.map(t => [t.region, t.els, t.sib]));
      for (const id of ids) await cdp.send('CSS.forcePseudoState', { nodeId: id, forcedPseudoClasses: [] });
      await quiet();
      const delta = [];
      if (recs) for (const [k, ps, v] of recs) { const b = base.get(k + '|' + ps) || {}, d = {}; let n = 0; for (let i = 0; i < v.length; i++) { const j = fidx[i], bv = j in b ? b[j] : full.mode[j]; if (v[i] !== bv) { d[j] = v[i]; n++; } } if (n) delta.push([k, ps, d]); }
      rec.forced[bkey] = recs ? delta : 'missing';
    }
    await page.evaluate(() => window.__vb.finish()); lap('forced');
  }
  await ctx.close();
  return rec;
}

/* The page is served over local HTTP as UTF-8: it has no <meta charset>, and from file:// Chromium's guess at the
   encoding varies between runs. Anything not on this server (the web fonts) is blocked. */
function serve(file) {
  const body = fs.readFileSync(file);
  const srv = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); r.end(body); });
  return new Promise(res => srv.listen(0, '127.0.0.1', () => res({ url: `http://127.0.0.1:${srv.address().port}/hotstep-tiers.html`, close: () => srv.close() })));
}
async function capture(name, o = {}) {
  const pw = loadPW();
  if (o.classmap) classMap = JSON.parse(fs.readFileSync(o.classmap, 'utf8'));
  const file = path.resolve(o.page || path.join(repo, 'dist-test', 'index.html'));
  const html = fs.readFileSync(file, 'utf8'), custom = customProps(html);
  const dir = path.join(OUT, name), shots = path.join(dir, 'shots');
  fs.mkdirSync(shots, { recursive: true });
  const only = o.only ? new RegExp(o.only) : null;
  const list = STATES.filter(s => !only || only.test(s.id));
  const planDir = o.planFrom ? path.join(OUT, o.planFrom) : null;
  const server = await serve(file);
  const browser = await pw.chromium.launch();
  const workers = +(o.workers || 6); let next = 0, done = 0; const t0 = Date.now(), problems = [];
  await Promise.all(Array.from({ length: workers }, async () => {
    while (next < list.length) {
      const st = list[next++];
      let plan = null, propsIn = null; if (planDir) { const f = path.join(planDir, st.id + '.json.gz'); if (fs.existsSync(f)) { const b = ungz(f); plan = st.pseudo ? b.plan : null; propsIn = b.props || null; } }
      let rec; try { rec = await runState(browser, st, server.url, custom, plan, o.noShots ? null : shots, propsIn); }
      catch (e) { rec = { id: st.id, crash: String(e.stack || e) }; }
      fs.writeFileSync(path.join(dir, st.id + '.json.gz'), gz(rec));
      done++; if (rec.crash || rec.errs?.length || rec.fails?.length) problems.push(`${st.id}: ${rec.crash ? 'CRASH ' + rec.crash.split('\n')[0] : ''}${(rec.errs || []).join('; ')} ${(rec.fails || []).join('; ')}`);
      if (!o.quiet) process.stdout.write(`\r${done}/${list.length} ${st.id.padEnd(40)}`);
    }
  }));
  await browser.close(); server.close();
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ page: file, at: new Date().toISOString(), states: list.map(s => s.id) }, null, 1));
  console.log(`\ncaptured ${list.length} states into ${path.relative(repo, dir)} in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  if (problems.length) console.log('setup notes (same in both captures is fine):\n  ' + problems.join('\n  '));
}

/* absolute values per key|pseudo. Elements inside <head> are left out: they are never rendered, and a new <meta> renumbers them */
const absolute = rec => { const m = new Map(); for (const [k, ps, d] of rec.els) { if (k.startsWith('html>head:') || !inScope(k)) continue; const v = rec.mode.slice(); for (const i in d) v[i] = d[i]; m.set(k + '|' + ps, v); } return m; };

async function compare(a, b, o = {}) {
  const A = path.join(OUT, a), B = path.join(OUT, b), max = +(o.max || 40);
  const ids = fs.readdirSync(A).filter(f => f.endsWith('.json.gz')).map(f => f.slice(0, -8)).filter(id => !o.only || new RegExp(o.only).test(id));
  let states = 0, elRecs = 0, forcedRecs = 0, diffs = 0; const lines = [], byProp = new Map();
  const note = (st, s) => { diffs++; if (lines.length < max) lines.push(st + ': ' + s); };
  for (const id of ids) {
    const fb = path.join(B, id + '.json.gz'); if (!fs.existsSync(fb)) { note(id, 'missing in ' + b); continue; }
    const ra = ungz(path.join(A, id + '.json.gz')), rb = ungz(fb); states++;
    if (ra.crash || rb.crash) { note(id, 'crash ' + (ra.crash || rb.crash).split('\n')[0]); continue; }
    if (JSON.stringify(ra.fails) !== JSON.stringify(rb.fails)) note(id, `setup differs: ${JSON.stringify(ra.fails)} vs ${JSON.stringify(rb.fails)}`);
    if (JSON.stringify(ra.errs) !== JSON.stringify(rb.errs)) note(id, `page errors differ: ${JSON.stringify(ra.errs)} vs ${JSON.stringify(rb.errs)}`);
    const pa = ra.props, pb = rb.props; const ib = new Map(pb.map((p, i) => [p, i]));
    const va = absolute(ra), vb = absolute(rb);
    const cmp = (label, x, y) => { for (let i = 0; i < pa.length; i++) { const j = ib.get(pa[i]); const yv = j === undefined ? '(no such property)' : y[j]; if (x[i] !== yv) { note(id, `${label} ${pa[i]}: ${JSON.stringify(x[i])} -> ${JSON.stringify(yv)}`); byProp.set(pa[i], (byProp.get(pa[i]) || 0) + 1); } } };
    for (const [k, x] of va) { elRecs++; const y = vb.get(k); if (!y) { note(id, `${k} only in ${a}`); continue; } cmp(k, x, y); }
    for (const k of vb.keys()) if (!va.has(k)) note(id, `${k} only in ${b}`);
    for (const fk of Object.keys(ra.forced || {})) {
      const fa = ra.forced[fk], fbv = rb.forced?.[fk];
      if (fbv === undefined) { note(id, `forced ${fk} not recorded in ${b}`); continue; }
      if (fa === 'missing' || fbv === 'missing') { if (fa !== fbv) note(id, `forced ${fk}: ${fa === 'missing' ? 'missing' : 'present'} vs ${fbv === 'missing' ? 'missing' : 'present'}`); continue; }
      const inBody = ([k]) => !k.startsWith('html>head:') && inScope(k), ma = new Map(fa.filter(inBody).map(([k, ps, d]) => [k + '|' + ps, d])), mb = new Map(fbv.filter(inBody).map(([k, ps, d]) => [k + '|' + ps, d]));
      for (const key of new Set([...ma.keys(), ...mb.keys()])) {
        forcedRecs++;
        const base = va.get(key) || ra.mode, x = base.slice(), da = ma.get(key) || {}; for (const i in da) x[i] = da[i];
        const baseB = vb.get(key) || rb.mode, y = baseB.slice(), db = mb.get(key) || {}; for (const i in db) y[i] = db[i];
        cmp(`[forced ${JSON.parse(fk).map(([e, f]) => e.map((k, i) => k + ':' + f[i].join(':')).join(' & ')).join(' | ').slice(0, 160)}] ${key}`, x, y);
      }
    }
  }
  /* screenshots: identical bytes, else count differing pixels in a browser canvas */
  let shots = 0; const sa = path.join(A, 'shots'), sb = path.join(B, 'shots'), pixel = [], noise = [];
  if (fs.existsSync(sa)) for (const f of fs.readdirSync(sa)) { if (o.only && !new RegExp(o.only).test(f.replace(/.png$/, ''))) continue; const g = path.join(sb, f); shots++;
    if (!fs.existsSync(g)) { note('shot ' + f, 'missing in ' + b); continue; }
    const x = fs.readFileSync(path.join(sa, f)), y = fs.readFileSync(g); if (!x.equals(y)) pixel.push([f, x, y]); }
  if (pixel.length) { const pw = loadPW(), br = await pw.chromium.launch(), pg = await br.newPage();
    for (const [f, x, y] of pixel) { const n = await pg.evaluate(async ([x, y]) => { const load = s => new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = 'data:image/png;base64,' + s; });
        const [i1, i2] = await Promise.all([load(x), load(y)]); if (i1.width !== i2.width || i1.height !== i2.height) return `size ${i1.width}x${i1.height} vs ${i2.width}x${i2.height}`;
        const c = new OffscreenCanvas(i1.width, i1.height), g = c.getContext('2d'); g.drawImage(i1, 0, 0); const d1 = g.getImageData(0, 0, c.width, c.height).data; g.clearRect(0, 0, c.width, c.height); g.drawImage(i2, 0, 0); const d2 = g.getImageData(0, 0, c.width, c.height).data;
        /* rasterising is not bit-exact between runs (a level or two on blurred shadows and gradients): a pixel counts
           when a channel moves by more than 8 of 255 */
        let n = 0; for (let i = 0; i < d1.length; i += 4) if (Math.max(Math.abs(d1[i] - d2[i]), Math.abs(d1[i + 1] - d2[i + 1]), Math.abs(d1[i + 2] - d2[i + 2]), Math.abs(d1[i + 3] - d2[i + 3])) > 8) n++; return n; }, [x.toString('base64'), y.toString('base64')]);
      /* and a few dozen such pixels (antialiasing on an edge) are noise: listed, not counted */
      if (typeof n === 'string' || n > 64) note('shot ' + f, `${n} pixels differ`); else if (n) noise.push(`${f}: ${n} pixels`); }
    await br.close(); }
  console.log(`compared ${states} states, ${elRecs} element records, ${forcedRecs} forced-state records, ${shots} screenshots`);
  if (noise.length) console.log('screenshot noise (64 pixels or fewer, ignored): ' + noise.join(', '));
  if (!diffs) { console.log('RESULT: identical'); return 0; }
  console.log(`RESULT: ${diffs} differences` + (diffs > lines.length ? ` (first ${lines.length} shown)` : ''));
  for (const l of lines) console.log('  ' + l);
  if (byProp.size) console.log('by property: ' + [...byProp].sort((p, q) => q[1] - p[1]).slice(0, 20).map(([p, n]) => `${p} ${n}`).join(', '));
  return 1;
}

if (cmd === 'list') { for (const s of STATES) console.log(`${s.id}  ${s.w}x${s.h} ${s.scheme}${s.reduced ? ' reduced' : ''}${s.pseudo ? ' +pseudo' : ''}${s.shot ? ' +shot' : ''}`); console.log(STATES.length + ' states'); }
else if (cmd === 'capture' && args[1]) await capture(args[1], { page: opt('page'), only: opt('only'), workers: opt('workers'), planFrom: opt('plan-from'), noShots: flag('no-shots'), classmap: opt('classmap') });
else if (cmd === 'compare' && args[2]) process.exitCode = await compare(args[1], args[2], { max: opt('max'), only: opt('only') });
else if (cmd === 'check' && args[1]) { const now = args[1] + '-now'; fs.rmSync(path.join(OUT, now), { recursive: true, force: true }); await capture(now, { page: opt('page'), only: opt('only'), workers: opt('workers'), planFrom: args[1], noShots: flag('no-shots') }); process.exitCode = await compare(args[1], now, { max: opt('max'), only: opt('only') }); }
else { console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(0, 9).join('\n')); process.exitCode = 2; }
