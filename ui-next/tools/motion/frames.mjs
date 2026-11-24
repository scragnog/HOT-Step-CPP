// Motion, frame by frame, against its approved samples: each motion of motions.mjs is set up and set off on the test
// build, every animation is then held and moved to 0, 16, 32 ... ms, and at each step the boxes that move are read:
// elements by their on-screen box (a running transform included), the tier change's parts by their View Transition
// group's box and their old and new images' opacity. Each step must agree with the motion's approved samples
// (tools/motion/approved/<motion>.json, committed) within TOL px and OTOL of opacity, and each motion must move (but
// for one marked `still`). --approve records the current build's samples as the approved ones, after a deliberate
// change to a motion.
//
//   node tools/motion/frames.mjs [--only <regex>] [--page <file>] [--approve] [--shots] [--workers n] [--list]
//
// --shots also writes a screenshot of each step of each motion to tools/motion/out/<motion>/ (000.png, 016.png ...).
// --workers: motions captured at once (default: half the machine's threads, 6 on 12).
// Needs Playwright (PLAYWRIGHT_MODULE, as the other tools) and the test build (built first when it is missing or older
// than the sources).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { serve } from '../behaviour/harness.mjs';
import { ensureTestBuild, testPage } from '../test-build.mjs';
import { MOTIONS } from './motions.mjs';

const require = createRequire(import.meta.url);
const loadPW = () => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch (e) {} } throw new Error('Playwright not found: set PLAYWRIGHT_MODULE'); };
const here = path.dirname(fileURLToPath(import.meta.url)), ui = path.resolve(here, '..', '..'), APPROVED = path.join(here, 'approved');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; }, flag = k => args.includes('--' + k);
const only = opt('only') ? new RegExp(opt('only')) : null, approve = flag('approve'), workers = +opt('workers', Math.max(1, Math.floor(os.availableParallelism() / 2)));
/* the tolerance: a box may move by TOL px, an opacity by OTOL, from its approved sample at the same step (the boxes
   are read to a tenth of a pixel; two runs of one build agree to that) */
const TOL = 0.5, OTOL = 0.01;

if (flag('list')) { for (const m of MOTIONS) console.log(m.id.padEnd(28), fs.existsSync(path.join(APPROVED, m.id + '.json')) ? 'approved' : 'NOT APPROVED', ' ', m.what); process.exit(0); }
const page = opt('page') ? path.resolve(ui, opt('page')) : testPage;
if (!opt('page')) ensureTestBuild();

/** Runs one motion: the steps it read, by time, each a map of key to [x, y, w, h] or [opacity]. */
async function capture(browser, url, m, shotsDir) {
  const ctx = await browser.newContext({ viewport: { width: m.w || 1600, height: m.h || 900 }, colorScheme: m.scheme || 'light', reducedMotion: m.reduced ? 'reduce' : 'no-preference', deviceScaleFactor: 1 });
  const page = await ctx.newPage(), errs = [];
  /* smooth scrolling runs on the browser's own clock, which no step holds: every scroll is instant */
  await page.addInitScript(() => { const inst = o => (o && typeof o === 'object' && o.behavior === 'smooth') ? { ...o, behavior: 'instant' } : o;
    for (const P of [Element.prototype, window]) for (const k of ['scrollTo', 'scrollBy', 'scroll']) { const f = P[k]; if (f) P[k] = function (a, b) { return typeof a === 'object' ? f.call(this, inst(a)) : f.call(this, a, b); }; }
    const siv = Element.prototype.scrollIntoView; Element.prototype.scrollIntoView = function (o) { return siv.call(this, inst(o)); }; });
  page.on('pageerror', e => errs.push(e.message));
  try {
    await page.goto(url); await page.waitForTimeout(900);
    await page.evaluate(m.scene); await page.waitForTimeout(500);
    if (m.before) { await m.before(page); await page.waitForTimeout(300); }
    /* the timeline holds still: nothing moves until it is moved by hand */
    const cdp = await ctx.newCDPSession(page); await cdp.send('Animation.enable'); await cdp.send('Animation.setPlaybackRate', { playbackRate: 0 });
    await page.evaluate(() => { const o = document.startViewTransition?.bind(document); window.__vt = null; if (o) document.startViewTransition = cb => (window.__vt = o(cb)); });
    await m.act(page);
    const end = await page.evaluate(async () => {
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      if (window.__vt) await window.__vt.ready.catch(() => {});
      const as = document.getAnimations().filter(a => { const t = a.effect?.getComputedTiming(); return t && Number.isFinite(t.endTime); });
      window.__as = as; as.forEach(a => a.pause());
      return Math.max(0, ...as.map(a => a.effect.getComputedTiming().endTime));
    });
    const steps = {};
    for (let t = 0; t <= Math.min(Math.max(end, m.until || 0), 700) + 16; t += 16) {
      const at = await page.evaluate(([t, keys, opac]) => {
        window.__as.forEach(a => { try { a.currentTime = t; } catch (e) {} });
        const out = {}, root = document.documentElement, r4 = r => [r.left, r.top, r.width, r.height].map(v => Math.round(v * 10) / 10);
        for (const k of keys) {
          if (k.startsWith('vt:')) { /* a View Transition part: its group's box, its images' opacity */
            const n = k.slice(3), g = getComputedStyle(root, `::view-transition-group(${n})`);
            if (g.width === 'auto' || !g.width) continue;
            const tr = new DOMMatrix(g.transform === 'none' ? undefined : g.transform);
            out[k] = [tr.m41, tr.m42, parseFloat(g.width), parseFloat(g.height)].map(v => Math.round(v * 10) / 10);
            for (const side of ['old', 'new']) { const s = getComputedStyle(root, `::view-transition-${side}(${n})`); out[`${k}:${side}`] = [s.display === 'none' ? 0 : Math.round(parseFloat(s.opacity) * 100) / 100]; }
            continue;
          }
          const els = [...document.querySelectorAll(k.split('|')[0])], key = k.split('|')[1];
          for (const el of els) { if (!el.getClientRects().length) continue; const id = key ? el.getAttribute(key) : ''; out[k.split('|')[0] + (id ? `[${id}]` : '')] = r4(el.getBoundingClientRect()); if (opac) out[k.split('|')[0] + (id ? `[${id}]` : '') + ':op'] = [Math.round(parseFloat(getComputedStyle(el).opacity) * 100) / 100]; }
        }
        return out;
      }, [t, m.keys, !!m.opacity]);
      steps[t] = at;
      if (shotsDir) await page.screenshot({ path: path.join(shotsDir, `${String(t).padStart(3, '0')}.png`) });
    }
    return { steps, errs };
  } finally { await ctx.close(); }
}

const commit = (() => { const r = spawnSync('git', ['rev-parse', '--short=10', 'HEAD'], { cwd: ui, encoding: 'utf8' }); return r.status === 0 ? r.stdout.trim() : ''; })();
const pw = loadPW(), browser = await pw.chromium.launch(), server = await serve(page);
let failed = 0;
const list = MOTIONS.filter(m => !only || only.test(m.id)), ran = list.length, got = new Map();
try {
  /* the motions are captured several at a time (each in a page of its own, its timeline held by hand), then judged in order */
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(workers, list.length) }, async () => {
    while (next < list.length) {
      const m = list[next++], dir = flag('shots') ? path.join(here, 'out', m.id) : null;
      if (dir) fs.mkdirSync(dir, { recursive: true });
      got.set(m.id, await capture(browser, server.url, m, dir));
    }
  }));
  for (const m of list) {
    const b = got.get(m.id);
    const times = Object.keys(b.steps), first = JSON.stringify(b.steps[0]);
    const moving = times.filter(t => +t > 0 && JSON.stringify(b.steps[t]) !== first).length;
    const errs = b.errs.map(e => 'page error: ' + e);
    if (!moving && !m.still) errs.push('nothing moved: the motion was not set off');
    const file = path.join(APPROVED, m.id + '.json');
    if (approve) {
      if (errs.length) { failed++; console.log(`FAIL ${m.id}: not approved`); errs.forEach(e => console.log('    ' + e)); continue; }
      fs.mkdirSync(APPROVED, { recursive: true });
      /* one line per step, so a re-approval's diff shows which steps moved */
      const head = JSON.stringify({ motion: m.id, what: m.what, commit }).slice(0, -1);
      fs.writeFileSync(file, `${head},"steps":{\n${times.map(t => JSON.stringify(t) + ':' + JSON.stringify(b.steps[t])).join(',\n')}\n}}\n`);
      console.log(`ok   ${m.id}: ${times.length} steps of 16ms approved, ${moving} of them in motion`);
      continue;
    }
    if (!fs.existsSync(file)) { failed++; console.log(`FAIL ${m.id}: no approved samples (node tools/motion/frames.mjs --approve --only "^${m.id}$")`); continue; }
    const a = JSON.parse(fs.readFileSync(file, 'utf8')).steps;
    const worst = new Map();
    for (const t of new Set([...Object.keys(a), ...times])) {
      const ca = a[t] ?? {}, cb = b.steps[t] ?? {};
      for (const k of new Set([...Object.keys(ca), ...Object.keys(cb)])) {
        const x = ca[k], y = cb[k], op = x?.length === 1 || y?.length === 1;
        const d = !x || !y ? Infinity : Math.max(...x.map((v, i) => Math.abs(v - y[i])));
        if (d <= (op ? OTOL : TOL)) continue;
        const w = worst.get(k); if (!w || d > w.d) worst.set(k, { d, t, x, y });
      }
    }
    const left = [...worst].map(([k, w]) => `    ${k} at ${w.t}ms: approved ${JSON.stringify(w.x) ?? '(none)'}, now ${JSON.stringify(w.y) ?? '(none)'}${w.d === Infinity ? '' : ` (${Math.round(w.d * 100) / 100})`}`);
    const ok = !left.length && !errs.length;
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${m.id}: ${times.length} steps of 16ms, ${moving} of them in motion${left.length ? `, ${left.length} keys differ from the approved samples` : ''}`);
    for (const l of left.slice(0, 12)) console.log(l); if (left.length > 12) console.log(`    ... ${left.length - 12} more`);
    for (const e of errs) console.log('    ' + e);
  }
} finally { await browser.close(); server.close(); }
if (!ran) { console.log(`no motion matches ${opt('only')}`); process.exit(1); }
console.log(approve ? (failed ? `${failed} motion(s) not approved` : `${ran} motions approved into ${path.relative(ui, APPROVED)}: commit them`)
  : failed ? `${failed} motion(s) differ from their approved samples; after a deliberate change, look (--shots), then --approve --only <motion>` : `every motion moves as approved (${ran} motions)`);
process.exit(failed ? 1 : 0);
