// Motion against the concept, frame by frame (package 9): each motion is set up and set off on the concept page and on
// the rebuild, every animation is then held and moved to 0, 16, 32 ... ms (the way the concept's reviewers captured the
// tier change), and at each step the boxes that move are read: elements by their on-screen box (a running transform
// included), the tier change's parts by their View Transition group's box and their old and new images' opacity. The
// two pages must agree within 2px and 0.05 of opacity at every step, but for the differences accepted in
// tools/motion/accepted.json (each with its reason and the most it accepts).
//
//   node tools/motion/frames.mjs [--only <regex>] [--concept <file>] [--page <file>] [--shots] [--list]
//
// --shots also writes a screenshot of each step of each page to tools/motion/out/<motion>/ (concept-064.png ...).
// Needs Playwright (PLAYWRIGHT_MODULE, as the other tools) and the test build (npm run build:test).
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { serve } from '../behaviour/harness.mjs';
import { MOTIONS } from './motions.mjs';

const require = createRequire(import.meta.url);
const loadPW = () => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch (e) {} } throw new Error('Playwright not found: set PLAYWRIGHT_MODULE'); };
const here = path.dirname(fileURLToPath(import.meta.url)), ui = path.resolve(here, '..', '..');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; }, flag = k => args.includes('--' + k);
const concept = path.resolve(opt('concept') || path.join(process.env.HOTSTEP_CONCEPT || path.join(ui, '..', '..', 'HOT-Step-UI'), 'hotstep-tiers.html'));
const rebuild = path.resolve(ui, opt('page', 'dist-test/index.html'));
const only = opt('only') ? new RegExp(opt('only')) : null;
const accepted = JSON.parse(fs.readFileSync(path.join(here, 'accepted.json'), 'utf8')).accepted.map(a => ({ ...a, re: new RegExp(a.match), seen: 0 }));
const TOL = 2, OTOL = 0.05;

if (flag('list')) { for (const m of MOTIONS) console.log(m.id.padEnd(28), m.what); process.exit(0); }

/** Runs one motion on one page: the steps it read, by time, each a map of key to [x, y, w, h] or [opacity]. */
async function capture(browser, url, m, shotsDir) {
  const ctx = await browser.newContext({ viewport: { width: m.w || 1600, height: m.h || 900 }, colorScheme: m.scheme || 'light', reducedMotion: m.reduced ? 'reduce' : 'no-preference', deviceScaleFactor: 1 });
  const page = await ctx.newPage(), errs = [];
  /* smooth scrolling runs on the browser's own clock, which no step holds: every scroll is instant on both pages */
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
    const steps = new Map();
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
      steps.set(t, at);
      if (shotsDir) await page.screenshot({ path: path.join(shotsDir, `${String(t).padStart(3, '0')}.png`) });
    }
    return { steps, errs };
  } finally { await ctx.close(); }
}

const pw = loadPW(), browser = await pw.chromium.launch(), sc = await serve(concept), sr = await serve(rebuild);
let failed = 0;
try {
  for (const m of MOTIONS) {
    if (only && !only.test(m.id)) continue;
    const dir = flag('shots') ? path.join(here, 'out', m.id) : null;
    if (dir) { fs.mkdirSync(path.join(dir, 'concept'), { recursive: true }); fs.mkdirSync(path.join(dir, 'rebuild'), { recursive: true }); }
    const a = await capture(browser, sc.url, m, dir && path.join(dir, 'concept')), b = await capture(browser, sr.url, m, dir && path.join(dir, 'rebuild'));
    const worst = new Map(), lines = [];
    let frames = 0, moving = 0;
    for (const [t, ca] of a.steps) {
      const cb = b.steps.get(t) ?? {}; frames++;
      for (const k of new Set([...Object.keys(ca), ...Object.keys(cb)])) {
        const x = ca[k], y = cb[k], op = x?.length === 1 || y?.length === 1;
        const d = !x || !y ? Infinity : Math.max(...x.map((v, i) => Math.abs(v - y[i])));
        if (d <= (op ? OTOL : TOL)) continue;
        const w = worst.get(k); if (!w || d > w.d) worst.set(k, { d, t, x, y });
      }
      const t0 = a.steps.get(0); if (t > 0 && JSON.stringify(ca) !== JSON.stringify(t0)) moving++;
    }
    const left = [];
    for (const [k, w] of worst) {
      const acc = accepted.find(x => x.re.test(`${m.id} ${k}`) && (w.d === Infinity ? x.most === 'missing' : typeof x.most === 'number' && w.d <= x.most));
      if (acc) { acc.seen++; continue; }
      left.push(`    ${k} at ${w.t}ms: concept ${JSON.stringify(w.x)}, rebuild ${JSON.stringify(w.y)}${w.d === Infinity ? '' : ` (${Math.round(w.d * 10) / 10})`}`);
    }
    const errs = [...a.errs.map(e => 'concept: ' + e), ...b.errs.map(e => 'rebuild: ' + e)];
    const ok = !left.length && !errs.length && (moving > 0 || !!m.conceptStill);
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${m.id}: ${frames} steps of 16ms, ${moving} of them in motion on the concept, ${worst.size - left.length} accepted keys${left.length ? `, ${left.length} differ` : ''}`);
    if (!moving && !m.conceptStill) console.log('    nothing moved on the concept page: the motion was not set off');
    for (const l of left.slice(0, 12)) console.log(l); if (left.length > 12) console.log(`    ... ${left.length - 12} more`);
    for (const e of errs) console.log('    ' + e);
  }
} finally { await browser.close(); sc.close(); sr.close(); }
for (const x of accepted) if (x.seen > (x.n ?? Infinity)) { console.log(`accepted "${x.why}" seen ${x.seen} times, ${x.n} accepted`); failed++; }
console.log(failed ? `${failed} motion(s) differ from the concept` : 'every motion moves as the concept moves');
process.exit(failed ? 1 : 0);
