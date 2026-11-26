// A tier change on a full desk as the screen shows it: headed Chromium on the GPU, traced, read from the frames the
// display presented (viz's DirectRenderer::DrawFrame), not from the page's rAF loop (tier.cjs, which never sees the GPU
// or what reaches the screen). Every module, pedal and side unit of the engine on the desk, each tier pair, clicked as a
// user clicks the tier switch. For each, from the click's pointerdown:
//   answered  the first presented frame drawn after the main thread committed the tier switch's new state
//   moving    the first presented frame drawn after the main thread committed the motion's start (the View
//             Transition's ready, or the first animation the change plays: the old faces starting to go)
//   gap       the longest wait between presented frames from the click until the motion has ended
//   raster    the GPU's raster time over that span (RasterDecoderImpl::DoEndRasterCHROMIUM), and its longest task
//   main      the renderer's main thread busy over that span
// Medians of `--runs` (3); the browser is warmed first (one change per engine, unmeasured).
//
//   node tools/perf/tier-frames.cjs [--engines ace,mm3,yue] [--pairs quick-studio,...] [--runs 3] [--again] [--json out]
//   --again: the new tier visited before (there and back first);  --reduced: reduced motion;  PERF_BUILD as lib.cjs
const fs = require('fs'), path = require('path');
const { chromium, url } = require('./lib.cjs');

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const RUNS = +arg('runs', 3), AGAIN = process.argv.includes('--again'), REDUCED = process.argv.includes('--reduced');
const ENGINES = arg('engines', 'ace,mm3,yue').split(',');
const PAIRS = arg('pairs', 'quick-studio,studio-quick,studio-lab,lab-studio,quick-lab,lab-quick').split(',').map(p => p.split('-'));
const PAGE = url('dist-test');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const med = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
const r0 = x => (x == null ? null : Math.round(x));

/* in the page: marks the trace can place (user timing). checked: the switch's new state is in the DOM; moving: the
   motion is started (the View Transition's ready, or the first animation started after the click); done: it ended */
const PROBE = `(() => {
  const T = window.__tf = { on: false, moving: 0, anims: [] };
  const mark = n => { if (T.on && !T[n]) { T[n] = performance.now(); performance.mark('tier:' + n); } };
  new MutationObserver(() => mark('checked')).observe(document, { subtree: true, attributes: true, attributeFilter: ['aria-checked'] });
  const o = Document.prototype.startViewTransition;
  if (o) Document.prototype.startViewTransition = function (cb) {
    const t = o.call(this, cb); if (T.on) { T.vt = true; t.ready.then(() => mark('moving'), () => {}); t.finished.then(() => mark('done'), () => mark('done')); } return t;
  };
  const a = Element.prototype.animate;
  Element.prototype.animate = function (...x) { const an = a.apply(this, x); if (T.on && !T.vt) { mark('moving'); T.anims.push(an); } return an; };
  /* the hand-drawn motion ends when the page takes html.tier-morph off; under reduced motion, when its animations end */
  addEventListener('DOMContentLoaded', () => new MutationObserver(() => { if (T.moving && !T.vt && !document.documentElement.classList.contains('tier-morph') && T.morph) mark('done'); if (document.documentElement.classList.contains('tier-morph')) T.morph = true; }).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }));
  T.end = () => T.vt || T.morph ? null : Promise.race([Promise.all(T.anims.filter(an => Number.isFinite(an.effect?.getComputedTiming().endTime)).map(an => an.finished.catch(() => {}))), new Promise(r => setTimeout(r, 1500))]).then(() => mark('done'));
})();`;

function scene(engine, tier) {
  return `S.backend = ${JSON.stringify(engine)}; render();
    const on = d => !d.only || d.only.includes(${JSON.stringify(engine)});
    S.tier = ${JSON.stringify(tier)}; S.mods = Object.keys(MODS).filter(id => on(MODS[id]));
    S.pedals = [...PEDALS, ...EXTRAS].filter(on).map(p => p.id); S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; });
    S.qAll = true; render(); scrollTo(0, 0);`;
}

const CATS = ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'disabled-by-default-devtools.timeline.frame', 'viz', 'gpu', 'blink.user_timing'];

function read(ev) {
  const thr = {}; for (const e of ev) if (e.ph === 'M' && e.name === 'thread_name') thr[e.pid + ':' + e.tid] = e.args.name;
  const down = ev.find(e => e.name === 'EventDispatch' && e.args?.data?.type === 'pointerdown');
  if (!down) throw new Error('no pointerdown in the trace');
  const t0 = down.ts, ms = t => (t - t0) / 1000;
  const mark = n => { const m = ev.find(e => e.name === 'tier:' + n && e.ts >= t0); return m ? m.ts : null; };
  const main = e => thr[e.pid + ':' + e.tid] === 'CrRendererMain';
  const commits = ev.filter(e => e.name === 'Commit' && e.ph === 'X' && main(e) && e.ts >= t0).map(e => e.ts).sort((a, b) => a - b);
  const draws = ev.filter(e => e.name === 'DirectRenderer::DrawFrame' && e.ph === 'X' && e.ts >= t0 - 100000).map(e => e.ts + e.dur).sort((a, b) => a - b);
  /* the first frame presented after the first commit at or after a mark */
  const shown = t => { if (t == null) return null; const c = commits.find(x => x >= t); if (c == null) return null; const d = draws.find(x => x > c); return d == null ? null : ms(d); };
  const checked = mark('checked'), moving = mark('moving'), done = mark('done') ?? (moving ?? t0) + 400000;
  const end = done + 20000, span = draws.filter(t => t >= t0 && t <= end);
  const pts = [t0, ...span]; let gap = 0, gapAt = 0;
  for (let i = 1; i < pts.length; i++) if (pts[i] - pts[i - 1] > gap) { gap = pts[i] - pts[i - 1]; gapAt = pts[i - 1]; }
  const raster = ev.filter(e => e.name === 'RasterDecoderImpl::DoEndRasterCHROMIUM' && e.ph === 'X' && e.ts >= t0 && e.ts <= end);
  const tasks = ev.filter(e => e.name === 'RunTask' && e.ph === 'X' && main(e) && e.ts + e.dur >= t0 && e.ts <= end);
  return {
    answered: r0(shown(checked)), moving: r0(shown(moving)), gap: r0(gap / 1000), gapAt: r0(ms(gapAt)), done: r0(ms(done)),
    raster: r0(raster.reduce((s, e) => s + e.dur, 0) / 1000), rasterMax: r0(Math.max(0, ...raster.map(e => e.dur)) / 1000),
    main: r0(tasks.reduce((s, e) => s + Math.min(e.ts + e.dur, end) - Math.max(e.ts, t0), 0) / 1000),
    frames: span.length,
  };
}

async function measure(browser, engine, from, to, keep) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, reducedMotion: REDUCED ? 'reduce' : 'no-preference' });
  await ctx.addInitScript(PROBE);
  const page = await ctx.newPage(), errs = []; page.on('pageerror', e => errs.push(e.message));
  try {
    await page.goto(PAGE); await page.waitForSelector('#go'); await sleep(900);
    await page.evaluate(scene(engine, from)); await sleep(1200);
    if (AGAIN) for (const k of [to, from]) { await page.click(`#tiers [data-tier="${k}"]`); await sleep(900); }
    const b = await page.evaluate(to => { const r = document.querySelector(`#tiers [data-tier="${to}"]`).getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }, to);
    await page.mouse.move(b.x, b.y); await sleep(300);
    const cdp = await ctx.newCDPSession(page), ev = [];
    cdp.on('Tracing.dataCollected', e => { for (const x of e.value) ev.push(x); });
    const complete = new Promise(r => cdp.once('Tracing.tracingComplete', r));
    await cdp.send('Tracing.start', { traceConfig: { includedCategories: CATS }, transferMode: 'ReportEvents' });
    await sleep(250);
    await page.evaluate(() => { window.__tf.on = true; });
    await page.mouse.down(); await page.mouse.up();
    await sleep(150); await page.evaluate(() => window.__tf.end());
    await sleep(1000);
    await cdp.send('Tracing.end'); await complete;
    const tier = await page.evaluate(() => { window.__tf.on = false; return document.querySelector('#tiers [aria-checked="true"]')?.dataset.tier; });
    if (tier !== to) throw new Error(`the tier is ${tier}, not ${to}`);
    if (keep) fs.writeFileSync(keep, JSON.stringify({ traceEvents: ev }));
    return { ...read(ev), errs: errs.slice(0, 2) };
  } finally { await ctx.close(); }
}

(async () => {
  const browser = await chromium.launch({ headless: false, args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'] });
  const out = { page: PAGE, at: new Date().toISOString(), runs: RUNS, again: AGAIN, reduced: REDUCED, rows: [] };
  try {
    for (const e of ENGINES) await measure(browser, e, 'studio', 'lab').catch(() => {});
    console.log(`${PAGE}${AGAIN ? ' (return)' : ''}${REDUCED ? ' (reduced)' : ''}\nengine change           answered moving  gap (at)  done  raster (max)  main  frames`);
    for (const e of ENGINES) for (const [from, to] of PAIRS) {
      const rs = []; for (let i = 0; i < RUNS; i++) rs.push(await measure(browser, e, from, to, process.env.TIER_KEEP && i === 0 ? `${process.env.TIER_KEEP}-${e}-${from}-${to}.json` : null));
      const keys = Object.keys(rs[0]).filter(k => k !== 'errs'), r = Object.fromEntries(keys.map(k => [k, med(rs.map(x => x[k]))]));
      r.gapMax = Math.max(...rs.map(x => x.gap));
      out.rows.push({ engine: e, from, to, ...r, all: rs });
      console.log(`${e.padEnd(6)} ${(from + ' > ' + to).padEnd(16)} ${String(r.answered).padStart(8)} ${String(r.moving).padStart(6)} ${String(r.gap).padStart(4)} (${r.gapAt}) ${String(r.done).padStart(5)} ${String(r.raster).padStart(6)} (${r.rasterMax}) ${String(r.main).padStart(5)} ${String(r.frames).padStart(6)}  worst gap ${r.gapMax}${rs[0].errs.length ? ' ERR ' + rs[0].errs[0] : ''}`);
    }
  } finally { await browser.close(); }
  const file = arg('json'); if (file) { fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true }); fs.writeFileSync(file, JSON.stringify(out, null, 1)); }
})();
