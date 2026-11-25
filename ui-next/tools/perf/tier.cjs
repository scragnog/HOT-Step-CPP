// What a tier change costs on a full desk: every module, pedal and side unit of the engine on the desk, each tier pair
// in both directions, clicked as a user clicks the tier switch, watched for a second in real time. For each:
//   first   the click (its input's timeStamp) to the end of the first frame that shows the change moving (the View
//           Transition's first frame; without one, the frame that paints the new tier);
//   longest the longest frame (rAF to rAF) from the click on;
//   main    the main thread's busy time over the second (CDP Performance), and its script, style and layout parts;
//   dropped frames longer than 25 ms during the motion (the click to its --t-move, 280 ms, after the first frame);
//   settled the click to the end of the motion and of the last long frame (> 25 ms) after it;
//   models  the views' models asked again and found changed, and the tiles measured (the test layer's counters).
// Medians of `--runs` (3). The browser is warmed up first (one change per engine, unmeasured), as perf.cjs does.
//
//   node tools/perf/tier.cjs [--engines ace,mm3,yue] [--pairs quick-studio,studio-quick,...] [--runs 3] [--json out]
//   node tools/perf/tier.cjs --split [...]   where the time goes, by source file, on a source-mapped build of the test
//        page (npx vite build --mode test --config tools/perf/vite.instr.mjs, PERF_PROD=1 keeps React's production build)
//   PERF_BUILD=<folder holding dist-test/> to measure another build.
const fs = require('fs'), path = require('path');
const { launch, BUILD, url } = require('./lib.cjs');

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const RUNS = +arg('runs', 3), SPLIT = process.argv.includes('--split');
const ENGINES = arg('engines', 'ace,mm3,yue').split(',');
const PAIRS = arg('pairs', 'quick-studio,studio-quick,studio-lab,lab-studio,quick-lab,lab-quick').split(',').map(p => p.split('-'));
const PAGE = SPLIT ? 'file:///' + path.join(BUILD, 'tools/perf/results/instr/dist-test/index.html').split(path.sep).join('/') : url('dist-test');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const med = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) / 2)] : null; };
const r1 = x => (x == null ? null : Math.round(x * 10) / 10);
const MOVE = 280;
/* --mods a,b / --pedals a,b (or none): another desk than the full one */
const MODS = arg('mods') ? arg('mods').split(',') : null, PEDS = arg('pedals') ? arg('pedals').split(',').filter(x => x !== 'none') : null;

/* in the page, before its scripts: the frames, the click's time, the View Transition's moments, the tier's commit */
const PROBE = `(() => {
  const T = window.__tier = { frames: [], down: 0, vt: null, shown: 0, on: false };
  const tick = ts => { if (T.on) T.frames.push(ts); requestAnimationFrame(tick); }; requestAnimationFrame(tick);
  addEventListener('pointerdown', e => { if (T.on && !T.down) T.down = e.timeStamp; }, true);
  const o = Document.prototype.startViewTransition;
  if (o) Document.prototype.startViewTransition = function (cb) {
    const v = T.vt = { start: performance.now() };
    const t = o.call(this, () => { v.cb0 = performance.now(); const r = cb(); v.cb1 = performance.now(); return r; });
    t.ready.then(() => { v.ready = performance.now(); }, () => { v.skipped = true; });
    t.finished.then(() => { v.finished = performance.now(); }, () => {});
    return t;
  };
  new MutationObserver(() => { if (T.on && !T.shown) T.shown = performance.now(); }).observe(document, { subtree: true, attributes: true, attributeFilter: ['aria-checked'] });
})();`;

function scene(engine, tier) {
  return `S.backend = ${JSON.stringify(engine)}; render();
    const on = d => !d.only || d.only.includes(${JSON.stringify(engine)});
    S.tier = ${JSON.stringify(tier)}; S.mods = ${MODS ? JSON.stringify(MODS) : 'Object.keys(MODS).filter(id => on(MODS[id]))'};
    S.pedals = ${PEDS ? JSON.stringify(PEDS) : '[...PEDALS, ...EXTRAS].filter(on).map(p => p.id)'}; S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; });
    S.qAll = true; render(); scrollTo(0, 0);`;
}

/* ------------------------------------------------------------------ the split: self time by source file */
let smc = null, L0 = 0, C0 = 0;
function mapper() {
  const dir = path.join(BUILD, 'tools/perf/results/instr/dist-test'), html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8').split('\n');
  const { SourceMapConsumer } = require(path.join(__dirname, '..', '..', 'node_modules', 'source-map-js'));
  L0 = html.findIndex(l => l.startsWith('<script type="module">')); C0 = '<script type="module">'.length;
  const map = fs.readdirSync(path.join(dir, 'assets')).find(f => f.endsWith('.map'));
  smc = new SourceMapConsumer(JSON.parse(fs.readFileSync(path.join(dir, 'assets', map), 'utf8')));
}
const GROUPS = [
  ['react', /node_modules\/(react|react-dom|scheduler)\//],
  ['core (models)', /^src\/(derive|definitions)/],
  ['pack', /^src\/app\/desk\/(layout|pack)\.ts/],
  ['motion setup', /^src\/app\/(shell\/tier|features\/motion)/],
  ['view', /^src\/app\//],
  ['other script', /./],
];
function split(profile) {
  const byId = new Map(profile.nodes.map(n => [n.id, n])), parent = new Map();
  for (const n of profile.nodes) for (const c of n.children || []) parent.set(c, n.id);
  const fileOf = new Map(), cache = new Map();
  const where = cf => {
    if (!cf.url) return null;
    const k = cf.lineNumber + ':' + cf.columnNumber; if (cache.has(k)) return cache.get(k);
    const p = cf.lineNumber < L0 ? null : smc.originalPositionFor({ line: cf.lineNumber - L0 + 1, column: cf.lineNumber === L0 ? cf.columnNumber - C0 : cf.columnNumber });
    const s = p?.source ? p.source.replace(/^.*?(src\/|node_modules\/)/, '$1') : '?'; cache.set(k, s); return s;
  };
  for (const n of profile.nodes) fileOf.set(n.id, where(n.callFrame));
  const out = {}, add = (g, ms) => { out[g] = (out[g] || 0) + ms; };
  const dts = profile.timeDeltas;
  for (let i = 0; i < profile.samples.length; i++) {
    const dt = (dts[i + 1] ?? dts[i]) / 1000, n = byId.get(profile.samples[i]), name = n.callFrame.functionName;
    if (/^\((idle|program|root)\)$/.test(name)) continue;
    if (name === '(garbage collector)') { add('gc', dt); continue; }
    /* a native call (a forced layout, a style read, a DOM write) is put on the script that made it */
    let id = n.id; while (id !== undefined && !fileOf.get(id)) id = parent.get(id);
    const f = id === undefined ? '?' : fileOf.get(id), native = id !== n.id;
    const g = process.env.TIER_BY === 'file' ? f : process.env.TIER_BY === 'line' ? `${n.callFrame.functionName} ${f}:${(smc && n.callFrame.url ? smc.originalPositionFor({ line: n.callFrame.lineNumber - L0 + 1, column: n.callFrame.columnNumber }).line : '')}` : GROUPS.find(([, re]) => re.test(f))[0];
    add(native ? g + ' (native)' : g, dt);
  }
  if (process.env.TIER_FNS) { const want = new Set(process.env.TIER_FNS.split(',')), inc = {};
    for (let i = 0; i < profile.samples.length; i++) { const dt = (dts[i + 1] ?? dts[i]) / 1000, seen = new Set(); let id = profile.samples[i]; if (/^\((idle|program|root)\)$/.test(byId.get(id).callFrame.functionName)) continue; while (id !== undefined) { const f = byId.get(id).callFrame.functionName; if (want.has(f) && !seen.has(f)) { seen.add(f); inc[f] = (inc[f] || 0) + dt; } id = parent.get(id); } }
    for (const [k, v] of Object.entries(inc)) out['total ' + k] = v; }
  return Object.fromEntries(Object.entries(out).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, r1(v)]));
}

async function measure(browser, engine, from, to) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await ctx.addInitScript(PROBE);
  const page = await ctx.newPage(), errs = []; page.on('pageerror', e => errs.push(e.message));
  try {
    await page.goto(PAGE); await page.waitForSelector('#go'); await sleep(900);
    await page.evaluate(scene(engine, from)); await sleep(1200);
    const b = await page.evaluate(to => { const r = document.querySelector(`#tiers [data-tier="${to}"]`).getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }, to);
    await page.mouse.move(b.x, b.y); await sleep(300);
    const cdp = await ctx.newCDPSession(page); await cdp.send('Performance.enable', { timeDomain: 'threadTicks' });
    const metric = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
    if (SPLIT) { await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 100 }); await cdp.send('Profiler.start'); }
    await page.evaluate(() => { window.__t0 = { ...(window.__TALLY || {}) }; window.__tier.on = true; });
    const m0 = await metric();
    await page.mouse.down(); await page.mouse.up();
    await sleep(1000);
    const m1 = await metric();
    const prof = SPLIT ? (await cdp.send('Profiler.stop')).profile : null;
    const T = await page.evaluate(() => { const T = window.__tier; T.on = false; const c = window.__TALLY || {}, c0 = window.__t0; T.tally = Object.fromEntries(Object.entries(c).map(([k, v]) => [k, v - (c0[k] || 0)]).filter(([, v]) => v)); T.tier = document.querySelector('#tiers [aria-checked="true"]')?.dataset.tier; return T; });
    if (T.tier !== to) throw new Error(`the tier is ${T.tier}, not ${to}`);
    const fr = T.frames.filter(t => t >= T.down - 17), gaps = fr.slice(1).map((t, i) => [fr[i], t - fr[i]]);
    if (process.env.TIER_GAPS) console.log(gaps.filter(g => g[1] > 20).map(([t, d]) => `${Math.round(t + d - T.down)}:${Math.round(d)}`).join(" "), "vt", T.vt && Math.round(T.vt.ready - T.down), T.vt && Math.round(T.vt.finished - T.down));
    const vt = T.vt && !T.vt.skipped && T.vt.ready ? T.vt : null;
    /* the first frame that shows the change: the View Transition's first (it ends at the next frame), or the frame after the commit */
    const at = vt ? vt.ready : T.shown, i0 = fr.findIndex(t => t > at), firstEnd = vt ? fr[i0] : fr[i0 + 1];
    const moveEnd = (firstEnd ?? T.down) + MOVE;
    const longs = gaps.filter(([, d]) => d > 25);
    const ms = k => r1(((m1[k] ?? 0) - (m0[k] ?? 0)) * 1000);
    return {
      first: firstEnd ? r1(firstEnd - T.down) : null,
      longest: r1(Math.max(0, ...gaps.map(g => g[1]))),
      main: ms('TaskDuration'), script: ms('ScriptDuration'), style: ms('RecalcStyleDuration'), layout: ms('LayoutDuration'),
      dropped: gaps.filter(([t, d]) => d > 25 && t + d > (firstEnd ?? T.down) - 1 && t < moveEnd).length,
      longFrames: longs.length,
      settled: r1(Math.max(vt?.finished ?? 0, ...longs.map(([t, d]) => t + d)) - T.down),
      update: vt ? r1(vt.cb1 - vt.cb0) : null,
      models: T.tally['model'] ?? null, changed: T.tally['model.changed'] ?? null, measured: T.tally['measure.tiles'] ?? 0,
      tally: T.tally,
      ...(prof ? { split: split(prof) } : {}),
      errs: errs.slice(0, 2),
    };
  } finally { await ctx.close(); }
}

(async () => {
  if (SPLIT) mapper();
  const browser = await launch(), out = { page: PAGE, at: new Date().toISOString(), runs: RUNS, rows: [] };
  try {
    for (const e of ENGINES) await measure(browser, e, 'studio', 'lab').catch(() => {});
    console.log(`${PAGE}\n${'engine'.padEnd(5)} ${'change'.padEnd(15)} first longest main (script style layout) dropped settled update | models asked, changed, tiles measured`);
    for (const e of ENGINES) for (const [from, to] of PAIRS) {
      const rs = []; for (let i = 0; i < RUNS; i++) rs.push(await measure(browser, e, from, to));
      const keys = Object.keys(rs[0]).filter(k => k !== 'errs' && k !== 'split'), r = Object.fromEntries(keys.map(k => [k, med(rs.map(x => x[k]))]));
      if (SPLIT) { const gs = [...new Set(rs.flatMap(x => Object.keys(x.split)))]; r.split = Object.fromEntries(gs.map(g => [g, med(rs.map(x => x.split[g] ?? 0))]).sort((a, b) => b[1] - a[1])); }
      out.rows.push({ engine: e, from, to, ...r, errs: rs.flatMap(x => x.errs).slice(0, 2) });
      console.log(`${e.padEnd(5)} ${(from + ' > ' + to).padEnd(15)} ${String(r.first).padStart(5)} ${String(r.longest).padStart(7)} ${String(r.main).padStart(5)} (${r.script} ${r.style} ${r.layout}) ${String(r.dropped).padStart(3)} ${String(r.settled).padStart(7)} ${String(r.update).padStart(6)} | ${r.models}, ${r.changed}, ${r.measured}${rs[0].errs.length ? ' ERR ' + rs[0].errs[0] : ''}`);
      if (r.split) console.log('      ' + Object.entries(r.split).map(([k, v]) => `${k} ${v}`).join(', '));
    }
  } finally { await browser.close(); }
  const file = arg('json'); if (file) { fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true }); fs.writeFileSync(file, JSON.stringify(out, null, 1)); }
})();
