// Measures the rebuild's interactions, one action at a time, each in a fresh page on a saved desk (empty, typical,
// full), with real input (Playwright through CDP: trusted pointer and key events).
//
//   node perf.cjs [--throttle 4] [--runs 3] [--desks empty,typical,full] [--only name,name] [--page file|test] [--out results/x.json]
//   PERF_BUILD=<folder with dist-file/ and dist-test/> to measure another build (default: ui-next itself).
//
// Per action it records: the largest Event Timing duration of the action's interactions (INP as Chrome computes it,
// 8 ms steps, only >= 16 ms reported), "next frame" (input timeStamp to a task queued from the next animation frame, per
// input event; the max and the median over the action's inputs), main-thread time (CDP Performance: TaskDuration,
// ScriptDuration, LayoutDuration, RecalcStyleDuration deltas, settle time included), long tasks, long animation
// frames, and for drags the frames painted against the frames due.
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT } = require('./lib.cjs');
const PROBE = fs.readFileSync(path.join(__dirname, 'probe-page.js'), 'utf8');

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const RUNS = +arg('runs', 3), DESKS = arg('desks', 'empty,typical,full').split(','), ONLY = arg('only', '') ? arg('only').split(',') : null;
const THROTTLE = +arg('throttle', 1), PAGE = arg('page', 'file'), OUT = arg('out', path.join(ROOT, 'results', `perf-${Date.now()}.json`));

const { A, settle, sleep } = require('./actions.cjs');

/* ------------------------------------------------------------------ measuring one action */
const M = ['TaskDuration', 'ScriptDuration', 'LayoutDuration', 'RecalcStyleDuration', 'LayoutCount', 'RecalcStyleCount', 'JSHeapUsedSize', 'Nodes'];
async function metrics(cdp) { const r = await cdp.send('Performance.getMetrics'); const o = {}; for (const m of r.metrics) if (M.includes(m.name)) o[m.name] = m.value; return o; }
const med = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const r1 = x => x == null ? null : Math.round(x * 10) / 10;

async function measure(browser, desk, a) {
  const { ctx, page, url } = await openWith(browser, desk, PAGE);
  await ctx.addInitScript(PROBE);
  const errs = []; page.on('pageerror', e => errs.push(e.message));
  await page.goto(url); await page.waitForSelector('#go'); await sleep(900);
  const cdp = await ctx.newCDPSession(page); await cdp.send('Performance.enable', { timeDomain: 'threadTicks' });
  if (THROTTLE > 1) await cdp.send('Emulation.setCPUThrottlingRate', { rate: THROTTLE });
  try {
    const why = a.skip ? await a.skip(page, desk) : null;
    if (why) return { skip: why };
    if (a.prep) await a.prep(page, desk);
    await settle(page, 400);
    await page.evaluate(() => { window.__perf.reset(); });
    if (a.drag) await page.evaluate(() => window.__perf.startFrames());
    const m0 = await metrics(cdp), t0 = Date.now();
    await a.act(page, desk);
    const wall = Date.now() - t0;
    const fr = (a.drag || a.long) ? await page.evaluate(() => window.__perf.stopFrames()) : [];
    await settle(page, 300);
    const m1 = await metrics(cdp);
    const P = await page.evaluate(() => window.__perf);
    const inter = new Map(); for (const e of P.events) if (e.id) inter.set(e.id, Math.max(inter.get(e.id) || 0, e.dur));
    const inps = [...inter.values()];
    const lats = P.marks.filter(m => m.type !== 'pointermove' && m.type !== 'dragover').map(m => m.lat), moves = P.marks.filter(m => m.type === 'pointermove' || m.type === 'dragover').map(m => m.lat);
    const gaps = fr.slice(1).map((t, i) => t - fr[i]);
    const span = fr.length > 1 ? fr[fr.length - 1] - fr[0] : 0;
    return {
      inp: inps.length ? Math.max(...inps) : null, inpMed: med(inps), nInter: inps.length,
      evMax: P.events.length ? Math.max(...P.events.map(e => e.dur)) : null,
      nf: lats.length ? r1(Math.max(...lats)) : null, nfMed: r1(med(lats)), nIn: lats.length,
      moveMax: moves.length ? r1(Math.max(...moves)) : null, moveMed: r1(med(moves)), nMoves: moves.length,
      task: r1((m1.TaskDuration - m0.TaskDuration) * 1000), script: r1((m1.ScriptDuration - m0.ScriptDuration) * 1000), layout: r1((m1.LayoutDuration - m0.LayoutDuration) * 1000), style: r1((m1.RecalcStyleDuration - m0.RecalcStyleDuration) * 1000),
      layouts: m1.LayoutCount - m0.LayoutCount, styles: m1.RecalcStyleCount - m0.RecalcStyleCount, nodes: m1.Nodes,
      lt: P.lt.length, ltSum: r1(P.lt.reduce((s, x) => s + x.dur, 0)), ltMax: r1(Math.max(0, ...P.lt.map(x => x.dur))),
      loaf: P.loaf.length, loafMax: r1(Math.max(0, ...P.loaf.map(x => x.dur))),
      frames: fr.length || undefined, fps: span ? r1((fr.length - 1) / span * 1000) : undefined, gapMax: gaps.length ? r1(Math.max(...gaps)) : undefined, over33: gaps.filter(g => g > 33.4).length || 0,
      wall, errs: errs.slice(0, 3),
    };
  } catch (e) { return { error: String(e.message || e).slice(0, 200) }; }
  finally { await ctx.close(); }
}

(async () => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  const browser = await launch();
  const out = { page: PAGE, build: process.env.PERF_BUILD || 'ui-next', at: new Date().toISOString(), priority: process.env.PERF_PRIORITY || 'normal', throttle: THROTTLE, compositing: process.env.PERF_SOFT ? 'software' : 'gpu', runs: [] };
  for (let run = 0; run < RUNS; run++) {
    for (const desk of DESKS) for (const a of A) {
      if (ONLY && !ONLY.includes(a.name)) continue;
      const r = await measure(browser, desk, a);
      out.runs.push({ run, desk, action: a.name, ...r });
      const s = r.skip ? `skip (${r.skip})` : r.error ? `ERROR ${r.error}` : `inp ${r.inp} nf ${r.nf}/${r.nfMed} move ${r.moveMax ?? '-'}/${r.moveMed ?? '-'} task ${r.task} script ${r.script} layout ${r.layout} style ${r.style} lt ${r.lt}/${r.ltMax}${r.fps ? ` fps ${r.fps} gapMax ${r.gapMax} >33ms ${r.over33}` : ''}${r.errs?.length ? ' ERR ' + r.errs[0] : ''}`;
      console.log(`${run} ${desk.padEnd(7)} ${a.name.padEnd(20)} ${s}`);
      fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
    }
  }
  await browser.close();
  console.log('wrote', OUT);
})();
