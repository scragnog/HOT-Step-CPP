// A timeline trace of one action on the product build: how many layouts and style recalcs it ran, how long each
// took, which script forced them (a layout inside script is a forced synchronous layout), and the paint and raster
// time after it (what Event Timing's figure includes beyond the main thread).
//   node trace.cjs [--desks full] [--only remove,tier-switch] [--out results/trace.json]
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT } = require('./lib.cjs');
const { A, settle, sleep } = require('./actions.cjs');
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const DESKS = arg('desks', 'full').split(','), ONLY = arg('only', '') ? arg('only').split(',') : null, OUT = arg('out', path.join(ROOT, 'results', `trace-${Date.now()}.json`));
const PROBE = fs.readFileSync(path.join(__dirname, 'probe-page.js'), 'utf8');
const CATS = 'devtools.timeline,disabled-by-default-devtools.timeline,disabled-by-default-devtools.timeline.stack,blink,cc,gpu,viz,disabled-by-default-devtools.timeline.frame';

async function run(browser, desk, a) {
  const { ctx, page, url } = await openWith(browser, desk);
  await ctx.addInitScript(PROBE);
  await page.goto(url); await page.waitForSelector('#go'); await sleep(900);
  const cdp = await ctx.newCDPSession(page);
  try {
    const why = a.skip ? await a.skip(page, desk) : null; if (why) return { skip: why };
    if (a.prep) await a.prep(page, desk);
    await settle(page, 400);
    const events = []; const onData = d => events.push(...d.value);
    cdp.on('Tracing.dataCollected', onData);
    await cdp.send('Tracing.start', { categories: CATS, transferMode: 'ReportEvents' });
    await a.act(page, desk); await settle(page, 300);
    const done = new Promise(r => cdp.once('Tracing.tracingComplete', r)); await cdp.send('Tracing.end'); await done;
    cdp.off('Tracing.dataCollected', onData);
    /* the renderer's main thread (CrRendererMain), and every thread's busy time by name */
    const names = {}; for (const e of events) if (e.ph === 'M' && e.name === 'thread_name') names[e.pid + ':' + e.tid] = e.args.name;
    const main = Object.keys(names).find(k => names[k] === 'CrRendererMain');
    const threads = {}; for (const e of events) if (e.name === 'RunTask' && e.dur) { const n = names[e.pid + ':' + e.tid] || '?'; threads[n] = (threads[n] || 0) + e.dur / 1000; }
    const on = e => e.pid + ':' + e.tid === main;
    const X = events.filter(e => e.ph === 'X' && e.dur !== undefined);
    const sum = (n, f = on) => { const es = X.filter(e => e.name === n && f(e)); return { n: es.length, ms: Math.round(es.reduce((s, e) => s + e.dur, 0) / 100) / 10 }; };
    /* a layout or style recalc that started inside a script call on the main thread was forced by it */
    const calls = X.filter(e => on(e) && (e.name === 'FunctionCall' || e.name === 'EventDispatch' || e.name === 'FireAnimationFrame' || e.name === 'TimerFire' || e.name === 'v8.callFunction'));
    const inScript = e => calls.some(c => c.ts <= e.ts && e.ts + e.dur <= c.ts + c.dur);
    const forced = X.filter(e => on(e) && e.name === 'Layout' && inScript(e));
    const forcedBy = {};
    for (const e of forced) { const st = e.args?.beginData?.stackTrace?.[0]; const k = st ? `${st.functionName || '(anon)'}:${st.lineNumber}` : '?'; forcedBy[k] = forcedBy[k] || { n: 0, ms: 0 }; forcedBy[k].n++; forcedBy[k].ms += e.dur / 1000; }
    const other = e => !on(e);
    return {
      layout: sum('Layout'), forcedLayout: { n: forced.length, ms: Math.round(forced.reduce((s, e) => s + e.dur, 0) / 100) / 10 }, style: sum('UpdateLayoutTree'),
      paint: sum('Paint'), prePaint: sum('PrePaint'), layerize: sum('Layerize'), commit: sum('Commit'),
      raster: { n: X.filter(e => other(e) && /RasterTask|RasterizerTaskImpl::RunOnWorkerThread/.test(e.name)).length, ms: Math.round(X.filter(e => other(e) && /RasterTask/.test(e.name)).reduce((s, e) => s + e.dur, 0) / 100) / 10 },
      script: sum('FunctionCall'), events: sum('EventDispatch'),
      threads: Object.fromEntries(Object.entries(threads).map(([k, v]) => [k, Math.round(v)]).sort((x, y) => y[1] - x[1]).slice(0, 6)),
      forcedBy: Object.entries(forcedBy).sort((x, y) => y[1].ms - x[1].ms).slice(0, 8).map(([k, v]) => [k, v.n, Math.round(v.ms * 10) / 10]),
    };
  } catch (e) { return { error: String(e.message || e).slice(0, 200) }; }
  finally { await ctx.close(); }
}

(async () => {
  const browser = await launch();
  const out = {};
  for (const desk of DESKS) for (const a of A) {
    if (ONLY && !ONLY.includes(a.name)) continue;
    const r = await run(browser, desk, a); out[`${desk} ${a.name}`] = r;
    console.log(`${desk} ${a.name}:`, r.skip || r.error || `layouts ${r.layout.n} (${r.layout.ms} ms; forced ${r.forcedLayout.n}, ${r.forcedLayout.ms} ms) style ${r.style.n} (${r.style.ms} ms) paint ${r.paint.n} (${r.paint.ms} ms) raster ${r.raster.ms} ms commit ${r.commit.ms} ms | threads ${JSON.stringify(r.threads)} | forced by ${JSON.stringify(r.forcedBy)}`);
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true }); fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  await browser.close();
})();
