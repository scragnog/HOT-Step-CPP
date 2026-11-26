// CPU with the page doing nothing, and while a run is in progress.
//   node idle.cjs [--desks empty,full] [--secs 10] [--out results/idle.json]
// For each desk: 10 s untouched (after 3 s to settle), then one run of the simulated engine (Generate pressed, 4.5 s).
// Reports the renderer main thread's busy time (CDP Performance TaskDuration, thread ticks), the renderer process's
// CPU time (SystemInfo.getProcessInfo), and from a timeline trace what ran: frames begun, style recalcs, layouts, paints,
// timers, animation frame callbacks, script calls.
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT } = require('./lib.cjs');
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const DESKS = arg('desks', 'empty,full').split(','), SECS = +arg('secs', 10), OUT = arg('out', path.join(ROOT, 'results', `idle-${Date.now()}.json`));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const COUNT = ['BeginMainThreadFrame', 'UpdateLayoutTree', 'Layout', 'Paint', 'TimerFire', 'FireAnimationFrame', 'FunctionCall', 'Animation', 'RunTask', 'Commit', 'EventDispatch'];

async function window_(browser, cdp, page, ms, what) {
  const bcdp = await browser.newBrowserCDPSession();
  const cpu = async () => { try { const r = await bcdp.send('SystemInfo.getProcessInfo'); return r.processInfo.filter(p => p.type === 'renderer').reduce((s, p) => s + p.cpuTime, 0); } catch (e) { return null; } };
  const events = []; const onData = d => { for (const e of d.value) if (COUNT.includes(e.name)) events.push(e); };
  cdp.on('Tracing.dataCollected', onData);
  await cdp.send('Tracing.start', { categories: 'devtools.timeline,disabled-by-default-devtools.timeline.frame,blink.animations', transferMode: 'ReportEvents' });
  const m0 = await cdp.send('Performance.getMetrics'), c0 = await cpu(), t0 = Date.now();
  if (what) await what();
  await sleep(ms - (Date.now() - t0));
  const m1 = await cdp.send('Performance.getMetrics'), c1 = await cpu(), wall = Date.now() - t0;
  const done = new Promise(r => cdp.once('Tracing.tracingComplete', r)); await cdp.send('Tracing.end'); await done; cdp.off('Tracing.dataCollected', onData);
  const g = (m, n) => m.metrics.find(x => x.name === n).value;
  const counts = {}; let fnMs = 0;
  for (const e of events) { if (e.ph === 'X' || e.ph === 'B' || e.ph === 'I' || e.ph === 'i' || e.ph === 'n' || e.ph === 'b') counts[e.name] = (counts[e.name] || 0) + 1; if (e.name === 'FunctionCall' && e.dur) fnMs += e.dur / 1000; }
  const fnBy = {}; for (const e of events) if (e.name === 'FunctionCall' && e.args?.data) { const k = `${e.args.data.functionName || '?'}@${e.args.data.lineNumber ?? ''}`; fnBy[k] = (fnBy[k] || 0) + 1; }
  await bcdp.detach();
  const s = wall / 1000;
  return { wallS: +s.toFixed(1), mainThreadMsPerS: +((g(m1, 'TaskDuration') - g(m0, 'TaskDuration')) * 1000 / s).toFixed(1), scriptMsPerS: +((g(m1, 'ScriptDuration') - g(m0, 'ScriptDuration')) * 1000 / s).toFixed(1), rendererCpuPct: c0 == null ? null : +((c1 - c0) / s * 100).toFixed(1), perS: Object.fromEntries(Object.entries(counts).map(([k, v]) => [k, +(v / s).toFixed(1)])), fnMsPerS: +(fnMs / s).toFixed(1), topCalls: Object.entries(fnBy).sort((a, b) => b[1] - a[1]).slice(0, 6) };
}

(async () => {
  const browser = await launch();
  const out = {};
  for (const desk of DESKS) {
    const { ctx, page, url } = await openWith(browser, desk);
    await page.goto(url); await page.waitForSelector('#go'); await sleep(3000);
    const cdp = await ctx.newCDPSession(page); await cdp.send('Performance.enable', { timeDomain: 'threadTicks' });
    out[desk] = { idle: await window_(browser, cdp, page, SECS * 1000) };
    await page.mouse.click(5, 400); await sleep(3000);
    out[desk].idleAfterClick = await window_(browser, cdp, page, SECS * 1000);
    out[desk].run = await window_(browser, cdp, page, 4500, async () => { const b = await page.$('#go'); await b.click(); });
    console.log(desk, JSON.stringify(out[desk], null, 1));
    await ctx.close();
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true }); fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  await browser.close();
})();
