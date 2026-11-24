// Lists every Layout and style recalc of one action with what Blink says about it (dirty objects, the layout root,
// elements whose style was recalculated), to see why a layout was large. node layout-detail.cjs <action> [desk]
const fs = require('fs'), path = require('path');
const { launch, openWith } = require('./lib.cjs');
const { A, settle, sleep } = require('./actions.cjs');
const PROBE = fs.readFileSync(path.join(__dirname, 'probe-page.js'), 'utf8');
(async () => {
  const name = process.argv[2], desk = process.argv[3] || 'full', a = A.find(x => x.name === name);
  const browser = await launch(); const { ctx, page, url } = await openWith(browser, desk); await ctx.addInitScript(PROBE);
  await page.goto(url); await page.waitForSelector('#go'); await sleep(900);
  const cdp = await ctx.newCDPSession(page);
  if (a.prep) await a.prep(page, desk); await settle(page, 400);
  const ev = []; cdp.on('Tracing.dataCollected', d => ev.push(...d.value));
  await cdp.send('Tracing.start', { categories: 'devtools.timeline,disabled-by-default-devtools.timeline,disabled-by-default-devtools.timeline.stack,disabled-by-default-devtools.timeline.invalidationTracking', transferMode: 'ReportEvents' });
  await a.act(page, desk); await settle(page, 300);
  const done = new Promise(r => cdp.once('Tracing.tracingComplete', r)); await cdp.send('Tracing.end'); await done;
  const t0 = Math.min(...ev.filter(e => e.ts).map(e => e.ts));
  for (const e of ev.filter(e => (e.name === 'Layout' || e.name === 'UpdateLayoutTree') && e.ph === 'X' && e.dur > 1000).sort((x, y) => x.ts - y.ts)) {
    const b = e.args?.beginData || {}, en = e.args?.endData || {}, st = b.stackTrace?.[0];
    console.log(`${((e.ts - t0) / 1000).toFixed(1).padStart(7)} ${e.name.padEnd(16)} ${(e.dur / 1000).toFixed(1).padStart(6)} ms  dirty ${b.dirtyObjects ?? ''}/${b.totalObjects ?? ''} partial ${b.partialLayout ?? ''} elements ${e.args?.elementCount ?? ''} root ${JSON.stringify(en.layoutRoots?.[0]?.nodeName ?? '')} by ${st ? st.functionName + ':' + st.lineNumber + ':' + st.columnNumber : '(frame)'}`);
  }
  const inv = ev.filter(e => /Invalidat/.test(e.name)).slice(0, 400);
  const by = {}; for (const e of inv) { const d = e.args?.data || {}; const k = `${e.name} ${d.reason || ''} ${d.selectors?.[0]?.selector || d.changedClass || d.changedAttribute || d.changedId || ''}`; by[k] = (by[k] || 0) + 1; }
  console.log('invalidations:', Object.entries(by).sort((x, y) => y[1] - x[1]).slice(0, 15));
  await browser.close();
})();
