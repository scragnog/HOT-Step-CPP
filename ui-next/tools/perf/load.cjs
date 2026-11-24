// Load: from navigation to the page being usable, on a fresh profile and with saved desks.
//   node load.cjs [--runs 7] [--desks fresh,typical,full] [--throttle 1] [--out results/load.json]
// "Usable" is the end of the last long task (> 50 ms) before 1 s with none, measured from navigation start (the
// page's time origin), with the go button and the saved desk's tiles on the page. Also: DOMContentLoaded, first
// contentful paint, the main thread's script, style and layout time to that point, the heap and the DOM size.
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT } = require('./lib.cjs');
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const RUNS = +arg('runs', 7), DESKS = arg('desks', 'fresh,typical,full').split(','), THROTTLE = +arg('throttle', 1);
const OUT = arg('out', path.join(ROOT, 'results', `load-${Date.now()}.json`)), PAGE = arg('page', 'file');
const PROBE = fs.readFileSync(path.join(__dirname, 'probe-page.js'), 'utf8');
const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };

(async () => {
  const browser = await launch();
  const out = { throttle: THROTTLE, runs: [] };
  for (let r = 0; r < RUNS; r++) for (const desk of DESKS) {
    const { ctx, page, url } = await openWith(browser, desk === 'fresh' ? null : desk, PAGE);
    await ctx.addInitScript(PROBE);
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Performance.enable', { timeDomain: 'threadTicks' });
    if (THROTTLE > 1) await cdp.send('Emulation.setCPUThrottlingRate', { rate: THROTTLE });
    await page.goto(url, { waitUntil: 'load' });
    /* wait for 1 s with no long task (up to 10 s) */
    const res = await page.evaluate(() => new Promise(done => {
      const P = window.__perf, t0 = performance.now();
      const check = () => {
        const now = performance.now(), last = P.lt.reduce((m, x) => Math.max(m, x.start + x.dur), 0);
        if ((now - Math.max(last, 0) > 1000 && document.querySelector('#go')) || now - t0 > 10000) {
          const nav = performance.getEntriesByType('navigation')[0], fcp = performance.getEntriesByName('first-contentful-paint')[0];
          done({ usable: Math.round(Math.max(last, nav.domContentLoadedEventEnd)), lastLongTask: Math.round(last), dcl: Math.round(nav.domContentLoadedEventEnd), load: Math.round(nav.loadEventEnd), fcp: fcp ? Math.round(fcp.startTime) : null,
            longTasks: P.lt.map(x => [Math.round(x.start), Math.round(x.dur)]), tiles: document.querySelectorAll('#bento > .tile[data-m]').length, nodes: document.querySelectorAll('*').length });
        } else setTimeout(check, 100);
      };
      check();
    }));
    const m = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(x => [x.name, x.value]));
    const row = { run: r, desk, ...res, script: Math.round(m.ScriptDuration * 1000), task: Math.round(m.TaskDuration * 1000), layout: Math.round(m.LayoutDuration * 1000), style: Math.round(m.RecalcStyleDuration * 1000), heapMB: +(m.JSHeapUsedSize / 1048576).toFixed(1) };
    out.runs.push(row);
    console.log(`${r} ${desk.padEnd(7)} usable ${row.usable} ms (dcl ${row.dcl}, fcp ${row.fcp}, load ${row.load}) long tasks ${JSON.stringify(row.longTasks)} | main thread ${row.task} (script ${row.script}, style ${row.style}, layout ${row.layout}) heap ${row.heapMB} MB, ${row.tiles} tiles, ${row.nodes} nodes`);
    await ctx.close();
  }
  for (const desk of DESKS) { const rs = out.runs.filter(x => x.desk === desk); console.log(`${desk}: usable median ${med(rs.map(x => x.usable))} (min ${Math.min(...rs.map(x => x.usable))}, max ${Math.max(...rs.map(x => x.usable))}); fcp median ${med(rs.map(x => x.fcp))}; main thread median ${med(rs.map(x => x.task))}`); }
  fs.mkdirSync(path.dirname(OUT), { recursive: true }); fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  await browser.close();
})();
