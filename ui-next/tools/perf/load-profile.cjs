// Where the load's time goes, on the instrumented build: a CPU profile from navigation to 3 s after, mapped to the
// source, with the layout counters. PERF_BUILD=../ui-instr node load-profile.cjs [desk]
const fs = require('fs'), path = require('path');
const { launch, openWith } = require('./lib.cjs');
const { digest } = require('./prof-lib.cjs');
(async () => {
  const desk = process.argv[2] || 'full';
  const browser = await launch(); const { ctx, page, url } = await openWith(browser, desk === 'fresh' ? null : desk);
  const cdp = await ctx.newCDPSession(page);
  await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 100 }); await cdp.send('Profiler.start');
  await page.goto(url); await page.waitForTimeout(3000);
  const { profile } = await cdp.send('Profiler.stop');
  const d = digest(profile), cnt = await page.evaluate(() => window.__cnt);
  console.log('busy', d.busy, 'ms; counts', JSON.stringify(cnt));
  console.log('files:', d.files.slice(0, 14).map(([f, ms]) => `${f} ${ms}`).join(' | '));
  console.log('self:'); for (const [l, ms] of d.self.slice(0, 20)) console.log(`  ${String(ms).padStart(7)}  ${l}`);
  console.log('total:'); for (const [l, ms] of d.total.slice(0, 30)) console.log(`  ${String(ms).padStart(7)}  ${l}`);
  await browser.close();
})();
