// Finds where an action's time goes, on the instrumented build (tools/perf/vite.instr.ts: unminified, source-mapped,
// React's profiling build, the page's counters on):
//   - a CPU profile of the action (V8 sampling, 50 us), self and total time by function, mapped to the source file and
//     line through the build's source map;
//   - the page's counters (src/app/tally.ts): models asked again, packs, tiles measured, layouts solved;
//   - React: commits, and the components that rendered, by name, with their own render time.
//
//   npx vite build --mode file --config tools/perf/vite.instr.ts
//   PERF_BUILD=tools/perf/results/instr node tools/perf/profile.cjs [--desks full] [--only knob-keys,remove] [--top 25]
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT, BUILD } = require('./lib.cjs');
const { A, settle, sleep } = require('./actions.cjs');

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const DESKS = arg('desks', 'full').split(','), ONLY = arg('only', '') ? arg('only').split(',') : null, TOP = +arg('top', 25);
const OUT = arg('out', path.join(ROOT, 'results', `profile-${Date.now()}.json`));
const PROBES = ['probe-page.js', 'probe-react.js'].map(f => fs.readFileSync(path.join(__dirname, f), 'utf8'));

const { digest } = require('./prof-lib.cjs');

async function run(browser, desk, a) {
  const { ctx, page, url } = await openWith(browser, desk, 'file');
  for (const p of PROBES) await ctx.addInitScript(p);
  await page.goto(url); await page.waitForSelector('#go'); await sleep(900);
  const cdp = await ctx.newCDPSession(page);
  try {
    const why = a.skip ? await a.skip(page, desk) : null; if (why) return { skip: why };
    if (a.prep) await a.prep(page, desk);
    await settle(page, 400);
    await page.evaluate(() => { window.__cnt0 = { ...(window.__TALLY || {}) }; window.__rxReset(); });
    await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 50 }); await cdp.send('Profiler.start');
    await a.act(page, desk); await settle(page, 300);
    const { profile } = await cdp.send('Profiler.stop');
    const st = await page.evaluate(() => {
      const R = window.__rx, c = window.__TALLY || {}, c0 = window.__cnt0 || {};
      const rn = Object.entries(R.rendered).sort((a, b) => b[1] - a[1]).slice(0, 30).map(([n, k]) => [n, k, Math.round(R.self[n] * 10) / 10]);
      return { counts: Object.fromEntries(Object.entries(c).map(([k, v]) => [k, v - (c0[k] || 0)]).filter(([, v]) => v)), react: { commits: R.commits, rendered: R.total, mounts: R.mounts, walkMs: Math.round(R.walkMs), byName: rn } };
    });
    return { ...st, cpu: digest(profile) };
  } catch (e) { return { error: String(e.message || e).slice(0, 300) }; }
  finally { await ctx.close(); }
}

(async () => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  const browser = await launch();
  const out = {};
  for (const desk of DESKS) for (const a of A) {
    if (ONLY && !ONLY.includes(a.name)) continue;
    const r = await run(browser, desk, a); out[`${desk} ${a.name}`] = r;
    if (r.skip || r.error) { console.log(desk, a.name, r.skip || r.error); continue; }
    console.log(`
=== ${desk} ${a.name}: cpu busy ${r.cpu.busy} ms; react commits ${r.react.commits}, rendered ${r.react.rendered} (mounts ${r.react.mounts})`);
    console.log('  counts:', JSON.stringify(r.counts));
    console.log('  rendered:', r.react.byName.slice(0, 14).map(x => `${x[0]} ${x[1]} (${x[2]}ms)`).join(', '));
    console.log('  self:'); for (const [l, ms] of r.cpu.self.slice(0, 14)) console.log(`    ${String(ms).padStart(7)}  ${l}`);
    console.log('  native calls, by the script that made them:'); for (const [l, ms] of r.cpu.natives.slice(0, 12)) console.log(`    ${String(ms).padStart(7)}  ${l}`);
    console.log('  files:', r.cpu.files.slice(0, 10).map(([f, ms]) => `${f} ${ms}`).join(' | '));
    fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  }
  await browser.close();
  console.log('wrote', OUT);
})();
