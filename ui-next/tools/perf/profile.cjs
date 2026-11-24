// Finds where an action's time goes, on the instrumented build (ui-instr: unminified, source-mapped, React's profiling
// build, useModel and the desk store counted; see report.md "Scripts"):
//   - a CPU profile of the action (V8 sampling, 50 us), self and total time by function, mapped to the source file and
//     line through the build's source map;
//   - useModel: how many views asked the core for their model, how long that took, how many models really changed;
//   - the desk store: how many notifications, and the time in each listener, by where it subscribed;
//   - React: commits, and the components that rendered, by name, with their own render time.
//
//   PERF_BUILD=../ui-instr node profile.cjs [--desks full] [--only knob-keys,remove] [--top 25] [--out results/profile.json]
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT, BUILD } = require('./lib.cjs');
const { A, settle, sleep } = require('./actions.cjs');

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const DESKS = arg('desks', 'full').split(','), ONLY = arg('only', '') ? arg('only').split(',') : null, TOP = +arg('top', 25);
const OUT = arg('out', path.join(ROOT, 'results', `profile-${Date.now()}.json`));
const PROBES = ['probe-page.js', 'probe-react.js'].map(f => fs.readFileSync(path.join(__dirname, f), 'utf8'));

const { digest, mapTag } = require('./prof-lib.cjs');

async function run(browser, desk, a) {
  const { ctx, page, url } = await openWith(browser, desk, 'file');
  for (const p of PROBES) await ctx.addInitScript(p);
  await page.goto(url); await page.waitForSelector('#go'); await sleep(900);
  const cdp = await ctx.newCDPSession(page);
  try {
    const why = a.skip ? await a.skip(page, desk) : null; if (why) return { skip: why };
    if (a.prep) await a.prep(page, desk);
    await settle(page, 400);
    await page.evaluate(() => { const u = window.__um, d = window.__ds; window.__um0 = JSON.parse(JSON.stringify(u)); window.__ds0 = JSON.parse(JSON.stringify(d)); window.__cnt0 = JSON.parse(JSON.stringify(window.__cnt || {})); window.__rxReset(); });
    await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 50 }); await cdp.send('Profiler.start');
    await a.act(page, desk); await settle(page, 300);
    const { profile } = await cdp.send('Profiler.stop');
    const st = await page.evaluate(() => {
      const u = window.__um, u0 = window.__um0, d = window.__ds, d0 = window.__ds0, R = window.__rx;
      const by = Object.entries(u.byKey).map(([k, v]) => [k, v[0] - (u0.byKey[k]?.[0] || 0), Math.round((v[1] - (u0.byKey[k]?.[1] || 0)) * 10) / 10, v[2] - (u0.byKey[k]?.[2] || 0)]).filter(x => x[1]).sort((a, b) => b[2] - a[2]);
      const ls = Object.entries(d.by).map(([k, v]) => [k, v[0] - (d0.by[k]?.[0] || 0), Math.round((v[1] - (d0.by[k]?.[1] || 0)) * 10) / 10]).filter(x => x[1]).sort((a, b) => b[2] - a[2]);
      const rn = Object.entries(R.rendered).sort((a, b) => b[1] - a[1]).slice(0, 30).map(([n, c]) => [n, c, Math.round(R.self[n] * 10) / 10]);
      const c = window.__cnt || {}, c0 = window.__cnt0 || {};
      const counts = Object.fromEntries(Object.entries(c).map(([k, v]) => [k, [v[0] - (c0[k]?.[0] || 0), Math.round((v[1] - (c0[k]?.[1] || 0)) * 10) / 10]]).filter(([, v]) => v[0]));
      return {
        counts,
        useModel: { asked: u.calls - u0.calls, computed: u.computes - u0.computes, fnMs: Math.round(u.fnMs - u0.fnMs), jsonMs: Math.round(u.jsonMs - u0.jsonMs), changed: u.changed - u0.changed, byKey: by.slice(0, 15) },
        store: { notifications: d.sets - d0.sets, listenerMs: Math.round(d.ms - d0.ms), listeners: ls.slice(0, 12) },
        react: { commits: R.commits, rendered: R.total, mounts: R.mounts, walkMs: Math.round(R.walkMs), byName: rn },
      };
    });
    st.store.listeners = st.store.listeners.map(([t, n, ms]) => [mapTag(t), n, ms]);
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
    console.log(`\n=== ${desk} ${a.name}: cpu busy ${r.cpu.busy} ms; store notifications ${r.store.notifications}, listeners ${r.store.listenerMs} ms; useModel asked ${r.useModel.asked}, computed ${r.useModel.computed} (${r.useModel.fnMs} + ${r.useModel.jsonMs} json ms), changed ${r.useModel.changed}; react commits ${r.react.commits}, rendered ${r.react.rendered} (mounts ${r.react.mounts})`);
    console.log('  counts:', JSON.stringify(r.counts));
    console.log('  listeners:', r.store.listeners.slice(0, 6).map(x => `${x[0]} x${x[1]} ${x[2]}ms`).join(' | '));
    console.log('  useModel by key:', r.useModel.byKey.slice(0, 8).map(x => `${x[0]} n${x[1]} ${x[2]}ms ch${x[3]}`).join(' | '));
    console.log('  rendered:', r.react.byName.slice(0, 14).map(x => `${x[0]} ${x[1]} (${x[2]}ms)`).join(', '));
    console.log('  self:'); for (const [l, ms] of r.cpu.self.slice(0, 14)) console.log(`    ${String(ms).padStart(7)}  ${l}`);
    console.log('  files:', r.cpu.files.slice(0, 10).map(([f, ms]) => `${f} ${ms}`).join(' | '));
    fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  }
  await browser.close();
  console.log('wrote', OUT);
})();
