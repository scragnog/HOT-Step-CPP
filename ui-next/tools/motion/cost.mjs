// What each motion costs the main thread on a full desk (package 9, for the performance pass): every ACE-Step module
// and the board with five pedals, at 1920x1080, the motion set off as a user does and watched for 700ms in real time.
// For each: the main thread's busy time over the window (CDP Performance: script, style, layout, the whole task time),
// the longest task the action itself took (the time to the next frame), and the frames dropped while it moved (rAF
// intervals over 1.5 frames, and the longest). Each motion is run `--runs` times (default 3) and the median kept.
//
//   node tools/motion/cost.mjs [--page dist-test/index.html] [--only <regex>] [--runs 3] [--json out.json]
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { serve } from '../behaviour/harness.mjs';

const require = createRequire(import.meta.url);
const loadPW = () => { for (const m of [process.env.PLAYWRIGHT_MODULE, 'playwright'].filter(Boolean)) { try { return require(m); } catch (e) {} } throw new Error('Playwright not found: set PLAYWRIGHT_MODULE'); };
const here = path.dirname(fileURLToPath(import.meta.url)), ui = path.resolve(here, '..', '..');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const file = path.resolve(ui, opt('page', 'dist-test/index.html')), only = opt('only') ? new RegExp(opt('only')) : null, runs = +opt('runs', 3);

const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'guidance', 'latent', 'perf', 'post'];
const PEDS = ['ss', 'lift', 'gain', 'norm', 'nat'];
const full = (tier, mods = ACE, pedals = PEDS) => new Function(`S.tier = ${JSON.stringify(tier)}; S.mods = ${JSON.stringify(mods)}; S.pedals = ${JSON.stringify(pedals)}; S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = true; render(); scrollTo(0, 0);`);
/* a full Quick desk long enough to fold (the test build's two stand-ins), folded or open */
const folded = open => new Function(`['tx1', 'tx2'].forEach((id, i) => { defineModule(id, { at: 300 + i, name: 'Stand-in ' + (i + 1), cat: 'Engine', w: 1, keys: [] }); defineFace(id, () => '<span>stand-in</span>'); }); S.tier = 'quick'; S.mods = ${JSON.stringify([...ACE, 'tx1', 'tx2'])}; S.pedals = ${JSON.stringify(PEDS)}; S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = ${open}; render(); scrollTo(0, 0);`);
const LINES = 'a\nb\nc\nd\ne\nf';
const COSTS = [
  { id: 'tier studio to lab', scene: full('studio'), act: p => p.click('#tiers [data-tier="lab"]') },
  { id: 'tier lab to quick', scene: full('lab'), act: p => p.click('#tiers [data-tier="quick"]') },
  { id: 'tier quick to studio', scene: full('quick'), act: p => p.click('#tiers [data-tier="studio"]') },
  { id: 'add a tile', scene: full('lab', ACE.filter(m => m !== 'seed')), act: p => p.click('#browser .bcard[data-additem="seed"]') },
  { id: 'remove a tile', scene: full('lab'), act: p => p.click('.tile[data-m="avoid"] .tile-head .x') },
  { id: 'move a tile by key', scene: full('lab'), before: p => p.focus('[data-move="takes"]'), act: p => p.keyboard.press('ArrowLeft') },
  { id: 'a pedal joins', scene: full('lab', ACE, PEDS.filter(x => x !== 'lift')), act: p => p.evaluate(() => addItem('pd:lift')) },
  { id: 'a pedal leaves', scene: full('lab'), act: p => p.click('[data-pedalx="lift"]') },
  { id: 'the fold opens', scene: folded(false), act: p => p.evaluate(() => document.querySelector('#bento > [data-qall]').click()) },
  { id: 'the fold folds', scene: folded(true), act: p => p.evaluate(() => document.querySelector('#bento > [data-qall]').click()) },
  { id: 'prompt grows', scene: full('quick'), before: p => p.focus('#prompt'), act: p => p.keyboard.insertText(LINES) },
  { id: 'song info opens', scene: full('quick'), act: p => p.click('#songInfoBtn') },
];

async function measure(browser, url, c) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  try {
    await page.goto(url); await page.waitForTimeout(900);
    await page.evaluate(c.scene); await page.waitForTimeout(800);
    if (c.before) { await c.before(page); await page.waitForTimeout(300); }
    const cdp = await ctx.newCDPSession(page); await cdp.send('Performance.enable');
    const metric = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
    await page.evaluate(() => { const f = window.__f = []; const tick = t => { f.push(t); if (f.length < 400) requestAnimationFrame(tick); }; requestAnimationFrame(tick); });
    await page.waitForTimeout(100);
    const m0 = await metric(), t0 = await page.evaluate(() => performance.now());
    await c.act(page);
    await page.waitForTimeout(700);
    const m1 = await metric();
    const fr = await page.evaluate(t0 => { const f = window.__f.filter(t => t >= t0 - 20 && t <= t0 + 700), d = f.slice(1).map((t, i) => t - f[i]); return { first: f.length ? f[0] - t0 : 0, n: d.length, dropped: d.filter(x => x > 25).length, longest: Math.max(0, ...d) }; }, t0);
    const ms = k => Math.round(((m1[k] ?? 0) - (m0[k] ?? 0)) * 10000) / 10;
    return { task: ms('TaskDuration'), script: ms('ScriptDuration'), style: ms('RecalcStyleDuration'), layout: ms('LayoutDuration'), layouts: (m1.LayoutCount ?? 0) - (m0.LayoutCount ?? 0), dropped: fr.dropped, longest: Math.round(fr.longest) };
  } finally { await ctx.close(); }
}

const pw = loadPW(), browser = await pw.chromium.launch(), srv = await serve(file), out = {};
const med = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
try {
  console.log(`${path.relative(ui, file)}: main thread over 700ms after the action (median of ${runs})`);
  console.log('motion'.padEnd(22), 'task ms', 'script', 'style', 'layout', 'layouts', 'dropped', 'longest frame ms');
  for (const c of COSTS) {
    if (only && !only.test(c.id)) continue;
    const rs = []; for (let i = 0; i < runs; i++) rs.push(await measure(browser, srv.url, c));
    const r = Object.fromEntries(Object.keys(rs[0]).map(k => [k, med(rs.map(x => x[k]))])); out[c.id] = r;
    console.log(c.id.padEnd(22), String(r.task).padStart(7), String(r.script).padStart(6), String(r.style).padStart(5), String(r.layout).padStart(6), String(r.layouts).padStart(7), String(r.dropped).padStart(7), String(r.longest).padStart(6));
  }
} finally { await browser.close(); srv.close(); }
if (opt('json')) fs.writeFileSync(opt('json'), JSON.stringify(out, null, 1));
