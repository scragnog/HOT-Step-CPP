// Records the concept page's desk layouts for the pure layout's tests (src/app/desk/pack.ts, tools/test-view.ts): in
// each state, what the page measured (deskMeasure: the columns and each tile's widths and heights) and where its
// solver put each tile (deskSolve), into tools/fixtures/pack.json. The states are every engine at every tier with
// every module the engine uses on the desk, at the five desktop widths, and a few made-up desks.
//
//   PLAYWRIGHT_MODULE=<node_modules/playwright> node tools/concept/collect-pack.mjs [<concept>/hotstep-tiers.html]

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..', '..');
const require = createRequire(import.meta.url);
const pw = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const page0 = path.resolve(process.argv[2] || path.join(root, '..', '..', 'HOT-Step-UI', 'hotstep-tiers.html'));
const ENGINES = ['ace', 'mm3', 'yue', 'sa3', 'hm'], TIERS = ['quick', 'studio', 'lab'], WIDTHS = [1920, 1600, 1440, 1280, 1000];

const browser = await pw.chromium.launch();
const out = [];
for (const w of WIDTHS) {
  const ctx = await browser.newContext({ viewport: { width: w, height: 1000 }, reducedMotion: 'reduce' });
  await ctx.route(u => !u.href.startsWith('file:'), r => r.abort());
  const page = await ctx.newPage();
  await page.goto('file:///' + page0.split(path.sep).join('/'));
  await page.waitForTimeout(400);
  const scenes = [];
  for (const bk of ENGINES) for (const tier of TIERS) scenes.push({ id: `${bk}-${tier}-everything`, bk, tier, mods: 'all', pedals: ['ss', 'norm', 'lift', 'wh'] });
  scenes.push({ id: 'ace-studio-small', bk: 'ace', tier: 'studio', mods: ['avoid', 'seed', 'post'], pedals: ['gain'] });
  scenes.push({ id: 'ace-lab-slice', bk: 'ace', tier: 'lab', mods: ['avoid', 'guidance', 'post'], pedals: ['tile', 'refiner', 'ss', 'ppvae', 'lift', 'nat', 'gain', 'vst', 'norm', 'wh'] });
  scenes.push({ id: 'yue-lab-slice', bk: 'yue', tier: 'lab', mods: ['yuebatch', 'post'], pedals: ['refiner', 'ss', 'nat', 'gain', 'vst', 'norm', 'wh'] });
  for (const sc of scenes) {
    const rec = await page.evaluate(sc => {
      S.backend = sc.bk; S.tier = sc.tier; S.qAll = true;
      S.mods = sc.mods === 'all' ? Object.keys(MODS).filter(m => !MODS[m].source && m !== 'post' && (!MODS[m].only || MODS[m].only.includes(sc.bk))).concat('post') : sc.mods;
      S.pedals = sc.pedals; render();
      const g = document.querySelector('#bento'), m = deskMeasure(g), res = deskSolve(m.cols, m.items);
      packNow(false);
      return { cols: m.cols, items: m.items, out: res.out.map(r => ({ i: r.i, c: r.c, w: r.w, y: r.y, h: r.h })), plan: res.plan };
    }, sc);
    out.push({ id: `w${w}-${sc.id}`, ...rec });
  }
  await ctx.close();
}
await browser.close();
const file = path.join(root, 'tools', 'fixtures', 'pack.json');
fs.writeFileSync(file, JSON.stringify({ about: 'Desk layouts the concept page chose: what it measured (cols, items) and where its solver put each tile (out, plan). Recorded by tools/concept/collect-pack.mjs; checked by tools/test-view.ts.', cases: out }, null, 1).replace(/\n/g, '\r\n'));
console.log(`recorded ${out.length} layouts into tools/fixtures/pack.json`);
