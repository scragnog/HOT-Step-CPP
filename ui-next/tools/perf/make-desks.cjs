// Writes the saved desks the measurements load: empty, typical and full, by setting the scene through the test
// build's layer (window.S, render) and reading what the page autosaves.
const fs = require('fs'), path = require('path');
const { launch, chromium, PAGES, DESKS } = require('./lib.cjs');
const SCENES = {
  empty: S => { S.tier = 'quick'; S.mods = []; S.pedals = []; },
  typical: S => { S.tier = 'studio'; S.mods = ['tempo', 'seed', 'sampler', 'adapters', 'timbre']; S.pedals = ['ppvae', 'gain', 'norm']; },
  full: S => { S.tier = 'lab'; S.mods = Object.keys(MODS).filter(id => !MODS[id].only || MODS[id].only.includes('ace')); S.pedals = [...PEDALS.map(p => p.id), ...EXTRAS.map(p => p.id)].filter(id => !PED[id].only || PED[id].only.includes('ace')); },
};
(async () => {
  fs.mkdirSync(DESKS, { recursive: true });
  const browser = await launch();
  for (const [name, fn] of Object.entries(SCENES)) {
    const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
    const page = await ctx.newPage();
    await page.goto(PAGES.test); await page.waitForTimeout(800);
    await page.evaluate(`(${fn.toString()})(S); render();`);
    await page.waitForTimeout(1500);
    const out = await page.evaluate(() => { const o = {}; for (const k of Object.keys(localStorage)) o[k] = localStorage.getItem(k); return { ls: o, mods: S.mods, pedals: S.pedals, tier: S.tier, nodes: document.querySelectorAll('*').length, tiles: document.querySelectorAll('.tile').length }; });
    fs.writeFileSync(path.join(DESKS, name + '.json'), JSON.stringify(out.ls, null, 1));
    console.log(name, out.tier, out.mods.length, 'mods', out.pedals.length, 'pedals', out.nodes, 'nodes', out.tiles, 'tiles', Object.keys(out.ls));
    await ctx.close();
  }
  await browser.close();
})();
