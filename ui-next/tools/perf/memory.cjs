// Memory over a long mixed session: the heap (after a forced garbage collection), the DOM's nodes, listeners and
// documents, after load and after every round of mixed actions on the full desk.
//   node memory.cjs [--rounds 12] [--desk full] [--no-generate] [--out results/memory.json]
// (--no-generate leaves out Generate, whose new song cards are content the page keeps on purpose, up to 24 shown)
// One round: turn a knob by keys and by drag, type in the prompt and the lyric sheet, flip a toggle, open and choose
// in a dropdown, switch tier there and back, switch engine there and back, remove a module and add it back from the
// panel, open and close the picker, the popover, the notes and the song menu, undo three times and redo once, and
// press Generate (the run completes during the next round).
const fs = require('fs'), path = require('path');
const { launch, chromium, openWith, ROOT } = require('./lib.cjs');
const { click, focus, vis, drag, settle, sleep } = require('./actions.cjs');
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const NOGEN = process.argv.includes('--no-generate'), ROUNDS = +arg('rounds', 12), DESK = arg('desk', 'full'), OUT = arg('out', path.join(ROOT, 'results', `memory-${Date.now()}.json`));

/* each step on its own: one that cannot be done on this desk now (a control another step moved) is skipped and said */
async function round(p, i) {
  const skipped = [];
  const step = async (name, fn) => { try { await fn(); } catch (e) { skipped.push(name); } };
  await step('knob keys', async () => { await focus(p, '.tile[data-m="sampler"] .knob'); for (let k = 0; k < 5; k++) await p.keyboard.press(k % 2 ? 'ArrowDown' : 'ArrowUp'); });
  await step('pedal knob drag', async () => drag(p, await vis(p, '.pedal[data-pid="ss"] .knob'), 20, 0, i % 2 ? 2 : -2));
  await step('prompt', async () => { await click(p, '#prompt'); await p.keyboard.type(` r${i}`, { delay: 20 }); });
  await step('lyrics', async () => { await click(p, 'textarea[data-bind="lyrics.text"]'); await p.keyboard.type(` line ${i}`, { delay: 20 }); });
  await step('toggle', async () => { await click(p, '.led-btn'); await settle(p, 50); });
  await step('dropdown', async () => { await click(p, '[data-pick]'); await settle(p, 100); await click(p, '#dd [role="option"]:not([aria-selected="true"])'); await settle(p, 100); });
  await step('tier', async () => { await click(p, '[data-tier="studio"]'); await settle(p, 100); await click(p, '[data-tier="lab"]'); await settle(p, 100); });
  await step('engine', async () => { await click(p, '#engBtn'); await settle(p, 100); await click(p, '.ep-card[data-backend="mm3"]'); await settle(p, 200); await click(p, '.ep-card[data-backend="ace"]'); await settle(p, 200); await p.keyboard.press('Escape'); });
  await step('remove and add', async () => { await click(p, '.x[data-remove="avoid"]'); await settle(p, 200); await click(p, '#brList [data-additem="avoid"]'); await settle(p, 200); });
  await step('picker', async () => { await p.keyboard.press('F9'); await settle(p, 300); await p.keyboard.press('Escape'); await settle(p, 300); });
  await step('popover', async () => { await click(p, '#ideasBtn'); await settle(p, 100); await p.keyboard.press('Escape'); });
  await step('notes', async () => { await click(p, '#notesBtn'); await settle(p, 300); await p.keyboard.press('Escape'); await settle(p, 200); });
  await step('song menu', async () => { await click(p, '[data-songmenu]'); await settle(p, 100); await p.keyboard.press('Escape'); });
  await step('undo', async () => { await p.mouse.click(5, 400); for (let k = 0; k < 3; k++) await p.keyboard.press('Control+z'); await p.keyboard.press('Control+Shift+z'); });
  if (!NOGEN) await step('generate', async () => { await click(p, '#go'); await settle(p, 200); });
  if (skipped.length) console.log('round', i, 'skipped', skipped.join(', '));
}

(async () => {
  const browser = await launch();
  const { ctx, page, url } = await openWith(browser, DESK);
  await ctx.addInitScript(require('fs').readFileSync(require('path').join(__dirname, 'probe-page.js'), 'utf8'));
  await ctx.addInitScript(() => { window.__perf.on = false; });
  const errs = []; page.on('pageerror', e => errs.push(e.message));
  await page.goto(url); await page.waitForSelector('#go'); await sleep(1500);
  const cdp = await ctx.newCDPSession(page); await cdp.send('Performance.enable');
  const snap = async label => {
    await cdp.send('HeapProfiler.collectGarbage'); await sleep(200); await cdp.send('HeapProfiler.collectGarbage');
    const m = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(x => [x.name, x.value]));
    const dom = await cdp.send('Memory.getDOMCounters');
    const ls = await page.evaluate(() => { let n = 0; for (const k of Object.keys(localStorage)) n += (localStorage.getItem(k) || '').length; return n; });
    const r = { label, heapMB: +(m.JSHeapUsedSize / 1048576).toFixed(2), nodes: dom.nodes, listeners: dom.jsEventListeners, documents: dom.documents, pageNodes: await page.evaluate(() => document.querySelectorAll('*').length), localStorageKB: Math.round(ls / 1024) };
    console.log(JSON.stringify(r)); return r;
  };
  const out = [await snap('after load')];
  const t0 = Date.now();
  for (let i = 1; i <= ROUNDS; i++) { await round(page, i); await sleep(4000); out.push(await snap(`after round ${i}`)); }
  console.log('session', Math.round((Date.now() - t0) / 1000), 's; errors', errs.slice(0, 3));
  fs.mkdirSync(path.dirname(OUT), { recursive: true }); fs.writeFileSync(OUT, JSON.stringify({ desk: DESK, rounds: out, errs }, null, 1));
  await browser.close();
})();
