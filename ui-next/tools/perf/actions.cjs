// The actions measured, shared by perf.cjs (timings, product build) and profile.cjs (causes, instrumented build).
const sleep = ms => new Promise(r => setTimeout(r, ms));
/* ------------------------------------------------------------------ helpers used by the actions */
const vis = (page, sel) => page.evaluate(sel => { const el = [...document.querySelectorAll(sel)].find(e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden'); if (!el) return null; el.scrollIntoView({ block: 'center' }); const r = el.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height }; }, sel);
const has = async (page, sel) => !!(await vis(page, sel));
const click = async (page, sel) => { const b = await vis(page, sel); if (!b) throw new Error('no ' + sel); await page.mouse.click(b.x, b.y); };
const focus = (page, sel) => page.evaluate(sel => { const el = [...document.querySelectorAll(sel)].find(e => e.getClientRects().length); el.scrollIntoView({ block: 'center' }); el.focus(); }, sel);
const settle = (page, ms = 250) => page.evaluate(ms => window.__perf.settle(ms), ms);
/** A pointer drag paced at ~60 Hz: down, `n` moves of (dx, dy), up. */
async function drag(page, b, n, dx, dy, pace = 16) {
  await page.mouse.move(b.x, b.y); await page.mouse.down();
  let t = Date.now();
  for (let i = 1; i <= n; i++) { await page.mouse.move(b.x + dx * i, b.y + dy * i); const w = t + pace - Date.now(); if (w > 0) await sleep(w); t = Date.now(); }
  await page.mouse.up();
}
const firstOf = async (page, sels) => { for (const s of sels) if (await has(page, s)) return s; return null; };
const KNOB = ['.tile[data-m="sampler"] .knob', '.tile:not([data-m="post"]) .knob', '.knob'];
const PEDAL_KNOB = ['.pedal[data-pid="ss"] .knob', '.pedal .knob'];
const TOGGLE = ['.led-btn', '#quickLyrics .ltog, #quickLyrics button[aria-pressed]', 'button[aria-pressed]:not([data-tier])'];
const MOD_FOR = { empty: 'tempo', typical: 'avoid', full: 'avoid' };
const removeVia = async (page, id) => { if (await has(page, `.x[data-remove="${id}"]`)) { await click(page, `.x[data-remove="${id}"]`); await settle(page, 600); } };

/* ------------------------------------------------------------------ the actions
   prep: unmeasured; act: measured; skip: returns a reason when the action does not apply to this desk */
const A = [];
const def = (name, o) => A.push({ name, ...o });

def('prompt-type', { prep: async p => { await click(p, '#prompt'); }, act: async p => { await p.keyboard.type('warm tape hiss, bells', { delay: 40 }); } });
def('lyrics-type', { prep: async p => { await click(p, 'textarea[data-bind="lyrics.text"]'); }, act: async p => { await p.keyboard.type('[Verse] soft light on the water', { delay: 40 }); } });
def('knob-drag', { skip: async p => (await firstOf(p, KNOB)) ? null : 'no knob', act: async p => { const b = await vis(p, await firstOf(p, KNOB)); await drag(p, b, 40, 0, -2); }, drag: true });
def('pedal-knob-drag', { skip: async p => (await firstOf(p, PEDAL_KNOB)) ? null : 'no pedal', act: async p => { const b = await vis(p, await firstOf(p, PEDAL_KNOB)); await drag(p, b, 40, 0, -2); }, drag: true });
def('knob-keys', { skip: async p => (await firstOf(p, KNOB)) ? null : 'no knob', prep: async p => focus(p, await firstOf(p, KNOB)), act: async p => { for (let i = 0; i < 10; i++) { await p.keyboard.press('ArrowUp'); await sleep(60); } } });
/* a fader is drawn by a loaded adapter (Adapters, a DiT adapter's cartridge): the prep loads the first one offered */
const loadAdapter = async p => { if (await has(p, '[data-fader]')) return; await p.locator('.tile[data-m="adapters"] .sbtn').first().click(); await settle(p, 300); await p.locator('#pop [data-loadad]').first().click(); await settle(p, 600); };
def('fader-drag', { skip: async p => (await has(p, '.tile[data-m="adapters"]')) ? null : 'no Adapters tile on this desk', prep: loadAdapter, act: async p => { const b = await vis(p, '[data-fader="ag.self"], [data-fader]'); await drag(p, b, 40, 0, -2); }, drag: true });
def('toggle', { skip: async p => (await firstOf(p, TOGGLE)) ? null : 'no toggle', act: async p => { const s = await firstOf(p, TOGGLE); await click(p, s); await settle(p, 100); await click(p, s); } });
def('dropdown-open', { skip: async p => (await has(p, '[data-pick]')) ? null : 'no dropdown', act: async p => click(p, '[data-pick]') });
/* the first open of a page load also loads a web font the list uses (a whole-document layout); a second open shows the list's own cost */
def('dropdown-reopen', { skip: async p => (await has(p, '[data-pick]')) ? null : 'no dropdown', prep: async p => { await click(p, '[data-pick]'); await settle(p, 600); await p.keyboard.press('Escape'); await settle(p, 300); }, act: async p => click(p, '[data-pick]') });
def('dropdown-choose', { skip: async p => (await has(p, '[data-pick]')) ? null : 'no dropdown', prep: async p => { await click(p, '[data-pick]'); await settle(p, 300); }, act: async p => click(p, '#dd [role="option"]:not([aria-selected="true"])') });
def('add-from-panel', { prep: async (p, d) => removeVia(p, MOD_FOR[d]), act: async (p, d) => click(p, `#brList [data-additem="${MOD_FOR[d]}"]`) });
def('add-from-picker', { prep: async (p, d) => { await removeVia(p, MOD_FOR[d]); await p.keyboard.press('F9'); await settle(p, 500); }, act: async (p, d) => click(p, `#pickerWrap .fl-it[data-pkid="${MOD_FOR[d]}"]`) });
def('remove', { skip: async p => (await has(p, '.x[data-remove="tempo"]')) ? null : 'nothing to remove', act: async p => click(p, '.x[data-remove="tempo"]') });
def('reorder-drag', { skip: async p => (await p.locator('#bento > .tile[data-m] .tile-head[data-drag]').count()) >= 3 ? null : 'too few tiles', drag: true,
  act: async p => {
    const a = await vis(p, '.tile[data-m="tempo"] .tile-head'), b = await p.evaluate(() => { const t = [...document.querySelectorAll('#bento > .tile[data-m]')].filter(x => x.dataset.m !== 'tempo' && x.dataset.m !== 'lyrics'); const r = t[2].getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + 30 }; });
    await p.mouse.move(a.x - 60, a.y); await p.mouse.down();
    const n = 30; for (let i = 1; i <= n; i++) { await p.mouse.move(a.x - 60 + (b.x - a.x + 60) * i / n, a.y + (b.y - a.y) * i / n); await sleep(16); }
    await p.mouse.up();
  } });
/* dragging a card from the Modules panel onto the desk (HTML5 drag and drop): a module (its preview is the tile drawn
   by React off the page, dnd.tsx drawn()) and a pedal (onto the Pedalboard, or the Pedalboard previewed when it is not
   on the desk) */
const dragCard = async (p, id) => {
  const a = await vis(p, `#brList [data-drag="${id}"]`); if (!a) throw new Error('no card ' + id);
  const b = await p.evaluate(() => { const r = document.getElementById('bento').getBoundingClientRect(); return { x: r.x + r.width / 2, y: Math.min(innerHeight - 40, Math.max(r.y + 40, 300)) }; });
  await p.mouse.move(a.x, a.y); await p.mouse.down();
  const n = 30; for (let i = 1; i <= n; i++) { await p.mouse.move(a.x + (b.x - a.x) * i / n, a.y + (b.y - a.y) * i / n); await sleep(16); }
  await p.mouse.up();
};
const MOD_DRAG = { empty: 'sampler', typical: 'avoid', full: 'avoid' }, PEDAL_DRAG = { empty: 'pd:lift', typical: 'pd:lift', full: 'pd:lift' };
def('module-drag-add', { drag: true, prep: async (p, d) => removeVia(p, MOD_DRAG[d]), act: async (p, d) => dragCard(p, MOD_DRAG[d]) });
def('pedal-drag-add', { drag: true, skip: async (p, d) => d === 'full' ? 'every pedal is on the board' : null, act: async (p, d) => dragCard(p, PEDAL_DRAG[d]) });
def('reorder-keys', { skip: async p => (await has(p, '.grip[data-move="tempo"]')) && (await p.locator('#bento > .tile[data-m]').count()) >= 3 ? null : 'too few tiles', prep: async p => focus(p, '.grip[data-move="tempo"]'), act: async p => { for (const k of ['ArrowRight', 'ArrowRight', 'ArrowLeft']) { await p.keyboard.press(k); await sleep(150); } } });
def('tier-switch', { act: async p => { const t = await p.evaluate(() => document.querySelector('[data-tier][aria-checked="true"]').dataset.tier); const to = t === 'lab' ? 'studio' : 'lab'; await click(p, `[data-tier="${to}"]`); await settle(p, 400); await click(p, `[data-tier="${t}"]`); } });
def('engine-open', { act: async p => click(p, '#engBtn') });
def('engine-switch', { prep: async p => { await click(p, '#engBtn'); await settle(p, 300); }, act: async p => click(p, '.ep-card[data-backend="mm3"]') });
def('engine-switch-back', { prep: async p => { await click(p, '#engBtn'); await settle(p, 300); await click(p, '.ep-card[data-backend="mm3"]'); await settle(p, 600); }, act: async p => click(p, '.ep-card[data-backend="ace"]') });
def('engine-close', { prep: async p => { await click(p, '#engBtn'); await settle(p, 300); }, act: async p => p.keyboard.press('Escape') });
def('undo', { prep: async p => { await click(p, await firstOf(p, TOGGLE)); await settle(p, 300); await p.mouse.click(5, 300); }, act: async p => p.keyboard.press('Control+z') });
def('redo', { prep: async p => { await click(p, await firstOf(p, TOGGLE)); await settle(p, 300); await p.mouse.click(5, 300); await p.keyboard.press('Control+z'); await settle(p, 300); }, act: async p => p.keyboard.press('Control+Shift+z') });
def('picker-open', { act: async p => p.keyboard.press('F9') });
def('picker-close', { prep: async p => { await p.keyboard.press('F9'); await settle(p, 500); }, act: async p => p.keyboard.press('Escape') });
def('popover-open', { act: async p => click(p, '#ideasBtn') });
def('popover-close', { prep: async p => { await click(p, '#ideasBtn'); await settle(p, 300); }, act: async p => p.keyboard.press('Escape') });
def('notes-open', { act: async p => click(p, '#notesBtn') });
def('notes-close', { prep: async p => { await click(p, '#notesBtn'); await settle(p, 500); }, act: async p => p.keyboard.press('Escape') });
def('song-menu-open', { act: async p => click(p, '[data-songmenu]') });
def('song-menu-close', { prep: async p => { await click(p, '[data-songmenu]'); await settle(p, 300); }, act: async p => p.keyboard.press('Escape') });
def('generate-press', { act: async p => click(p, '#go') });
/* the queue progressing: one run of the simulated engine (3.6 s), measured from the press until it is done */
def('generate-run', { long: true, act: async p => { await click(p, '#go'); await p.evaluate(() => window.__perf.startFrames()); await sleep(4500); } });

module.exports = { A, sleep, settle, click, vis, has, focus, drag };
