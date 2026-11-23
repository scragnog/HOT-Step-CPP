// The shared controls no module the page draws today can show yet (they stand in containers later packages build: a
// fader bank, an LCD row, a terminal line, a rack group, the plans deck, the seed's flaps), operated in the parts
// gallery of the test build (src/app/testing/gallery.ts: every part on every surface, bound to the gallery's values:
// n1 0 to 10 in steps of 0.5, default 5, in dB; c Soft, Hard or Wide; b a switch, off; t a text). These run on the
// rebuild only (the concept has no gallery); what each checks is the concept's contract for the control, as
// 04-controls.mjs and 05-modules.mjs check it where the concept draws it. When the containers land, the concept's own
// scenarios take over (docs/09, package 1's notes).

const gallery = (tier = 'lab', vals = {}) => async t => {
  await t.js(() => { window.__g = id => `.tile[data-m="zz-g-tile"] ${id}`; window.__toast = () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''); });
  await t.do(`${tier} gallery`, ([tier, vals]) => { S.backend = 'ace'; S.tier = tier; S.mods = ['zz-g-tile']; S.pedals = []; Object.assign(S.vals, vals); render(); }, [tier, vals]);
};
const G = sel => `.tile[data-m="zz-g-tile"] ${sel}`;

export default [
  { id: 'fader-keys-drag-and-reset', run: async t => {
    await gallery()(t);
    const F = G('[data-fader="zzg.tile.n1"]');
    await t.ok('a fader is a slider with its value and its unity line', () => { const f = document.querySelector(__g('[data-fader="zzg.tile.n1"]')); return (f.getAttribute('role') === 'slider' && f.getAttribute('aria-valuetext') === '5.0 dB' && f.style.getPropertyValue('--u') !== '' && /5\.0 dB/.test(f.parentElement.querySelector('output').textContent)) || f.outerHTML.slice(0, 200); });
    await t.focus(F); await t.press('ArrowUp');
    await t.eq('an arrow steps', () => get('zzg.tile.n1'), 5.5);
    await t.press('Shift+ArrowUp'); await t.eq('Shift steps ten', () => get('zzg.tile.n1'), 10);
    await t.press('Home'); await t.eq('Home', () => get('zzg.tile.n1'), 0);
    await t.mouseDrag(F, [[0, -30], [0, -30]], [0.5, 0.9]);
    await t.ok('a press travels there and the cap follows the hand up', () => +get('zzg.tile.n1') > 3 || get('zzg.tile.n1'));
    await t.dblclick(F); await t.eq('a double-click resets it', () => get('zzg.tile.n1'), 5);
    await t.focus(F); await t.press('End'); await t.press('Delete');
    await t.eq('Delete resets it (S-18)', () => get('zzg.tile.n1'), 5);
  } },
  { id: 'lcd-number-types-and-steps', run: async t => {
    await gallery()(t);
    const N = G('.lcd-num[data-num="zzg.tile.n1"]');
    await t.ok('the big LCD number shows the value as typed, its label beside it', () => document.querySelector(__g('.lcd-num[data-num="zzg.tile.n1"]')).value === '5' && /AMOUNT/.test(document.querySelector(__g('.lcd-numw .lcd-unit')).textContent));
    await t.click(N); await t.type('7.5'); await t.press('Enter');
    await t.eq('typed and Enter sets it', () => get('zzg.tile.n1'), 7.5);
    await t.press('ArrowDown'); await t.eq('an arrow steps it', () => get('zzg.tile.n1'), 7);
    await t.press('Control+a'); await t.type('99'); await t.press('Enter');
    await t.ok('out of range is clamped and said', () => (get('zzg.tile.n1') === 10 && /Amount: 0\.0 to 10\.0\. Set to 10\.0\./.test(__toast())) || [get('zzg.tile.n1'), __toast()]);
    await t.mouseDrag(N, [[0, 15], [0, 15]]);
    await t.ok('dragging it down scrubs it', () => +get('zzg.tile.n1') < 10 || get('zzg.tile.n1'));
  } },
  { id: 'switches-in-a-modules-look', run: async t => {
    await gallery()(t);
    for (const [sel, words] of [['.lcd-btns [data-ledtog="zzg.tile.b"]', 'an LCD key'], ['.flag[data-ledtog="zzg.tile.b"]', 'a terminal flag'], ['.rec[data-ledtog="zzg.tile.b"]', 'a record button'], ['.lock[data-ledtog="zzg.tile.b"]', 'the padlock']]) {
      await t.click(G(sel));
      await t.ok(`${words} switches on, and says so`, s => { const el = document.querySelector(window.__g(s)); return get('zzg.tile.b') === true && el.getAttribute('aria-pressed') === 'true' && el.classList.contains('on') && el.classList.contains('changed'); }, sel);
      await t.click(G(sel));
      await t.ok(`${words} switches off`, s => get('zzg.tile.b') === false && document.querySelector(window.__g(s)).getAttribute('aria-pressed') === 'false', sel);
    }
    await t.ok('each press was one step, named in the page\'s words', () => UNDO.length >= 8 && /Boost/.test(document.querySelector('#undoBtn').getAttribute('aria-label')));
  } },
  { id: 'power-chip-switches-a-group-by-its-number', run: async t => {
    await gallery()(t);
    const P = G('[data-power="zzg.tile.n1"]');
    await t.ok('on while its number is not the off value: lit, its label alone', () => { const b = document.querySelector(__g('[data-power="zzg.tile.n1"]')); return (b.classList.contains('on') && b.getAttribute('aria-pressed') === 'true' && b.textContent === 'Amount') || b.outerHTML; });
    await t.click(P);
    await t.ok('a press sets the off value, and the chip says "· off"', () => { const b = document.querySelector(__g('[data-power="zzg.tile.n1"]')); return (get('zzg.tile.n1') === 0 && !b.classList.contains('on') && /· off$/.test(b.textContent) && /Switch it on/.test(b.title)) || [get('zzg.tile.n1'), b.textContent]; });
  } },
  { id: 'chips-choose-on-paper-and-lcd', run: async t => {
    await gallery()(t);
    await t.click(G('.tchips [data-v="wide"]'));
    await t.ok('a terminal chip chooses', () => get('zzg.tile.c') === 'Wide' && document.querySelector(__g('.tchips [data-v="wide"]')).getAttribute('aria-pressed') === 'true');
    await t.click(G('.lcd-bars [data-v="soft"]'));
    await t.ok('an LCD chip chooses', () => get('zzg.tile.c') === 'Soft' && document.querySelectorAll(__g('.lcd-bars [aria-pressed="true"]')).length === 1);
    await t.click(G('.chipsw [data-v="hard"]'));
    await t.ok('a chip selector chooses, its label marked changed', () => get('zzg.tile.c') === 'Hard' && document.querySelector(__g('.chipsw')).classList.contains('changed'));
  } },
  { id: 'look-back-strip-grows-into-the-past', run: async t => {
    await gallery()(t);
    const L = G('[data-codes="zzg.tile.n1"]');
    await t.ok('the strip lights the window from now backwards', () => { const c = document.querySelector(__g('[data-codes="zzg.tile.n1"]')), lit = c.querySelectorAll('i.w'); return (lit.length === 80 && c.lastElementChild.classList.contains('w') && !c.firstElementChild.classList.contains('w')) || lit.length; });
    await t.focus(L); await t.press('ArrowLeft');
    await t.eq('Left reaches further back', () => get('zzg.tile.n1'), 5.5);
    await t.press('ArrowRight'); await t.press('ArrowRight');
    await t.eq('Right less', () => get('zzg.tile.n1'), 4.5);
    await t.press('End'); await t.eq('End is the whole window', () => get('zzg.tile.n1'), 10);
    await t.mouseDrag(L, [[2, 0]], [0.75, 0.5]);
    await t.ok('a hand along it sets how far back it reaches', () => Math.abs(get('zzg.tile.n1') - 2.5) <= 0.5 || get('zzg.tile.n1'));
  } },
  { id: 'chip-model-and-stamp-pickers', run: async t => {
    await gallery()(t);
    await t.click(G('.pchip[data-pick="zzg.tile.c"]'));
    await t.ok('a chip picker opens the page\'s list on the chosen one', () => document.querySelector('#dd').classList.contains('dd-chip') && document.activeElement.matches('#dd [aria-selected="true"]'));
    await t.press('ArrowDown'); await t.press('Enter');
    await t.ok('the arrows and Enter choose, and focus is back', () => (get('zzg.tile.c') === 'Hard' && document.querySelector('#dd').hidden && document.activeElement.matches('.pchip')) || [get('zzg.tile.c'), document.activeElement.className]);
    await t.click(G('.pk-model[data-pick="zzg.tile.c"]'));
    await t.ok('a model picker opens its rows', () => document.querySelector('#dd').classList.contains('dd-model') && document.querySelectorAll('#dd .opt-model').length === 3);
    await t.press('Escape');
    await t.ok('Escape closes it and refocuses', () => document.querySelector('#dd').hidden && document.activeElement.matches('.pk-model'));
    await t.click(G('.stamp-wrap [data-pick="zzg.tile.c"]'));
    await t.ok('a stamp picker opens its stamps with a filter', () => document.querySelector('#dd').classList.contains('dd-stamp') && !!document.querySelector('#dd #ddq') && !document.querySelector('#dd .stamps-rec'));
    await t.click('#dd .opt-stamp:nth-child(1)');
    await t.click(G('.stamp-wrap [data-pick="zzg.tile.c"]'));
    await t.click('#dd .stamps:not(.stamps-rec .stamps) .opt-stamp:nth-child(2)');
    await t.click(G('.stamp-wrap [data-pick="zzg.tile.c"]'));
    await t.ok('the language used last comes first', () => [...document.querySelectorAll('#dd .stamps-rec .opt')].map(b => b.querySelector('small').textContent).join() === 'Soft' || document.querySelector('#dd').innerHTML.slice(0, 200));
    await t.press('s');
    await t.ok('a letter filters', () => document.activeElement.id === 'ddq' && [...document.querySelectorAll('#dd .stamps:not(.stamps-rec .stamps) .opt')].filter(b => !b.hidden).length === 1);
    await t.press('Tab');
    await t.ok('Tab closes it and carries on from its button', () => document.querySelector('#dd').hidden && document.activeElement !== document.body && !document.querySelector('#dd').contains(document.activeElement) && !document.activeElement.matches('.pk-stamp'));
  } },
  { id: 'lcd-picker-cycles-and-lists', run: async t => {
    await gallery()(t);
    await t.click(G('.pk-lcdwrap [data-cycle="zzg.tile.c|1"]'));
    await t.eq('the arrow beside an LCD picker cycles forward', () => get('zzg.tile.c'), 'Hard');
    await t.click(G('.pk-lcdwrap [data-cycle="zzg.tile.c|1"]')); await t.click(G('.pk-lcdwrap [data-cycle="zzg.tile.c|1"]'));
    await t.eq('and wraps', () => get('zzg.tile.c'), 'Soft');
    await t.click(G('.pk-lcdwrap [data-cycle="zzg.tile.c|-1"]'));
    await t.eq('and back', () => get('zzg.tile.c'), 'Wide');
    await t.click(G('.pk-lcdwrap [data-pick="zzg.tile.c"]'));
    await t.ok('its list is the LCD list', () => document.querySelector('#dd').classList.contains('dd-lcd') && document.querySelectorAll('#dd .opt-lcd').length === 3);
    await t.press('Home'); await t.press('Enter');
    await t.eq('Home and Enter', () => get('zzg.tile.c'), 'Soft');
  } },
];
