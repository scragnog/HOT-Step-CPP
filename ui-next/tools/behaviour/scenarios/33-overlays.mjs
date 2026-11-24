// The picker, the popover, tooltips, touch, the focus ring, the keys that walk the desk and drag and drop (package 8),
// as scenarios. Each is an older scenario's own checks with the steps that needed another package's part taken out
// (the prompt, the workflows list, the seed manager, the takes' reels, the generation queue).

const desk = (tier, mods, pedals = [], backend = 'ace') => t => t.do(`${tier} desk`, ([tier, mods, pedals, backend]) => {
  if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
  S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); render();
}, [tier, mods, pedals, backend], 400);

export default [
  /* 07-overlays-keys/picker without its Ctrl+Enter check (the generation queue, package 6) */
  { id: 'picker-opens-filters-walks-and-adds', w: 1600, run: async t => {
    await desk('studio', ['tempo'])(t);
    await t.click('[data-openpicker]');
    await t.ok('the Picker button opens it, focus in the filter', () => !document.querySelector('#pickerWrap').hidden && document.activeElement.id === 'pkSearch');
    await t.press('Escape');
    await t.ok('Escape closes it and returns focus', () => document.querySelector('#pickerWrap').hidden && !!document.activeElement.closest('[data-openpicker]'));
    await t.press('F9');
    await t.ok('F9 opens it', () => !document.querySelector('#pickerWrap').hidden);
    await t.type('see');
    await t.ok('typing filters and ranks', () => document.querySelector('#pkCount').textContent === '1 match' && !!document.querySelector('.fl-it[data-pkid="seed"]:not(.nomatch)'));
    await t.press('Enter');
    await t.ok('Enter adds the first match and keeps picking', () => S.mods.includes('seed') && /Added Seed/.test(document.querySelector('#pkInfo').textContent));
    await t.press('Escape');
    await t.ok('Escape with a filter clears the filter first', () => document.querySelector('#pkSearch').value === '' && !document.querySelector('#pickerWrap').hidden);
    await t.click('#pkBar [data-pkcat="Pedals"]');
    await t.ok('a category narrows the wall', () => S.pkCat === 'Pedals' && document.querySelectorAll('#pkLeft .fl-it.dim').length > 0);
    await t.click('#pkBar [data-pkcat="All"]');
    await t.focus('#pkSearch'); await t.press('ArrowDown');
    await t.ok('ArrowDown goes into the wall', () => document.activeElement.matches('.fl-it'));
    const first = await t.js(() => document.activeElement.dataset.pkid);
    await t.press('ArrowRight');
    await t.ok('arrows move between drawings', id => document.activeElement.dataset.pkid !== id && document.activeElement.matches('.fl-it'), first);
    await t.press('End'); await t.press('Home');
    await t.ok('Home and End', id => document.activeElement.dataset.pkid === id, first);
    await t.press('Tab');
    await t.ok('Tab cycles inside the picker', () => document.querySelector('#pickerWrap').contains(document.activeElement));
    await t.focus('.fl-it[data-pkid="avoid"]'); await t.press(' ');
    await t.ok('Space adds the focused drawing', () => S.mods.includes('avoid'));
    await t.focus('.fl-it[data-pkid="guidance"]'); await t.press('Enter');
    await t.ok('a Lab module offers Lab in the info line', () => !!document.querySelector('#pkInfo [data-labadd="guidance"]') && document.activeElement.matches('#pkInfo [data-labadd]'));
    await t.press('Enter');
    await t.ok('which switches to Lab and adds it, the picker staying open', () => S.tier === 'lab' && S.mods.includes('guidance') && !document.querySelector('#pickerWrap').hidden && /Lab now/.test(document.querySelector('#pkInfo').textContent));
    await t.click('.fl-it[data-pkid="tempo"]');
    await t.ok('a click on one already on the desk closes the picker and shows it', () => document.querySelector('#pickerWrap').hidden);
    await t.press('F9'); await t.press('Tab'); await t.press('Control+z');
    await t.ok('Ctrl+Z under the picker undoes nothing and says so', () => S.mods.includes('guidance') && /Close this first/.test(document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.hover('.fl-it[data-pkid="timbre"]');
    await t.ok('pointing shows its description', () => /Timbre reference/.test(document.querySelector('#pkInfo').textContent));
    await t.click('#pkClose');
    await t.ok('× closes it', () => document.querySelector('#pickerWrap').hidden);
    await t.press('F9'); await t.click('#pickerWrap .fl-col h2');
    await t.ok('a click on the backdrop closes it', () => document.querySelector('#pickerWrap').hidden);
    await t.press('F9'); await t.press('F9');
    await t.ok('F9 toggles it', () => document.querySelector('#pickerWrap').hidden);
  } },
  /* 02-dnd-panels/drag-move-tile without the workflow's changed mark (package 7) */
  { id: 'drag-moves-a-tile-and-cancels', w: 1600, run: async t => {
    await desk('studio', ['tempo', 'seed', 'takes', 'sampler'])(t);
    await t.drag('.tile[data-m="tempo"] [data-drag="move:tempo"]', '.tile[data-m="sampler"]', { at: [0.9, 0.95] });
    await t.ok('a tile dragged by its header moves', () => S.mods.indexOf('tempo') > 0 || S.mods.join());
    await t.ok('the move is one step to undo, and the drag leaves nothing behind', () => UNDO.length === 1 && !document.querySelector('#bento .ghost, .tile.moving, .tile[data-mo]') && !document.body.classList.contains('dnd'));
    const moved = await t.js(() => S.mods.join());
    await t.drag('.tile[data-m="takes"] [data-drag="move:takes"]', '.tile[data-m="seed"]', { drop: false });
    await t.eq('a cancelled move puts it back', () => [S.mods.join(), !!document.querySelector('.tile.moving'), UNDO.length], [moved, false, 1]);
    await t.drag('#browser .bcard[data-drag="pd:gain"]', '.tile[data-m="seed"]', { drop: false });
    await t.ok('a cancelled pedal drag adds nothing and leaves nothing glowing', () => !S.pedals.length && !document.querySelector('.tile.glow, #bento .ghost'));
  } },
  /* the Lab offer in the popover: Escape and Tab out of it, as every popover (17-popover.js, 16-overlays.js) */
  { id: 'lab-offer-popover-keys', w: 1600, run: async t => {
    await desk('studio', ['sampler'])(t);
    await t.focus('#browser .bcard[data-additem="guidance"]'); await t.press('Enter');
    await t.ok('Enter on a Lab card offers Lab, focus on its first choice', () => !document.querySelector('#pop').hidden && document.activeElement.matches('#pop [data-labadd="guidance"]'));
    await t.press('Escape');
    await t.ok('Escape closes it and gives focus back to the card', () => document.querySelector('#pop').hidden && document.activeElement.matches('.bcard[data-additem="guidance"]') && !S.mods.includes('guidance'));
    await t.press('Enter'); await t.press('Tab'); await t.press('Tab');
    await t.ok('Tab past its last button closes it', () => document.querySelector('#pop').hidden);
    await t.click('#browser .bcard[data-additem="guidance"]');
    await t.click('#bento');
    await t.ok('a click outside closes it', () => document.querySelector('#pop').hidden && !S.mods.includes('guidance'));
  } },
  /* 07-overlays-keys/focus-tooltips-keys through the top bar and a knob, not the prompt and the workflows list */
  { id: 'focus-ring-and-keyboard-tips', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['sampler'])(t);
    await t.focus('#engBtn'); await t.press('Tab');
    await t.ok('Tab turns on the keyboard focus ring', () => document.documentElement.classList.contains('kbnav'));
    await t.do('wait', () => {}, null, 600);
    await t.ok('a control reached by keyboard shows its title as a tip, every digit kept', () => /the 31\.8 GB card/.test(document.querySelector('#tip')?.textContent || '') || document.querySelector('#tip')?.textContent);
    await t.press('Escape');
    await t.ok('Escape hides the tip', () => document.querySelector('#tip').hidden);
    await t.click('.top .brand');
    await t.ok('a pointer press turns the ring off', () => !document.documentElement.classList.contains('kbnav'));
    await t.click('.tile[data-m="sampler"] .knob[data-knob="samp.cfg"]');
    await t.ok('a pointer-focused knob does not show the ring', () => document.activeElement.matches('.knob') && !document.activeElement.matches(':focus-visible'));
    await t.press('ArrowUp');
    await t.ok('the first arrow key brings it up', () => document.activeElement.matches('.knob:focus-visible'));
  } },
  /* 09-undo-storage/hover-tooltip on the save state, which both pages draw (the import button is the workflows') */
  { id: 'hover-tip-holds-the-title-back', run: async t => {
    await t.hover('#saveState', 500);
    await t.ok('a resting pointer gets the page\'s own tip, and the browser\'s is held back (C-31)', () => { const tip = document.querySelector('#tip'), b = document.querySelector('#saveState'); return (!!tip && !tip.hidden && tip.textContent === 'Your desk, songs and settings are kept in this browser as you work' && !b.hasAttribute('title')) || [tip?.hidden, tip?.textContent, b.getAttribute('title')]; });
    await t.hover('#ask', 50);
    await t.ok('it goes when the pointer leaves, and the title comes back', () => document.querySelector('#tip').hidden && document.querySelector('#saveState').title === 'Your desk, songs and settings are kept in this browser as you work');
  } },
  /* 18-tips-touch/tip-waits-out-the-quiet-after-a-press and 14-lab-desk/tip-keeps-off-the-read-out, on a knob */
  { id: 'knob-tip-waits-and-keeps-off-its-value', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['sampler', 'guidance'])(t);
    await t.click('.tile[data-m="guidance"] .tile-head h3');
    await t.hover('.tile[data-m="sampler"] [data-knob="samp.cfg"]', 700);
    await t.ok('no tip yet, inside the quiet after the press', () => { const tip = document.querySelector('#tip'); return !tip || tip.hidden; });
    await t.do('rest', () => {}, null, 2000);
    await t.ok('the tip opens once the quiet is over, the pointer still resting (L-06)', () => { const tip = document.querySelector('#tip'); return (!!tip && !tip.hidden && tip.textContent.length > 3) || 'no tip'; });
    await t.ok('and it never covers the knob\'s value (L-06)', () => { const a = document.querySelector('#tip').getBoundingClientRect(), b = document.querySelector('[data-out="samp.cfg"]').getBoundingClientRect(); return a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom || JSON.stringify([a, b]); });
  } },
  /* 14-lab-desk/tip-keeps-off-the-read-out where it matters: an idle pedal's LED, whose tip would open over the pedal's
     knobs and labels above it */
  { id: 'tip-keeps-off-the-pedals-read-outs', w: 1440, h: 900, run: async t => {
    await desk('lab', ['post'], ['vst', 'gain'])(t);
    await t.hover('.pedal[data-pid="vst"] .led', 900);
    await t.ok('the idle LED tip opens clear of everything its pedal reads out (L-06)', () => { const tip = document.querySelector('#tip'); if (!tip || tip.hidden) return 'no tip'; const a = tip.getBoundingClientRect();
      const hit = [...document.querySelectorAll('.pedal[data-pid="vst"] :is(output, [data-out], .lcd, .kl)')].map(e => e.getBoundingClientRect()).filter(b => b.width && !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom));
      return !hit.length || JSON.stringify([a, hit]); });
  } },
  /* 07-overlays-keys/touch-tips and 18-tips-touch/a-held-press-still-presses, on the tier switch */
  { id: 'a-held-finger-shows-the-tip-and-still-presses', w: 820, h: 1180, touch: true, run: async t => {
    await t.do('long press', () => { const b = document.querySelector('#saveState'), r = b.getBoundingClientRect(); b.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch', clientX: r.left + 4, clientY: r.top + 4 })); }, null, 700);
    await t.ok('pressing and holding shows the tip', () => { const tip = document.querySelector('#tip'); return (!!tip && !tip.hidden && /kept in this browser/.test(tip.textContent)) || 'no tip'; });
    await t.do('later', () => {}, null, 3000);
    await t.ok('and it goes by itself', () => document.querySelector('#tip').hidden);
    const r = await t.js(() => { const b = document.querySelector('#tiers [data-tier="studio"]').getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; });
    const cdp = await t.page.context().newCDPSession(t.page);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: r.x, y: r.y }] }); await t.settle(800);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await t.settle(300); await cdp.detach();
    await t.eq('a tier held for 0.8 s is chosen (S-01)', () => S.tier, 'studio');
  } },
  /* 16-engines-sizes-input/touch-scroll-or-control on a desk both pages draw */
  { id: 'a-swipe-scrolls-and-a-rest-turns', w: 390, h: 844, touch: true, run: async t => {
    await desk('lab', ['tempo', 'sampler', 'guidance'])(t);
    const cdp = await t.page.context().newCDPSession(t.page);
    const swipe = async (sel, dx, dy, hold) => { const r = await t.js(sel => { const el = document.querySelector(sel); el.scrollIntoView({ block: 'center', behavior: 'instant' }); const b = el.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; }, sel); await t.settle(100);
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: r.x, y: r.y, id: 1 }] }); if (hold) await t.settle(hold);
      for (let i = 1; i <= 10; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: r.x + dx * i / 10, y: r.y + dy * i / 10, id: 1 }] }); await t.page.waitForTimeout(16); }
      await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await t.settle(200); };
    const before = await t.js(() => JSON.stringify(S.vals));
    await swipe('[data-knob="samp.cfg"]', 0, -160); await swipe('[data-ruler="tempo.dur"]', 0, -160); await swipe('[data-num="samp.steps"]', 0, -60); await swipe('.xy[data-xy]', 0, -120);
    await t.ok('a swipe up or down that starts on a knob, ruler, number box or pad changes none of them (S-29)', b => JSON.stringify(S.vals) === b, before);
    await swipe('[data-knob="samp.cfg"]', 0, -60, 300);
    await t.ok('resting a finger on a knob first, then dragging, turns it', () => +get('samp.cfg') > 9);
    await swipe('[data-ruler="tempo.dur"]', -60, 4);
    await t.ok('a sideways drag on the ruler moves it', () => S.vals['tempo.dur.auto'] === false);
    await t.eq('each touch edit is an undo step of its own', () => UNDO.length, 2);
    await cdp.detach();
  } },
  /* 16-engines-sizes-input/f6-moves-between-modules from the tier switch, on a desk set by the scene */
  { id: 'f6-walks-the-desk', run: async t => {
    await desk('lab', ['tempo', 'avoid', 'guidance', 'sampler', 'seed', 'takes'])(t);
    await t.focus('#tiers [aria-checked="true"]'); await t.press('F6');
    const first = await t.js(() => document.activeElement.closest('.tile')?.dataset.m);
    await t.ok('F6 from before the desk enters it at its first module (S-33)', () => !!document.activeElement.closest('#bento > .tile'));
    await t.press('F6'); await t.press('F6');
    await t.ok('each press is the next module in reading order', f => { const ts = [...document.querySelectorAll('#bento > .tile[data-m]')]; return ts.indexOf(document.activeElement.closest('.tile')) === ts.findIndex(x => x.dataset.m === f) + 2; }, first);
    await t.press('Shift+F6');
    await t.ok('Shift+F6 goes back one', f => { const ts = [...document.querySelectorAll('#bento > .tile[data-m]')]; return ts.indexOf(document.activeElement.closest('.tile')) === ts.findIndex(x => x.dataset.m === f) + 1; }, first);
    await t.press('F9'); await t.press('F6');
    await t.ok('not while the picker is open', () => document.querySelector('#pickerWrap').contains(document.activeElement));
  } },
];
