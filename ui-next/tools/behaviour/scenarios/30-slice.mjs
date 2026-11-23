// The slice's own claims, as scenarios that hold on the concept page too (run them there with --page): a number
// control's keys, undo and redo by key and by button with what they say, the autosave and its restore, the theme, an
// engine switch from its panel, the tier keys, adding and removing, and adding what is already there. Written for the
// rebuild's acceptance (tools/behaviour/acceptance/slice.json), where the concept's own scenarios did not test a slice
// feature at all; each uses only what both pages share (S, render, get, the markup's data-* hooks).

const desk = (tier, mods, pedals = [], vals = {}) => t => t.do(`${tier} desk`, ([tier, mods, pedals, vals]) => { S.tier = tier; S.mods = mods; S.pedals = pedals; Object.assign(S.vals, vals); render(); }, [tier, mods, pedals, vals]);
const KNOB = '.pedal[data-pid="ss"] [data-knob="pp.ss.str"]';

export default [
  { id: 'knob-keys-keep-the-value-contract', run: async t => {
    await desk('lab', ['post'], ['ss'], { 'pp.ss.on': true })(t);
    await t.focus(KNOB);
    await t.press('ArrowUp');
    await t.eq('an arrow steps once', () => [get('pp.ss.str'), document.querySelector('[data-knob="pp.ss.str"]').getAttribute('aria-valuenow')], [31, '31']);
    await t.press('Shift+ArrowUp');
    await t.eq('Shift steps ten', () => get('pp.ss.str'), 41);
    await t.press('PageDown');
    await t.eq('Page Down steps ten', () => get('pp.ss.str'), 31);
    await t.press('End');
    await t.eq('End goes to the top', () => get('pp.ss.str'), 60);
    await t.press('Shift+ArrowUp');
    await t.ok('past the end it stays there, and says nothing: the clamp message is for a typed value', () => (get('pp.ss.str') === 60 && !/10 to 60/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''))) || [get('pp.ss.str'), (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')]);
    await t.press('Home');
    await t.eq('Home goes to the bottom', () => get('pp.ss.str'), 10);
    await t.ok('each press is a step, named in the page\'s words', () => (UNDO.length >= 5 && /Refine/.test(document.querySelector('#undoBtn span').textContent)) || [UNDO.length, document.querySelector('#undoBtn span').textContent]);
  } },
  { id: 'undo-and-redo-by-key-and-button', run: async t => {
    await desk('lab', ['post'], ['ss'], { 'pp.ss.on': true })(t);
    await t.press('Control+z');
    await t.ok('with nothing to undo, Ctrl+Z says so (A-19)', () => /Nothing to undo/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) || (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''));
    await t.focus(KNOB); await t.press('ArrowUp');
    await t.ok('the step shows on Undo', () => !document.querySelector('#undoBtn').hidden && /Refine/.test(document.querySelector('#undoBtn').getAttribute('aria-label')));
    await t.press('Control+z');
    await t.eq('Ctrl+Z puts it back', () => get('pp.ss.str'), 30);
    await t.ok('and says what it undid, in the toast and to a screen reader', () => (/^Undone: Refine/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && /Undone: Refine/.test(document.querySelector('#say').textContent)) || [(document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''), document.querySelector('#say').textContent]);
    await t.ok('Redo stands ready', () => !document.querySelector('#redoBtn').hidden);
    await t.press('Control+Shift+z');
    await t.eq('Ctrl+Shift+Z does it again', () => get('pp.ss.str'), 31);
    await t.ok('and says so', () => (/^Redone: Refine/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && /Redone: Refine/.test(document.querySelector('#say').textContent)) || [(document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''), document.querySelector('#say').textContent]);
    await t.press('Control+z');
    await t.press('Control+y');
    await t.eq('Ctrl+Y redoes too', () => get('pp.ss.str'), 31);
    await t.click('#undoBtn');
    await t.eq('the Undo button undoes', () => get('pp.ss.str'), 30);
    await t.click('#redoBtn');
    await t.eq('the Redo button redoes', () => get('pp.ss.str'), 31);
    await t.click('#toast .toast-act');
    await t.eq('the toast\'s Undo undoes', () => get('pp.ss.str'), 30);
  } },
  { id: 'the-desk-is-kept-and-comes-back', run: async t => {
    await desk('studio', ['avoid', 'post'], ['ss'], { 'pp.ss.on': true })(t);
    await t.focus(KNOB); await t.press('ArrowUp'); await t.press('ArrowUp');
    await t.do('let the autosave run', () => {}, null, 1200);
    await t.reload();
    await t.eq('after a reload the desk is as it was: tier, modules, pedals, values', () => [S.tier, S.mods.join(), S.pedals.join(), get('pp.ss.str')], ['studio', 'avoid,post', 'ss', 32]);
    await t.ok('and drawn', () => !!document.querySelector('#bento .tile[data-m="avoid"]') && document.querySelector('[data-knob="pp.ss.str"]')?.getAttribute('aria-valuenow') === '32');
  } },
  { id: 'theme-is-chosen-kept-and-keyed', run: async t => {
    await t.click('#themeSeg [data-themeset="light"]');
    await t.eq('Light', () => [document.documentElement.dataset.theme, document.querySelector('#themeSeg [aria-checked="true"]').dataset.themeset], ['light', 'light']);
    await t.click('#themeSeg [data-themeset="dark"]');
    await t.eq('Dark', () => document.documentElement.dataset.theme, 'dark');
    await t.reload();
    await t.eq('kept over a reload', () => [document.documentElement.dataset.theme, document.querySelector('#themeSeg [aria-checked="true"]').dataset.themeset], ['dark', 'dark']);
    await t.focus('#themeSeg [data-themeset="dark"]'); await t.press('ArrowRight');
    await t.eq('the arrows go round to System, which follows the system', () => [document.documentElement.dataset.theme ?? null, document.activeElement.dataset.themeset], [null, 'system']);
  } },
  { id: 'engine-switch-from-its-panel', run: async t => {
    await desk('studio', ['avoid', 'post'], ['ss'])(t);
    await t.click('#engBtn');
    await t.ok('the panel opens on the engine in use', () => !document.querySelector('#engPanel').hidden && document.activeElement?.matches('.ep-card[aria-checked="true"]'));
    await t.click('#engPanel .ep-card[data-backend="sa3"]');
    await t.eq('a card switches the engine', () => S.backend, 'sa3');
    await t.ok('the toast says where the desk is now, to a screen reader too, and carries Undo', () => (/^On Stable Audio 3/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && /On Stable Audio 3/.test(document.querySelector('#say').textContent) && !!document.querySelector('#toast .toast-act')) || [(document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''), document.querySelector('#say').textContent]);
    await t.ok('Avoid, which Stable Audio 3 does not use, waits parked', () => !!document.querySelector('#bento .pchip[data-pm="avoid"]'));
    await t.click('#toast .toast-act');
    await t.eq('Undo goes back', () => S.backend, 'ace');
  } },
  { id: 'tier-keys-and-the-tier-switch', run: async t => {
    await desk('studio', ['avoid', 'post'], ['ss'])(t);
    const n = await t.js(() => UNDO.length);
    await t.press('Alt+3');
    await t.eq('Alt+3 is Lab', () => S.tier, 'lab');
    await t.press('Alt+1');
    await t.eq('Alt+1 is Quick', () => S.tier, 'quick');
    await t.focus('#tiers [aria-checked="true"]'); await t.press('ArrowRight');
    await t.eq('the arrows move along the switch', () => [S.tier, document.activeElement.dataset.tier], ['studio', 'studio']);
    await t.click('#tiers [data-tier="lab"]');
    await t.eq('a click chooses', () => S.tier, 'lab');
    await t.eq('a tier is a view, not an undo step', n0 => UNDO.length === n0, true, n);
  } },
  { id: 'add-and-remove-with-undo', run: async t => {
    await desk('studio', [])(t);
    await t.click('.desk-empty [data-additem="tempo"]');
    await t.ok('a starter adds its module, and focus lands on it (Q-16)', () => (S.mods.includes('tempo') && document.activeElement?.dataset.move === 'tempo') || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('#browser .bcard[data-additem="avoid"]');
    await t.ok('a card adds its module', () => S.mods.includes('avoid') && !!document.querySelector('#bento .tile[data-m="avoid"]'));
    await t.do('add it again', () => addItem('avoid'), null, 50);
    await t.ok('adding what is already there rings it', () => document.querySelector('#bento .tile[data-m="avoid"]').classList.contains('flash'));
    await t.click('.tile[data-m="avoid"] .tile-head .x');
    await t.ok('× removes it, says so with Undo, to a screen reader too', () => (!S.mods.includes('avoid') && /^Removed Avoid/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && /Removed Avoid/.test(document.querySelector('#say').textContent) && !!document.querySelector('#toast .toast-act')) || [S.mods, (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''), document.querySelector('#say').textContent]);
    await t.ok('focus goes to the neighbouring tile', () => document.activeElement?.dataset.move === 'tempo' || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('#toast .toast-act');
    await t.ok('Undo brings it back where it was', () => S.mods.join() === 'tempo,avoid' || S.mods.join());
    await t.do('add a pedal', () => addItem('pd:ss'));
    await t.ok('a pedal brings the Pedalboard', () => S.mods.includes('post') && S.pedals.includes('ss'));
    await t.do('add it again', () => addItem('pd:ss'), null, 50);
    await t.ok('adding a pedal already there rings it', () => document.querySelector('.pedal[data-pid="ss"]').classList.contains('here'));
    await t.click('.pedal[data-pid="ss"] [data-pedalx="ss"]');
    await t.ok('taking the last pedal off takes the Pedalboard too, said', () => (!S.mods.includes('post') && /^Took StableStep off the board/.test((document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')) && /Took StableStep off the board/.test(document.querySelector('#say').textContent)) || [S.mods, (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || '')]);
  } },
];
