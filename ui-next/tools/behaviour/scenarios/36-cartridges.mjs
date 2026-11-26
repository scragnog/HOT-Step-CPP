// The instruments of package 3b, as scenarios: the
// Adapters' cartridges (the adapter list, the stack, section masks and the runtime they force, phase windows, removing
// with the directives renumbered, the planner adapter, Quick's flat cartridge, YuE2's pair and the column beside one
// cartridge), and the planner's XY pad (its top-p off rail, its fade with the planner off). Where a hook's value is a choice,
// it is compared without case.

const desk = (tier, mods, vals = {}, backend = 'ace', more = {}) => async t => {
  await t.js(() => { window.__c = { toast: () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''),
    undo: () => (document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent),
    tile: () => document.querySelector('.tile[data-m="adapters"]'), text: s => document.querySelector(s)?.textContent.replace(/ /g, ' ') ?? null }; });
  await t.do(`${backend} ${tier} desk`, ([tier, mods, vals, backend, more]) => {
    if (typeof lyricsBlock === 'function') window.lyricsBlock = () => '';
    if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
    S.backend = backend; S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, more); render();
  }, [tier, mods, vals, backend, more]);
};
const TWO = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }];
const LYRICS = '[Verse]\na\nb\n[Chorus]\nc\n[Verse 2]\nd';

export default [
  { id: 'adapter-list-loads-and-stacks', run: async t => {
    await desk('studio', ['adapters'], {}, 'ace', { slots: [TWO[0]] })(t);
    await t.ok('one cartridge: its words and the way to a second stand beside it (Q-19)', () => !!__c.tile().querySelector('.rig-right .rig-side [data-cartadd2]') && /Two small matrices/.test(__c.text('.tile[data-m="adapters"] .rig-side p')));
    await t.click('.tile[data-m="adapters"] [data-cartadd2]');
    await t.eq('the list opens under it, the files of its kind but the one loaded', () => [document.querySelector('#pop').hidden, document.querySelectorAll('#pop [data-loadad]').length, !!document.querySelector('#pop [data-loadad^="tape-lofi-v2|"]')], [false, 5, false]);
    await t.ok('focus is on its first file', () => document.activeElement?.matches('#pop .cartpick'));
    await t.press('ArrowRight');
    await t.ok('the arrows move through the grid (L-22)', () => document.activeElement === document.querySelectorAll('#pop .cartpick')[1]);
    await t.press('Escape');
    await t.ok('Escape closes it and gives focus back to its button', () => document.querySelector('#pop').hidden && document.activeElement?.matches('[data-cartadd2]'));
    await t.click('.tile[data-m="adapters"] [data-cartadd2]');
    await t.click('#pop [data-loadad="folk-voice|LoKr"]');
    await t.ok('a second adapter loaded while Stack is Single stacks them, in one step (L-22)', () => S.slots.length === 2 && !!document.querySelector('.tile[data-m="adapters"] .stackbar') && document.querySelectorAll('.tile[data-m="adapters"] .cart:not(.empty)').length === 2);
    await t.eq('the step is named for the file', () => __c.undo(), 'Load folk-voice');
    await t.ok('focus lands on its fader (L-07)', () => document.activeElement?.matches('[data-m="adapters"] [data-fader="s1"]') || document.activeElement?.outerHTML.slice(0, 80));
    await t.ok('in a stack each cartridge says what it sends', () => __c.text('.tile[data-m="adapters"] [data-eff="0"]') === 'sends 0.38' && __c.text('.tile[data-m="adapters"] .cart .slot') === '#1 · DiT');
  } },
  { id: 'unused-adapters-and-stack-them', run: async t => {
    await desk('studio', ['adapters'], {}, 'ace', { slots: TWO })(t);
    await t.ok('while Stack is Single one cartridge is used, no empty slot beside it', () => __c.tile().querySelectorAll('.cart:not(.empty)').length === 1 && !__c.tile().querySelector('[data-cartadd]'));
    await t.eq('the other is said loaded but not used, with the way to stack them (A-14)', () => __c.text('.tile[data-m="adapters"] .unused .lbl'), 'folk-voice is loaded but not used while Stack is Single.');
    await t.click('.tile[data-m="adapters"] .unused .sbtn');
    await t.ok('Stack them stacks them', () => document.querySelectorAll('.tile[data-m="adapters"] .cart:not(.empty)').length === 2 && !!document.querySelector('.tile[data-m="adapters"] .stackbar') && !document.querySelector('.tile[data-m="adapters"] .unused'));
  } },
  { id: 'section-masks-force-runtime', run: async t => {
    await desk('lab', ['adapters'], { 'ad.adv': 'Advanced', 'ad.mode': 'Merge', 'lyrics.text': LYRICS }, 'ace', { slots: TWO })(t);
    await t.eq('each cartridge has a chip per section, tagged by its initial (L-25)', () => [...document.querySelectorAll('.tile[data-m="adapters"] [data-secband="0"] button')].map(b => b.textContent + ':' + b.className), ['V1:full', 'C:full', 'V2:full']);
    await t.click('.tile[data-m="adapters"] [data-mask="0-1"]');
    await t.ok('a press writes a directive into the lyrics', () => /\[Chorus\]\{#1=0\.5; #2=1\}/.test(get('lyrics.text')) || get('lyrics.text'));
    await t.eq('a step named for its section and adapter (L-09)', () => __c.undo(), 'Chorus: tape-lofi-v2 full → half');
    await t.focus('.tile[data-m="adapters"] [data-mask="0-1"]'); await t.press('Enter');
    await t.ok('Enter steps it again', () => /\[Chorus\]\{#1=0; #2=1\}/.test(get('lyrics.text')) && document.querySelector('[data-mask="0-1"]').className === 'off');
    await t.ok('directives force Runtime, and the foot says what returns after (A-30)', () => /Runtime while section directives are set · Merge returns after/.test(__c.text('[data-adfoot]')));
    await t.eq('Apply as shows Runtime pressed, the others disabled; the choice is kept', () => [[...document.querySelectorAll('[data-adfoot] [data-seg="ad.mode"]')].map(b => b.textContent.toLowerCase() + ':' + b.getAttribute('aria-pressed') + ':' + b.disabled), get('ad.mode')],
      [['merge:false:true', 'runtime:true:false', 'low-rank:false:true'], 'Merge']);
    await t.mouseDrag('.tile[data-m="adapters"] [data-mask="1-0"]', [[30, 0], [30, 0]], [0.5, 0.5]);
    await t.ok('a drag across the strip paints the first chip\'s level on the rest, one step', () => /\[Verse\]\{#1=1; #2=0\.5\}/.test(get('lyrics.text')) && /\[Chorus\]\{#1=0; #2=0\.5\}/.test(get('lyrics.text')) && __c.undo() === 'Section masks of folk-voice' || [get('lyrics.text'), __c.undo()]);
  } },
  { id: 'phase-window-keys-drag-and-steps', run: async t => {
    await desk('lab', ['adapters'], { 'ad.adv': 'Advanced' }, 'ace', { slots: TWO })(t);
    await t.focus('.tile[data-m="adapters"] [data-phase="0"] [data-ph="0"]'); await t.press('ArrowRight');
    await t.eq('a key moves a handle a step, named for its adapter (L-01)', () => [S.slots[0].ph, __c.undo()], [[5, 100], 'Phase of tape-lofi-v2 0–100% → 5–100%']);
    await t.press('End');
    await t.eq('End takes it to its own end, 5% short of the other', () => S.slots[0].ph, [95, 100]);
    await t.ok('the handles are sliders with their ranges', () => { const h = document.querySelector('[data-phase="0"] [data-ph="1"]'); return h.getAttribute('aria-valuemin') === '100' && /to 100% of the run/.test(h.getAttribute('aria-valuetext')); });
    await t.ok('and the foot says Runtime is used while phase windows are set', () => /Runtime while phase windows are set/.test(__c.text('[data-adfoot]')));
    await t.mouseDrag('.tile[data-m="adapters"] [data-phase="1"]', [[-30, 0]], [0.97, 0.5]);
    await t.ok('a drag takes the nearer handle, on the 5% grid', () => { const p = S.slots[1].ph; return !!p && p[0] === 0 && p[1] < 100 && p[1] % 5 === 0 || JSON.stringify(p); });
  } },
  { id: 'removing-renumbers-the-directives', run: async t => {
    await desk('lab', ['adapters'], { 'ad.adv': 'Advanced', s1: 1.5, 'lyrics.text': '[Verse]{#1=0.5; #2=1}\na\n[Chorus]{#1=1; #2=0}\nb' }, 'ace', { slots: TWO })(t);
    await t.click('.tile[data-m="adapters"] [data-cartx="0"]');
    await t.eq('the later adapter and its strength move up', () => [S.slots.map(s => s.file), get('s0')], [['folk-voice'], 1.5]);
    await t.eq('its directives are renumbered, and the toast says so', () => [get('lyrics.text'), __c.toast()], ['[Verse]\na\n[Chorus]{#1=0}\nb', 'Removed tape-lofi-v2. The lyric directives were renumbered to match']);
    await t.click('#toast .toast-act');
    await t.eq('Undo in the toast puts all of it back', () => [S.slots.length, get('s1'), get('lyrics.text')], [2, 1.5, '[Verse]{#1=0.5; #2=1}\na\n[Chorus]{#1=1; #2=0}\nb']);
  } },
  { id: 'planner-adapter-from-its-own-list', w: 1920, h: 1080, run: async t => {
    await desk('lab', ['adapters'], {}, 'ace', { slots: [TWO[0]] })(t);
    await t.ok('one adapter: the bank stands beside the cartridge, the tile under 600px (L-05)', () => { const c = __c.tile().querySelector('.cart:not(.empty)').getBoundingClientRect(), f = __c.tile().querySelector('.faders').getBoundingClientRect(); return f.left > c.right && __c.tile().offsetHeight < 600 || __c.tile().offsetHeight; });
    await t.ok('Lab offers the planner adapter beside the DiT adapter', () => __c.text('.tile[data-m="adapters"] [data-plannerpick]') === '+ Planner adapter' && __c.text('.tile[data-m="adapters"] .cart .slot') === 'DiT');
    await t.ok('a bank fader is named in full, its short tape carrying the full name (V-01)', () => { const f = __c.tile().querySelector('.faders .fdr'); return f.querySelector('.fdr-l').title === 'Self-Attn' && f.querySelector('[data-fader]').getAttribute('aria-label') === 'Self-Attn strength'; });
    await t.click('.tile[data-m="adapters"] [data-plannerpick]');
    await t.eq('its list: the planner LM adapters, and what they change', () => [document.querySelectorAll('#pop [data-pickplanner]').length, __c.text('#pop h4'), /what the planner writes/.test(__c.text('#pop .lbl'))], [2, 'Planner LM adapters', true]);
    await t.click('#pop [data-pickplanner="folk-voice-planner (LoRA)"]');
    await t.ok('it loads as a planner cartridge, with its note where a fader would be', () => get('ad.planner') === 'folk-voice-planner (LoRA)' && document.querySelector('#pop').hidden && __c.text('.tile[data-m="adapters"] .cart.h-plan .pl-note') === 'Shapes the plan, not the sound');
    await t.ok('focus lands on its × (L-07)', () => document.activeElement?.matches('[data-m="adapters"] .cart.h-plan .x') || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('.tile[data-m="adapters"] .cart.h-plan .x');
    await t.eq('and comes out', () => get('ad.planner'), 'None');
  } },
  { id: 'quick-flat-cartridge-and-the-rest-said', w: 1920, h: 1080, run: async t => {
    await desk('quick', ['adapters', 'avoid', 'timbre', 'seed'], { 'ad.adv': 'Advanced' }, 'ace', { slots: [{ file: 'synthwave-80s', type: 'DoRA' }, TWO[0]] })(t);
    await t.ok('the first lies flat with its art beside its name, its strength a choice row', () => !!document.querySelector('.qcart.t-dora .vec-art') && __c.text('.qcart .lab-n') === 'synthwave-80s' && __c.text('.qcart em') === 'DoRA' && !!__c.tile().querySelector('[role="radiogroup"]'));
    await t.eq('the rest in one line, with the way to Studio (Q-18)', () => __c.text('.tile[data-m="adapters"] .q-more'), '2 adapters, stacked. Edit in Studio');
    await t.click('.tile[data-m="adapters"] .q-more .linkbtn');
    await t.eq('which goes to Studio', () => S.tier, 'studio');
  } },
  { id: 'yue-pair-and-hm-one-column', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['adapters'], {}, 'yue', { slots: [{ file: 'folk-voice-ar', type: 'LoRA' }] })(t);
    await t.eq('the AR cartridge: its strength beside Plan and Compose (E-05)', () => [...document.querySelectorAll('.tile[data-m="adapters"] .fdr-pair .fdr-l')].map(x => x.textContent), ['Strength', 'Plan', 'Compose']);
    await t.ok('the AR composer is the planner\'s housing, the NAR slot the sound\'s', () => __c.tile().querySelector('.cart:not(.empty)').classList.contains('h-plan') && __c.tile().querySelector('[data-cartadd]').classList.contains('h-sound'));
    await t.ok('the empty slot names the NAR renderer, the column says what it does (E-10)', () => __c.text('.tile[data-m="adapters"] [data-cartadd]') === '+ NAR · renderer adapter' && /second slot takes a NAR renderer adapter/.test(__c.tile().textContent));
    await desk('studio', ['adapters'], {}, 'hm', { slots: [{ file: 'heart-folk-lm', type: 'LoRA' }] })(t);
    await t.ok('HeartMuLa: one column, its one limit said beside the cartridge (S-14, S-21)', () => __c.tile().getBoundingClientRect().width < 400 && /HeartMuLa takes one adapter/.test(__c.tile().textContent));
  } },
  { id: 'adapters-another-engine-keeps', run: async t => {
    await desk('studio', ['adapters'], {}, 'ace', { slots: TWO })(t);
    await t.do('switch to YuE2', () => switchEngine('yue'));
    await t.ok('the empty tile says ACE-Step keeps its adapters for when you switch back (B-02)', () => /^No AR · composer adapter loaded · ACE-Step 1\.5 keeps its 2 adapters for when you switch back/.test(__c.tile().textContent.replace(/^[\s\S]*?Adapters\s*/, '')) || __c.tile().textContent);
    await t.do('one adapter here', () => { S.slots = [{ file: 'folk-voice-ar', type: 'LoRA' }]; render(); });
    await t.ok('and with one loaded, under the rig', () => __c.text('.tile[data-m="adapters"] .ad-away') === 'ACE-Step 1.5 keeps its 2 adapters for when you switch back.');
  } },
  { id: 'xy-off-rail-and-planner-off', run: async t => {
    await desk('lab', ['planner'], { 'mm.topp': 0 }, 'mm3')(t);
    await t.ok('top-p off: the puck on its rail along the top, the rail lit (E-28, E-32)', () => { const x = document.querySelector('.tile[data-m="planner"] .xy'); return x.classList.contains('has-yoff') && x.classList.contains('y-off') && /top-p off/i.test(x.querySelector('.xy-rail span').textContent); });
    await t.focus('.tile[data-m="planner"] .xy'); await t.press('ArrowDown');
    await t.ok('down leaves the rail for the widest cut', () => +get('mm.topp') === 1 && !document.querySelector('.tile[data-m="planner"] .xy').classList.contains('y-off'));
    await t.press('ArrowUp');
    await t.ok('and up from the widest is off again', () => +get('mm.topp') === 0 && document.querySelector('.tile[data-m="planner"] .xy').classList.contains('y-off'));
    await desk('lab', ['planner'], { 'lm.on': false }, 'ace')(t);
    await t.ok('with the planner off the pad stands back, says why, and stays operable (Q-17)', () => { const x = document.querySelector('.tile[data-m="planner"] .xy'); return getComputedStyle(x).opacity === '0.45' && x.title === 'Not used while the planner is off'; });
    await t.do('quick', () => { S.tier = 'quick'; render(); });
    await t.ok('at Quick How adventurous stands back and says why under it', () => { const m = document.querySelector('.tile[data-m="planner"] .mood'); return getComputedStyle(m).opacity === '0.45' && document.querySelector('.tile[data-m="planner"] .lm-note')?.textContent === 'Not used while the planner is off'; });
  } },
];
