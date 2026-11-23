// Round 3, the desk on ACE-Step (review A): packing, state that survives reload and undo, changed marks, typed fields,
// overrides between modules, Quick's choice rows, focus after removals, undo for structural changes, keyboard contracts.
const desk = (tier, mods, extra) => t => t.do(`${tier} desk`, ([tier, mods, extra]) => { S.tier = tier; S.mods = mods; S.pedals = extra?.pedals || []; if (extra?.vals) Object.assign(S.vals, extra.vals); if (extra?.slots) S.slots = extra.slots; render(); }, [tier, mods, extra]);
const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'repetition', 'sampler', 'guidance', 'perf', 'latent', 'seed', 'takes'];
/* the desk's packing moved to 12-desk-layout.mjs (round 4: the order is the layout) */
export default [
  { id: 'adapter-phase-survives-reload-and-undo', run: async t => {
    await desk('lab', ['adapters'], { vals: { 'ad.adv': 'Advanced' }, slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }, { file: 'my-band-ep', type: 'LoRA' }] })(t);
    await t.focus('.tile[data-m="adapters"] [data-phase="0"] [data-ph="0"]'); await t.press('PageUp'); await t.press('ArrowRight'); await t.press('ArrowRight');
    await t.eq('the phase handle steps by keys (A-24)', () => S.slots[0].ph, [35, 100]);
    await t.ok('the handle is a full slider', () => { const h = document.querySelector('[data-phase="0"] [data-ph="0"]'); return h.getAttribute('aria-valuemin') === '0' && h.getAttribute('aria-valuemax') === '95' && /35% of the run/.test(h.getAttribute('aria-valuetext')); });
    await t.do('let autosave run', () => {}, null, 600);
    await t.reload();
    await t.eq('a phase window comes back after a reload (A-05)', () => S.slots[0].ph, [35, 100]);
    await t.click('.tile[data-m="adapters"] [data-cartx="2"]');
    await t.press('Control+z');
    await t.eq('and after removing another adapter and undoing it', () => [S.slots.length, S.slots[0].ph], [3, [35, 100]]);
    await t.ok('the footer says Runtime is used while phase windows are set', () => /Runtime while phase windows are set/.test(document.querySelector('[data-adfoot]').textContent));
  } },
  { id: 'changed-marks-survive-render', run: async t => {
    await desk('lab', ['tempo', 'planner', 'seed', 'takes', 'timbre'], { pedals: [] })(t);
    await t.click('.tile[data-m="tempo"] [data-ledtog="tempo.beatio"]');
    await t.click('.tile[data-m="planner"] [data-ledtog="lm.on"]');
    await t.do('set BPM, key, takes and the blend', () => { S.vals['tempo.bpm.auto'] = false; setVal('tempo.bpm', 128); setVal('tempo.key', 'E minor'); setVal('takes.n', 3); setVal('timbre.str', .5); setVal('lm.topp', .5); });
    const marks = () => ['tempo.beatio', 'lm.on', 'tempo.bpm', 'tempo.key', 'takes.n', 'timbre.str'].map(k => [k, [...document.querySelectorAll(`#bento [data-ctl="${k}"]`)].length > 0 && [...document.querySelectorAll(`#bento [data-ctl="${k}"]`)].every(e => e.classList.contains('changed'))]);
    const want = [['tempo.beatio', true], ['lm.on', true], ['tempo.bpm', true], ['tempo.key', true], ['takes.n', true], ['timbre.str', true]];
    await t.eq('every changed setting is marked (A-06)', marks, want);
    await t.do('a re-render', () => render());
    await t.eq('and stays marked after a re-render', marks, want);
    await t.click('#tiers [data-tier="studio"]'); await t.click('#tiers [data-tier="lab"]');
    await t.eq('and after a tier switch', marks, want);
    await t.ok('an XY pad marks only the axis that changed', () => document.querySelector('.tile[data-m="planner"] .xy-ax.y').classList.contains('changed') && !document.querySelector('.tile[data-m="planner"] .xy-ax.x').classList.contains('changed'));
  } },
  { id: 'typed-field-commits-on-leave', run: async t => {
    await desk('lab', ['latent', 'sampler'])(t);
    await t.click('.tile[data-m="latent"] [data-bind="lat.ts"]'); await t.type('0.5, 0.9, 2'); await t.press('Tab');
    await t.ok('an invalid list is never committed, not even its valid prefix; the text stays, marked, with its reason (A-08, L-13)', () => get('lat.ts') === '' && document.querySelector('[data-bind="lat.ts"]').value === '0.5, 0.9, 2' && document.querySelector('[data-bind="lat.ts"]').getAttribute('aria-invalid') === 'true' && !!document.querySelector('[data-err="lat.ts"]').textContent);
    await t.click('.tile[data-m="latent"] [data-bind="lat.ts"]'); await t.press('Control+a'); await t.type('0.9, 0.6, 0.3, 0'); await t.press('Enter');
    await t.eq('a valid list commits on Enter', () => get('lat.ts'), '0.9, 0.6, 0.3, 0');
    await t.ok('the Sampler says custom timesteps replace its steps and schedule (A-09)', () => { const s = document.querySelector('.tile[data-m="sampler"]'); return /4 custom timesteps from Latent/.test(s.querySelector('[data-scoperead]').textContent) && s.querySelector('.numbox.overridden .num').value === '4' && !!s.querySelector('.overridden [inert] [data-pick="samp.sched"]'); });
    await t.click('.tile[data-m="sampler"] [data-showmod="latent"]', 50);
    await t.ok('its link takes you to the module that sets them', () => document.querySelector('.tile[data-m="latent"]').classList.contains('flash'));
  } },
  { id: 'quick-rows-tell-the-truth', run: async t => {
    await desk('studio', ['sampler', 'planner'])(t);
    await t.do('20 steps, guidance 6.5, temperature 0.95', () => { setVal('samp.steps', 20); setVal('samp.cfg', 6.5); setVal('lm.temp', .95); });
    await t.click('#tiers [data-tier="quick"]');
    await t.ok('a value between the options presses none of them (A-29)', () => [...document.querySelectorAll('.qchoice[data-q="samp.steps"] button, .qchoice[data-q="samp.cfg"] button, .mood button')].every(b => b.getAttribute('aria-checked') === 'false'));
    await t.ok('and the row says what the value is', () => /20 steps/.test(document.querySelector('.qchoice[data-q="samp.steps"] .kl').textContent) && /temperature 0.95/.test(document.querySelector('.mood .kl').textContent));
    await t.click('.qchoice[data-q="samp.steps"] [data-setq="samp.steps|12"]');
    await t.ok('pressing an option makes it the value, and pressed', () => +get('samp.steps') === 12 && document.querySelector('[data-setq="samp.steps|12"]').getAttribute('aria-checked') === 'true' && !/steps,/.test(document.querySelector('.qchoice[data-q="samp.steps"] .kl').textContent));
  } },
  { id: 'focus-after-pedal-removal-and-undo', run: async t => {
    await desk('lab', ['tempo', 'post'], { pedals: ['tile', 'ss', 'norm'] })(t);
    await t.focus('.pedal[data-pid="ss"] .foot'); await t.press('Shift+Tab'); await t.press('Shift+Tab');
    await t.do('remove StableStep from its tab', () => document.querySelector('.pedal[data-pid="ss"] .px').focus());
    await t.press('Enter');
    await t.ok('focus goes to the next pedal, not the first tile (A-18)', () => document.activeElement?.closest('.pedal')?.dataset.pid === 'norm' || document.activeElement?.outerHTML.slice(0, 80));
    await t.click('#toast .toast-act');
    await t.ok('after Undo, focus is on the pedal that came back', () => S.pedals.includes('ss') && document.activeElement?.closest('.pedal')?.dataset.pid === 'ss' || document.activeElement?.outerHTML.slice(0, 80));
  } },
  { id: 'undo-covers-structure', run: async t => {
    await desk('studio', ['tempo', 'seed'])(t);
    await t.do('empty the history', () => { UNDO.length = 0; REDO.length = 0; paintUndo(); });
    await t.press('Control+z');
    await t.ok('with nothing to undo, it says so (A-19)', () => /Nothing to undo/.test(document.querySelector('#toast .toast-msg').textContent));
    await t.do('add Takes from the panel', () => addItem('takes'));
    await t.focus('.tile[data-m="tempo"] .grip'); await t.press('ArrowRight'); await t.press('ArrowRight');
    await t.eq('two key presses on one tile are one move', () => [S.mods.join(), UNDO.length], ['seed,takes,tempo', 2]);
    await t.press('Control+z');
    await t.eq('undo puts the move back', () => S.mods.join(), 'tempo,seed,takes');
    await t.press('Control+z');
    await t.eq('and then the add', () => S.mods.join(), 'tempo,seed');
  } },
  { id: 'keyboard-contracts', run: async t => {
    await desk('lab', ['takes', 'sampler', 'planner', 'tempo'])(t);
    await t.focus('.tile[data-m="takes"] [data-takes]'); await t.press('ArrowRight'); await t.press('ArrowRight');
    await t.eq('the reels are one slider: arrows step it (A-23)', () => [+get('takes.n'), document.activeElement.matches('[data-takes]'), document.activeElement.getAttribute('aria-valuenow')], [3, true, '3']);
    await t.press('End'); await t.eq('End takes it to the most', () => +get('takes.n'), 9);
    await t.ok('the reels are not tab stops of their own', () => [...document.querySelectorAll('.tile[data-m="takes"] .reel2')].every(r => r.tabIndex === -1));
    await t.focus('.tile[data-m="sampler"] [data-num="samp.steps"]'); await t.do('type 20', () => { document.activeElement.value = '20'; }); await t.press('Enter');
    await t.ok('Enter in a number box commits and keeps focus (A-24)', () => +get('samp.steps') === 20 && document.activeElement.matches('[data-num="samp.steps"]'));
    await t.press('Escape'); await t.ok('Escape keeps focus too', () => document.activeElement.matches('[data-num="samp.steps"]'));
    await t.focus('.tile[data-m="planner"] .xy'); await t.press('End'); await t.press('PageDown');
    await t.eq('an XY pad answers End and Page Down', () => [+get('lm.temp'), +get('lm.topp')], [2, 0.82]);
    await t.focus('.tile[data-m="sampler"] .knob[data-knob="samp.shift"]'); await t.press('ArrowUp');
    await t.ok('a nudge takes Shift off auto, and its auto pill unlights (A-11)', () => !isAuto('samp.shift') && !document.querySelector('[data-auto="samp.shift"]').classList.contains('on'));
    await t.click('.tile[data-m="sampler"] [data-auto="samp.shift"]');
    await t.ok('the auto pill puts it back on auto', () => isAuto('samp.shift') && document.querySelector('[data-auto="samp.shift"]').classList.contains('on'));
    await t.click('.tile[data-m="tempo"] [data-tap]'); await t.do('wait', () => {}, null, 1700); await t.click('.tile[data-m="tempo"] [data-tap]');
    await t.ok('tap tempo reaches the field\'s own range (A-25)', () => +get('tempo.bpm') < 40 || get('tempo.bpm'));
  } },
  { id: 'repaint-region-keeps-its-place', run: async t => {
    await t.do('a song to repaint', () => { S.mods = ['repaint']; S.source = { song: S.songs[0].id }; S.region = [0.3, 0.6]; S.tier = 'lab'; render(); });
    await t.click('[data-rpwave]', null, [0.85, 0.5]);
    await t.eq('a plain click outside the region leaves it as it was (A-32)', () => S.region, [0.3, 0.6]);
    await t.mouseDrag('[data-rpwave]', [[40, 0]], [0.45, 0.5]);
    await t.ok('a drag inside the region carries it whole', () => S.region[0] > 0.3 && Math.abs((S.region[1] - S.region[0]) - 0.3) < 1e-6);
    await t.ok('and the spoken range follows', () => { const c = document.querySelector('[data-rpwave]'); return c.getAttribute('aria-valuetext') === `${mmss(S.region[0] * curSong().dur)} to ${mmss(S.region[1] * curSong().dur)}`; });
  } },
  { id: 'offline-fonts-keep-the-layout', run: async t => {
    await desk('lab', ['tempo'], { pedals: ['tile', 'norm'] })(t);
    await t.do('fonts settle', () => document.fonts.ready, null, 300);
    await t.ok('with the web fonts blocked, the matched fallback faces are in use (A-03)', () => [...document.fonts].some(f => f.family.replace(/"/g, '') === 'Geist Mono FB' && f.status === 'loaded'));
    await t.ok('and the tiled decoder LCD is whole', () => { const b = document.querySelector('.pedal[data-pid="tile"] .pk-lcd'); return b && b.scrollWidth <= b.clientWidth + 1; });
  } },
];
