// The desk's fix round (review r1, RD-01 to RD-18, and RA-23): each defect the review found, driven the way a person
// reaches it. A module comes from its card in the Modules panel, an engine from the engine panel, a tier from its key, a
// file from the file field, a value from the control's own keys; the test layer's scenes are used only to read what the
// page holds (S, get, UNDO). Unless its `why` says the concept has the defect too, each holds on the concept page
// (run it there with --page).

const W = { w: 1600, h: 900 };
const helpers = async t => {
  await t.js(() => { window.A = '.tile[data-m="adapters"]'; window.ctls = () => [...new Set([...document.querySelectorAll('.tile[data-m="sampler"] [data-ctl]')].map(e => e.dataset.ctl))]; window.__d = {
    undo: () => (document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent),
    toast: () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''),
    text: s => document.querySelector(s)?.textContent.replace(/ /g, ' ').replace(/\s+/g, ' ').trim() ?? null,
    shown: s => { const e = document.querySelector(s); return !!e && !e.hidden && e.getBoundingClientRect().height > 0; },
    box: s => { const r = document.querySelector(s)?.getBoundingClientRect(); return r ? [Math.round(r.width), Math.round(r.height)] : null; },
  }; });
};
const engine = async (t, id) => { await t.click('#engBtn'); await t.click(`#engPanel .ep-card[data-backend="${id}"]`); await t.press('Escape'); };
const add = (t, id) => t.click(`#browser .bcard[data-additem="${id}"]`);
const A = '.tile[data-m="adapters"]';

export default [
  { id: 'first-adapter-from-the-empty-strip', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+2');
    await add(t, 'adapters');
    await t.ok('the empty tile is its strip, with the way to load one', () => /No adapter loaded/.test(__d.text(`${A} .mod-strip`)) && !!document.querySelector(`${A} .mod-strip .sbtn`));
    await t.click(`${A} .mod-strip .sbtn`);
    await t.ok('a click on it opens the adapter list under it (RD-01)', () => __d.shown('#pop') && /DiT adapters for ACE-Step/i.test(__d.text('#pop h4')) && document.querySelectorAll('#pop .cartpick').length > 0 && document.querySelector(`${A} .mod-strip .sbtn`).getAttribute('aria-expanded') === 'true');
    await t.ok('focus is on its first file', () => document.activeElement?.matches('#pop .cartpick'));
    await t.press('Enter');
    await t.eq('choosing one loads it, a step named for the file', () => [S.slots.map(s => s.file), __d.undo()], [['tape-lofi-v2'], 'Load tape-lofi-v2']);
    await t.ok('the tile now holds its cartridge', () => !!document.querySelector(`${A} .cart:not(.empty)`) && !document.querySelector(`${A} .mod-strip`));
    await t.press('Control+z');
    await t.ok('Undo takes it back to the strip', () => S.slots.length === 0 && !!document.querySelector(`${A} .mod-strip .sbtn`));
    await t.focus(`${A} .mod-strip .sbtn`); await t.press('Enter');
    await t.ok('Enter on the strip\'s button opens the list too', () => __d.shown('#pop') && document.querySelectorAll('#pop .cartpick').length > 0);
    await t.press('Escape');
    await t.ok('Escape closes it, focus back on the button', () => !__d.shown('#pop') && document.activeElement?.matches(`${A} .mod-strip .sbtn`));
  } },
  { id: 'first-planner-adapter-and-other-engines', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await add(t, 'adapters');
    await t.eq('at Lab the strip offers both kinds', () => [...document.querySelectorAll(`${A} .mod-strip .sbtn`)].map(b => b.textContent), ['+ DiT adapter', '+ Planner adapter']);
    await t.click(`${A} .mod-strip .sbtn:nth-child(2)`);
    await t.ok('the planner adapter has its own list', () => __d.shown('#pop') && document.querySelectorAll('#pop .cartpick').length > 0);
    await t.click('#pop .cartpick');
    await t.ok('and one chosen is the planner adapter', () => get('ad.planner') !== '' && !document.querySelector(`${A} .mod-strip`));
    for (const [e, n] of [['yue', 'AR · composer'], ['mm3', 'Planner LM'], ['hm', 'Global LM']]) {
      await engine(t, e);
        await t.click(`${A} .mod-strip .sbtn`);
      await t.ok(`${n}: the list opens (RD-01)`, () => __d.shown('#pop') && document.querySelectorAll('#pop .cartpick').length > 0);
      await t.click('#pop .cartpick');
      await t.ok(`${n}: the first adapter loads`, () => S.slots.length === 1 && /^Load /.test(__d.undo()));
    }
  } },
  { id: 'a-dim-keeps-a-control-only-for-its-reason', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await engine(t, 'mm3');
    await add(t, 'sampler');
    await t.ok('with Lua plugins off, no shift and no schedule on MiniMax-Music3 (RD-02)', () => { const c = ctls(); return !c.includes('samp.shift') && !c.includes('samp.sched') || c; });
    await t.click('.tile[data-m="sampler"] [data-ctl="samp.plugins"]');
    await t.ok('with them on, both are there', () => { const c = ctls(); return c.includes('samp.shift') && c.includes('samp.sched') || c; });
    await engine(t, 'hm');
    await t.ok('HeartMuLa keeps its own Lua switch, off: no flow shift', () => !ctls().includes('samp.shift') || ctls());
    await t.click('.tile[data-m="sampler"] [data-ctl="samp.plugins"]');
    await t.ok('on, the flow shift is there', () => ctls().includes('samp.shift') || ctls());
  } },
  { id: 'held-takes-said-where-they-are', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+2');
    await add(t, 'takes');
    await t.click('.tile[data-m="takes"] [data-take="4"]');
    await t.eq('four takes', () => get('takes.n'), 4);
    await engine(t, 'sa3');
    await t.eq('on a one-song engine the strip says the four wait (RD-03)', () => __d.text('.tile[data-m="takes"] .ms-why .tk-held'), 'Your 4 takes return on ACE-Step 1.5');
    await t.press('Alt+1');
    await t.ok('at Quick the strip still takes the whole row (RD-04)', () => document.querySelector('.tile[data-m="takes"]').classList.contains('w4') && Math.abs(document.querySelector('.tile[data-m="takes"]').getBoundingClientRect().width - document.querySelector('#bento').getBoundingClientRect().width) < 2);
    await engine(t, 'ace');
    await t.press('Alt+2');
    await t.click('.tile[data-m="takes"] [data-take="8"]');
    await engine(t, 'mm3');
    await t.eq('on an engine that takes fewer, the reels say what waits (RD-03)', () => __d.text('.tile[data-m="takes"] .tk-held'), 'Your 8 takes return on ACE-Step 1.5');
    await t.click('.tile[data-m="takes"] [data-take="2"]');
    await t.ok('a count changed here is the one: the note goes', () => !document.querySelector('.tile[data-m="takes"] .tk-held'));
  } },
  { id: 'no-empty-housing-at-quick', ...W, run: async t => {
    await helpers(t);
    await engine(t, 'sa3');
    await t.press('Alt+1');
    await add(t, 'tempo');
    await t.ok('Stable Audio 3\'s Length at Quick has no empty key-wheel housing above it (RD-04)', () => !document.querySelector('.tile[data-m="tempo"] .tempo') && document.querySelector('.tile[data-m="tempo"] .tile-body').firstElementChild?.matches('.dur'));
  } },
  { id: 'the-reference-sleeve-draws-its-wave', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+2');
    await add(t, 'timbre');
    await t.ok('empty, the sleeve has no waveform', () => !document.querySelector('.tile[data-m="timbre"] canvas.minwave'));
    await t.file('.tile[data-m="timbre"] input[type="file"]', 'my-ref.wav', 'audio/wav');
    await t.eq('a chosen file loads', () => S.timbreFile, 'my-ref.wav');
    await t.ok('and its waveform is drawn under its name (RD-05)', () => { const c = document.querySelector('.tile[data-m="timbre"] .sl-txt canvas.minwave'); if (!c || !c.width) return false; const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; for (let i = 3; i < d.length; i += 4) if (d[i]) return true; return false; });
  } },
  { id: 'the-plan-text-is-as-tall-as-the-concept-draws-it', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+2');
    await add(t, 'planner');
    await t.eq('the plan block ends with the cursor\'s own line (RD-07)', () => { const c = document.querySelector('.tile[data-m="planner"] .term .cursor'); return Math.round(c.parentElement.getBoundingClientRect().height); }, 113);
  } },
  { id: 'a-plugin-without-controls-says-so', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await engine(t, 'yue');
    await add(t, 'sampler');
    await t.do('the solver Ancestral Euler', () => setVal('samp.solver', 'Ancestral Euler (exp.)'));
    await t.eq('its card says it has no controls of its own (RD-06)', () => __d.text('.tile[data-m="sampler"] .plugctl.none'), 'Ancestral EulerThis solver has no controls of its own; it runs on its plugin defaults unless a profile sets them.');
    await t.do('the schedule Plan-Loose', () => setVal('samp.sched', 'Plan-Loose (exp.)'));
    await t.eq('a schedule says it too', () => [...document.querySelectorAll('.tile[data-m="sampler"] .plugctl.none b')].map(b => b.textContent), ['Ancestral Euler', 'Plan-Loose']);
  } },
  { id: 'repaint-rings-its-tile-and-leaves-with-its-song', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+2');
    await t.click('#latest .song[data-id="1"] [data-songmenu]');
    await t.click('#menu [data-sa="repaint"]');
    await t.ok('Repaint arrives working on the song, its tile ringed as the source (RD-09)', () => !!S.source && document.querySelector('.tile[data-m="repaint"]').classList.contains('source'));
    await t.click('.tile[data-m="repaint"] .x');
    await t.eq('removed, it takes its song with it (RA-23)', () => S.source ?? null, null);
    await t.press('Control+z');
    await t.ok('Undo brings both back', () => !!S.source && !!document.querySelector('.tile[data-m="repaint"]'));
  } },
  { id: 'mm3-quick-tempo-note-reads-whole', ...W, run: async t => {
    await helpers(t);
    await engine(t, 'mm3');
    await t.press('Alt+1');
    await add(t, 'tempo');
    await t.eq('the note\'s action stands inside its sentence (RD-10)', () => __d.text('.tile[data-m="tempo"] .cap-note'), 'MiniMax-Music3 reads tempo from the caption. Compose caption writes it in.');
  } },
  { id: 'a-pedal-knob-step-names-its-pedal', ...W, why: 'RD-11 holds on the concept; RD-18 (Pre-VST gain said once) is the concept\'s fault.', run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await engine(t, 'mm3');
    await add(t, 'pd:refiner');
    await t.focus('[data-knob="pp.refiner.mix"]'); await t.press('ArrowDown');
    await t.eq('a bare Mix on the board says whose it is (RD-11)', () => __d.undo(), 'MM3 latent refiner mix 100% → 99%');
    await add(t, 'pd:gain');
    await t.focus('[data-knob="pp.gain"]'); await t.press('ArrowUp');
    await t.eq('the Pre-VST gain\'s Gain is said once (RD-18)', () => __d.undo(), 'Pre-VST gain 0.0 dB → 0.5 dB');
  } },
  { id: 'a-key-at-the-end-of-the-range-is-no-step', ...W, why: 'RD-14: the concept records "Mix 100% → 100%" (its fault).', run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await engine(t, 'mm3');
    await add(t, 'pd:refiner');
    const n = await t.js(() => UNDO.length);
    await t.focus('[data-knob="pp.refiner.mix"]'); await t.press('ArrowUp'); await t.press('End');
    await t.eq('at its maximum, ArrowUp and End change nothing and record nothing (RD-14)', n0 => [UNDO.length - n0, 'pp.refiner.mix' in S.vals], [0, false], n);
  } },
  { id: 'yue-cfg-rescale-steps-by-hundredths', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await engine(t, 'yue');
    await add(t, 'planner');
    await t.focus('.tile[data-m="planner"] [data-scrub="lm.resc"]'); await t.press('ArrowUp'); await t.press('ArrowUp');
    await t.eq('two presses: 0.02 (RD-12)', () => get('lm.resc'), 0.02);
  } },
  { id: 'continue-part-undo-says-back-to', ...W, run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await engine(t, 'yue');
    await add(t, 'cont');
    await t.file('.tile[data-m="cont"] input[type="file"]', 'take-01.wav', 'audio/wav');
    await t.focus('[data-contwave]'); await t.press('ArrowRight');
    await t.ok('the move is a step that says from and to', () => / → /.test(__d.undo()) || __d.undo());
    await t.press('Control+z');
    await t.ok('its Undo says where the part went back to (RD-13)', () => /^Undone: Part to continue from back to \d+:\d\d–\d+:\d\d$/.test(document.querySelector('#say')?.textContent.trim() || __d.toast()) || [__d.toast(), document.querySelector('#say')?.textContent]);
  } },
  { id: 'score-preview-from-the-planner-alone', ...W, run: async t => {
    await helpers(t);
    await engine(t, 'yue');
    await t.press('Alt+2');
    await add(t, 'planner');
    await t.js(() => [...document.querySelectorAll('.tile[data-m="planner"] button')].find(b => /score preview/.test(b.textContent))?.setAttribute('data-fixpv', ''));
    await t.click('.tile[data-m="planner"] [data-fixpv]');
    await t.ok('the planner\'s "score preview" opens the preview with no lead sheet on the desk', () => __d.shown('#scoreModal') || [...document.querySelectorAll('.tile[data-m="planner"] button')].map(b => b.textContent).join('|'));
  } },
  { id: 'mm3-planner-seed-reads-as-a-token', ...W, run: async t => {
    await helpers(t);
    await engine(t, 'mm3');
    await t.press('Alt+3');
    await add(t, 'planner');
    await t.eq('its empty --seed says "follows" (RD-18)', () => document.querySelector('.tile[data-m="planner"] [data-ctl="mm.seed"]')?.placeholder, 'follows');
  } },
  { id: 'side-unit-lip-focus-is-a-whole-ring', ...W, why: 'RD-15: the concept draws a red line along the bottom (its fault; the owner\'s rule 1).', run: async t => {
    await helpers(t);
    await t.press('Alt+3');
    await add(t, 'ex:wh');
    await t.focus('[data-xid="wh"] .u-pow'); await t.press('Shift+Tab'); await t.press('Tab');
    await t.ok('the power button\'s keyboard ring is drawn inside it (RD-15)', () => document.activeElement?.matches('[data-xid="wh"] .u-pow') && getComputedStyle(document.activeElement).outlineOffset === '-2px' || [document.activeElement?.outerHTML.slice(0, 60), getComputedStyle(document.activeElement).outlineOffset]);
  } },
];
