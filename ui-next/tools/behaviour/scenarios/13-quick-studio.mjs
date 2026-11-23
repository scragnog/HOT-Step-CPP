// Round 4, Quick and Studio on ACE-Step (review Q): the tier switch stays under the pointer, focus lands on what was acted
// on, every tier tells the truth about the value underneath, the choice rows are radio groups, and the pedals line up.
const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const desk = (tier, mods, pedals = []) => t => t.do(`${tier} desk`, ([tier, mods, pedals]) => { S.tier = tier; S.mods = [...mods]; S.pedals = [...pedals]; pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); S.qAll = true; render(); }, [tier, mods, pedals], 400);
const box = sel => { const r = document.querySelector(sel)?.getBoundingClientRect(); return r ? [Math.round(r.left), Math.round(r.width)] : null; };
export default [
  { id: 'tier-switch-stays-under-the-pointer', w: 1920, h: 1080, run: async t => {
    await desk('quick', ACE)(t);
    const at = await t.js(box, '#tiers');
    for (const k of ['studio', 'lab', 'quick', 'lab']) {
      await t.click(`#tiers [data-tier="${k}"]`);
      await t.eq(`${k}: the tier switch keeps its place (Q-03)`, box, at, '#tiers');
      await t.eq(`${k}: focus stays on the pressed tier (Q-16)`, () => document.activeElement?.dataset.tier, k);
    }
    await t.press('Alt+2');
    await t.eq('Alt+2 chooses Studio from anywhere (Q-43)', () => S.tier, 'studio');
    await t.do('scroll down the page', () => scrollTo(0, 900));
    await t.ok('the bar with the tier switch stays in view while the page scrolls (Q-14)', () => { const r = document.querySelector('#tiers').getBoundingClientRect(); return r.top >= 50 && r.bottom < 160; });
  } },
  { id: 'focus-lands-on-what-arrived', w: 1920, h: 1080, run: async t => {
    await desk('studio', [])(t);
    await t.click('.desk-empty [data-additem="tempo"]');
    await t.eq('a starter: focus on the new tile\'s grip (Q-16)', () => document.activeElement?.dataset.move, 'tempo');
    await t.click('#browser .bcard[data-additem="seed"]');
    await t.eq('a click-add from the panel: focus on the new tile\'s grip', () => document.activeElement?.dataset.move, 'seed');
    await desk('quick', [])(t);
    await t.ok('an empty Quick desk offers the starters under the sheet (Q-42)', () => document.querySelectorAll('#bento .q-empty [data-additem]').length === 3);
  } },
  { id: 'every-tier-tells-the-truth', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['tempo', 'planner', 'sampler', 'seed', 'latent'])(t);
    await t.click('.tile[data-m="seed"] [data-dice]');
    await t.click('.tile[data-m="seed"] [data-ledtog="seed.lock"]');
    await t.ok('Studio: an unlocked seed reads RANDOM', () => document.querySelector('[data-seednum]').classList.contains('rand'));
    await t.do('set key, signature and LM CFG', () => { setVal('tempo.key', 'D major'); setVal('tempo.sig', '3/4'); setVal('lm.cfg', 5); });
    await t.click('#tiers [data-tier="quick"]');
    await t.ok('Quick: the same unlocked seed reads RANDOM too, and Random each time is pressed (Q-06)', () => document.querySelector('[data-seednum]').classList.contains('rand') && document.querySelector('[data-seedrand]').getAttribute('aria-pressed') === 'true');
    await t.ok('Quick: a tile names the values it does not show (Q-07)', () => /Also set: Key D major · Time signature 3\/4/.test(document.querySelector('.tile[data-m="tempo"] .q-also')?.textContent || '') && /LM CFG 5\.0/.test(document.querySelector('.tile[data-m="planner"] .q-also')?.textContent || ''));
    await t.ok('a tile with nothing hidden changed has no such line', () => !document.querySelector('.tile[data-m="sampler"] .q-also'));
    await t.click('.tile[data-m="tempo"] .q-also [data-tier]');
    await t.ok('its link opens the tier that shows them, ringing them', () => S.tier === 'studio' && !!document.querySelector('.tile[data-m="tempo"] [data-ctl="tempo.key"].ring-here, .tile[data-m="tempo"] .ring-here'));
    await t.do('custom timesteps, then Quick', () => { setVal('lat.ts', '1, 0.8, 0.5, 0.2, 0'); setVal('samp.steps', 50); S.tier = 'quick'; render(); });
    await t.ok('Quick says the custom timesteps instead of a Quality that would not apply (Q-11)', () => /5 custom timesteps/.test(document.querySelector('.tile[data-m="sampler"]').textContent) && !document.querySelector('.qchoice[data-q="samp.steps"]'));
    await t.do('no custom timesteps', () => { setVal('lat.ts', ''); render(); });
    await t.ok('a step count no option matches is said without naming a tier (Q-11)', () => { const l = document.querySelector('.qchoice[data-q="samp.steps"] .kl').textContent; return /50 steps/.test(l) && !/Studio|Lab/.test(l); });
    await t.ok('no option is checked, and a marker stands beyond High (Q-30)', () => [...document.querySelectorAll('.qchoice[data-q="samp.steps"] button')].every(b => b.getAttribute('aria-checked') === 'false') && !document.querySelector('.qchoice[data-q="samp.steps"] > .q-mark').hidden);
    await t.ok('the row carries the changed dot (Q-12)', () => document.querySelector('.qchoice[data-q="samp.steps"]').classList.contains('changed'));
  } },
  { id: 'choice-rows-are-radio-groups', w: 1920, h: 1080, run: async t => {
    await desk('quick', ['sampler', 'planner'])(t);
    await t.ok('one Tab stop per row (Q-30)', () => [...document.querySelectorAll('.qchoice[data-q] button, .mood button')].filter(b => b.tabIndex === 0).length === document.querySelectorAll('.qchoice[data-q], .mood').length);
    await t.focus('.qchoice[data-q="samp.steps"] button[tabindex="0"]');
    await t.press('ArrowRight');
    await t.ok('an arrow chooses the next option and focus follows', () => { const bs = [...document.querySelectorAll('.qchoice[data-q="samp.steps"] button')]; return String(get('samp.steps')) === bs[2].dataset.setq.split('|')[1] && document.activeElement === bs[2] && bs[2].getAttribute('aria-checked') === 'true'; });
    await t.ok('the Planner and Sampler rows share no word (Q-36)', () => { const w = s => [...document.querySelectorAll(s + ' button')].map(b => b.textContent); const a = w('.qchoice[data-q="samp.cfg"]'), b = w('.mood'); return a.every(x => !b.includes(x)); });
    await t.click('.tile[data-m="planner"] [data-ledtog="lm.on"]');
    await t.ok('with the planner off, what only it reads stands back and says so (Q-17)', () => document.querySelector('.tile[data-m="planner"]').classList.contains('lm-off') && /Not used while the planner is off/.test(document.querySelector('.tile[data-m="planner"]').textContent) && +getComputedStyle(document.querySelector('.tile[data-m="planner"] .mood')).opacity < .5);
  } },
  { id: 'pedals-line-up-and-keep-their-state', w: 1920, h: 1080, run: async t => {
    await desk('studio', ['post'], ['ss', 'nat', 'gain', 'vst', 'master', 'norm'])(t);
    const mid = sel => { const p = document.querySelector(sel), r = p.getBoundingClientRect(), l = p.querySelector('.led').getBoundingClientRect(); return Math.round(l.top + l.height / 2 - r.top); };
    await t.eq('Final normalizer\'s LED is level with every other pedal\'s (Q-09)', mid, await t.js(mid, '.pedal[data-pid="nat"]'), '.pedal[data-pid="norm"]');
    await t.ok('Pre-VST gain\'s lower edge is a darker grey of its own face, not a colour (Q-10)', () => { const m = getComputedStyle(document.querySelector('.pedal[data-pid="gain"]')).boxShadow.match(/rgb\((\d+), (\d+), (\d+)\)/); const [r, g, b] = m.slice(1).map(Number); return Math.max(r, g, b) - Math.min(r, g, b) < 30; });
    await t.ok('an engaged pedal with nothing to run shows amber and says why (Q-19)', () => document.querySelector('.pedal[data-pid="master"]').classList.contains('idle') && /reference/.test(document.querySelector('.pedal[data-pid="master"] .led').title));
    await t.click('.tile[data-m="post"] [data-ledtog="pp.on"]');
    await t.ok('with the chain bypassed, an engaged pedal still shows it is engaged (Q-18)', () => document.querySelector('.pedal[data-pid="ss"]').classList.contains('held'));
    await t.click('.pedal[data-pid="ss"] .foot');
    await t.ok('and a press on its switch shows at once', () => !document.querySelector('.pedal[data-pid="ss"]').classList.contains('held'));
    await t.ok('Tab meets the switch before the remove tab (Q-21)', () => { const p = document.querySelector('.pedal[data-pid="nat"]'), f = [...p.querySelectorAll('button, [tabindex="0"]')]; return f.indexOf(p.querySelector('.foot')) < f.indexOf(p.querySelector('.px.tab')); });
  } },
  { id: 'small-things-hold-still', w: 1920, h: 1080, run: async t => {
    await desk('quick', ['seed', 'tempo', 'sampler'])(t);
    const h0 = await t.js(() => document.querySelector('.tile[data-m="seed"]').offsetHeight);
    await t.click('.tile[data-m="seed"] [data-seededit]');
    await t.eq('typing a seed does not change the tile\'s height (Q-23)', () => document.querySelector('.tile[data-m="seed"]').offsetHeight, h0);
    await t.press('Escape');
    await desk('studio', ['sampler', 'tempo'])(t);
    await t.ok('the Sampler strip\'s label rows are one height, Shift\'s auto pill included (Q-24)', () => { const hs = [...document.querySelectorAll('.samp-strip .kn .kl')].map(l => Math.round(l.getBoundingClientRect().height)); return new Set(hs).size === 1 || hs.join(','); });
    await t.ok('and an automatic value says what auto means', () => /^≈ /.test(document.querySelector('.samp-strip [data-out="samp.shift"]').textContent));
    await t.focus('.tile[data-m="tempo"] .wheel [aria-checked="true"]');
    await t.do('two wheel keys inside one frame', () => { const w = () => document.querySelector('.tile[data-m="tempo"] .wheel [tabindex="0"], .tile[data-m="tempo"] .wheel [aria-checked="true"]'); setVal('tempo.key', 'B♭ minor'); ['ArrowRight', 'ArrowUp'].forEach(k => (document.activeElement.closest('.wheel') ? document.activeElement : w()).dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }))); });
    await t.ok('both count, from the value (Q-33)', () => /major/.test(get('tempo.key')) && get('tempo.key') !== 'D♭ major');
  } },
];
