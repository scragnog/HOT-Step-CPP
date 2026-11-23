// The instruments of package 3a, each in the module the concept draws it in, as scenarios that hold on the concept
// page too (run them there with --page): Tempo's key wheel, TAP and the prompt chips; the Seed's split-flap and its
// saved seeds; the Takes reels and what they cost; the Sampler's scope and its interval handles; the latent curve;
// the repaint region; the timbre reference's recent files and its blend; Performance trades' corner words; and the knob
// that can be automatic. Each uses only what both pages share (S, render, get, setVal, UNDO, the markup's data-*
// hooks and the concept's own classes, which the instruments emit).

const desk = (tier, mods, vals = {}, backend = 'ace', more = {}) => async t => {
  await t.js(() => { window.__i = { toast: () => (document.querySelector('#toast').hidden ? '' : document.querySelector('#toast .toast-msg')?.textContent || ''),
    undo: () => (document.querySelector('#undoBtn').hidden ? '' : document.querySelector('#undoBtn span').textContent), text: s => document.querySelector(s)?.textContent.replace(/\u00a0/g, ' ') ?? null,
    /* the prompt's text: the concept's box, or the rebuild's value while its prompt card is not drawn */
    prompt: () => document.querySelector('#prompt')?.value ?? get('prompt.text'),
    setPrompt: v => { const p = document.querySelector('#prompt'); if (p) { p.value = v; p.dispatchEvent(new Event('input', { bubbles: true })); } else { S.vals['prompt.text'] = v; render(); } } }; });
  await t.do(`${backend} ${tier} desk`, ([tier, mods, vals, backend, more]) => {
    if (typeof lyricsBlock === 'function') window.lyricsBlock = () => '';
    if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
    S.backend = backend; S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, more); render();
  }, [tier, mods, vals, backend, more]);
};
const SONG = { song: 1, title: 'Studio sketch 01', style: 'Lo-fi hip hop, mellow electric piano', dur: 171, hue: 215 };

export default [
  { id: 'key-wheel-stands-beside-the-lcd', run: async t => {
    await desk('studio', ['tempo'])(t);
    await t.ok('the LCD and the wheel stand side by side in one housing', () => { const w = document.querySelector(`.tile[data-m="tempo"] .tempo`); return !!w && !w.classList.contains('solo') && w.children[0].matches('.lcd') && w.children[1].matches('svg.wheel'); });
    await t.eq('majors outside, minors inside, Auto in the middle', () => [document.querySelectorAll(`.tile[data-m="tempo"] .wheel path.maj`).length, document.querySelectorAll(`.tile[data-m="tempo"] .wheel path.min`).length, !!document.querySelector(`.tile[data-m="tempo"] .wheel circle[role="radio"]`)], [12, 12, true]);
    await t.eq('keys are spelled as a musician spells them (G-D-25)', () => ['A# major', 'D# minor', 'F# major'].map(k => document.querySelector(`.tile[data-m="tempo"] .wheel path[data-key="${k}"]`).getAttribute('aria-label')), ['B♭ major', 'E♭ minor', 'F♯ major']);
    await t.click(`.tile[data-m="tempo"] .wheel path[data-key="A# major"]`);
    await t.eq('a segment sets the key, the LCD spells it', () => [get('tempo.key'), document.querySelector(`.tile[data-m="tempo"] [data-keylcd]`).textContent], ['A# major', 'B♭ MAJOR']);
    await t.ok('the chosen segment is checked and filled', () => { const s = document.querySelector(`.tile[data-m="tempo"] .wheel path.sel`); return s?.dataset.key === 'A# major' && s.getAttribute('aria-checked') === 'true'; });
    await t.focus(`.tile[data-m="tempo"] .wheel [aria-checked="true"]`); await t.press('ArrowDown');
    await t.eq('down moves to the minor ring, focus with it', () => [get('tempo.key'), document.activeElement.dataset.key], ['G minor', 'G minor']);
    await t.press('Enter');
    await t.eq('Enter keeps the segment in focus chosen', () => get('tempo.key'), 'G minor');
    await t.do('quick', () => { S.tier = 'quick'; render(); });
    await t.ok('at Quick the LCD stands alone (solo)', () => { const w = document.querySelector(`.tile[data-m="tempo"] .tempo`); return !!w && w.classList.contains('solo') && !w.querySelector('.wheel') && !!w.querySelector('.lcd'); });
    await t.do('stable audio, studio', () => { if (typeof enterEngine === 'function') enterEngine('sa3'); S.backend = 'sa3'; S.tier = 'studio'; render(); });
    await t.ok('on Stable Audio 3 the LCD holds Beat I/O alone, without a wheel', () => { const w = document.querySelector(`.tile[data-m="tempo"] .tempo`); return !!w && w.classList.contains('solo') && !w.querySelector('.wheel') && !!w.querySelector('[data-ledtog="tempo.beatio"]'); });
  } },
  { id: 'tap-tempo-arms-then-sets', run: async t => {
    await desk('lab', ['tempo'])(t);
    const u0 = await t.js(() => UNDO.length);
    await t.click(`.tile[data-m="tempo"] [data-tap]`);
    await t.eq('the first tap arms it: it asks for another', () => [document.querySelector(`.tile[data-m="tempo"] [data-tap]`).textContent, document.querySelector(`.tile[data-m="tempo"] [data-tap]`).classList.contains('armed'), isAuto('tempo.bpm')], ['AGAIN', true, true]);
    await t.do('half a second', () => {}, null, 500);
    await t.click(`.tile[data-m="tempo"] [data-tap]`);
    await t.ok('the second sets the BPM from their spacing, off auto', () => +get('tempo.bpm') > 30 && +get('tempo.bpm') <= 120 && !isAuto('tempo.bpm') && document.querySelector(`.tile[data-m="tempo"] [data-tap]`).textContent === 'TAP');
    await t.do('a third, 400ms on', () => {}, null, 400);
    await t.click(`.tile[data-m="tempo"] [data-tap]`);
    await t.ok('a third goes on from the run', () => !isAuto('tempo.bpm'));
    await t.eq('and the run is one step', u0 => UNDO.length - u0, 1, u0);
    await t.do('the run ends', () => {}, null, 2600);
    await t.ok('the button rests again', () => !document.querySelector(`.tile[data-m="tempo"] [data-tap]`).classList.contains('armed'));
  } },
  { id: 'prompt-chips-are-items-of-the-prompt', run: async t => {
    await desk('studio', ['tempo'], {}, 'sa3')(t);
    await t.do('a prompt with a sentence', () => __i.setPrompt('a slow build at 90bpm, warm pads'));
    await t.eq('the chips, none pressed: a tempo inside a sentence is not an item (S-04)', () => [...document.querySelectorAll(`.tile[data-m="tempo"] [data-capadd]`)].map(b => b.getAttribute('aria-pressed')).filter(x => x === 'true').length, 0);
    await t.click(`.tile[data-m="tempo"] [data-capadd="120 BPM"]`);
    await t.eq('a chip goes in as an item after a comma', () => __i.prompt(), 'a slow build at 90bpm, warm pads, 120 BPM');
    await t.eq('and shows pressed', () => document.querySelector(`.tile[data-m="tempo"] [data-capadd="120 BPM"]`).getAttribute('aria-pressed'), 'true');
    await t.click(`.tile[data-m="tempo"] [data-capadd="140 BPM"]`);
    await t.eq('another of its group replaces it', () => __i.prompt(), 'a slow build at 90bpm, warm pads, 140 BPM');
    await t.click(`.tile[data-m="tempo"] [data-capadd="A minor"]`);
    await t.eq('a key is a group of its own', () => __i.prompt(), 'a slow build at 90bpm, warm pads, 140 BPM, A minor');
    await t.click(`.tile[data-m="tempo"] [data-capadd="140 BPM"]`);
    await t.eq('pressed again, it comes out', () => __i.prompt(), 'a slow build at 90bpm, warm pads, A minor');
    await t.eq('each press is a step named for the chip', () => __i.undo(), 'Take 140 BPM out of the prompt');
  } },
  { id: 'split-flap-rolls-and-takes-a-typed-seed', run: async t => {
    await desk('studio', ['seed'])(t);
    await t.eq('ten cells, RANDOM in the middle', () => [document.querySelectorAll(`.tile[data-m="seed"] [data-seednum] .flap`).length, document.querySelector(`.tile[data-m="seed"] [data-seednum]`).textContent, document.querySelector(`.tile[data-m="seed"] [data-seednum]`).classList.contains('rand')], [10, 'RANDOM', true]);
    await t.click(`.tile[data-m="seed"] [data-dice]`);
    await t.ok('Roll once keeps a fresh seed', () => Number.isInteger(S.seed) && get('seed.lock') === true);
    await t.do('the roll lands', () => {}, null, 900);
    await t.ok('the cells land on the seed, right-aligned, and rest', s => { const el = document.querySelector(`.tile[data-m="seed"] [data-seednum]`); return el.textContent === String(S.seed) && !el.classList.contains('rolling') && !el.classList.contains('rand') && el.querySelectorAll('.flap').length === 10; });
    await t.ok('the roll is named as the seed', () => /^Seed random → \d+$/.test(__i.undo()) || __i.undo());
    await t.click(`.tile[data-m="seed"] [data-seededit]`);
    await t.ok('pressing the cells opens the field in their place', () => document.activeElement.matches('[data-seedin]') && document.querySelector(`.tile[data-m="seed"] [data-seednum]`).hidden);
    await t.do('clear', () => { document.querySelector('[data-seedin]').value = ''; });
    await t.type('12 345'); await t.press('Enter');
    await t.eq('a typed seed, spaces ignored, is kept', () => [S.seed, get('seed.lock'), document.activeElement.matches('[data-seednum]')], [12345, true, true]);
    await t.click(`.tile[data-m="seed"] [data-ledtog="seed.lock"]`);
    await t.eq('the lock off: the next run draws its own, said RANDOM', () => [get('seed.lock'), document.querySelector(`.tile[data-m="seed"] [data-seednum]`).classList.contains('rand'), document.querySelector(`.tile[data-m="seed"] [data-seedrand]`).getAttribute('aria-pressed')], [false, true, 'true']);
    await t.do('lab, with the planner', () => { S.tier = 'lab'; S.mods = ['seed', 'planner']; S.vals['seed.lock'] = true; render(); });
    await t.eq('the LM seed follows the kept seed (Lab)', () => __i.text(`.tile[data-m="seed"] [data-lmseed]`), 'uses 12345 too');
    await t.click(`.tile[data-m="seed"] [data-ledtog="lm.seedlock"]`);
    await t.eq('switched off, the LM draws its own', () => [get('lm.seedlock'), __i.text(`.tile[data-m="seed"] [data-lmseed]`)], [false, 'LM seed: random']);
    await t.do('quick', () => { S.tier = 'quick'; render(); });
    await t.ok('Quick has no lock: Random each time carries the changed mark', () => !document.querySelector(`.tile[data-m="seed"] [data-ledtog="seed.lock"]`) && document.querySelector(`.tile[data-m="seed"] [data-seedrand]`).dataset.ctl === 'seed.lock');
  } },
  { id: 'saved-seeds-save-star-search-and-delete', run: async t => {
    await desk('studio', ['seed'])(t);
    await t.click(`.tile[data-m="seed"] [data-star]`);
    await t.ok('with a random seed there is nothing to save, and it says so', () => !document.querySelector('.sm-save') && /Roll or load a seed first/.test(__i.toast()));
    await t.click(`.tile[data-m="seed"] [data-loadseed="77104"]`);
    await t.eq('a saved seed loads and is kept', () => [S.seed, get('seed.lock')], [77104, true]);
    await t.ok('its row is the current one', () => document.querySelector(`.tile[data-m="seed"] [data-loadseed="77104"]`).closest('.sm-row').classList.contains('cur'));
    await t.click(`.tile[data-m="seed"] [data-star]`);
    await t.ok('the form says it updates the saved one', () => /Already saved/.test(__i.text('.sm-save')) && document.querySelector('.sm-save [data-smf="n"]').value === 'airy chorus');
    await t.do('describe it', () => { document.querySelector('[data-smf="d"]').value = 'open vowels'; });
    await t.click('.sm-save [data-smsave]');
    await t.ok('Update is one step, with Undo in its toast', () => /Updated saved seed 77104/.test(__i.toast()) && document.querySelector('#toast .toast-act')?.textContent === 'Undo' && __i.text(`.tile[data-m="seed"] [data-loadseed="77104"] .sm-dt`) === 'open vowels');
    await t.click(`.tile[data-m="seed"] [data-smstar="77104"]`);
    await t.ok('a star puts it first', () => document.querySelector(`.tile[data-m="seed"] [data-stars] .sm-row`).querySelectorAll('[data-smstar="77104"]').length === 1 && document.querySelector(`.tile[data-m="seed"] [data-smstar="77104"]`).getAttribute('aria-pressed') === 'true');
    await t.click(`.tile[data-m="seed"] [data-smq]`); await t.type('drum');
    await t.eq('search finds by tag', () => [...document.querySelectorAll(`.tile[data-m="seed"] [data-stars] [data-loadseed]`)].map(b => b.dataset.loadseed), ['1834920213']);
    await t.press('Escape');
    await t.eq('Escape clears the search first', () => [document.querySelector(`.tile[data-m="seed"] [data-smq]`).value, document.querySelectorAll(`.tile[data-m="seed"] [data-stars] .sm-row`).length], ['', 2]);
    await t.click(`.tile[data-m="seed"] [data-smrand]`);
    await t.eq('a random saved seed is never the one loaded (L-22)', () => S.seed, 1834920213);
  } },
  { id: 'takes-reels-preview-keys-and-cost', run: async t => {
    await desk('studio', ['takes'])(t);
    await t.eq('the cost of one take on the card', () => [__i.text(`.tile[data-m="takes"] .tk-gb`), __i.text(`.tile[data-m="takes"] .tk-cost > span:last-child`)], ['15.6 of 31.8 GB', '+2.3 GB per take']);
    await t.hover(`.tile[data-m="takes"] [data-take="3"]`);
    await t.eq('pointing at a reel previews its count', () => [__i.text(`.tile[data-m="takes"] .tk-lcd b`), document.querySelector(`.tile[data-m="takes"] .tk-lcd`).classList.contains('pv')], ['3', true]);
    await t.click(`.tile[data-m="takes"] [data-take="3"]`);
    await t.eq('a reel sets the count, and the slider holds the focus', () => [get('takes.n'), document.activeElement.matches('[data-takes]'), document.querySelectorAll(`.tile[data-m="takes"] .reel2.on`).length], [3, true, 3]);
    await t.eq('the cost follows', () => __i.text(`.tile[data-m="takes"] .tk-gb`), '20.2 of 31.8 GB');
    await t.press('ArrowRight');
    await t.eq('the arrows move it; the preview stands down (L-32)', () => [get('takes.n'), document.querySelector(`.tile[data-m="takes"] .tk-reels`).classList.contains('kbd'), document.querySelector(`.tile[data-m="takes"] .tk-lcd`).classList.contains('pv')], [4, true, false]);
    await t.press('Home');
    await t.eq('Home is one take', () => get('takes.n'), 1);
    await t.do('minimax', () => { if (typeof enterEngine === 'function') enterEngine('mm3'); S.backend = 'mm3'; render(); });
    await t.eq('MiniMax-Music3: four reels, each extra take costs its own', () => [document.querySelectorAll(`.tile[data-m="takes"] .reel2`).length, __i.text(`.tile[data-m="takes"] .tk-cost > span:last-child`)], [4, '+3.4 GB per extra take']);
  } },
  { id: 'scope-draws-what-the-controls-say', run: async t => {
    await desk('studio', ['sampler'], { 'samp.steps': 8 })(t);
    const pts = () => document.querySelector(`.tile[data-m="sampler"] [data-scope] polyline`).getAttribute('points');
    await t.eq('a dot a step while there are 32 or fewer', () => document.querySelectorAll(`.tile[data-m="sampler"] [data-scope] .sdot`).length, 9);
    const p0 = await t.js(pts);
    await t.do('the Power schedule', () => { S.vals['samp.sched'] = 'Power'; render(); });
    await t.ok('the curve is the schedule its menu pictures', p0 => document.querySelector('.tile[data-m="sampler"] [data-scope] polyline').getAttribute('points') !== p0, p0);
    const p1 = await t.js(pts);
    await t.do('a steeper exponent', () => { S.vals['sch.exp'] = 3; render(); });
    await t.ok('the curve follows the shape\'s slider', p1 => document.querySelector('.tile[data-m="sampler"] [data-scope] polyline').getAttribute('points') !== p1, p1);
    await t.eq('guidance on every step, by default', () => __i.text(`.tile[data-m="sampler"] [data-scoperead]`), 'guidance on every step');
    await t.do('a window and a cutoff', () => { S.mods = ['sampler', 'perf']; S.vals['samp.cs'] = 0.2; S.vals['perf.cfgcut'] = 0.6; render(); });
    await t.ok('the window is a band, the cutoff hatched', () => !!document.querySelector(`.tile[data-m="sampler"] [data-scope] rect[fill="#f5b54422"]`) && !!document.querySelector(`.tile[data-m="sampler"] [data-scope] rect[fill="url(#hatch)"]`));
    await t.ok('and the read-out says where guidance runs, numbers in bold, with the way to the cutoff', () => /^guidance from 0\.20 to 0\.60 \(CFG cutoff\)/.test(__i.text(`.tile[data-m="sampler"] [data-scoperead]`)) && document.querySelectorAll(`.tile[data-m="sampler"] [data-scoperead] b`).length === 2 && !!document.querySelector(`.tile[data-m="sampler"] [data-scoperead] button.scope-link`));
    await t.do('custom timesteps', () => { S.mods = ['sampler', 'latent']; S.vals['lat.ts'] = '0.9, 0.6, 0.3, 0'; render(); });
    await t.ok('custom timesteps replace the schedule: one dot each, and said', () => document.querySelectorAll('.tile[data-m="sampler"] [data-scope] .sdot').length === 4 && /4 custom timesteps from Latent & correction/.test(__i.text('.tile[data-m="sampler"] [data-scoperead]')));
    await t.do('minimax', () => { if (typeof enterEngine === 'function') enterEngine('mm3'); S.backend = 'mm3'; S.mods = ['sampler']; render(); });
    await t.ok('MiniMax-Music3 without Lua plugins samples natively', () => /native sampling/.test(__i.text(`.tile[data-m="sampler"] [data-scoperead]`)));
  } },
  { id: 'interval-handles-keep-apart', run: async t => {
    await desk('lab', ['sampler'])(t);
    await t.eq('two handles, at the run\'s ends', () => [...document.querySelectorAll(`.tile[data-m="sampler"] [data-ivl]`)].map(h => h.dataset.ivl), ['samp.cs', 'samp.ce']);
    await t.focus(`.tile[data-m="sampler"] [data-ivl="samp.cs"]`); await t.press('End');
    await t.eq('the start never passes the end (A-31)', () => get('samp.cs'), 0.99);
    await t.focus(`.tile[data-m="sampler"] [data-ivl="samp.ce"]`); await t.press('Home');
    await t.eq('nor the end the start', () => get('samp.ce'), 1);
    await t.ok('the window that holds no step turns guidance off, and says so', () => /guidance off/.test(__i.text(`.tile[data-m="sampler"] [data-scoperead]`)));
    await t.mouseDrag(`.tile[data-m="sampler"] [data-ivl="samp.cs"]`, [[-200, 0]]);
    await t.ok('a handle follows the hand', () => get('samp.cs') < 0.99);
  } },
  { id: 'latent-curve-follows-its-pad', run: async t => {
    await desk('lab', ['latent'])(t);
    const peak = () => { const p = document.querySelector('.tile[data-m="latent"] [data-dist] polyline:last-of-type').getAttribute('points').split(' ').map(x => x.split(',').map(Number)); return p.reduce((a, b) => (b[1] < a[1] ? b : a))[0]; };
    const x0 = await t.js(peak);
    await t.ok('grey and green start as one curve, at the middle', x0 => Math.abs(x0 - 150) < 2, x0);
    await t.do('shift', () => { S.vals['lat.shift'] = 1; render(); });
    await t.ok('the green curve moves with the shift', p => { const q = document.querySelector('.tile[data-m="latent"] [data-dist] polyline:last-of-type').getAttribute('points').split(' ').map(x => x.split(',').map(Number)); return q.reduce((a, b) => (b[1] < a[1] ? b : a))[0] > 180; });
    await t.ok('with the pad laid over it', () => !!document.querySelector('.tile[data-m="latent"] .xy-wrap > .xy.xy-overlay'));
  } },
  { id: 'repaint-region-moves-in-the-song-times', run: async t => {
    await desk('studio', ['repaint', 'takes'], {}, 'ace', { source: SONG, region: [0.35, 0.55] })(t);
    await t.eq('the song, and what will be repainted, in its times', () => [__i.text('.tile[data-m="repaint"] .songref b'), __i.text('[data-rpcap]')], ['Studio sketch 01', 'Will repaint 1:00 to 1:34']);
    await t.focus('[data-rpwave]'); await t.press('ArrowRight');
    await t.ok('a key moves the region a hundredth, a step named in the song\'s times (Q-24)', () => Math.abs(S.region[0] - 0.36) < 1e-9 && /^Repaint region 1:00–1:34 → 1:02–1:36$/.test(__i.undo()) || [S.region, __i.undo()]);
    await t.press('Shift+ArrowRight');
    await t.ok('Shift moves its end', () => Math.abs(S.region[1] - 0.57) < 1e-9 || S.region);
    await t.press('End');
    await t.ok('End takes it to the song\'s end, its length kept (L-36)', () => Math.abs(S.region[1] - 1) < 1e-9 && Math.abs(S.region[0] - 0.79) < 1e-6 || S.region);
    await t.click('[data-rpwave]', null, [0.2, 0.5]);
    await t.ok('a plain click outside leaves it (A-32)', () => Math.abs(S.region[0] - 0.79) < 1e-6 || S.region);
    await t.mouseDrag('[data-rpwave]', [[-60, 0]], [0.9, 0.5]);
    await t.ok('a drag inside carries it whole, one step', () => S.region[0] < 0.79 && Math.abs(S.region[1] - S.region[0] - 0.21) < 1e-6 && /^Repaint region /.test(__i.undo()) || [S.region, __i.undo()]);
    await t.ok('the spoken range follows', () => { const m = s => `${Math.floor(Math.round(s) / 60)}:${String(Math.round(s) % 60).padStart(2, '0')}`; return document.querySelector('[data-rpwave]').getAttribute('aria-valuetext') === `${m(S.region[0] * 171)} to ${m(S.region[1] * 171)}`; });
  } },
  { id: 'timbre-recent-files-and-blend', run: async t => {
    await desk('studio', ['timbre'], {}, 'ace', { refs: ['reference-mix.wav', 'my-band-master.flac', 'live-room-take.wav'], timbreFile: '' })(t);
    await t.eq('three recent files under their label', () => [...document.querySelectorAll('.tile[data-m="timbre"] .refs > .pref')].map(b => b.dataset.pref), ['reference-mix.wav', 'my-band-master.flac', 'live-room-take.wav']);
    await t.ok('with no track the blend stands back and says why', () => document.querySelector('[data-venn]').classList.contains('is-off') && /Load a track to blend it in/.test(__i.text('.venn-cap')));
    await t.click('.tile[data-m="timbre"] .pref[data-pref="my-band-master.flac"]');
    await t.eq('a recent file loads as the reference, a step named for it', () => [S.timbreFile, __i.undo()], ['my-band-master.flac', 'Load my-band-master.flac as the timbre reference']);
    await t.ok('the blend comes forward', () => !document.querySelector('[data-venn]').classList.contains('is-off') && !document.querySelector('.venn-cap .off-why'));
    await t.focus('[data-venn]'); await t.press('ArrowLeft');
    await t.eq('a key nudges the blend, the circles say it in percent', () => [get('timbre.str'), __i.text('[data-vennpct]')], [0.95, '95%']);
    await t.do('four recent', () => { S.refs = ['a.wav', 'b.wav', 'c.wav', 'd.wav']; render(); });
    await t.eq('more than three: the two latest and the rest behind one button (L-21)', () => [...document.querySelectorAll('.tile[data-m="timbre"] .refs > button')].map(b => b.textContent), ['a.wav', 'b.wav', '2 more…']);
  } },
  { id: 'perf-pad-corner-words', run: async t => {
    await desk('lab', ['sampler', 'perf'])(t);
    await t.eq('the corner words stand outside the pad, above and below its corners (A-07)', () => [...document.querySelectorAll('.tile[data-m="perf"] .xy-frame > .xy-cap span')].map(s => s.textContent), ['fastest', '', '', 'most faithful']);
  } },
  { id: 'knob-auto-and-symbol-labels', run: async t => {
    await desk('studio', ['sampler'], { 'samp.sched': 'Beta (Custom)' })(t);
    await t.ok('a knob that can be automatic carries its auto pill, lit, and stands in the auto colour (A-11, L-11)', () => !!document.querySelector(`.tile[data-m="sampler"] .kn.has-auto [data-auto="samp.shift"].on`) && document.querySelector(`.tile[data-m="sampler"] [data-knob="samp.shift"]`).classList.contains('is-auto'));
    await t.focus(`.tile[data-m="sampler"] [data-knob="samp.shift"]`); await t.press('ArrowUp');
    await t.eq('turning it leaves auto', () => [isAuto('samp.shift'), document.querySelector(`.tile[data-m="sampler"] [data-auto="samp.shift"]`).getAttribute('aria-pressed')], [false, 'false']);
    await t.click(`.tile[data-m="sampler"] [data-auto="samp.shift"]`);
    await t.eq('the pill puts it back', () => isAuto('samp.shift'), true);
    await t.eq('a symbol label is printed as written', () => [...document.querySelectorAll(`.tile[data-m="sampler"] .schctl .kn .kl.sym`)].map(k => k.textContent), ['α', 'β']);
  } },
];
