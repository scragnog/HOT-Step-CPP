// The engine panel and the models (package 5), as scenarios: each engine's card (its tagline, output rate, songs per Generate, the testing badge), the models' heading
// and rows at each tier on each engine, a model row's file, quantisation, size and share of the card, its list and a
// choice from it (the step named by the model's role, the panel left open, the meter following), the panel's keys with
// a model row in it, and the VRAM meter's figures: what is loaded, what the next Generate needs, its LEDs and peak, and
// when a run will be refused or will not fit at once. An option is found by its words, which are its value in the
// simulated server.

/* what the panel and the meter show, installed in the page for the checks (which run there): the models heading, each
   row's role, file name, quantisation and size, and the meter's read-out and accessible name */
const helpers = t => t.js(() => { window.__m = {
  rows: () => [...document.querySelectorAll('#engPanel .pk-model')].map(b => [b.querySelector('.pk-kind')?.textContent, b.querySelector('.pk-val')?.textContent, b.querySelector('.qb')?.textContent ?? '', b.querySelector('.gb')?.textContent ?? '']),
  head: () => [document.querySelector('#engPanel .ep-mh b')?.textContent, document.querySelector('#engPanel .ep-mh span')?.textContent],
  meter: () => [document.querySelector('#emUse').textContent, document.querySelector('#engMeter').getAttribute('aria-label')],
}; });
const desk = (tier, mods = [], vals = {}, backend = 'ace') => async t => {
  await helpers(t);
  await t.do(`${backend} ${tier} desk`, ([tier, mods, vals, backend]) => {
    if (backend !== S.backend) { if (typeof enterEngine === 'function') enterEngine(backend); else S.backend = backend; }
    S.backend = backend; S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); render();
  }, [tier, mods, vals, backend], 900);
};
const tier = (t, tr) => t.do(`${tr}`, tr => { S.tier = tr; render(); }, tr, 900);
const open = async t => { await t.click('#engBtn'); };

export default [
  { id: 'engine-cards-say-what-each-engine-is', run: async t => {
    await desk('studio')(t); await open(t);
    await t.eq('each card: its name and its tagline in two lines', () => [...document.querySelectorAll('#engPanel .ep-card')].map(c => [c.querySelector('b').textContent, c.querySelector('small').innerHTML.replace(/<br\s*\/?>/, ' | ')]),
      [['ACE-Step 1.5', 'Lyrics and fields | Repaint'], ['MiniMax-Music3', 'Structured caption | 8B planner'], ['YuE2', 'Lead-sheet composer | long songs'], ['Stable Audio 3', 'Prompt only | exact length'], ['HeartMuLa', 'Tags + lyrics | in testing']]);
    await t.eq('the output rate, short, with its words as the tip', () => [...document.querySelectorAll('#engPanel .ep-specs i:first-child')].map(i => [i.textContent, i.title]),
      [['48k', '48 kHz output'], ['44.1k', '44.1 kHz output'], ['48k', '48 kHz output'], ['44.1k', '44.1 kHz output'], ['48k', '48 kHz output']]);
    await t.eq('how many songs one Generate makes, and its tip (ACE-Step\'s maximum is the definitions\', below)', () => [...document.querySelectorAll('#engPanel .ep-specs i:last-child')].slice(1).map(i => [i.textContent, i.title]),
      [['×4', 'Up to 4 songs per Generate'], ['×4', 'Up to 4 songs per Generate'], ['×1', 'One song per Generate'], ['×1', 'One song per Generate']]);
    await t.ok('ACE-Step makes several songs a Generate', () => /^×[89]$/.test(document.querySelector('#engPanel .ep-card[data-backend="ace"] .ep-specs i:last-child').textContent) && /^Up to [89] songs per Generate$/.test(document.querySelector('#engPanel .ep-card[data-backend="ace"] .ep-specs i:last-child').title));
    await t.eq('only HeartMuLa wears the testing badge, inside its scope', () => [...document.querySelectorAll('#engPanel .ep-badge')].map(b => [b.closest('.ep-card').dataset.backend, b.parentElement.className, b.textContent]), [['hm', 'ep-scope', 'testing']]);
  } },
  { id: 'models-follow-the-tier-and-the-engine', run: async t => {
    await desk('studio')(t); await open(t);
    await t.eq('Studio: every component of ACE-Step, each by its role, on the file the server selected', () => [__m.head(), __m.rows()],
      [['ACE-Step 1.5 models', 'Each one shows what it costs in VRAM'], [['Sound', 'XL Turbo', 'Q8_0', '5.3 GB'], ['Planner', '4B', 'Q8_0', '4.5 GB'], ['Decoder', 'ScragVAE', '', '0.3 GB']]]);
    await tier(t, 'quick');
    await t.eq('Quick shows the sound model only, and says Studio shows the rest', () => [__m.head(), __m.rows().map(r => r[0])], [['ACE-Step 1.5 models', 'Studio shows every component'], ['Sound']]);
    await t.click('#engPanel [data-backend="hm"]');
    await t.eq('on HeartMuLa at Quick: its variant and its language model, and that it is in testing', () => [__m.head(), __m.rows().map(r => r.slice(0, 2))],
      [['HeartMuLa models', 'Studio shows every component · in integration testing'], [['Variant', 'HeartMuLa 3B'], ['Language', 'Global']]]);
    await tier(t, 'lab');
    await t.eq('at Lab, HeartMuLa\'s four, named by what each model is (S-32)', () => __m.rows().map(r => r[0]), ['Variant', 'Language', 'Decoder', 'Codec']);
    await t.ok('a variant has no size of its own', () => document.querySelector('#engPanel [data-pick="model.var"] .gb') === null && document.querySelector('#engPanel [data-pick="model.var"] .mem') === null);
    await t.click('#engPanel [data-backend="mm3"]');
    await t.eq('MiniMax-Music3: sound, planner and detail', () => __m.rows(), [['Sound', 'Flow DiT', 'Q8_0', '2.6 GB'], ['Planner', '8B planner', 'Q8_0', '9.1 GB'], ['Detail', 'Depth decoder', 'Q8_0', '0.7 GB']]);
    await t.click('#engPanel [data-backend="yue"]');
    await t.eq('YuE2: sound and decoder', () => __m.rows().map(r => r.slice(0, 2)), [['Sound', 'YuE2'], ['Decoder', 'Standard VAE']]);
    await t.click('#engPanel [data-backend="sa3"]');
    await t.eq('Stable Audio 3: its sound model', () => __m.rows(), [['Sound', 'SA3 DiT', 'BF16', '2.9 GB']]);
  } },
  { id: 'a-model-row-names-its-file-and-its-share-of-the-card', run: async t => {
    await desk('studio')(t); await open(t);
    await t.eq('it is named as its row is, with the technical name after (C-28)', () => [...document.querySelectorAll('#engPanel .pk-model')].map(b => b.getAttribute('aria-label')),
      ['Sound model (DiT model): XL Turbo · Q8_0', 'Planner model (LM model): 4B · Q8_0', 'Decoder model (VAE decoder): ScragVAE']);
    await t.ok('its bar is its share of the 31.8 GB card', () => { const w = parseFloat(document.querySelector('#engPanel [data-pick="model.dit"] .mem i').style.getPropertyValue('--w')); return Math.abs(w - 5.3 / 31.8 * 100) < 0.01 || w; });
    await t.ok('nothing is marked changed on the engine\'s own files', () => !document.querySelector('#engPanel .pk-model.changed'));
  } },
  { id: 'choosing-a-model-from-its-list', run: async t => {
    await desk('studio')(t); await open(t);
    await t.click('#engPanel [data-pick="model.dit"]');
    await t.ok('its list: one row a model, the quantisations as chips, headed by the role and the card', () => (document.querySelector('#dd').classList.contains('dd-model') && /^Sound model\s*VRAM on a 31\.8 GB card$/.test(document.querySelector('#dd .dd-model-title').textContent.trim())
      && [...document.querySelectorAll('#dd .opt-mrow')].map(r => r.querySelector('b').textContent + ':' + r.querySelectorAll('.opt-q').length).slice(0, 2).join() === 'XL Turbo:3,Turbo:2') || document.querySelector('#dd').textContent.slice(0, 200));
    await t.ok('the file in use is the one chosen, and has focus', () => document.activeElement.matches('#dd [aria-selected="true"]') && document.activeElement.dataset.opt === 'XL Turbo · Q8_0');
    await t.click('#dd [data-opt="Turbo · Q4_K_M"]');
    await t.ok('a click chooses it, closes the list and leaves the panel open, focus on the row', () => (get('model.dit') === 'Turbo · Q4_K_M' && document.querySelector('#dd').hidden && !document.querySelector('#engPanel').hidden && document.activeElement.dataset.pick === 'model.dit') || [get('model.dit'), document.querySelector('#dd').hidden, document.querySelector('#engPanel').hidden, document.activeElement.className]);
    await t.eq('the step is named by the model\'s role (F-12)', () => UNDO.at(-1)?.label, 'Sound model XL Turbo · Q8_0 → Turbo · Q4_K_M');
    await t.eq('the row shows the new file, marked changed', () => [__m.rows()[0], document.querySelector('#engPanel [data-pick="model.dit"]').classList.contains('changed')], [['Sound', 'Turbo', 'Q4_K_M', '1.4 GB'], true]);
    await t.eq('the meter takes off what the file saves', () => __m.meter(), ['9.4', 'VRAM: 9.4 GB loaded, about 11.7 GB for the next Generate, on a 31.8 GB card']);
    await t.press('Escape'); await t.press('Control+z'); await open(t);
    await t.eq('Undo puts the engine\'s file back, and the meter with it', () => [get('model.dit'), __m.rows()[0][1], __m.meter()[0]], ['XL Turbo · Q8_0', 'XL Turbo', '13.3']);
  } },
  { id: 'the-panels-keys-with-a-model-row-in-it', run: async t => {
    await desk('studio')(t); await open(t);
    await t.ok('the panel opens on the engine in use', () => document.activeElement.matches('#engPanel .ep-card[aria-checked="true"][data-backend="ace"]'));
    await t.press('Tab');
    await t.ok('Tab goes from the engines to the first model row', () => document.activeElement.dataset.pick === 'model.dit' || document.activeElement.outerHTML.slice(0, 80));
    await t.press('ArrowRight');
    await t.ok('an arrow on a model row moves no engine', () => (document.activeElement.dataset.pick === 'model.dit' && S.backend === 'ace' && document.querySelector('#dd').hidden) || document.activeElement.outerHTML.slice(0, 80));
    await t.press('ArrowDown');
    await t.ok('ArrowDown opens its list on the file in use', () => !document.querySelector('#dd').hidden && document.activeElement.dataset.opt === 'XL Turbo · Q8_0');
    await t.press('ArrowDown'); await t.press('Enter');
    await t.ok('the arrows and Enter choose another', () => (get('model.dit') !== 'XL Turbo · Q8_0' && document.querySelector('#dd').hidden && document.activeElement.dataset.pick === 'model.dit') || [get('model.dit'), document.activeElement.className]);
    await t.press('ArrowDown'); await t.press('Escape');
    await t.ok('Escape closes only the list, focus back on its row', () => document.querySelector('#dd').hidden && !document.querySelector('#engPanel').hidden && document.activeElement.dataset.pick === 'model.dit');
    await t.press('Escape');
    await t.ok('then the panel, focus back on its button', () => document.querySelector('#engPanel').hidden && document.activeElement.id === 'engBtn');
  } },
  { id: 'the-vram-meter-adds-up-the-engine', run: async t => {
    await desk('studio')(t);
    await t.eq('ACE-Step: its three files, its text encoder and working memory loaded; the planner\'s track for the next run', () => __m.meter(), ['13.3', 'VRAM: 13.3 GB loaded, about 15.6 GB for the next Generate, on a 31.8 GB card']);
    await t.eq('the LEDs: lit to what is loaded, outlined to what the next run needs', () => [document.querySelectorAll('#emBar i.on').length, document.querySelectorAll('#emBar i.nx').length], [13, 3]);
    await t.ok('the peak stands at what the next run needs, and the ticks at 8, 16 and 24 GB', () => { const pk = parseFloat(document.querySelector('#emBar .em-peak').style.left), em = [...document.querySelectorAll('#emBar em')].map(e => parseFloat(e.style.left));
      return (Math.abs(pk - 15.64 / 31.8 * 100) < 0.01 && em.length === 3 && em.every((x, i) => Math.abs(x - (i + 1) * 8 / 31.8 * 100) < 0.01)) || [pk, em]; });
    await t.ok('its tip and its read-out name the card', () => /the 31\.8 GB card/.test(document.querySelector('#engMeter').title) && /^\/ 31\.8\s*$/.test(document.querySelector('#engMeter .em-of').textContent));
    await desk('studio', ['planner'], { 'lm.on': false })(t);
    await t.eq('without planning first, a take adds nothing', () => __m.meter()[1], 'VRAM: 13.3 GB loaded, about 13.3 GB for the next Generate, on a 31.8 GB card');
    await desk('studio', [], {}, 'mm3')(t);
    await t.eq('MiniMax-Music3 holds its own', () => __m.meter(), ['16.2', 'VRAM: 16.2 GB loaded, about 16.2 GB for the next Generate, on a 31.8 GB card']);
    await desk('studio', [], {}, 'sa3')(t);
    await t.eq('Stable Audio 3', () => __m.meter()[0], '5.7');
  } },
  { id: 'the-vram-meter-says-when-a-run-will-not-fit', run: async t => {
    await desk('studio', ['takes'], { 'takes.n': 4 }, 'mm3')(t);
    await t.do('the F16 files', () => { setVal('model.dit', 'Flow DiT · F16'); setVal('model.lm', '8B planner · F16'); setVal('model.x', 'Depth decoder · F16'); }, null, 900);
    await t.ok('four takes of MiniMax-Music3 on F16 are refused: the meter turns over and shows what they need (G-B-08)', () => (document.querySelector('#engMeter').classList.contains('over') && !document.querySelector('#engMeter').classList.contains('tight') && __m.meter()[0] === '37.4' && /27\.2 GB loaded, about 37\.4 GB/.test(__m.meter()[1])) || __m.meter());
    await t.ok('its last LEDs are red, and every one past the card is outlined', () => document.querySelectorAll('#emBar i.r').length === 4 && document.querySelectorAll('#emBar i.on, #emBar i.nx').length === 32);
    await t.do('one take', () => setVal('takes.n', 1), null, 900);
    await t.ok('one take fits: the read-out is what is loaded again', () => !document.querySelector('#engMeter').classList.contains('over') && __m.meter()[0] === '27.2');
    await desk('studio', ['takes'], { 'takes.n': 8 })(t);
    await t.do('the 4B BF16 planner', () => setVal('model.lm', '4B · BF16'), null, 900);
    await t.ok('eight takes of ACE-Step do not fit at once: tight, and the read-out is what they need', () => (document.querySelector('#engMeter').classList.contains('tight') && __m.meter()[0] === '35.6') || __m.meter());
  } },
  { id: 'the-panel-follows-an-engine-switch', run: async t => {
    await desk('lab')(t); await open(t);
    await t.do('a planner of its own on ACE-Step', () => setVal('model.lm', '1.7B · Q8_0'));
    await t.click('#engPanel [data-backend="mm3"]');
    await t.eq('the panel is MiniMax-Music3\'s at once, the chosen card in focus', () => [__m.head()[0], __m.rows()[1].slice(0, 2), document.activeElement.dataset.backend], ['MiniMax-Music3 models', ['Planner', '8B planner'], 'mm3']);
    await t.click('#engPanel [data-backend="ace"]');
    await t.eq('back on ACE-Step its own planner is back', () => [__m.rows()[1].slice(0, 3), __m.meter()[0]], [['Planner', '1.7B', 'Q8_0'], '10.8']);
  } },
  { id: 'a-narrow-panel-brings-its-engine-into-view', run: async t => {
    await t.resize(760, 1024);
    await desk('studio', [], {}, 'hm')(t); await open(t);
    await t.ok('the engines are a strip that scrolls sideways, HeartMuLa\'s card in view (B-21)', () => { const cs = document.querySelector('#engPanel .ep-cards'), c = cs.querySelector('[aria-checked="true"]').getBoundingClientRect(), b = cs.getBoundingClientRect();
      return (cs.scrollWidth > cs.clientWidth && cs.scrollLeft > 0 && c.left >= b.left - 1 && c.right <= b.right + 1) || [cs.scrollWidth, cs.clientWidth, cs.scrollLeft]; });
    await t.ok('the panel stays inside the window', () => { const r = document.querySelector('#engPanel').getBoundingClientRect(); return (r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight) || [r.left, r.right, r.bottom]; });
  } },
];
