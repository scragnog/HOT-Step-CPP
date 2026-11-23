// The state matrix. Each state: { id, w, h, scheme, reduced, pseudo, shot, run(h) }.
//   pseudo: also force :hover / :focus / :focus-visible / :focus-within / :active on one element per selector that uses them
//   shot:   also take a full-page screenshot
//   raw:    record straight after run(), without running the clock (for states that last one task)
// h (see vb.mjs): h.S(obj) assigns to the page's S (vals merged) and renders; h.js(fn, arg); h.click(sel); h.press(key);
// h.settle(ms); h.expect(fn, arg) fails the state's setup check when fn returns false (reported, never fatal).
// Add a state by adding an entry; keep ids stable, since baselines are compared by id.

const ENGINES = ['ace', 'mm3', 'yue', 'sa3', 'hm'];
const TIERS = ['quick', 'studio', 'lab'];
const WIDTHS = [1920, 1600, 1440, 1280, 1000, 760, 390];
const heightFor = w => (w <= 390 ? 844 : w <= 760 ? 1024 : w <= 1000 ? 800 : w <= 1440 ? 900 : 1080);

/* every module the engine supports, the Pedalboard with every pedal and side unit */
const full = (bk, tier) => async h => {
  await h.js(([bk, tier]) => {
    S.backend = bk; S.tier = tier;
    S.mods = Object.keys(MODS).filter(m => !MODS[m].source && m !== 'post' && (!MODS[m].only || MODS[m].only.includes(bk)));
    S.mods.push('post'); S.pedals = [...PEDALS, ...EXTRAS].map(p => p.id); render();
  }, [bk, tier]);
  await h.settle(900);
};
const studio = (bk = 'ace', mods = ['tempo', 'seed', 'takes', 'sampler', 'adapters', 'post'], tier = 'studio') => async h => {
  await h.js(([bk, mods, tier]) => { S.backend = bk; S.tier = tier; S.mods = mods; S.pedals = ['ss', 'lift', 'norm', 'q']; render(); }, [bk, mods, tier]);
  await h.settle(700);
};
const LYRICS = '[Verse 1]\nStreetlights hum a borrowed tune\nI count the windows, one by one\n\n[Chorus]\nHold the night, hold the night\nUntil the morning comes undone\n\n[Bridge]\nQuiet now\n\n[Outro]\nOne by one';

const LIST = [];
/* An open dropdown places itself from its button's box when it opens; placing it once more after the page has settled
   makes its position independent of how the desk's packing and scrolling went (it moved by a pixel between runs). */
const reDD = async h => { if (await h.js(() => { const d = document.querySelector('#dd'); if (!d || d.hidden || typeof placeDD !== 'function') return false; placeDD(); return true; })) await h.settle(); };
const add = s => LIST.push({ w: 1440, scheme: 'light', reduced: false, pseudo: false, shot: false, ...s, h: s.h || heightFor(s.w || 1440), run: async h => { await s.run(h); await reDD(h); } });

/* 1. every engine x tier, everything on the desk, light and dark, at 1600 */
for (const bk of ENGINES) for (const tier of TIERS) for (const scheme of ['light', 'dark'])
  add({ id: `desk-${bk}-${tier}-${scheme}`, w: 1600, scheme, run: full(bk, tier), pseudo: scheme === 'light' || (bk === 'ace' && tier === 'lab'), shot: tier !== 'studio' || bk === 'ace' });

/* 2. widths: every engine x tier in light, Lab in dark */
for (const w of WIDTHS) for (const bk of ENGINES) for (const tier of TIERS) {
  add({ id: `w${w}-${bk}-${tier}`, w, run: full(bk, tier), shot: bk === 'ace' && (w === 390 || w === 1000 || w === 1920) && tier === 'lab', pseudo: bk === 'ace' && tier === 'lab' && (w === 390 || w === 1000) });
  if (tier === 'lab' || (bk === 'ace' && tier === 'quick')) add({ id: `w${w}-${bk}-${tier}-dark`, w, scheme: 'dark', run: full(bk, tier) });
}

/* 3. reduced motion */
for (const bk of ENGINES) add({ id: `reduced-${bk}-lab`, w: 1600, reduced: true, run: full(bk, 'lab') });
add({ id: 'reduced-ace-quick-dark', w: 1440, reduced: true, scheme: 'dark', run: full('ace', 'quick') });

/* 3b. touch screens: no hover, coarse pointer */
const TOUCH = [{ name: 'hover', value: 'none' }, { name: 'any-hover', value: 'none' }, { name: 'pointer', value: 'coarse' }, { name: 'any-pointer', value: 'coarse' }];
for (const [w, tier] of [[1440, 'studio'], [1440, 'lab'], [390, 'lab'], [1000, 'quick']]) add({ id: `touch-w${w}-ace-${tier}`, w, media: TOUCH, run: full('ace', tier) });
add({ id: 'touch-w1440-picker', w: 1440, media: TOUCH, run: async h => { await full('ace', 'studio')(h); await h.press('F9'); await h.settle(900); } });

/* 4. every module, including the ones the engine does not use (parked) */
for (const bk of ['sa3', 'hm', 'yue']) add({ id: `parked-${bk}-studio`, w: 1600, pseudo: bk === 'sa3', run: async h => {
  await h.js(bk => { S.backend = bk; S.tier = 'studio'; S.mods = Object.keys(MODS).filter(m => !MODS[m].source); S.pedals = [...PEDALS, ...EXTRAS].map(p => p.id); render(); }, bk); await h.settle(900);
} });
add({ id: 'repaint-source', w: 1600, pseudo: true, shot: true, run: async h => {
  await studio('ace', ['tempo', 'seed'])(h);
  await h.js(() => { S.mods = S.mods.filter(m => !MODS[m].source); S.source = { song: 1 }; S.region = [0.35, 0.55]; S.mods.unshift('repaint'); render(); }); await h.settle(900);
  await h.expect(() => !!document.querySelector('.tile[data-m="repaint"]'));
} });
add({ id: 'repaint-source-lab-dark', w: 1440, scheme: 'dark', run: async h => {
  await full('ace', 'lab')(h);
  await h.js(() => { S.source = { song: 2 }; S.mods.unshift('repaint'); render(); }); await h.settle(900);
} });
add({ id: 'empty-desk-studio', w: 1440, pseudo: true, run: async h => { await h.js(() => { S.tier = 'studio'; S.mods = []; S.pedals = []; render(); }); await h.settle(); } });
add({ id: 'empty-desk-quick', w: 1440, shot: true, pseudo: true, run: async h => { await h.settle(); } });
add({ id: 'hidden-lab-note', w: 1440, run: async h => { await full('ace', 'lab')(h); await h.click('#tiers [data-tier="studio"]'); await h.settle(900); } });

/* 5. overlays, light and dark, and some at small widths */
const OVER = {
  engine: async h => { await h.click('#engBtn'); await h.expect(() => !document.querySelector('#engPanel').hidden); },
  'engine-model-dd': async h => { await h.click('#engBtn'); await h.click('#engPanel [data-variant="model"]'); await h.expect(() => !document.querySelector('#dd').hidden); },
  log: async h => { await h.click('#engMeter'); await h.expect(() => !document.querySelector('#logs').hidden); },
  notes: async h => { await h.click('#notesBtn'); await h.expect(() => !document.querySelector('#notes').hidden); },
  picker: async h => { await h.press('F9'); await h.settle(900); await h.expect(() => !document.querySelector('#pickerWrap').hidden); },
  'picker-filter': async h => { await h.press('F9'); await h.settle(600); await h.page.keyboard.type('se'); await h.settle(600); await h.press('ArrowDown'); await h.settle(); },
  'picker-cat': async h => { await h.press('F9'); await h.settle(600); await h.js(() => document.querySelectorAll('#pkBar button')[2]?.click()); await h.settle(); },
  ideas: async h => { await h.click('#ideasBtn'); await h.expect(() => !document.querySelector('#pop').hidden); },
  'dd-solver': async h => { await h.click('.tile[data-m="sampler"] [data-pick="samp.solver"]'); await h.expect(() => !document.querySelector('#dd').hidden); },
  'dd-sched': async h => { await h.click('.tile[data-m="sampler"] [data-pick="samp.sched"]'); await h.expect(() => !document.querySelector('#dd').hidden); },
  'dd-lcd': async h => { await h.click('[data-variant="lcd"]'); await h.expect(() => !document.querySelector('#dd').hidden); },
  'dd-chip': async h => { await h.js(() => { S.backend = 'mm3'; render(); }); await h.settle(700); await h.click('[data-variant="chip"]'); await h.expect(() => !document.querySelector('#dd').hidden); },
  'dd-stamp': async h => { await h.click('[data-variant="stamp"]'); await h.expect(() => !document.querySelector('#dd').hidden); },
  'song-menu': async h => { await h.page.hover('#latest .song:not(.pending)'); await h.click('#latest .song [data-songmenu]'); await h.expect(() => !document.querySelector('#menu').hidden); },
  toast: async h => { await h.js(() => toast('Saved the desk as a workflow called Evening')); await h.settle(); },
  'toast-action': async h => { await h.js(() => toast('Removed Sampler', { label: 'Undo', fn: () => {} })); await h.settle(); },
  save: async h => { await h.click('#saveBtn'); await h.expect(() => !document.querySelector('#saveRow').hidden); },
  'save-wf': async h => { await h.click('#saveBtn'); await h.click('[data-saveas="wf"]'); },
  drafts: async h => { await h.click('#draftsBtn'); await h.expect(() => !document.querySelector('#pop').hidden); },
  'drafts-saved': async h => { await h.js(() => { S.drafts = [{ name: 'Night drive', backend: 'ace', at: Date.UTC(2026, 0, 1, 9), prompt: 'synthwave night drive', vals: { 'lyrics.text': '[Verse]\nx' } }, { name: 'Folk idea', backend: 'yue', at: Date.UTC(2026, 0, 1, 8), prompt: 'warm folk', vals: {} }]; }); await h.click('#draftsBtn'); },
};
const overSetup = studio('ace', ['tempo', 'seed', 'takes', 'sampler', 'adapters', 'post']);
for (const [name, fn] of Object.entries(OVER)) for (const scheme of ['light', 'dark']) {
  const lang = name === 'dd-stamp' ? studio('ace', ['tempo', 'seed', 'takes', 'sampler', 'adapters', 'post'], 'lab') : overSetup;
  add({ id: `ov-${name}-${scheme}`, w: 1440, scheme, pseudo: scheme === 'light', shot: scheme === 'light' && ['engine', 'picker', 'notes', 'dd-solver', 'toast-action'].includes(name), run: async h => { await lang(h); await fn(h); await h.settle(); } });
}
add({ id: 'ov-score-yue', w: 1440, pseudo: true, shot: true, run: async h => { await full('yue', 'studio')(h); await h.js(() => openScore()); await h.settle(); await h.expect(() => !document.querySelector('#scoreModal').hidden); } });
add({ id: 'ov-score-yue-dark', w: 1440, scheme: 'dark', run: async h => { await full('yue', 'studio')(h); await h.js(() => openScore()); await h.settle(); } });
add({ id: 'ov-row-menu', w: 1440, pseudo: true, run: async h => {
  await h.js(() => { S.workflows = [{ name: 'My desk', tier: 'studio', mods: ['tempo', 'seed'], pedals: [] }]; saveUser(); renderPresets(); }); await h.settle();
  await h.js(() => document.querySelector('#presetList [data-rowmore]').click()); await h.settle(); await h.expect(() => !document.querySelector('#pop').hidden);
} });
for (const bk of ['mm3', 'yue', 'sa3', 'hm']) {
  add({ id: `ov-engine-${bk}`, w: 1440, run: async h => { await full(bk, 'studio')(h); await h.click('#engBtn'); await h.settle(); } });
  add({ id: `ov-dd-solver-${bk}`, w: 1440, run: async h => { await full(bk, 'lab')(h); await h.js(() => document.querySelector('[data-pick="samp.solver"], [data-pick="samp.sched"]')?.click()); await h.settle(); } });
}
for (const w of [1000, 390]) for (const name of ['engine', 'engine-model-dd', 'notes', 'picker', 'dd-solver', 'toast-action', 'drafts', 'song-menu', 'log', 'ideas'])
  add({ id: `ov-${name}-w${w}`, w, run: async h => { await overSetup(h); await OVER[name](h); await h.settle(); } });
add({ id: 'ov-picker-dark-w390', w: 390, scheme: 'dark', run: async h => { await overSetup(h); await OVER.picker(h); await h.settle(); } });
add({ id: 'ov-picker-reduced', w: 1440, reduced: true, run: async h => { await overSetup(h); await OVER.picker(h); await h.settle(); } });
add({ id: 'ov-engine-reduced', w: 1440, reduced: true, run: async h => { await overSetup(h); await OVER.engine(h); await h.settle(); } });

/* 6. populated states */
const POP = {
  adapters: async h => { await full('ace', 'lab')(h); await h.S({ slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'vocal-air', type: 'LoKr' }, { file: 'drum-bus', type: 'LoRA' }] }); },
  'adapters-studio': async h => { await full('ace', 'studio')(h); await h.S({ slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'vocal-air', type: 'LoKr' }] }); },
  'adapters-mm3': async h => { await full('mm3', 'lab')(h); await h.S({ slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }] }); },
  seeds: async h => { await studio('ace', ['seed', 'takes', 'tempo'])(h); await h.S({ seed: 1834920213 }); await h.js(() => document.querySelector('.tile[data-m="seed"] [data-star]').click()); await h.settle(); },
  'seeds-lab': async h => { await studio('ace', ['seed', 'takes'], 'lab')(h); await h.S({ seed: 77104, smq: 'air' }); },
  lyrics: async h => { await studio('ace', ['tempo', 'seed'])(h); await h.S({ vals: { 'lyrics.text': LYRICS } }); },
  'lyrics-quick': async h => { await h.S({ vals: { 'lyrics.text': LYRICS } }); await h.page.fill('#prompt', 'warm acoustic folk, soft vocals'); await h.settle(); },
  'lyrics-yue': async h => { await full('yue', 'lab')(h); await h.S({ vals: { 'lyrics.text': LYRICS } }); },
  'lyrics-mm3': async h => { await full('mm3', 'studio')(h); await h.S({ vals: { 'lyrics.text': LYRICS } }); },
  running: async h => { await studio('ace', ['tempo', 'seed', 'takes'])(h); await h.S({ vals: { 'takes.n': 3 } }); await h.page.fill('#prompt', 'warm acoustic folk'); await h.click('#go', 1200); },
  'running-mm3': async h => { await full('mm3', 'studio')(h); await h.page.fill('#prompt', 'synth pop'); await h.click('#go', 2500); },
  'running-yue-build': async h => { await full('yue', 'lab')(h); await h.js(() => document.querySelector('.tile[data-m="build"] button[data-build], .tile[data-m="build"] .sbtn')?.click()); await h.settle(1500); },
  finished: async h => { await studio('ace', ['tempo', 'seed', 'takes'])(h); await h.S({ vals: { 'takes.n': 2 } }); await h.page.fill('#prompt', 'warm acoustic folk'); await h.click('#go', 40000); },
  error: async h => { await studio('ace', ['tempo'])(h); await h.js(() => { S.songs.unshift({ id: 900, batch: 900, title: 'Too big', style: 'x', dur: 100, hue: 10, bk: 'ace', error: 'Needs 41.0 GB, the card has 31.8 GB. Lower the takes.', retryN: 2 }); renderSongs(); }); await h.settle(); },
  player: async h => { await studio('ace', ['tempo'])(h); await h.js(() => togglePlay(S.songs[1].id)); await h.settle(1500); await h.expect(() => !document.querySelector('#player').hidden); },
  'player-dark': async h => { await studio('ace', ['tempo'])(h); await h.js(() => togglePlay(S.songs[0].id)); await h.settle(1500); },
  modified: async h => { await h.js(() => loadPreset(FACTORY.find(w => w.name === 'Radio master'))); await h.settle(); await h.js(() => { setVal('pp.ss.str', 40); S.mods.push('seed'); markDirty('layout'); render(); }); await h.settle(); },
  'modified-rev': async h => { await h.js(() => document.querySelector('#presetList [data-wfopen="Deep sampler"]')?.click()); await h.settle(); await h.js(() => document.querySelector('#presetList .pre.rev')?.click()); await h.settle(); await h.js(() => { setVal('samp.steps', 12); }); await h.settle(); },
  'rev-detail': async h => { await h.js(() => { S.wfOpen.add('Deep sampler'); S.revInfo = 'Deep sampler|ace|Quick ideas|0'; renderPresets(); }); await h.settle(); await h.expect(() => !!document.querySelector('.pdetail')); },
  rename: async h => { await h.js(() => { S.workflows = [{ name: 'My desk', tier: 'studio', mods: ['tempo', 'seed'], pedals: [] }]; saveUser(); renderPresets(); }); await h.settle(); await h.js(() => document.querySelector('#presetList [data-wfren]').click()); await h.settle(); await h.expect(() => !!document.querySelector('.pren')); },
  'user-workflows': async h => { await h.js(() => { S.workflows = [{ name: 'My desk', tier: 'studio', mods: ['tempo', 'seed'], pedals: [] }, { name: 'Lab desk', tier: 'lab', mods: ['sampler'], pedals: ['norm'] }]; S.revs['My desk'] = [{ name: 'Mine', backend: 'ace', vals: { 'samp.steps': 20 }, user: true }]; S.wfOpen.add('My desk'); saveUser(); renderPresets(); }); await h.settle(); },
  changed: async h => { await full('ace', 'studio')(h); await h.js(() => { ['samp.steps', 'tempo.bpm', 'takes.n', 'pp.ss.on', 'samp.solver'].forEach(k => { const c = C[k]; if (!c) return; setVal(k, c.type === 'toggle' || typeof c.def === 'boolean' ? !c.def : c.opts ? c.opts[1] : (c.def || 0) + 1); }); }); await h.settle(); },
  undo: async h => { await studio('ace', ['tempo', 'seed', 'takes'])(h); await h.click('.tile[data-m="seed"] .tile-head .x'); await h.settle(1200); },
  'kb-focus': async h => { await studio('ace', ['tempo', 'seed', 'takes', 'sampler'])(h); for (let i = 0; i < 14; i++) await h.press('Tab', 120); await h.settle(1500); },
  'kb-focus-knob': async h => { await full('ace', 'lab')(h); await h.js(() => document.querySelector('.tile[data-m="sampler"] .knob')?.focus()); await h.press('ArrowUp', 200); await h.settle(1500); },
  'song-info': async h => { await studio('ace', ['tempo'])(h); await h.click('#songInfoBtn'); },
  wildcards: async h => { await studio('ace', ['adapters'])(h); await h.S({ slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }], latent: 'night-drive-take2.latent' }); await h.page.fill('#prompt', '{dreamy|gritty} indie folk, {acoustic guitar|banjo}'); await h.settle(); },
  'mm3-caption': async h => { await studio('mm3', ['tempo'])(h); await h.page.fill('#prompt', 'warm indie folk, fingerpicked guitar'); await h.settle(); await h.click('#structBtn'); await h.settle(); },
  'yue-tags': async h => { await full('yue', 'studio')(h); await h.S({ vals: { 'cap.src': 'Custom' } }); await h.page.fill('#prompt', 'synthwave, gated drums'); await h.settle(); },
  'yue-cont': async h => { await full('yue', 'lab')(h); await h.js(() => { S.contFile = 'my-demo.wav'; S.contDur = 180; S.contSel = [10, 40]; render(); }); await h.settle(); },
  'timbre-ref': async h => { await full('ace', 'studio')(h); await h.S({ timbreFile: 'reference-mix.wav', masterRef: 'my-band-master.flac' }); },
  'bypass-chain': async h => { await full('ace', 'lab')(h); await h.js(() => document.querySelector('.master[data-ledtog="pp.on"]')?.click()); await h.settle(); },
  'many-songs': async h => { await h.js(() => { for (let i = 0; i < 14; i++) S.songs.push({ id: 200 + i, title: 'Song ' + i, style: 'Style ' + i, dur: 120 + i, hue: i * 25, bk: ['ace', 'mm3', 'yue', 'sa3', 'hm'][i % 5] }); renderSongs(); }); await h.settle(); },
};
/* states that need a few steps: building a YuE song section by section, the adapter stack with section masks, menus
   with an armed delete, the other pages, the VRAM meter near full, and so on */
const clickJS = (h, sel) => h.js(sel => { const e = document.querySelector(sel); if (e) e.click(); return !!e; }, sel);
Object.assign(POP, {
  'build-planned': async h => { await full('yue', 'lab')(h); await h.S({ vals: { 'lyrics.text': LYRICS } }); await clickJS(h, '[data-bplan]'); await h.settle(3000); await h.expect(() => !!document.querySelector('.b-src')); },
  'build-takes': async h => { await full('yue', 'lab')(h); await h.S({ vals: { 'lyrics.text': LYRICS } }); await clickJS(h, '[data-bplan]'); await h.settle(3000); await clickJS(h, '[data-btake]'); await h.settle(800); await h.expect(() => !!document.querySelector('.b-take')); },
  'build-kept': async h => { await full('yue', 'lab')(h); await h.S({ vals: { 'lyrics.text': LYRICS } }); await clickJS(h, '[data-bplan]'); await h.settle(3000); await clickJS(h, '[data-btake]'); await h.settle(4000); await clickJS(h, '[data-bkeep]'); await h.settle(2000); await clickJS(h, '[data-btake]'); await h.settle(1500); },
  'adapter-stack': async h => { await full('ace', 'lab')(h); await h.S({ vals: { 'lyrics.text': LYRICS, 'ad.adv': 'Advanced' }, slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'vocal-air', type: 'LoKr' }, { file: 'drum-bus', type: 'DoRA' }] }); await h.expect(() => !!document.querySelector('.secband')); },
  'adapter-stack-studio': async h => { await full('ace', 'studio')(h); await h.S({ vals: { 'lyrics.text': LYRICS, 'ad.adv': 'Advanced' }, slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'vocal-air', type: 'LoKr' }] }); },
  'adapter-quick': async h => { await full('ace', 'quick')(h); await h.S({ slots: [{ file: 'tape-lofi-v2', type: 'DoRA' }] }); },
  'adapter-pick': async h => { await full('ace', 'lab')(h); await h.S({ vals: { 'ad.adv': 'Advanced' }, slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }] }); await clickJS(h, '[data-cartadd]'); await h.settle(); await h.expect(() => !document.querySelector('#pop').hidden); },
  'other-page': async h => { await clickJS(h, '[data-nav="Library"]'); await h.settle(); await h.expect(() => !document.querySelector('#other').hidden); },
  suggest: async h => { await studio('ace', ['tempo'])(h); await h.page.fill('#prompt', '[Verse]\nlate night city, 96 bpm'); await h.settle(); await h.expect(() => !!document.querySelector('#suggest button')); },
  'row-delete-armed': async h => { await h.js(() => { S.workflows = [{ name: 'My desk', tier: 'studio', mods: ['tempo', 'seed'], pedals: [] }]; saveUser(); renderPresets(); }); await h.settle(); await clickJS(h, '#presetList [data-rowmore]'); await h.settle(); await clickJS(h, '#pop [data-rowact="delpreset"]'); await h.settle(300); },
  'seed-delete-armed': async h => { await studio('ace', ['seed'], 'lab')(h); await clickJS(h, '.sm-del'); await h.settle(300); },
  'lyrics-wild': async h => { await studio('ace', ['tempo'])(h); await h.S({ vals: { 'lyrics.text': '[Verse]\n{Neon|Golden} lights on the {river|highway}\n\n[Chorus]\nHold on' } }); await h.settle(); },
  'tap-armed': async h => { await full('ace', 'studio')(h); await clickJS(h, '[data-tap]'); await h.settle(300); },
  'vram-full': async h => { await full('mm3', 'studio')(h); await h.js(() => { ['model.dit', 'model.lm', 'model.x'].forEach(k => { const f = C[k] && optsOf(k).find(x => /F16/i.test(x)); if (f) setVal(k, f); }); setVal('takes.n', 4); }); await h.settle(1500); },
  'vram-full-ace': async h => { await full('ace', 'studio')(h); await h.js(() => { ['model.dit', 'model.lm', 'model.x'].forEach(k => { const f = C[k] && optsOf(k).find(x => /F16|XL|4B/i.test(x)); if (f) setVal(k, f); }); setVal('takes.n', 9); }); await h.settle(1500); },
  'sched-composite': async h => { await full('ace', 'lab')(h); await h.js(() => setVal('samp.sched', 'Composite (2-Stage)')); await h.settle(); },
  'sched-beta': async h => { await full('ace', 'lab')(h); await h.js(() => setVal('samp.sched', 'Beta (Custom)')); await h.settle(); },
  'latent-on': async h => { await full('ace', 'lab')(h); await h.js(() => { setVal('dcw.on', true); setVal('den.on', true); setVal('lss.on', true); }); await h.settle(); },
  'chip-dd-auto': async h => { await full('mm3', 'studio')(h); await h.S({ vals: { 'lyrics.text': LYRICS, 'cap.src': 'Automatic' } }); await clickJS(h, '[data-pick="cap.src"]'); await h.settle(); },
});
for (const [name, fn] of Object.entries(POP)) add({ id: `pop-${name}`, w: 1440, pseudo: true, shot: ['adapters', 'seeds', 'lyrics', 'running', 'finished', 'error', 'player', 'modified'].includes(name), run: async h => { await fn(h); await h.settle(); } });
for (const name of ['adapters', 'seeds', 'lyrics', 'running', 'error', 'player-dark', 'modified', 'rev-detail', 'rename', 'kb-focus', 'changed', 'yue-cont'])
  add({ id: `pop-${name}-dark`, w: 1440, scheme: 'dark', run: async h => { await POP[name](h); await h.settle(); } });
for (const name of ['adapters', 'seeds', 'lyrics-quick', 'running', 'player', 'error'])
  add({ id: `pop-${name}-w390`, w: 390, run: async h => { await POP[name](h); await h.settle(); } });

/* 7. panels: collapsed, and drawers below 1180 */
add({ id: 'panel-left-collapsed', w: 1600, pseudo: true, run: async h => { await full('ace', 'studio')(h); await h.click('[data-collapse="left"]'); } });
add({ id: 'panel-right-collapsed', w: 1600, run: async h => { await full('ace', 'studio')(h); await h.click('[data-collapse="right"]'); } });
add({ id: 'panel-both-collapsed', w: 1600, shot: true, pseudo: true, run: async h => { await full('ace', 'lab')(h); await h.click('[data-collapse="left"]'); await h.click('[data-collapse="right"]'); } });
add({ id: 'panel-both-collapsed-dark', w: 1600, scheme: 'dark', run: async h => { await full('ace', 'lab')(h); await h.click('[data-collapse="left"]'); await h.click('[data-collapse="right"]'); } });
add({ id: 'panel-right-over-1280', w: 1280, pseudo: true, run: async h => { await full('ace', 'studio')(h); await h.js(() => { S.rOpen = true; paintPanels(); }); await h.settle(); } });
for (const w of [1000, 760, 390]) for (const side of ['left', 'right']) for (const scheme of ['light', 'dark'])
  add({ id: `drawer-${side}-w${w}-${scheme}`, w, scheme, pseudo: scheme === 'light' && w === 1000, shot: scheme === 'light' && w !== 760, run: async h => {
    await full('ace', 'studio')(h); await h.click(`.panel-toggles [data-open="${side}"]`); await h.settle(900);
  } });

/* 8. drag in progress */
const drag = (srcSel, toSel) => async h => {
  await h.js(srcSel => { const el = document.querySelector(srcSel); window.__dt = new DataTransfer(); el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: window.__dt })); }, srcSel);
  await h.settle(200);
  await h.js(toSel => { const t = document.querySelector(toSel), r = t.getBoundingClientRect(); const o = { bubbles: true, cancelable: true, dataTransfer: window.__dt, clientX: r.left + r.width * 0.3, clientY: r.top + 30 }; t.dispatchEvent(new DragEvent('dragenter', o)); t.dispatchEvent(new DragEvent('dragover', o)); }, toSel);
  await h.settle(900);
};
add({ id: 'drag-card', w: 1600, pseudo: true, shot: true, run: async h => { await studio()(h); await drag('#browser [data-drag]:not([data-drag^="move:"])', '.tile[data-m="seed"]')(h); await h.expect(() => document.body.classList.contains('dnd')); } });
add({ id: 'drag-move', w: 1600, run: async h => { await studio()(h); await drag('.tile[data-m="tempo"] [data-drag^="move:"]', '.tile[data-m="sampler"]')(h); } });
add({ id: 'drag-move-dark', w: 1600, scheme: 'dark', run: async h => { await studio()(h); await drag('.tile[data-m="tempo"] [data-drag^="move:"]', '.tile[data-m="sampler"]')(h); } });
add({ id: 'drag-picker-peek', w: 1600, run: async h => { await studio()(h); await h.press('F9'); await h.settle(900); await drag('#pkLeft [data-drag]', '.tile[data-m="seed"]')(h); } });
/* the drag image exists only until the next task: raw records it without running the clock */
add({ id: 'drag-image', w: 1600, raw: true, run: async h => { await studio()(h); await h.js(() => { const el = document.querySelector('#browser [data-drag]:not([data-drag^="move:"])'); el.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: new DataTransfer() })); }); await h.expect(() => !!document.querySelector('.drag-img')); } });
add({ id: 'drag-pedal', w: 1600, run: async h => { await studio()(h); await drag('#browser [data-drag^="pd:"]', '.tile[data-m="post"]')(h); } });

export const STATES = LIST;
