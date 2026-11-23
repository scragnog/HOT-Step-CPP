// The slice's states: the concept's matrix (states.mjs) cut to what the rebuild draws so far. Each state puts on the
// desk only the modules, pedals and side units the slice builds (docs/03, "Shared parts only"), on the same engines,
// tiers, widths, schemes and motion settings as the concept's own states, and is run on both pages with the same
// setup: on the concept page, first the parts of the page the slice does not build yet are withdrawn where they
// would change the desk's layout (the lyric sheet, which the concept puts at the head of the desk; withdraw() below),
// and nothing else is changed. SCOPES names the regions of the page compared: the elements the slice draws.
//
//   node tools/visual-baseline/vb.mjs capture concept-slice --states slice --scope slice --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-slice --states slice --scope slice --plan-from concept-slice
//   node tools/visual-baseline/vb.mjs compare concept-slice rebuild-slice --scope slice

const ENGINES = ['ace', 'mm3', 'yue', 'sa3', 'hm'];
const TIERS = ['quick', 'studio', 'lab'];
/* the five desktop sizes the concept supports; its 760 and 390 states are the phone and tablet layouts */
const WIDTHS = [1920, 1600, 1440, 1280, 1000];
const heightFor = w => (w <= 390 ? 844 : w <= 760 ? 1024 : w <= 1000 ? 800 : w <= 1440 ? 900 : 1080);
/* every pure-data module, pedal and side unit (docs/03-definitions-coverage.md, "Shared parts only") */
const MODS = ['avoid', 'guidance', 'yuebatch'];
const BOARD = ['tile', 'refiner', 'ss', 'ppvae', 'lift', 'nat', 'gain', 'vst', 'norm', 'wh'];
const ONLY = { avoid: ['ace'], guidance: ['ace', 'mm3', 'sa3', 'hm'], yuebatch: ['yue'] };

/* The concept puts its lyric sheet at the head of the desk at Studio and Lab, and under the prompt at Quick; the lyrics
   package has not been rebuilt, so on the concept page it is withdrawn, through the page's own global, before a state
   is set. The rebuild's page has no such global: there this does nothing. */
const withdraw = h => h.js(() => { if (typeof lyricsBlock === 'function') window.lyricsBlock = () => ''; });
const scene = (bk, tier, mods, pedals) => async h => {
  await withdraw(h);
  await h.js(([bk, tier, mods, pedals]) => { S.backend = bk; S.tier = tier; S.mods = mods; S.pedals = pedals; render(); }, [bk, tier, mods, pedals]);
  await h.settle(900);
};
/* every slice module the engine uses, the Pedalboard with every slice pedal and side unit (the concept's full()) */
const full = (bk, tier) => scene(bk, tier, [...MODS.filter(m => !ONLY[m] || ONLY[m].includes(bk)), 'post'], BOARD);

const LIST = [];
const add = s => LIST.push({ w: 1440, scheme: 'light', reduced: false, pseudo: false, shot: false, ...s, h: s.h || heightFor(s.w || 1440) });

/* 1. every engine x tier, light and dark, at 1600 */
for (const bk of ENGINES) for (const tier of TIERS) for (const scheme of ['light', 'dark'])
  add({ id: `slice-desk-${bk}-${tier}-${scheme}`, w: 1600, scheme, run: full(bk, tier), pseudo: scheme === 'light' });
/* 2. the desktop widths: every engine x tier in light, Lab in dark */
for (const w of WIDTHS) for (const bk of ENGINES) for (const tier of TIERS) {
  add({ id: `slice-w${w}-${bk}-${tier}`, w, run: full(bk, tier) });
  if (tier === 'lab') add({ id: `slice-w${w}-${bk}-${tier}-dark`, w, scheme: 'dark', run: full(bk, tier) });
}
/* 3. reduced motion */
for (const bk of ENGINES) add({ id: `slice-reduced-${bk}-lab`, w: 1600, reduced: true, run: full(bk, 'lab') });
/* 4. modules and pedals the engine does not use: parked */
for (const bk of ['sa3', 'hm', 'yue', 'mm3']) add({ id: `slice-parked-${bk}-studio`, w: 1600, run: scene(bk, 'studio', [...MODS, 'post'], BOARD) });
/* 5. the empty desk, and the note naming the Lab modules a tier hides */
add({ id: 'slice-empty-desk-studio', w: 1440, pseudo: true, run: scene('ace', 'studio', [], []) });
add({ id: 'slice-empty-desk-quick', w: 1440, run: scene('ace', 'quick', [], []) });
add({ id: 'slice-empty-desk-lab-dark', w: 1440, scheme: 'dark', run: scene('ace', 'lab', [], []) });
add({ id: 'slice-hidden-lab-note', w: 1440, run: async h => { await full('ace', 'lab')(h); await h.click('#tiers [data-tier="studio"]'); await h.settle(900); } });
/* 6. values set: each module's and pedal's controls changed, so the changed markers, the "Also set" lines, the plates
   and the revealed controls show */
const SET = { 'pp.ss.on': true, 'pp.ss.str': 44, 'pp.ss.blend': 'Crossover', 'pp.ss.seedf': false, 'pp.norm.on': true, 'pp.norm.target': 'Custom', 'pp.lift.on': true, 'pp.lift.den': 0.6, 'pp.wh.on': true, 'pp.wh.beam': 8, 'apg.mom': 0.4, 'apg.norm': 6, 'avoid.text': 'jazz, piano', 'pp.gain': 2.5, 'yue.noisevar': 3, 'yue.coalesce': false };
/* set as a scene (S.vals, then render), as the concept's own states set values: a script's setVal calls are no
   person's gesture, and the concept folds them into one step named after everything the script changed */
const changed = (bk, tier) => async h => { await full(bk, tier)(h); await h.js(o => { Object.assign(S.vals, o); render(); }, SET); await h.settle(900); };
for (const bk of ['ace', 'yue']) for (const tier of TIERS) add({ id: `slice-changed-${bk}-${tier}`, w: 1600, pseudo: tier !== 'quick', run: changed(bk, tier) });
add({ id: 'slice-changed-ace-lab-dark', w: 1600, scheme: 'dark', run: changed('ace', 'lab') });
add({ id: 'slice-changed-ace-studio-w1000', w: 1000, run: changed('ace', 'studio') });
/* 7. the chain bypassed, and a removal with the Undo button showing */
add({ id: 'slice-bypass-chain', w: 1600, run: async h => { await full('ace', 'lab')(h); await h.click('.master[data-ledtog="pp.on"]'); } });
add({ id: 'slice-bypass-chain-studio', w: 1600, run: async h => { await full('ace', 'studio')(h); await h.click('.master[data-ledtog="pp.on"]'); } });
add({ id: 'slice-undo', w: 1440, run: async h => { await full('ace', 'studio')(h); await h.click('.tile[data-m="avoid"] .tile-head .x'); await h.settle(1200); } });
add({ id: 'slice-quick-ace-dark', w: 1440, scheme: 'dark', reduced: true, run: full('ace', 'quick') });

export const STATES = LIST;

/** The regions compared: what the slice draws. The desk (every tile, the board, the parked tray, the empty desk), the
 *  note above it, the tier switch, Undo and Redo, and the top bar's brand, navigation, engine button and theme switch.
 *  Left out: the VRAM meter (the models package's), the prompt card and lyric sheet, the go bar's contents, Latest, the
 *  Workflows list and the overlays. */
export const SCOPES = {
  slice: ['#bento', '#hiddenNote', '#tiers', '#undoBtn', '#redoBtn', '#engBtn', '#themeSeg', '#notesBtn', 'html>body:1>header:1>div:0>button:0', 'html>body:1>header:1>div:0>nav:1'],
  desk: ['#bento'],
  panel: ['#browser'],
};
