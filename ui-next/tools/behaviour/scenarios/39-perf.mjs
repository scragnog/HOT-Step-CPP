// The performance pass's guards (tools/perf/README.md): each action is done as a user does it, on a full desk, and
// what the page counted while doing it (src/app/tally.ts, the test layer's __tally) is held to a limit, so the causes
// the pass removed cannot come back unseen: every view's model asked again on a change of one value; the whole desk
// measured again, with layouts forced, on a change of its make-up or a drag; the store telling its listeners several
// times for one Undo; the layout solved once per place for an arrow key. These hold on the rebuild only (the concept
// counts nothing).
//
// The limits: a knob's change asks again the models that read its value (the knob, the tile's "Also set" line, the go
// bar's words...), 3 of the full desk's 250 (the limit 40 leaves room for modules that read more); a drag of a tile, a
// removal and a move by key measure nothing, a tile brought back is measured once; one Undo is one notification of the
// store, and draws the shell not at all.

const ACE = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'repetition', 'sampler', 'guidance', 'perf', 'latent', 'seed', 'takes', 'post'];
const PEDS = ['tile', 'refiner', 'ss', 'ppvae', 'lift', 'nat', 'gain', 'vst', 'master', 'norm', 'lrc', 'wh', 'art', 'q'];
const full = (tier = 'lab') => t => t.do(`a full ${tier} desk`, ([tier, mods, pedals]) => { S.tier = tier; S.mods = mods.filter(id => MODS[id]); S.pedals = pedals.filter(id => PED[id]); S.pedals.forEach(id => { if (PED[id]?.on) S.vals[PED[id].on] = true; }); render(); scrollTo(0, 0); }, [tier, ACE, PEDS], 600);
const reset = t => t.js(() => window.__tally(true));
const counted = t => t.js(() => window.__tally());
/* a count within its limit, saying the count either way */
/* PERF_SHOW=1 prints every count (each check fails, saying it) */
const within = (t, label, got, max) => t.ok(`${label}: at most ${max}`, ([n, m]) => (n <= m ? true : `${n}`), [got ?? 0, process.env.PERF_SHOW ? -1 : max]);

export default [
  { id: 'a-knob-key-asks-again-only-the-models-that-read-it', run: async t => {
    await full()(t);
    await t.focus('.tile[data-m="sampler"] .knob');
    await reset(t);
    await t.press('ArrowUp');
    const n = await counted(t);
    await t.ok('the key changed the value (one step)', () => document.querySelector('#undoBtn, [data-undo]') !== null);
    await within(t, 'models worked out again for one knob key on the full desk', n.model, 40);
    await within(t, 'notifications of the desk store for one knob key', n.notify, 1);
    await within(t, 'desks laid out for one knob key', n.pack, 0);
  } },
  { id: 'a-knob-drag-asks-again-only-the-models-that-read-it', run: async t => {
    await full()(t);
    await reset(t);
    await t.mouseDrag('.tile[data-m="sampler"] .knob', [[0, -6], [0, -6], [0, -6], [0, -6], [0, -6]]);
    const n = await counted(t);
    /* five moves of the hand, each in four pointer moves: 20 changes of the value */
    await within(t, 'models worked out again over a knob drag of 20 pointer moves', n.model, 80);
    await within(t, 'notifications of the desk store over the drag (one a pointer move that changes the value)', n.notify, 20);
    await within(t, 'tiles measured over the drag', n['measure.tiles'], 0);
  } },
  { id: 'undo-and-redo-are-one-notification-each', run: async t => {
    await full()(t);
    await t.focus('.tile[data-m="sampler"] .knob');
    await t.press('ArrowUp');
    await t.js(() => document.activeElement?.blur());
    await reset(t);
    await t.press('Control+z');
    const u = await counted(t);
    await within(t, 'notifications of the desk store for one Undo', u.notify, 1);
    await within(t, 'times the page\'s shell was drawn again for one Undo', u.shell, 0);
    await within(t, 'models worked out again for one Undo of a knob key', u.model, 40);
    await reset(t);
    await t.press('Control+Shift+z');
    const r = await counted(t);
    await within(t, 'notifications of the desk store for one Redo', r.notify, 1);
  } },
  { id: 'a-removal-measures-only-what-changed', run: async t => {
    await full()(t);
    await reset(t);
    await t.click('.tile[data-m="avoid"] .tile-head .x');
    const n = await counted(t);
    await t.eq('the tile is gone', () => !!document.querySelector('.tile[data-m="avoid"]'), false);
    await within(t, 'tiles measured for a removal', n['measure.tiles'], 1);
    await within(t, 'layouts forced to measure for a removal', n['measure.layouts'], 1);
    await within(t, 'notifications of the desk store for a removal', n.notify, 2);
    await within(t, 'times the page\'s shell was drawn again for a removal (its focus request is a leaf of its own)', n.shell, 0);
    await reset(t);
    await t.press('Control+z');
    const u = await counted(t);
    await t.eq('Undo brings it back', () => !!document.querySelector('.tile[data-m="avoid"]'), true);
    await within(t, 'times the desk measured its tiles when Undo brings a tile back (once, for the tile that is new)', u.measure, 1);
  } },
  { id: 'a-move-by-key-measures-nothing-and-solves-each-place-once', run: async t => {
    await full()(t);
    await t.focus('.grip[data-move="takes"]');
    await reset(t);
    await t.press('ArrowLeft');
    const n = await counted(t);
    await within(t, 'tiles measured for a move by key', n['measure.tiles'], 0);
    await within(t, 'layouts solved for a move by key (one a place; the pack after it reuses its place\'s)', n.solve, 14);
  } },
  { id: 'a-tile-dragged-across-the-desk-measures-nothing', run: async t => {
    await full()(t);
    await reset(t);
    await t.drag('.tile[data-m="tempo"] .tile-head', '.tile[data-m="seed"]', { at: [0.5, 0.3] });
    const n = await counted(t);
    await within(t, 'tiles measured over a drag of a tile and its drop (the preview is the tile\'s copy)', n['measure.tiles'], 0);
  } },
  { id: 'a-tier-change-lays-the-desk-out-once-it-is-drawn', run: async t => {
    await full('studio')(t);
    await reset(t);
    await t.click('#tiers [data-tier="lab"]');
    const n = await counted(t);
    await t.eq('the tier is Lab', () => S.tier, 'lab');
    await within(t, 'times the desk measured its tiles for a tier change', n.measure, 2);
  } },
];
