// Package 7's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the Workflows
// panel, its save row, its row menu, the Drafts list and the Saved read-out, where the page draws them, as
// parts on their own (vb.mjs `parts`: each and everything under it, with its role, name, words and attributes): the
// panel fresh, with user workflows and revisions (a long name, a revision of yours under a factory workflow), modified
// rows (yours: Revert and Update; the factory's: Save as…), a revision's details on a modified revision, the save row
// as a revision (with the warning of a changed desk) and as a workflow (a refused name), a rename refused, the row menu
// (and Delete armed), the drafts empty and with drafts of two engines, and the read-out after a refused write.
//
//   npm run visual:all -- --only "^workflows$"            (and visual:approve, README.md)

/* a desk and the lists: user workflows and revisions in the concept's shape (testing/compat/workflows.ts reads them) */
const scene = (o = {}) => async h => {
  await h.js(o => {
    S.tier = o.tier ?? 'studio'; S.mods = o.mods ?? []; S.pedals = []; render();
    S.workflows = o.workflows ?? []; S.revs = Object.assign(Object.create(null), o.revs ?? {}); saveUser();
    if (o.open) loadPreset(findPreset(o.open), false);
    if (o.rev) applyRev(o.rev[0], (S.revs[o.rev[0]] || []).find(r => r.name === o.rev[1]) || FACTORY_REVS_OF(o.rev[0], o.rev[1]));
    for (const [k, v] of Object.entries(o.set ?? {})) setVal(k, v);
    if (o.add) addItem(o.add);
    for (const w of o.expand ?? []) S.wfOpen.add(w);
    renderPresets();
  }, o);
  await h.settle(900);
};
const then = (fn, ...more) => async h => { await fn(h); for (const m of more) { await m(h); await h.settle(600); } };
const click = sel => h => h.click(sel);
const type = text => h => h.js(t => { const f = document.activeElement; f.value = t; f.dispatchEvent(new Event('input', { bubbles: true })); }, text);
const key = k => h => h.js(k => document.activeElement.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true })), k);
const st = (id, run, parts, o = {}) => ({ id, w: 1920, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run, parts, ...o });

const PANEL = ['#leftPanel'], POP = ['#pop'];
const MINE = { workflows: [{ name: 'Mine', tier: 'studio', mods: ['tempo', 'sampler'], pedals: [] }, { name: 'A very long workflow name th', tier: 'lab', mods: ['seed'], pedals: [] }],
  revs: { Mine: [{ name: 'One', backend: 'ace', vals: { 'samp.steps': 20 }, slots: [] }, { name: 'Two', backend: 'sa3', vals: {}, slots: [] }], 'Deep sampler': [{ name: 'Polishing mine', backend: 'ace', vals: { 'samp.steps': 44 }, slots: [] }] } };

/* the concept's factory revisions are looked up by its own table; the rebuild's by the test layer (findPreset alone
   finds workflows) */
const factoryRev = h => h.js(() => { window.FACTORY_REVS_OF = (wf, n) => (typeof FACTORY_REVS === 'object' ? (FACTORY_REVS[wf] || []).find(r => r.name === n) : { name: n, backend: 'ace', vals: {} }); });
const withRevs = run => async h => { await factoryRev(h); await run(h); };

export const STATES = [
  st('wf-panel-fresh', withRevs(scene({ tier: 'quick' })), PANEL),
  st('wf-panel-fresh-dark', withRevs(scene({ tier: 'quick' })), PANEL, { scheme: 'dark' }),
  st('wf-panel-yours', withRevs(scene({ ...MINE, open: 'Mine', mods: ['tempo', 'sampler'], expand: ['Deep sampler'] })), PANEL),
  st('wf-panel-modified', withRevs(scene({ ...MINE, open: 'Mine', rev: ['Mine', 'One'], set: { 'samp.steps': 21 }, add: 'seed' })), PANEL),
  st('wf-panel-modified-dark', withRevs(scene({ ...MINE, open: 'Mine', rev: ['Mine', 'One'], set: { 'samp.steps': 21 }, add: 'seed' })), PANEL, { scheme: 'dark' }),
  st('wf-panel-factory-modified', withRevs(scene({ tier: 'lab', open: 'Deep sampler', rev: ['Deep sampler', 'Polishing'], set: { 'samp.steps': 22 }, add: 'seed' })), PANEL),
  st('wf-panel-details', withRevs(then(scene({ ...MINE, open: 'Mine', rev: ['Mine', 'One'], set: { 'samp.steps': 21 } }), click('.pre.rev[data-rn="One"] [data-rinfo]'))), PANEL),
  st('wf-panel-details-factory-dark', withRevs(then(scene({ tier: 'lab', expand: ['Lo-fi tape'] }), click('.pre.rev[data-rn="Tape at 82 BPM"] [data-rinfo]'))), PANEL, { scheme: 'dark' }),
  st('wf-save-row-rev', withRevs(then(scene({ open: 'Seed hunter' }), click('#saveBtn'))), PANEL),
  st('wf-save-row-rev-warn', withRevs(then(scene({ open: 'Seed hunter', add: 'tempo' }), click('#saveBtn'), click('[data-saveas="rev"]'))), PANEL),
  st('wf-save-row-refused-dark', withRevs(then(scene({ open: 'Seed hunter' }), click('#saveBtn'), click('[data-saveas="wf"]'), type('Songwriter'), click('#saveOk'))), PANEL, { scheme: 'dark' }),
  st('wf-save-row-replace', withRevs(then(scene({ ...MINE }), click('#saveBtn'), click('[data-saveas="wf"]'), type('mine'), click('#saveOk'))), PANEL),
  st('wf-rename-refused', withRevs(then(scene({ ...MINE }), click('.pre.wfrow[data-preset="Mine"] [data-rowmore]'), click('#pop [data-rowact="wfren"]'), type('Songwriter'), key('Enter'))), PANEL),
  st('wf-row-menu', withRevs(then(scene({ ...MINE, open: 'Mine' }), click('.pre.rev[data-rn="One"] [data-rowmore]'))), POP),
  st('wf-row-menu-armed-dark', withRevs(then(scene({ ...MINE }), click('.pre.wfrow[data-preset="Mine"] [data-rowmore]'), click('#pop [data-rowact="delpreset"]'))), POP, { scheme: 'dark' }),
  st('wf-drafts-empty', withRevs(then(scene({ tier: 'quick' }), click('#draftsBtn'))), POP),
  st('wf-drafts-two-engines-dark', withRevs(then(scene({ tier: 'quick' }), h => h.js(() => { S.drafts = [{ id: 'a', name: 'named one', backend: 'yue', at: Date.parse('2026-11-23T10:00:00'), prompt: 'other words here', vals: {} }, { id: 'b', name: '', backend: 'ace', at: Date.parse('2026-01-02T09:30:00'), prompt: 'first draft', vals: { 'lyrics.text': '[Verse]\nhello there' } }]; saveDrafts(); }), click('#draftsBtn'))), POP, { scheme: 'dark' }),
];

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids) and the scope it compares them in. */
export const VISUAL_SET = { name: 'workflows', only: '^wf-' };
