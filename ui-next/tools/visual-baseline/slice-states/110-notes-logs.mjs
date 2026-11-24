// Package 11's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the concept's
// overlay states for the notes and the engine log (states.mjs ov-notes, ov-log, at 1440 in both themes and at 1000 and
// 390), and besides them the notes scrolled to their foot on a short window (the × in reach, S-35, the edge fades), the
// log on the other engines and with a model changed, and the notes over the open log. The desk is the slice's (its
// modules, pedals and side unit drawn whole) and is not compared: the scope `notes` is the drawer, its scrim, the log
// and the VRAM meter.
//
//   node tools/visual-baseline/vb.mjs capture concept-notes --states slice --scope notes --only "^nl11-" --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-notes --states slice --scope notes --only "^nl11-" --plan-from concept-notes
//   node tools/visual-baseline/vb.mjs compare concept-notes rebuild-notes --scope notes --accept tools/visual-baseline/notes-accepted.json

const scene = (bk = 'ace', vals = {}) => async h => {
  await h.js(([bk, vals]) => { if (bk !== S.backend) { if (typeof enterEngine === 'function') enterEngine(bk); else S.backend = bk; } S.backend = bk; S.tier = 'studio'; S.mods = ['avoid', 'guidance', 'post']; S.pedals = ['gain', 'norm', 'wh']; Object.assign(S.vals, vals); render(); }, [bk, vals]);
  await h.settle(900);
};
const notes = async h => { await h.click('#notesBtn'); await h.settle(900); await h.expect(() => !document.querySelector('#notes').hidden); };
const log = async h => { await h.click('#engMeter'); await h.settle(900); await h.expect(() => !document.querySelector('#logs').hidden); };
const heightFor = w => (w <= 390 ? 844 : w <= 1000 ? 800 : 900);
const st = (id, run, o = {}) => ({ id, w: 1440, h: heightFor(o.w ?? 1440), scheme: 'light', reduced: false, pseudo: false, shot: false, run, ...o });
const both = (...fs) => async h => { for (const f of fs) await f(h); };

export const STATES = [
  /* the notes: in both themes (forced hover and focus in light), at 1000 and 390, and scrolled to their foot */
  st('nl11-notes-light', both(scene(), notes), { pseudo: true, shot: '#notes' }),
  st('nl11-notes-dark', both(scene(), notes), { scheme: 'dark' }),
  st('nl11-notes-w1000', both(scene(), notes), { w: 1000 }),
  st('nl11-notes-w390', both(scene(), notes), { w: 390 }),
  st('nl11-notes-foot', both(scene(), notes, async h => { await h.js(() => { const n = document.querySelector('#notes'); n.scrollTop = n.scrollHeight; n.dispatchEvent(new Event('scroll')); }); await h.settle(); }), { h: 700 }),
  /* the log: in both themes, at 1000 and 390, on the other engines, with a model changed, under the notes */
  st('nl11-log-light', both(scene(), log), { pseudo: true, shot: '#logs' }),
  st('nl11-log-dark', both(scene(), log), { scheme: 'dark' }),
  st('nl11-log-w1000', both(scene(), log), { w: 1000 }),
  st('nl11-log-w390', both(scene(), log), { w: 390, shot: '#logs' }),
  ...['mm3', 'yue', 'sa3', 'hm'].map(bk => st(`nl11-log-${bk}`, both(scene(bk), log))),
  st('nl11-log-changed', both(scene('ace', { 'model.dit': 'Turbo · Q4_K_M', 'model.vae': 'Regrind V9b' }), log)),
  st('nl11-log-under-notes-dark', both(scene(), log, notes), { scheme: 'dark' }),
];

/** The regions compared: the notes drawer and its scrim, the engine log, the VRAM meter that opens it. */
export const SCOPES = { notes: ['#notes', '#notesScrim', '#notesClose', '#notesTitle', '#logs', '#logEngine', '#logsClose', '#logText', '#engMeter', '#emBar', '#emRead', '#emUse'] };

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids), the scope it compares them in, and its accepted differences. */
export const VISUAL_SET = { name: 'notes', only: '^nl11-', scope: 'notes', accept: 'notes-accepted.json' };
