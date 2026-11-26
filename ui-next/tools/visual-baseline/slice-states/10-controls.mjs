// Package 1's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the shared
// controls where the page draws them, as parts on their own (vb.mjs `parts`: each control and everything
// under it, with its role, name, words and attributes), since the tiles around them still hold parts later packages
// build: Tempo's length ruler (on auto, set, on HeartMuLa, with Beat I/O and trim), the Sampler's number box and plugin
// pickers, its Quick rows (a value between its choices), the Planner's Quick pill and How adventurous, Build's foot of
// tokens, a side unit's chip selector, and the plugin lists open (the periodic table, the grid of curves, the cards).
//
//   npm run visual:all -- --only "^controls$"            (and visual:approve, README.md)

/* the lyric sheet stands at the head of the desk on both pages, as the concept puts it (package 4) */
const scene = (bk, tier, mods, vals = {}, pedals = []) => async h => {
  await h.js(([bk, tier, mods, vals, pedals]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = pedals; Object.assign(S.vals, vals); render(); }, [bk, tier, mods, vals, pedals]);
  await h.settle(900);
};
const open = (sel, ...setup) => async h => { await scene(...setup)(h); await h.click(sel); await h.settle(900); };
const st = (id, run, parts, o = {}) => ({ id, w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run, parts, ...o });

const DUR = ['.tile[data-m="tempo"] .dur'];
const SAMP = ['.tile[data-m="sampler"] .numbox', '.tile[data-m="sampler"] .pk-plugin'];
export const STATES = [
  st('controls-ruler-ace-studio', scene('ace', 'studio', ['tempo']), DUR),
  st('controls-ruler-ace-studio-set-dark', scene('ace', 'studio', ['tempo'], { 'tempo.dur.auto': false, 'tempo.dur': 300 }), DUR, { scheme: 'dark' }),
  st('controls-ruler-hm-studio', scene('hm', 'studio', ['tempo']), DUR),
  st('controls-ruler-sa3-lab-dark', scene('sa3', 'lab', ['tempo']), DUR, { scheme: 'dark' }),
  st('controls-ruler-ace-lab-markers', scene('ace', 'lab', ['tempo'], { 'tempo.beatio': true, 'tempo.beatbars': '8', 'tempo.dur': 20, 'tempo.dur.auto': false, 'tempo.trim': true }), DUR),
  st('controls-sampler-ace-lab', scene('ace', 'lab', ['sampler']), SAMP),
  st('controls-sampler-ace-lab-set-dark', scene('ace', 'lab', ['sampler'], { 'samp.steps': 20, 'samp.solver': 'Heun', 'samp.sched': 'Cosine' }), SAMP, { scheme: 'dark' }),
  st('controls-sampler-mm3-studio', scene('mm3', 'studio', ['sampler'], { 'samp.plugins': true }), SAMP),
  st('controls-quick-rows-between', scene('ace', 'quick', ['sampler'], { 'samp.steps': 20, 'samp.cfg': 11 }), ['.tile[data-m="sampler"] .qchoice']),
  st('controls-quick-rows-between-dark', scene('ace', 'quick', ['sampler'], { 'samp.steps': 40 }), ['.tile[data-m="sampler"] .qchoice'], { scheme: 'dark' }),
  st('controls-planner-quick', scene('ace', 'quick', ['planner']), ['.tile[data-m="planner"] .pill-tog', '.tile[data-m="planner"] .mood']),
  st('controls-planner-quick-set-dark', scene('ace', 'quick', ['planner'], { 'lm.on': false, 'lm.temp': 0.7 }), ['.tile[data-m="planner"] .pill-tog', '.tile[data-m="planner"] .mood'], { scheme: 'dark' }),
  st('controls-build-tokens', scene('yue', 'lab', ['build'], { 'lyrics.text': '[Verse]\na\n[Chorus]\nb' }), ['.tile[data-m="build"] .tok']),
  st('controls-build-tokens-set-dark', scene('yue', 'lab', ['build'], { 'lyrics.text': '[Verse]\na\n[Chorus]\nb', 'yue.btakes': 3, 'yue.bcond': 1.5 }), ['.tile[data-m="build"] .tok'], { scheme: 'dark' }),
  st('controls-unit-chips', scene('ace', 'lab', ['post'], { 'pp.q.on': true, 'pp.q.eval': 'Mastered' }, ['q']), ['.pedal[data-xid="q"] .chipsw']),
  st('controls-dd-solvers', open('.tile[data-m="sampler"] [data-pick="samp.solver"]', 'ace', 'lab', ['sampler']), ['#dd'], { shot: '#dd' }),
  st('controls-dd-schedules-dark', open('.tile[data-m="sampler"] [data-pick="samp.sched"]', 'ace', 'lab', ['sampler']), ['#dd'], { scheme: 'dark' }),
  st('controls-dd-sa3-samplers', open('.tile[data-m="sampler"] [data-pick="sa3.sampler"]', 'sa3', 'studio', ['sampler']), ['#dd']),
];

/** No region of the page is compared whole: every state compares its parts. */
export const SCOPES = {};

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids) and the scope it compares them in. */
export const VISUAL_SET = { name: 'controls', only: '^controls-' };
