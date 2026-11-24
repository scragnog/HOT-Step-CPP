// Package 3c's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the instruments of
// YuE2 and MiniMax-Music3 where the page draws them, as parts on their own (vb.mjs `parts`): Build by section
// empty, planned, with takes and a keep (its settings locked), at Quick and with a keep sung to old words; Continue from
// recording with its selection and ruler; the Lead sheet with its foot and the lyric-timing line; the score preview of a
// healthy, a long and a runaway plan and of a pasted sheet; and the Lab modules of the two engines (Flow renderer, Plans &
// endings, Re-planning, Plan adherence, Throughput). A build is given as a scene (S.build, takes included), so every run
// draws the same takes.
//
//   npm run visual:all -- --only "^instruments-3c$"            (and visual:approve, README.md)

/* the concept's page as it is, its lyric sheet in place (package 4 draws it) */
const scene = (bk, tier, mods, vals = {}, more = {}, after = null) => async h => {
  await h.js(([bk, tier, mods, vals, more]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, more); render(); }, [bk, tier, mods, vals, more]);
  await h.settle(900);
  if (after) { await h.js(after); await h.settle(900); }
};
const st = (id, run, parts, o = {}) => ({ id, w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run, parts, ...o });
const dark = { scheme: 'dark' };

const LYRICS = '[Verse]\nfirst line\nsecond line\n[Chorus]\nla la\n[Verse]\nthird line\n[Bridge]\nb';
const take = (id, hue, dur, w, words, note = '') => ({ id, ready: true, hue, dur, w, words, note });
const PLANNED = { planned: true, seed: 735257098, cur: 'Verse#1', plan: ['Verse#1', 'Chorus#1', 'Verse#2', 'Bridge#1'], takes: {}, kept: {}, locked: false, rendered: null };
const TAKES = { ...PLANNED, takes: { 'Verse#1': [take('t1a', 320, 24, 101, 'first line\nsecond line'), take('t1b', 110, 27, 202, 'first line\nsecond line', 'followed the score')] } };
const KEPT = { ...TAKES, kept: { 'Verse#1': 't1b' }, locked: true };
const STALE = { ...KEPT, takes: { 'Verse#1': [take('t1a', 320, 24, 101, 'old words'), take('t1b', 110, 27, 202, 'old words')] } };
const BUILD = ['.tile[data-m="build"] .tile-body'], CONT = ['.tile[data-m="cont"] .tile-body'], SCORE = ['.tile[data-m="score"] .tile-body'], MODAL = ['#scoreModal .sc-card'];
const lab = m => [`.tile[data-m="${m}"] .tile-body`];

export const STATES = [
  st('i3c-build-empty-lab', scene('yue', 'lab', ['build'], { 'lyrics.text': LYRICS }), BUILD),
  st('i3c-build-untagged-strip', scene('yue', 'studio', ['build'], { 'lyrics.text': '' }), ['.tile[data-m="build"]']),
  st('i3c-build-planned-studio-dark', scene('yue', 'studio', ['build'], { 'lyrics.text': LYRICS }, { build: PLANNED }), BUILD, dark),
  st('i3c-build-takes-quick', scene('yue', 'quick', ['build'], { 'lyrics.text': LYRICS }, { build: TAKES }), BUILD),
  st('i3c-build-kept-lab', scene('yue', 'lab', ['build'], { 'lyrics.text': LYRICS }, { build: KEPT }), BUILD),
  st('i3c-build-stale-lab-dark', scene('yue', 'lab', ['build'], { 'lyrics.text': LYRICS }, { build: STALE }), BUILD, dark),
  st('i3c-build-drift-studio', scene('yue', 'studio', ['build'], { 'lyrics.text': '[Intro]\nhi\n' + LYRICS }, { build: KEPT }), BUILD),
  st('i3c-cont-lab', scene('yue', 'lab', ['cont', 'score'], {}, { contFile: 'take.wav', contDur: 231, contSel: [10, 30] }), CONT),
  st('i3c-cont-build-dark', scene('yue', 'lab', ['cont', 'build', 'score'], {}, { contFile: 'a-much-longer-recording-name.flac', contDur: 140, contSel: [100, 130] }), CONT, dark),
  st('i3c-cont-empty', scene('yue', 'lab', ['cont', 'score']), CONT),
  st('i3c-score-studio', scene('yue', 'studio', ['score']), SCORE),
  st('i3c-score-lab-hard', scene('yue', 'lab', ['score', 'planner'], { 'yue.timing': 'Hard', 'yue.cfg.auto': false, 'yue.cfg': 2.1 }), SCORE),
  st('i3c-score-soft-dark', scene('yue', 'studio', ['score'], { 'yue.timing': 'Soft' }), SCORE, dark),
  st('i3c-score-abc', scene('yue', 'studio', ['score'], { 'yue.abc': 'X:1\nT:Stay\nM:4/4\nK:G\nG A B d | c B A G |' }), SCORE),
  st('i3c-score-notabc', scene('yue', 'studio', ['score'], { 'yue.abc': 'just some words' }), SCORE),
  st('i3c-modal-healthy', scene('yue', 'studio', ['score'], {}, {}, () => openScore(12)), MODAL),
  st('i3c-modal-runaway-dark', scene('yue', 'studio', ['score'], {}, {}, () => openScore(12345)), MODAL, dark),
  st('i3c-modal-long', scene('yue', 'studio', ['score'], { 'lyrics.text': '[Verse]\na\n[Chorus]\nb' }, {}, () => openScore(3)), MODAL),
  st('i3c-modal-pasted', scene('yue', 'studio', ['score'], { 'yue.abc': 'X:1\nT:Stay a little longer\nM:4/4\nL:1/4\nK:G\n"G" G A B d | "C" c B A G |]' }, {}, () => openScore()), MODAL),
  st('i3c-flow-lab', scene('mm3', 'lab', ['flow'], { 'mm.rend': 'TensorRT' }), lab('flow')),
  st('i3c-plans-lab-dark', scene('mm3', 'lab', ['plans', 'planner']), lab('plans'), dark),
  st('i3c-replan-lab', scene('yue', 'lab', ['replan', 'planner']), lab('replan')),
  st('i3c-adhere-lab', scene('yue', 'lab', ['adhere', 'sampler'], { 'yue.pcfg': 1.5 }), lab('adhere')),
  st('i3c-yuebatch-lab-dark', scene('yue', 'lab', ['yuebatch', 'takes'], { 'yue.chunk': 60 }), lab('yuebatch'), dark),
];

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids) and the scope it compares them in. */
export const VISUAL_SET = { name: 'instruments-3c', only: '^i3c-' };
