// Package 3a's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the instruments
// where the page draws them, as parts on their own (vb.mjs `parts`: each and everything under it, with its
// role, name, words and attributes): Tempo's LCD and key wheel on every engine and at Quick, the prompt chips, the
// split-flap with its lock, Roll and Random and the saved seeds, the reels and their cost, the Sampler's scope with its
// handles and read-out, the latent curve, the repaint region, the timbre blend and its recent files, Performance
// trades' corner words, and the Sampler's knobs (the auto pill, the symbol labels).
//
//   npm run visual:all -- --only "^instruments$"            (and visual:approve, README.md)

/* the concept's page as it is, its lyric sheet in place (package 4 draws it) */
const scene = (bk, tier, mods, vals = {}, more = {}) => async h => {
  await h.js(([bk, tier, mods, vals, more]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, more); render(); }, [bk, tier, mods, vals, more]);
  await h.settle(900);
};
const st = (id, run, parts, o = {}) => ({ id, w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run, parts, ...o });
const dark = { scheme: 'dark' };
const SONG = { song: 1, title: 'Studio sketch 01', style: 'Lo-fi hip hop, mellow electric piano', dur: 171, hue: 215 };

const TEMPO = ['.tile[data-m="tempo"] .tempo'], SEED = ['.tile[data-m="seed"] .tile-body'], TAKES = ['.tile[data-m="takes"] .tile-body'];
const SCOPE = ['.tile[data-m="sampler"] .scope'];
export const STATES = [
  st('instruments-tempo-ace-lab', scene('ace', 'lab', ['tempo']), TEMPO),
  st('instruments-tempo-ace-studio-key-dark', scene('ace', 'studio', ['tempo'], { 'tempo.key': 'A# major' }), TEMPO, dark),
  st('instruments-tempo-ace-quick', scene('ace', 'quick', ['tempo']), TEMPO),
  st('instruments-tempo-mm3-studio-key', scene('mm3', 'studio', ['tempo'], { 'tempo.key': 'D# minor', 'tempo.bpm.auto': false, 'tempo.bpm': 96 }), TEMPO),
  st('instruments-tempo-yue-lab-dark', scene('yue', 'lab', ['tempo']), TEMPO, dark),
  st('instruments-tempo-sa3-studio', scene('sa3', 'studio', ['tempo']), [...TEMPO, '.tile[data-m="tempo"] .cap-note']),
  st('instruments-tempo-hm-lab-dark', scene('hm', 'lab', ['tempo']), ['.tile[data-m="tempo"] .cap-note'], dark),
  st('instruments-seed-studio', scene('ace', 'studio', ['seed']), SEED),
  st('instruments-seed-kept-dark', scene('ace', 'studio', ['seed'], { 'seed.lock': true }, { seed: 77104 }), SEED, dark),
  st('instruments-seed-quick', scene('ace', 'quick', ['seed']), SEED),
  st('instruments-takes-ace', scene('ace', 'studio', ['takes'], { 'takes.n': 3 }), TAKES),
  st('instruments-takes-mm3-dark', scene('mm3', 'lab', ['takes'], { 'takes.n': 2 }), TAKES, dark),
  st('instruments-scope-ace-studio', scene('ace', 'studio', ['sampler'], { 'samp.steps': 8 }), SCOPE),
  st('instruments-scope-ace-lab-window-dark', scene('ace', 'lab', ['sampler', 'perf'], { 'samp.cs': 0.2, 'perf.cfgcut': 0.6 }), SCOPE, dark),
  st('instruments-scope-ace-lab-ts', scene('ace', 'lab', ['sampler', 'latent'], { 'lat.ts': '0.9, 0.6, 0.3, 0' }), SCOPE),
  st('instruments-scope-mm3-studio', scene('mm3', 'studio', ['sampler']), SCOPE),
  st('instruments-scope-ace-power', scene('ace', 'studio', ['sampler'], { 'samp.sched': 'Power', 'sch.exp': 3 }), [...SCOPE, '.tile[data-m="sampler"] .schctl']),
  st('instruments-knobs-sampler-dark', scene('ace', 'studio', ['sampler'], { 'samp.sched': 'Beta (Custom)' }), ['.tile[data-m="sampler"] .samp-strip .kn', '.tile[data-m="sampler"] .schctl'], dark),
  st('instruments-latent', scene('ace', 'lab', ['latent'], { 'lat.shift': 0.5, 'lat.rescale': 1.5 }), ['.tile[data-m="latent"] .scope']),
  st('instruments-repaint-studio', scene('ace', 'studio', ['repaint'], {}, { source: SONG, region: [0.35, 0.55] }), ['.tile[data-m="repaint"] .wave']),
  st('instruments-repaint-lab-dark', scene('ace', 'lab', ['repaint'], {}, { source: SONG, region: [0.2, 0.4] }), ['.tile[data-m="repaint"] .wave'], dark),
  st('instruments-timbre', scene('ace', 'studio', ['timbre'], {}, { refs: ['reference-mix.wav', 'my-band-master.flac', 'live-room-take.wav'], timbreFile: '' }), ['.tile[data-m="timbre"] .refs', '.tile[data-m="timbre"] .venn', '.tile[data-m="timbre"] .venn-cap']),
  st('instruments-timbre-loaded-dark', scene('ace', 'studio', ['timbre'], { 'timbre.str': 0.6 }, { refs: ['a.wav', 'b.wav', 'c.wav', 'd.wav'], timbreFile: 'b.wav' }), ['.tile[data-m="timbre"] .refs', '.tile[data-m="timbre"] .venn', '.tile[data-m="timbre"] .venn-cap'], dark),
  st('instruments-perf-corners', scene('ace', 'lab', ['sampler', 'perf']), ['.tile[data-m="perf"] .xy-frame']),
];

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids) and the scope it compares them in. */
export const VISUAL_SET = { name: 'instruments', only: '^instruments-' };
