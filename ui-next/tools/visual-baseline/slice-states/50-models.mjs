// Package 5's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the engine panel
// open (its cards and its models) on every engine at Quick, Studio and Lab, in both themes, at the desktop widths and
// with reduced motion; a model's list open; a model changed; and the VRAM meter in the top bar, at rest, over (a run
// that will be refused) and tight (one that does not fit at once). The desk is empty, or holds the Takes tile only, and
// is not compared: the scope `models` is the panel, the meter and the dropdown.
//
//   npm run visual:all -- --only "^models$"            (and visual:approve, README.md)

const scene = (bk, tier, mods = [], vals = {}) => async h => {
  await h.js(([bk, tier, mods, vals]) => { if (bk !== S.backend) { if (typeof enterEngine === 'function') enterEngine(bk); else S.backend = bk; } S.backend = bk; S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); render(); }, [bk, tier, mods, vals]);
  await h.settle(900);
};
const panel = (...setup) => async h => { await scene(...setup)(h); await h.click('#engBtn'); await h.settle(900); };
const list = (key, ...setup) => async h => { await panel(...setup)(h); await h.click(`#engPanel [data-pick="${key}"]`); await h.settle(900); };
const heightFor = w => (w <= 1000 ? 800 : w <= 1440 ? 900 : 1080);
const st = (id, run, o = {}) => ({ id, w: 1440, h: heightFor(o.w ?? 1440), scheme: 'light', reduced: false, pseudo: false, shot: false, run, ...o });

const ENGINES = ['ace', 'mm3', 'yue', 'sa3', 'hm'];
const F16 = { 'model.dit': 'Flow DiT · F16', 'model.lm': '8B planner · F16', 'model.x': 'Depth decoder · F16' };
export const STATES = [
  /* the panel on every engine at Studio (forced hover and focus on ACE-Step's), Quick and Lab, and in dark */
  ...ENGINES.map(bk => st(`models-panel-${bk}-studio`, panel(bk, 'studio'), { pseudo: bk === 'ace', shot: bk === 'ace' && '#engPanel' })),
  ...ENGINES.map(bk => st(`models-panel-${bk}-quick`, panel(bk, 'quick'))),
  ...ENGINES.map(bk => st(`models-panel-${bk}-lab-dark`, panel(bk, 'lab'), { scheme: 'dark', shot: bk === 'hm' && '#engPanel' })),
  /* the desktop widths, and reduced motion */
  ...[1920, 1600, 1280, 1000].map(w => st(`models-panel-w${w}`, panel('ace', 'studio'), { w, h: heightFor(w) })),
  st('models-panel-reduced', panel('mm3', 'lab'), { reduced: true }),
  /* a model's list, open */
  st('models-dd-ace-dit', list('model.dit', 'ace', 'studio'), { shot: '#dd' }),
  st('models-dd-yue-dit-dark', list('model.dit', 'yue', 'lab'), { scheme: 'dark' }),
  st('models-dd-hm-var', list('model.var', 'hm', 'lab')),
  st('models-dd-mm3-lm', list('model.lm', 'mm3', 'studio', [], { 'model.lm': '8B planner · Q6_K' })),
  /* models changed: the rows marked, the meter following */
  st('models-changed-ace', panel('ace', 'lab', [], { 'model.dit': 'Turbo · Q4_K_M', 'model.vae': 'Regrind V9b' }), { pseudo: true }),
  st('models-changed-hm-dark', panel('hm', 'studio', [], { 'model.var': 'HeartMuLa 3B RL', 'model.dit': 'Global · F16' }), { scheme: 'dark' }),
  /* the meter: at rest on each engine, over (four takes of MiniMax-Music3 on F16) and tight (eight of ACE-Step) */
  ...ENGINES.map(bk => st(`models-meter-${bk}`, scene(bk, 'studio'), { w: 1600, h: 1080 })),
  st('models-meter-over', scene('mm3', 'studio', ['takes'], { ...F16, 'takes.n': 4 }), { w: 1600, h: 1080 }),
  st('models-meter-over-dark', scene('mm3', 'studio', ['takes'], { ...F16, 'takes.n': 4 }), { w: 1600, h: 1080, scheme: 'dark' }),
  st('models-meter-tight', scene('ace', 'studio', ['takes'], { 'model.lm': '4B · BF16', 'takes.n': 8 }), { w: 1600, h: 1080 }),
];

/** The regions compared: the engine panel, the VRAM meter, the dropdown. */
export const SCOPES = { models: ['#engPanel', '#engMeter', '#dd'] };

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids) and the scope it compares them in. */
export const VISUAL_SET = { name: 'models', only: '^models-', scope: 'models' };
