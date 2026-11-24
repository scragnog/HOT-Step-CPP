// Quick's fold (A-28, Q-05): the desk at Quick with ten modules and the Pedalboard, four of them folded behind the
// button that names them, and open; with a value set, so the button names one fewer; in both themes and at the five
// desktop widths. No engine has enough modules Quick shows for the fold, so the states put the concept's two stand-ins
// on the desk, as its suite does (defineModule; the rebuild's are the test build's testing/standins.ts), their face
// a bare line of words on both pages (the suite's <p> has the browser's margins, the rebuild's plain note none). The scope is
// the button and the note above the desk, and a screenshot of the desk where the fold shows.
//
//   node tools/visual-baseline/vb.mjs capture concept-fold --states slice --scope fold --only "^fold-" --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-fold --states slice --scope fold --only "^fold-" --plan-from concept-fold
//   node tools/visual-baseline/vb.mjs compare concept-fold rebuild-fold --scope fold --accept tools/visual-baseline/fold-accepted.json

const TEN = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'tx1', 'tx2', 'post'];
const scene = (open, vals = {}) => async h => {
  await h.js(([open, vals, mods]) => {
    ['tx1', 'tx2'].forEach((id, i) => { defineModule(id, { at: 300 + i, name: 'Stand-in ' + (i + 1), cat: 'Engine', w: 1, keys: [] }); defineFace(id, () => '<span>stand-in</span>'); });
    S.backend = 'ace'; S.tier = 'quick'; S.mods = mods; S.pedals = ['lrc']; S.vals['pp.lrc.on'] = true; Object.assign(S.vals, vals); S.qAll = open; render();
  }, [open, vals, TEN]);
  await h.settle(900);
};
const heightFor = w => (w <= 1000 ? 800 : w <= 1440 ? 900 : 1080);
const st = (id, run, o = {}) => ({ id, w: 1600, h: heightFor(o.w ?? 1600), scheme: 'light', reduced: false, pseudo: false, shot: false, run, ...o });

export const STATES = [
  st('fold-closed-light', scene(false), { pseudo: true, shot: '#bento' }),
  st('fold-closed-dark', scene(false), { scheme: 'dark', pseudo: true }),
  st('fold-open-light', scene(true), { pseudo: true, shot: '#bento' }),
  st('fold-open-dark', scene(true), { scheme: 'dark' }),
  st('fold-set-value', scene(false, { 'takes.n': 3 })),
  ...[1920, 1440, 1280, 1000].map(w => st(`fold-closed-w${w}`, scene(false), { w })),
  ...[1920, 1280].map(w => st(`fold-open-w${w}`, scene(true), { w })),
];

/** The regions compared: the fold's button and the note above the desk (the tiles are every other set's; where they stand is
 *  the screenshot's, and the layout probe's in the fold's notes). */
export const SCOPES = { fold: ['#bento>button:0', '#hiddenNote'] };

/** This file's visual set, for npm run visual:all (visual-all.mjs). */
export const VISUAL_SET = { name: 'fold', only: '^fold-', scope: 'fold', accept: 'fold-accepted.json' };
