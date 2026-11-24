// Parts the slice's own modules do not draw, where the page draws them, as parts on their own (vb.mjs
// `parts`: each control and everything under it, keyed from its selector, with its role, name, words and attributes):
// MiniMax-Music3's Flow at Lab (its rack of LED switches, its foot with two labelled segmented choices), the
// Mastering pedal's VU meter, and a Quick choice row (the Sampler's). The tiles around them hold parts later packages
// build, so only these controls are compared.

const parts = (bk, tier, mods, pedals, sels, vals = {}) => async h => {
  await h.js(([bk, tier, mods, pedals, vals]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = pedals; Object.assign(S.vals, vals); render(); }, [bk, tier, mods, pedals, vals]);
  await h.settle(900);
  void sels;
};
const FLOW = ['.tile[data-m="flow"] .rackrow', '.tile[data-m="flow"] .tile-foot .foot-ctl'];
export const STATES = [
  { id: 'slice-parts-mm3-lab', w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run: parts('mm3', 'lab', ['flow', 'post'], ['master'], FLOW), parts: [...FLOW, '.pedal[data-pid="master"] .vu'] },
  { id: 'slice-parts-mm3-lab-dark', w: 1600, h: 1080, scheme: 'dark', reduced: false, pseudo: false, shot: false, partsOnly: true, run: parts('mm3', 'lab', ['flow', 'post'], ['master'], FLOW, { 'mm.rend': 'TensorRT', 'mm.stream': true }), parts: [...FLOW, '.pedal[data-pid="master"] .vu'] },
  { id: 'slice-parts-quick-row', w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run: parts('ace', 'quick', ['sampler'], [], []), parts: ['.tile[data-m="sampler"] .qchoice'] },
];
