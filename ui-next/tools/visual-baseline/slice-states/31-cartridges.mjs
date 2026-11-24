// Package 3b's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the Adapters'
// cartridges compared where the concept draws them, as parts on their own (vb.mjs `parts`): Quick's flat cartridge and
// its line, one cartridge with its column at Studio (ACE-Step Single, two loaded and one unused, YuE2's pair and its NAR
// slot, HeartMuLa's one), a stack with its bar and section masks, at Lab with phase windows, the planner adapter and
// the forced Apply as, YuE2's master section and rebase, the adapter list; and the planner's XY pad (MiniMax-Music3's
// top-p off rail, the pad and How adventurous with the planner off).
//
//   node tools/visual-baseline/vb.mjs capture concept-cartridges --states slice --only "^cartridges-" --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-cartridges --states slice --only "^cartridges-" --plan-from concept-cartridges
//   node tools/visual-baseline/vb.mjs compare concept-cartridges rebuild-cartridges --accept tools/visual-baseline/cartridges-accepted.json

/* the concept's page as it is, its lyric sheet in place (package 4 draws it) */
const scene = (bk, tier, mods, vals = {}, more = {}) => async h => {
  await h.js(([bk, tier, mods, vals, more]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = []; Object.assign(S.vals, vals); Object.assign(S, more); render(); }, [bk, tier, mods, vals, more]);
  await h.settle(900);
};
/* the adapter list, opened from its button */
const list = sel => async h => { await h.js(sel => document.querySelector(sel).click(), sel); await h.settle(400); };
const then = (...fs) => async h => { for (const f of fs) await f(h); };
const st = (id, run, parts, o = {}) => ({ id, w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run, parts, ...o });
const dark = { scheme: 'dark' };

const A = ['.tile[data-m="adapters"] .tile-body'];
const ONE = [{ file: 'tape-lofi-v2', type: 'LoRA' }], TWO = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }];
const THREE = [...TWO, { file: 'synthwave-80s', type: 'DoRA' }];
const LYRICS = '[Verse]\na\nb\n[Chorus]{#1=0.5; #2=1}\nc\n[Verse 2]\nd';
export const STATES = [
  st('cartridges-quick-one', scene('ace', 'quick', ['adapters'], {}, { slots: ONE }), A),
  st('cartridges-quick-two-dark', scene('ace', 'quick', ['adapters'], { 'ad.adv': 'Advanced' }, { slots: [{ file: 'synthwave-80s', type: 'DoRA' }, ONE[0]] }), A, dark),
  st('cartridges-studio-one', scene('ace', 'studio', ['adapters'], {}, { slots: ONE }), A),
  st('cartridges-studio-unused-dark', scene('ace', 'studio', ['adapters'], {}, { slots: TWO }), A, dark),
  st('cartridges-studio-stack', scene('ace', 'studio', ['adapters'], { 'ad.adv': 'Advanced', 'lyrics.text': LYRICS }, { slots: TWO }), A),
  st('cartridges-lab-stack-dark', scene('ace', 'lab', ['adapters'], { 'ad.adv': 'Advanced', 'ad.stack': 'Sum', 'lyrics.text': LYRICS }, { slots: THREE.map((s, i) => (i ? s : { ...s, ph: [20, 80] })) }), A, dark),
  st('cartridges-lab-planner-forced', scene('ace', 'lab', ['adapters'], { 'ad.adv': 'Advanced', 'ad.mode': 'Merge', 'ad.planner': 'folk-voice-planner (LoRA)', 'lyrics.text': LYRICS }, { slots: TWO }), A),
  st('cartridges-lab-one-planner', scene('ace', 'lab', ['adapters'], { 'ad.home': 'Turbo' }, { slots: ONE }), A),
  st('cartridges-yue-studio', scene('yue', 'studio', ['adapters'], {}, { slots: [{ file: 'folk-voice-ar', type: 'LoRA' }] }), A),
  st('cartridges-yue-lab-two-dark', scene('yue', 'lab', ['adapters'], {}, { slots: [{ file: 'folk-voice-ar', type: 'LoRA' }, { file: 'tape-warmth-nar', type: 'LoKr' }] }), A, dark),
  st('cartridges-hm-studio', scene('hm', 'studio', ['adapters'], {}, { slots: [{ file: 'heart-folk-lm', type: 'LoRA' }] }), A),
  st('cartridges-mm3-lab', scene('mm3', 'lab', ['adapters'], { 'ad.plan': 0.5 }, { slots: [{ file: 'folk-voice-planner', type: 'LoRA' }] }), A),
  st('cartridges-list', then(scene('ace', 'studio', ['adapters'], {}, { slots: ONE }), list('.tile[data-m="adapters"] [data-cartadd2]')), ['#pop']),
  st('cartridges-planner-list-dark', then(scene('ace', 'lab', ['adapters'], {}, { slots: ONE }), list('.tile[data-m="adapters"] [data-plannerpick]')), ['#pop'], dark),
  st('cartridges-xy-topp-off', scene('mm3', 'lab', ['planner'], { 'mm.topp': 0 }), ['.tile[data-m="planner"] .xy-frame']),
  st('cartridges-xy-planner-off-dark', scene('ace', 'lab', ['planner'], { 'lm.on': false }), ['.tile[data-m="planner"] .xy-frame'], dark),
  st('cartridges-mood-planner-off', scene('ace', 'quick', ['planner'], { 'lm.on': false }), ['.tile[data-m="planner"] .tile-body']),
];

/** This file's visual set, for npm run visual:all (visual-all.mjs): the states it compares (
 *  a regular expression on their ids), the scope it compares them in, and its accepted differences. */
export const VISUAL_SET = { name: 'cartridges', only: '^cartridges-', accept: 'cartridges-accepted.json' };
