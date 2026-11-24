// Package 2's states (one file of tools/visual-baseline/slice-states/, gathered by slice-states.mjs): the containers
// and read-outs compared where the concept draws them, as parts on their own (vb.mjs `parts`: each and everything under
// it, with its role, name, words and attributes), since the tiles around them still hold parts later packages build:
// the planner's terminal on every engine and with the planner off, the repetition guard on and off, Tempo's LCD, the
// rack groups of Plan adherence and Latent & correction, the plans deck and its Endings, the re-planning dice, the flow
// windows, the speed gauge, the Sampler's schedule strip and a YuE2 plugin's controls, the fader bank, the drop zones,
// and the side units' objects.
//
//   node tools/visual-baseline/vb.mjs capture concept-containers --states slice --only "^containers-" --page <concept>/hotstep-tiers.html
//   node tools/visual-baseline/vb.mjs capture rebuild-containers --states slice --only "^containers-" --plan-from concept-containers
//   node tools/visual-baseline/vb.mjs compare concept-containers rebuild-containers --accept tools/visual-baseline/containers-accepted.json

/* the concept's page as it is, its lyric sheet in place (package 4 draws it) */
const scene = (bk, tier, mods, vals = {}, pedals = [], more = {}) => async h => {
  await h.js(([bk, tier, mods, vals, pedals, more]) => { if (bk !== S.backend) enterEngine(bk); S.tier = tier; S.mods = mods; S.pedals = pedals; Object.assign(S.vals, vals); Object.assign(S, more); render(); }, [bk, tier, mods, vals, pedals, more]);
  await h.settle(900);
};
const st = (id, run, parts, o = {}) => ({ id, w: 1600, h: 1080, scheme: 'light', reduced: false, pseudo: false, shot: false, partsOnly: true, run, parts, ...o });
const dark = { scheme: 'dark' };

const TERM = ['.tile[data-m="planner"] .term'], REP = ['.tile[data-m="repetition"] .term'];
export const STATES = [
  st('containers-term-ace-lab', scene('ace', 'lab', ['planner']), TERM),
  st('containers-term-ace-studio-dark', scene('ace', 'studio', ['planner']), TERM, dark),
  st('containers-term-ace-off-dark', scene('ace', 'lab', ['planner'], { 'lm.on': false }), TERM, dark),
  st('containers-term-yue-lab', scene('yue', 'lab', ['planner']), TERM),
  st('containers-term-mm3-lab-dark', scene('mm3', 'lab', ['planner']), TERM, dark),
  st('containers-term-hm-studio', scene('hm', 'studio', ['planner']), TERM),
  st('containers-term-ace-plan-set', scene('ace', 'lab', ['planner', 'tempo'], { 'tempo.bpm.auto': false, 'tempo.bpm': 128, 'tempo.key': 'C# major', 'lyrics.text': '[Verse 1]\na\n[Chorus]\nb' }), ['.tile[data-m="planner"] [data-term]']),
  st('containers-guard-ace-lab', scene('ace', 'lab', ['planner', 'repetition']), REP),
  st('containers-guard-ace-off-dark', scene('ace', 'lab', ['planner', 'repetition'], { 'lm.rep': 1 }), REP, dark),
  st('containers-guard-yue-lab', scene('yue', 'lab', ['planner', 'repetition'], { 'yp.rep': 1 }), REP),
  st('containers-lcd-ace-lab', scene('ace', 'lab', ['tempo']), ['.tile[data-m="tempo"] .lcd']),
  st('containers-lcd-ace-studio-set-dark', scene('ace', 'studio', ['tempo'], { 'tempo.bpm.auto': false, 'tempo.bpm': 120, 'tempo.sig': '3/4', 'tempo.beatio': true }), ['.tile[data-m="tempo"] .lcd'], dark),
  st('containers-lcd-ace-lab-trim', scene('ace', 'lab', ['tempo'], { 'tempo.trim': true }), ['.tile[data-m="tempo"] .lcd']),
  st('containers-lcd-yue-lab-dark', scene('yue', 'lab', ['tempo']), ['.tile[data-m="tempo"] .lcd'], dark),
  st('containers-lcd-sa3-studio', scene('sa3', 'studio', ['tempo']), ['.tile[data-m="tempo"] .lcd']),
  st('containers-groups-adhere', scene('yue', 'lab', ['sampler', 'adhere']), ['.tile[data-m="adhere"] .scope', '.tile[data-m="adhere"] .ad-groups']),
  st('containers-groups-adhere-on-dark', scene('yue', 'lab', ['sampler', 'adhere'], { 'yue.pcfg': 1.5, 'yue.cads': 0.5, 'yue.pws': 1.4 }), ['.tile[data-m="adhere"] .scope', '.tile[data-m="adhere"] .ad-groups'], dark),
  st('containers-groups-latent', scene('ace', 'lab', ['sampler', 'latent']), ['.tile[data-m="latent"] .ad-groups', '.tile[data-m="latent"] .ts']),
  st('containers-groups-latent-on-dark', scene('ace', 'lab', ['sampler', 'latent'], { 'dcw.on': true, 'lss.on': true }), ['.tile[data-m="latent"] .ad-groups'], dark),
  st('containers-deck', scene('mm3', 'lab', ['planner', 'plans']), ['.tile[data-m="plans"] .deck', '.tile[data-m="plans"] .endings']),
  st('containers-deck-replay-dark', scene('mm3', 'lab', ['planner', 'plans'], { 'mm.reuse': false, 'mm.plank': 'plank 11-19 22.41', 'mm.hidpath': 'chorus-idea' }), ['.tile[data-m="plans"] .deck', '.tile[data-m="planner"] .term-bar'], dark),
  st('containers-dice', scene('yue', 'lab', ['planner', 'replan']), ['.tile[data-m="replan"] .dice-cap']),
  st('containers-dice-off-dark', scene('yue', 'lab', ['planner', 'replan'], { 'yue.replan': false, 'yue.tries': 6 }), ['.tile[data-m="replan"] .dice-cap'], dark),
  st('containers-flow', scene('mm3', 'lab', ['sampler', 'flow']), ['.tile[data-m="flow"] .flowviz', '.tile[data-m="flow"] [data-flowcap]']),
  st('containers-flow-hop-dark', scene('mm3', 'lab', ['sampler', 'flow'], { 'mm.hop': 160 }), ['.tile[data-m="flow"] .flowviz', '.tile[data-m="flow"] [data-flowcap]'], dark),
  st('containers-gauge', scene('ace', 'lab', ['sampler', 'perf'], { 'perf.cfgcut': 0.5 }), ['.tile[data-m="perf"] .gauge']),
  st('containers-sched-strip', scene('ace', 'lab', ['sampler'], { 'samp.sched': 'Beta (Custom)' }), ['.tile[data-m="sampler"] .schctl']),
  st('containers-sched-composite-dark', scene('ace', 'studio', ['sampler'], { 'samp.sched': 'Composite (2-Stage)' }), ['.tile[data-m="sampler"] .schctl'], dark),
  st('containers-bank', scene('ace', 'lab', ['adapters'], {}, [], { slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }] }), ['.tile[data-m="adapters"] .faders']),
  st('containers-bank-mm3-dark', scene('mm3', 'lab', ['adapters'], {}, [], { slots: [{ file: 'my-band-ep', type: 'LoRA' }] }), ['.tile[data-m="adapters"] .faders'], dark),
  st('containers-drops', scene('ace', 'lab', ['post', 'timbre'], {}, ['master']), ['.tile[data-m="timbre"] .drop', '.pedal[data-pid="master"] .drop']),
  st('containers-drops-loaded-dark', scene('ace', 'lab', ['post', 'timbre'], {}, ['master'], { masterRef: 'master-ref.wav', timbreFile: 'kick-loop.wav' }), ['.tile[data-m="timbre"] .drop', '.pedal[data-pid="master"] .drop'], dark),
  st('containers-drop-cont', scene('yue', 'lab', ['score', 'cont']), ['.tile[data-m="cont"] .drop']),
  st('containers-units', scene('ace', 'lab', ['post'], { 'lyrics.text': '[Chorus]\nla' }, ['lrc', 'art', 'q']), ['.pedal[data-xid="lrc"] .u-lcd', '.pedal[data-xid="q"] .u-badge', '.pedal[data-xid="art"] .u-art']),
  st('containers-units-yue-dark', scene('yue', 'studio', ['post'], {}, ['fa', 'art']), ['.pedal[data-xid="fa"] .u-status', '.pedal[data-xid="art"] .u-art'], dark),
];

/** No region of the page is compared whole: every state compares its parts. */
export const SCOPES = {};
