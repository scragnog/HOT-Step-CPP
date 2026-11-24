// Tests of the core's additions for the instruments (package 3a: src/derive/instruments.ts, the seed's and a span's
// step words in src/derive/steps.ts, the node rule for a bespoke part that houses its children in src/derive/desk.ts),
// each against what the concept does (25-module-tempo.js, 25-module-seed.js, 25-module-takes.js and 35-gobar.js
// vramNeed, 25-module-sampler.js sigmas and scopeRead, 25-module-latent.js distSVG, 25-module-repaint.js).
//
//   node tools/test-instruments.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Node, Tier, Value } from '../src/definitions/types.ts';
import {
  bindPart, capHas, capToggle, distOf, indexOf, nodeKeys, nodeShown, partAct, partModel, regionMove, seedTyped, sigmasOf, vramNeed, wheelWalk, listKey, selectionKey,
  type Change, type Desk, type Msg,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';
import { MODEL_FILES, FIGURES } from '../src/app/services/models.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, state: Desk['state'] = {}): Desk =>
  ({ engine, tier, vals, auto, state, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const def = (id: string) => DEFINITIONS.find(d => d.id === id)!;
const s = (m: unknown) => say(m as Msg);
const node = (id: string, part: string): Node => {
  let out: Node | undefined;
  const walk = (ns: Node[]) => { for (const x of ns) { if (!out && x.part === part) out = x; walk(x.children ?? []); } };
  walk(('face' in def(id) ? (def(id) as { face: Node[] }).face : []) ?? []);
  return out!;
};
const keyOf = (id: string, x: Node) => nodeKeys(def(id)).get(x) ?? '';
const model = (id: string, part: string, d: Desk) => partModel(ix, def(id), node(id, part), d, keyOf(id, node(id, part)), 'tile');
/** Acts on a part, as the page does, and returns the steps it took and what it said. */
function act(id: string, part: string, d0: Desk, name: string, arg?: unknown) {
  let d = d0; const steps: Change[] = [], told: string[] = [];
  const props = bindPart(ix, def(id), node(id, part), () => d, c => { steps.push(c); d = { ...d, vals: { ...d.vals, ...Object.fromEntries(Object.entries(c.vals).filter((e): e is [string, Value] => e[1] !== null)) }, auto: { ...d.auto, ...c.auto }, state: { ...d.state, ...c.state } }; });
  const done = partAct(ix, props.node, props, d, name, arg, m => told.push(s(m)));
  return { steps, told, done, desk: d };
}

/* ---- the key wheel */
{
  const keys = (def('tempo').params.find(p => p.key === 'tempo.key')!.options as { value: string }[]).map(o => o.value);
  ok('right goes round the ring', wheelWalk(keys, '', 'E minor', 'ArrowRight') === 'B minor');
  ok('up from a minor is its relative major', wheelWalk(keys, '', 'B minor', 'ArrowUp') === 'D major');
  ok('down from a minor is Auto, down from Auto nothing', wheelWalk(keys, '', 'D minor', 'ArrowDown') === '' && wheelWalk(keys, '', '', 'ArrowDown') === null);
  ok('from Auto, left is the last major (F)', wheelWalk(keys, '', '', 'ArrowLeft') === 'F major');
  ok('Home is Auto', wheelWalk(keys, '', 'A major', 'Home') === '');
  const m = model('tempo', 'keyWheel', desk('ace', 'studio', ['tempo'], { 'tempo.key': 'A# major' }))!;
  const w = m.instrument!.wheel!, bb = w.segs.find(x => x.value === 'A# major')!;
  ok('the wheel spells B♭ as a musician does, and knows it is chosen', bb.letter === 'B' && bb.acc === '♭' && bb.name === 'B♭ major' && bb.sel && !w.solo, bb);
  ok('twelve majors and twelve minors', w.segs.filter(x => x.ring === 'maj').length === 12 && w.segs.filter(x => x.ring === 'min').length === 12);
  const q = model('tempo', 'keyWheel', desk('ace', 'quick', ['tempo']))!;
  ok('at Quick the wheel edits nothing: its LCD stands alone', q.instrument!.wheel!.solo && q.params.length === 0);
  ok('on Stable Audio 3 it still houses the LCD (Beat I/O)', nodeShown(ix, node('tempo', 'keyWheel'), desk('sa3', 'studio', ['tempo'])) && !nodeShown(ix, node('tempo', 'keyWheel'), desk('heartmula', 'studio', ['tempo'])));
  const r = act('tempo', 'keyWheel', desk('ace', 'studio', ['tempo'], { 'tempo.key': 'E minor' }), 'walk', 'ArrowRight');
  ok('a key on the wheel is a step from the key in force', r.steps.length === 1 && s(r.steps[0]!.words.label) === 'Key E minor → B minor', r.steps.map(x => s(x.words.label)));
}

/* ---- TAP and the prompt chips */
{
  const r = act('tempo', 'tapTempo', desk('ace', 'lab', ['tempo']), 'tap', 500);
  ok('two taps half a second apart set 120 BPM, off auto', r.desk.vals['tempo.bpm'] === 120 && r.desk.auto['tempo.bpm'] === false, r.desk);
  const fast = act('tempo', 'tapTempo', desk('ace', 'lab', ['tempo']), 'tap', 100);
  ok('taps faster than its range are put at its end (A-25)', fast.desk.vals['tempo.bpm'] === 240);
  ok('a chip goes in as an item', J(capToggle('warm pads', '120 BPM')) === J({ text: 'warm pads, 120 BPM', on: true }));
  ok('another of its group replaces it', capToggle('warm pads, 120 BPM', '140 BPM').text === 'warm pads, 140 BPM');
  ok('a tempo inside a sentence is not an item, and stays (S-04)', !capHas('a build at 90 BPM', '90 BPM') && capToggle('a build at 90 BPM', '90 BPM').text === 'a build at 90 BPM, 90 BPM');
  ok('pressed again it comes out', J(capToggle('x, D major', 'D major')) === J({ text: 'x', on: false }));
  const d = desk('sa3', 'studio', ['tempo'], { 'prompt.text': 'pads' }), c = act('tempo', 'capChips', d, 'chip', '120 BPM');
  ok('a chip is a step of the prompt, named for it', c.steps.length === 1 && s(c.steps[0]!.words.label) === 'Add 120 BPM to the prompt' && c.desk.vals['prompt.text'] === 'pads, 120 BPM', c.steps.map(x => s(x.words.label)));
  const own = model('tempo', 'capChips', desk('sa3', 'studio', ['tempo'], { 'tempo.bpm': 96, 'tempo.key': 'A# minor' }, { 'tempo.bpm': false }))!;
  ok('the user\'s own tempo and key are offered first (S-11)', J(own.instrument!.caps!.chips.slice(0, 2).map(x => x.text)) === J(['96 BPM', 'Bb minor']), own.instrument!.caps!.chips);
}

/* ---- the seed */
{
  ok('a typed seed: spaces and commas ignored', J(seedTyped(' 12,345 ')) === J({ seed: 12345 }));
  ok('empty is random', J(seedTyped('')) === J({ seed: null }));
  ok('past 2^53 kept as its decimal text', J(seedTyped('18446744073709551615')) === J({ seed: '18446744073709551615' }));
  ok('a word or past 2^64-1 is refused', 'error' in seedTyped('12a') && 'error' in seedTyped('18446744073709551616'));
  const r = act('seed', 'splitFlap', desk('ace', 'studio', ['seed']), 'type', '4242');
  ok('typed, it is kept: the seed and its lock in one step, named as the seed', r.steps.length === 1 && r.desk.state.seed === 4242 && r.desk.vals['seed.lock'] === true && s(r.steps[0]!.words.label) === 'Seed random → 4242', r.steps.map(x => s(x.words.label)));
  const bad = act('seed', 'splitFlap', desk('ace', 'studio', ['seed']), 'type', 'nope');
  ok('a word is refused and said', !bad.done && !bad.steps.length && /A seed is a whole number/.test(bad.told[0] ?? ''), bad.told);
  const rnd = act('seed', 'splitFlap', desk('ace', 'studio', ['seed'], { 'seed.lock': true }, {}, { seed: 7 }), 'random');
  ok('Random each time: no seed, no lock', rnd.desk.state.seed === null && rnd.desk.vals['seed.lock'] === false && s(rnd.steps[0]!.words.label) === 'Seed 7 → random');
  const f = model('seed', 'splitFlap', desk('ace', 'lab', ['seed', 'planner'], { 'seed.lock': true }, {}, { seed: 77104 }))!.instrument!.flap!;
  ok('the cells: the seed right-aligned in ten', f.cells.join('') === '     77104' && !f.rand && f.lock, f);
  ok('the LM seed follows the kept seed', s(f.lm) === 'uses 77104 too');
  const noPlanner = model('seed', 'splitFlap', desk('ace', 'lab', ['seed'], { 'seed.lock': true }, {}, { seed: 77104 }))!.instrument!.flap!;
  ok('with the planner off the desk, no LM link', noPlanner.lm === null);
  const stars = [{ v: 1, n: 'one', d: '', tags: ['drums'] }, { v: 2, n: '', d: 'pads', tags: [], star: true }];
  const sv = act('seed', 'savedSeeds', desk('ace', 'studio', ['seed'], { 'seed.lock': true }, {}, { seed: 3, stars }), 'save', { name: '', desc: 'x', tags: 'a, b,' });
  ok('saving a seed: named seed_3 by default, tags split, a step', J((sv.desk.state.stars as unknown[])[0]) === J({ v: 3, n: 'seed_3', d: 'x', tags: ['a', 'b'] }) && s(sv.steps[0]!.words.label) === 'Save seed 3', [sv.desk.state.stars, sv.steps.map(x => s(x.words.label))]);
  const up = act('seed', 'savedSeeds', desk('ace', 'studio', ['seed'], { 'seed.lock': true }, {}, { seed: 2, stars }), 'save', { name: 'two' });
  ok('saving a saved one updates it, keeps its star, and offers Undo', s(up.steps[0]!.words.label) === 'Update seed 2' && (up.desk.state.stars as { star?: boolean }[])[0]!.star === true && up.told[0] === 'Updated saved seed 2');
  const st = act('seed', 'savedSeeds', desk('ace', 'studio', ['seed'], {}, {}, { stars }), 'star', '1');
  ok('a star is a step', s(st.steps[0]!.words.label) === 'Star one' && (st.desk.state.stars as { star?: boolean }[])[0]!.star === true);
  const rows = model('seed', 'savedSeeds', desk('ace', 'studio', ['seed'], {}, {}, { stars }))!.instrument!.stars!.rows;
  ok('starred first, unnamed as seed_<n>', J(rows.map(r => [r.v, r.name])) === J([[2, 'seed_2'], [1, 'one']]));
  const rs = act('seed', 'savedSeeds', desk('ace', 'studio', ['seed'], { 'seed.lock': true }, {}, { seed: 1, stars }), 'random', 0);
  ok('a random saved seed is never the one loaded (L-22), and is said', rs.desk.state.seed === 2 && rs.told[0] === 'Loaded saved seed 2', [rs.desk.state.seed, rs.told]);
}

/* ---- takes */
{
  /* the simulated server's catalogue and figures, as the page is told them (features/engine-panel.tsx) */
  const told = (d: Desk): Desk => ({ ...d, vram: FIGURES, lists: Object.fromEntries(Object.entries(MODEL_FILES).flatMap(([e, b]) => Object.entries(b).map(([k, f]) => [listKey({ from: 'models', bucket: k }, e as EngineId), f]))),
    selection: Object.fromEntries(Object.entries(MODEL_FILES).flatMap(([e, b]) => Object.entries(b).map(([k, f]) => [selectionKey(e as EngineId, k), f[0]!.value]))) });
  ok('one take on ACE-Step: 15.6 GB (35-gobar.js vramNeed)', vramNeed(ix, told(desk('ace', 'studio', ['takes'])), 1).need.toFixed(1) === '15.6');
  ok('each take adds the LM track', vramNeed(ix, told(desk('ace', 'studio', ['takes'])), 3).need.toFixed(1) === '20.2');
  const big = vramNeed(ix, told(desk('ace', 'studio', ['takes', 'adapters'], {}, {}, { slots: [{ file: 'a', type: 'LoRA' }] })), 8);
  ok('past the card ACE-Step plans in passes (an adapter tips eight takes over)', big.need > 31.8 && big.passes === 2 && !big.refuses, big);
  /* RA-01: the tile says what the go bar and the meter say, for the models chosen (one formula) */
  const mm = told(desk('minimax-m3', 'studio', ['takes'], { 'takes.n': 4, 'model.lm': '8B planner · F16' }));
  const tk = model('takes', 'takesReels', mm)!.instrument!.takes!, need = vramNeed(ix, mm);
  ok("the Takes tile's figure is vramNeed's, for the models chosen (RA-01)", s(tk.cost.text) === `${need.need.toFixed(1)} of 31.8 GB` && need.refuses && tk.cost.cls === 'over' && !!tk.cost.msg && s(tk.cost.msg) === `Won't fit. Try ${need.fits} takes.`, [s(tk.cost.text), need]);
  const small = told(desk('ace', 'studio', ['takes'], { 'model.dit': 'XL Turbo · MXFP4', 'model.lm': '0.6B · Q8_0' }));
  ok('a smaller model chosen costs less on the tile too', s(model('takes', 'takesReels', small)!.instrument!.takes!.cost.text) === `${vramNeed(ix, small).need.toFixed(1)} of 31.8 GB` && vramNeed(ix, small).need < 15);
  const t = model('takes', 'takesReels', told(desk('minimax-m3', 'studio', ['takes'], { 'takes.n': 2 })))!.instrument!.takes!;
  ok('MiniMax-Music3: four reels in one row, extra takes cost', t.max === 4 && t.cols === 4 && /per extra take/.test(s(t.cost.each)), t);
  ok('its words for one take and for more', s(t.labels[0]) === 'Take per run' && s(t.labels[1]) === 'Takes per run', t.labels.map(s));
  const a = model('takes', 'takesReels', desk('ace', 'studio', ['takes']))!.instrument!.takes!;
  ok("ACE-Step: eight reels, three a row (the footprint of the concept's nine)", a.max === 8 && a.cols === 3);
}

/* ---- the scope */
{
  const sc = node('sampler', 'samplerScope');
  const lin = sigmasOf(ix, sc, desk('ace', 'studio', ['sampler'], { 'samp.steps': 4 }, { 'samp.shift': false }, {}));
  ok('a linear schedule at shift 3: sigma = 3t/(1+2t), t from 1 to 0', J(lin.map(x => +x.toFixed(4))) === J([1, 0.9, 0.75, 0.5, 0]), lin);
  const ts = sigmasOf(ix, sc, desk('ace', 'studio', ['sampler', 'latent'], { 'lat.ts': '0.9, 0.5, 0' }));
  ok('custom timesteps from Latent & correction are drawn as they are', J(ts) === J([0.9, 0.5, 0]));
  const off = sigmasOf(ix, sc, desk('ace', 'studio', ['sampler'], { 'lat.ts': '0.9, 0.5, 0' }));
  ok('but not while Latent & correction is off the desk', off.length === 13, off.length);
  const pw = sigmasOf(ix, sc, { ...desk('ace', 'studio', ['sampler'], { 'samp.steps': 2, 'samp.sched': 'power', 'sch.exp': 2, 'samp.shift': 1 }, { 'samp.shift': false }), lists: { 'plugins.scheduler': [{ value: 'power', label: 'Power' }] } });
  ok('the Power shape reads its exponent', J(pw) === J([1, 0.25, 0]), pw);
  const m = model('sampler', 'samplerScope', desk('ace', 'studio', ['sampler', 'perf'], { 'samp.cs': 0.2, 'perf.cfgcut': 0.6 }))!.instrument!.scope!;
  ok('the window and the cutoff', !!m.band && !!m.cut && Math.abs(m.band.x - (6 + 288 * 0.2)) < 1e-9, m);
  ok('the read-out names the window, its numbers in bold, and offers the cutoff', s(m.read[0]!.text).replace(/\u0001/g, '|') === 'guidance from |0.20| to |0.60| (CFG cutoff)' && m.read[0]!.action === 1, m.read.map(r => s(r.text)));
  const none = model('sampler', 'samplerScope', desk('ace', 'studio', ['sampler'], { 'samp.cs': 0.95 }))!.instrument!.scope!;
  ok('a window that holds no step says guidance is off (A-31)', s(none.read[0]!.text) === 'guidance off: the window holds no step' && none.read[0]!.tone === 'warn');
  const mm = model('sampler', 'samplerScope', desk('minimax-m3', 'studio', ['sampler']))!.instrument!.scope!;
  ok('MiniMax-Music3 without Lua plugins: native sampling', s(mm.read[0]?.text) === 'native sampling', mm.read);
  const sa = model('sampler', 'samplerScope', desk('sa3', 'studio', ['sampler']))!.instrument!.scope!;
  ok('Stable Audio 3, with a sampler of its own, does not say native', !sa.read.length, sa.read);
  const lab = model('sampler', 'samplerScope', desk('ace', 'lab', ['sampler']))!.instrument!.scope!;
  ok('at Lab the interval has its two handles', J(lab.handles.map(h => h.key)) === J(['samp.cs', 'samp.ce']));
}

/* ---- the latent curve and the repaint region */
{
  const d = distOf(0, 1);
  ok('at rest the two curves are one', d.before === d.after && d.zero === 150);
  ok('shifted, the green curve moves', distOf(1, 1).after !== d.before);
  ok('a key moves the region a hundredth', J(regionMove(0.35, 0.55, { key: 'ArrowRight' })!.map(x => +x.toFixed(4))) === J([0.36, 0.56]));
  ok('Shift moves its end, never narrower than a fiftieth', J(regionMove(0.35, 0.36, { key: 'ArrowLeft', shift: true })) === J([0.35, 0.37]));
  ok('Home and End keep its length (L-36)', J(regionMove(0.3, 0.5, { key: 'End' })!.map(x => +x.toFixed(4))) === J([0.8, 1]));
  ok('a drag inside carries it whole', J(regionMove(0.3, 0.5, { kind: 'body', x0: 0.4, x: 0.6 })!.map(x => +x.toFixed(4))) === J([0.5, 0.7]));
  ok('a drag outside draws a new one', J(regionMove(0.3, 0.5, { kind: 'out', x0: 0.8, x: 0.6 })) === J([0.6, 0.8]));
  const src = { song: 1, title: 'S', style: 'lo-fi', dur: 100, hue: 215 };
  const r = act('repaint', 'repaintRegion', desk('ace', 'studio', ['repaint'], {}, {}, { source: src, region: { start: 35, end: 55 } }), 'move', { key: 'ArrowRight' });
  ok('a move is a step in the song\'s times (Q-24)', s(r.steps[0]?.words.label) === 'Repaint region 0:35–0:55 → 0:36–0:56', r.steps.map(x => s(x.words.label)));
  const m = model('repaint', 'repaintRegion', desk('ace', 'quick', ['repaint'], {}, {}, { source: src }))!.instrument!.region!;
  ok('the first region is 35% to 55%, said in the song\'s times; at Quick the wave takes the width', s(m.cap) === 'Will repaint 0:35 to 0:55' && m.one && !m.ctl, m);
  ok('at Lab its controls stand on the caption line', model('repaint', 'repaintRegion', desk('ace', 'lab', ['repaint'], {}, {}, { source: src }))!.instrument!.region!.ctl);
}

console.log(fails.length ? `\n${fails.length} failed:\n  ${fails.join('\n  ')}` : `instruments: ${n} checks of the instruments' core: ok`);
process.exitCode = fails.length ? 1 : 0;
