// Tests of the core's additions for the shared controls (package 1, src/derive/controls.ts and the setters it reaches
// through): each against what the concept does (20-controls.js, 21-numbox.js, 21-ruler.js, 21-fader.js, 22-pick.js,
// 25-module-tempo.js, 25-module-repetition.js, 25-module-adhere.js) and the pending findings the rebuild puts right
// (S-07, S-17, S-18, S-28, S-31).
//
//   node tools/test-controls.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Node, Tier } from '../src/definitions/types.ts';
import {
  autoEstimate, bindPart, boxText, choicePos, controlModel, indexOf, keptMsg, optionFaces, partModel, quantOf, rulerModel, rulerScale, rulerSnap,
  scrubValue, songTempo, type Change, type Desk, type Msg,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, auto: Desk['auto'] = {}): Desk =>
  ({ engine, tier, vals, auto, state: {}, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const def = (id: string) => DEFINITIONS.find(d => d.id === id)!;
const s = (m: unknown) => say(m as Msg);
const node = (id: string, part: string, param?: string): Node => {
  let hit: Node | undefined;
  const walk = (ns: Node[]) => { for (const x of ns) { if (x.part === part && (!param || x.param === param)) hit ??= x; walk(x.children ?? []); } };
  walk(('face' in def(id) ? (def(id) as { face: Node[] }).face : []) ?? []);
  return hit!;
};

/* ---- numbers in boxes (21-numbox.js numVal) */
ok('a box writes two places under a step of 1, none otherwise', boxText(12, 1) === '12' && boxText(1.05, 0.05) === '1.05' && boxText(0.5, 0.1) === '0.5');
{
  const d = desk('ace', 'studio', ['sampler']), m = controlModel(ix, node('sampler', 'numberBox', 'samp.steps'), d, {}, undefined)!;
  ok('a step count says the seconds it takes (25-module-sampler.js stepSec)', s(m.box!.seconds) === '≈ 1.3 s', s(m.box!.seconds));
  const t = desk('ace', 'studio', ['tempo']), bpm = { part: 'lcdNumber', param: 'tempo.bpm' } as Node, mb = controlModel(ix, bpm, t, {}, undefined)!;
  ok('the BPM on auto: an empty box, "≈120" while empty', mb.box!.text === '' && s(mb.box!.empty) === '≈120' && mb.box!.auto === true, mb.box);
  const mb2 = controlModel(ix, bpm, desk('ace', 'studio', ['tempo'], { 'tempo.bpm': 96 }, { 'tempo.bpm': false }), {}, undefined)!;
  ok('set, it shows the number', mb2.box!.text === '96' && mb2.box!.auto === false);
}

/* ---- the ruler (21-ruler.js) */
ok('a 10-minute scale is labelled every 2 minutes, minor ticks halving it', J(rulerScale(600)) === J({ maj: 20, min: 10, labels: [0, 120, 240, 360, 480, 600].map(v => ({ at: v / 6, text: `${Math.floor(v / 60)}:${String(v % 60).padStart(2, '0')}` })) }), rulerScale(600));
{
  const r = (e: EngineId, vals: Desk['vals'] = {}, auto: Desk['auto'] = {}) => rulerModel(ix, node('tempo', 'ruler', 'tempo.dur'), desk(e, 'studio', ['tempo'], vals, auto))!;
  ok('on auto the estimate: "≈ 3:00"', s(r('ace').caption) === '≈ 3:00' && r('ace').auto, s(r('ace').caption));
  ok('where the model ends the song, "up to 4:00" at the whole span (B-32)', s(r('heartmula').caption) === 'up to 4:00' && r('heartmula').at === 1, [s(r('heartmula').caption), r('heartmula').at]);
  ok('an exact length says no estimate (S-12)', s(r('sa3').caption) === '0:30');
  ok('near either end the cap stands flush with it', r('ace', { 'tempo.dur': 590 }, { 'tempo.dur': false }).capAt === 'end' && r('ace', { 'tempo.dur': 20 }, { 'tempo.dur': false }).capAt === 'start');
  const lab = rulerModel(ix, node('tempo', 'ruler', 'tempo.dur'), desk('ace', 'lab', ['tempo'], { 'tempo.beatio': true, 'tempo.beatbars': 8, 'tempo.dur': 20, 'tempo.trim': true }, { 'tempo.dur': false }))!;
  ok('Beat I/O longer than the song: each end half the song, and the line says what it needs (S-39)', lab.markers!.io![0]!.width === 10 / 600 && s(lab.markers!.note) === '8 bars each way needs about 0:32 at 120 BPM', [lab.markers!.io, s(lab.markers!.note)]);
  ok('the trim tag at the length', lab.markers!.trim!.at === 20 / 600);
  ok('no markers at Quick', rulerModel(ix, node('tempo', 'ruler', 'tempo.dur'), desk('ace', 'quick', ['tempo']))!.markers === null);
}
ok('the song\'s tempo: the BPM field, else a BPM in the prompt, else none (S-07)', songTempo({ value: 96, auto: false }, 'x, 120 BPM') === 96 && songTempo({ value: 96, auto: true }, 'folk, 140 bpm') === 140 && songTempo(undefined, 'folk') === null);
{
  const d = desk('sa3', 'studio', ['tempo']);
  ok('a dragged length snaps to 5 s (S-31)', rulerSnap(ix, 'tempo.dur', d, 0.0705, 600, 600) === 40, rulerSnap(ix, 'tempo.dur', d, 0.0705, 600, 600));
  ok('and to 15 s above 2:00', rulerSnap(ix, 'tempo.dur', d, 0.4, 600, 600) === 240 && rulerSnap(ix, 'tempo.dur', d, 0.3, 600, 600) === 180);
  ok('and to a labelled tick within 6px', rulerSnap(ix, 'tempo.dur', d, 121 / 600, 600, 300) === 120);
}
ok('auto stands for the default, or the maximum where it says "up to" (S-17)', autoEstimate(ix, 'tempo.dur', desk('ace', 'studio', ['tempo'])) === 180 && autoEstimate(ix, 'tempo.dur', desk('heartmula', 'studio', ['tempo'])) === 240);

/* ---- the scrub token (20-controls.js; S-28) */
{
  const d = desk('ace', 'lab', ['planner']);
  ok('a range of 500 steps or fewer is 220px end to end', Math.abs(scrubValue(ix, 'lm.temp', d, 0.85, 110, false)! - (0.85 + (ix.params.get('lm.temp')!.p.range![1] - ix.params.get('lm.temp')!.p.range![0]) / 2)) < 1e-9);
  const hm = desk('heartmula', 'lab', ['planner']);
  ok('a wide range moves with the value: top-K from 50, 100px right, lands under 200 (S-28)', (() => { const v = scrubValue(ix, 'hm.topk', hm, 50, 100, false)!; return v > 50 && v < 200; })(), scrubValue(ix, 'hm.topk', hm, 50, 100, false));
}

/* ---- a fader, a look-back strip, a power chip */
{
  const f = controlModel(ix, { part: 'fader', param: 'ad.plan', opts: { travel: 2 } } as Node, desk('ace', 'lab', ['adapters']), {}, undefined)?.fader;
  if (f) ok('a fader whose travel runs past its range stops there, the unity line at 1.00', f.stops && f.top < 1 && f.unity > 0, f);
  const lb = controlModel(ix, node('repetition', 'lookback', 'lm.win'), desk('ace', 'lab', ['repetition']), { unit: 'codes', who: 'the planner' }, undefined)!.lookback!;
  ok('the look-back strip lights its window of 160 bars and says its scale (25-module-repetition.js)', lb.lit === 40 && s(lb.far) === '51 s ago' && s(lb.mid) === 'the planner writes 5 codes a second', [lb.lit, s(lb.far), s(lb.mid)]);
  const pc = node('adhere', 'powerChip', 'yue.pcfg');
  const off = controlModel(ix, pc, desk('yue2', 'lab', ['adhere']), {}, 'Plan CFG')!.power!;
  ok('a power chip off says so on itself, and pressing it sets its on value (25-module-adhere.js)', !off.on && s(off.words) === 'Plan CFG · off' && off.to === 1.5, [off, s(off.words)]);
}

/* ---- a Quick row between its choices (20-controls.js choicePos) */
ok('a value between two choices stands between them', J(choicePos([8, 12, 30], 20)) === J({ pos: 1 + 8 / 18, beyond: 0 }));
ok('beyond either end, and a descending row', J(choicePos([8, 12, 30], 40)) === J({ pos: 2, beyond: 1 }) && J(choicePos([30, 12, 8], 10)) === J({ pos: 1.5, beyond: 0 }));
{
  const d = desk('ace', 'quick', ['sampler'], { 'samp.steps': 20 }), n0 = node('sampler', 'choiceRow', 'samp.steps');
  const m = partModel(ix, def('sampler'), n0, d, 'create.sampler.n.choiceRow.samp_2e_steps', 'quick')!;
  ok('the row says the value between its choices in its own words', m.row?.checked === null && !!m.control?.between && / · |20 steps/.test(say(m.row!.says)), [m.row?.checked, say(m.row!.says)]);
}

/* ---- a picker's option faces (22-pick.js) */
{
  const d: Desk = { ...desk('ace', 'lab', ['sampler']), lists: { 'plugins.solver': [{ value: 'euler', label: 'Euler' }, { value: 'heun', label: 'Heun' }, { value: 'md_x', label: 'MD Thing' }] } };
  const f = optionFaces(ix, 'samp.solver', d, 'plugin');
  ok('a plugin\'s place, symbol, family, hue and file', J(f[0]) === J({ n: 1, sym: 'Eu', family: 'Euler family', hue: 200, file: 'euler.lua' }) && f[2]!.family === 'MD community' && f[2]!.file === 'md_thing.lua', f);
  ok('a model\'s quantisation', J(quantOf('Turbo · Q4_K_M')) === J({ quant: 'Q4_K_M', qc: 'lo' }) && quantOf('ScragVAE') === null);
}

/* ---- the setters: typing, refusing, auto */
{
  const said: string[] = [], steps: Change[] = [];
  let d = desk('ace', 'studio', ['tempo'], { 'tempo.bpm': 96 }, { 'tempo.bpm': false });
  const b = bindPart(ix, def('tempo'), { part: 'lcdNumber', param: 'tempo.bpm' } as Node, () => d, c => { steps.push(c); }, m => said.push(s(m)));
  b.params['tempo.bpm']!.typed('');
  ok('an empty box puts a parameter with auto back on auto (21-numbox.js)', steps.length === 1 && steps[0]!.auto['tempo.bpm'] === true, steps);
  b.params['tempo.bpm']!.typed('fast');
  ok('a word in it is refused and said, auto offered', said.at(-1) === 'Type a number for BPM, or leave it empty for auto', said);
  d = desk('heartmula', 'studio', ['tempo']);
  const h = bindPart(ix, def('tempo'), { part: 'ruler', param: 'tempo.dur' } as Node, () => d, c => { steps.push(c); });
  h.params['tempo.dur']!.setAuto(false);
  ok('auto off starts where auto was showing (S-17)', steps.at(-1)!.vals['tempo.dur'] === 240, steps.at(-1));
  const r = desk('ace', 'lab', ['sampler'], { 'samp.retake': 0.5 }), told: string[] = [];
  const rs = bindPart(ix, def('sampler'), { part: 'textField', param: 'samp.rseed' } as Node, () => r, c => { steps.push(c); }, m => told.push(s(m)));
  const before = steps.length; rs.params['samp.rseed']!.set('x');
  ok('text that fails its check is refused and said with what is kept (12-values.js)', steps.length === before && /Kept “-1”\.$/.test(told.at(-1) ?? ''), told);
  const lat = desk('ace', 'lab', ['latent']);
  ok('a keeping field with nothing valid set says nothing is applied', s(keptMsg(ix, 'lat.ts', lat)!.says) === 'Nothing is applied until it is valid.' && !keptMsg(ix, 'lat.ts', lat)!.back);
  const mm = desk('minimax-m3', 'lab', ['planner'], { 'mm.seed': '12' });
  ok('and with a value, what is in force (E-33)', s(keptMsg(ix, 'mm.seed', mm)!.says) === 'Still using seed 12.', s(keptMsg(ix, 'mm.seed', mm)!.says));
}

if (fails.length) { console.log(`controls: ${fails.length} of ${n} failed`); fails.forEach(f => console.log('  ' + f)); process.exit(1); }
console.log(`controls: ${n} checks of the shared controls' core: ok`);
