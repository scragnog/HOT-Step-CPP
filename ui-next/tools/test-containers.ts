// Tests of the core's additions for the containers and read-outs (package 2: src/derive/containers.ts,
// src/derive/readouts.ts, a file's steps in src/derive/steps.ts, the node rule in src/derive/desk.ts), each against
// what the concept does (25-module-planner.js planText, 25-module-tempo.js pulseHTML, 25-module-replan.js,
// 25-module-flow.js, 25-module-perf.js, 25-module-adhere.js, 28-side-units.js, 13-files.js).
//
//   node tools/test-containers.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Node, Tier } from '../src/definitions/types.ts';
import { bindPart, fileRefused, indexOf, kidsOf, nodeKeys, nodeShown, partModel, readoutModel, sectionNames, spellKey, type Change, type Desk, type Msg } from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, state: Desk['state'] = {}): Desk =>
  ({ engine, tier, vals, auto, state, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const def = (id: string) => DEFINITIONS.find(d => d.id === id)!;
const s = (m: unknown) => say(m as Msg);
const nodes = (id: string, part: string): Node[] => {
  const out: Node[] = [];
  const walk = (ns: Node[]) => { for (const x of ns) { if (x.part === part) out.push(x); walk(x.children ?? []); } };
  walk(('face' in def(id) ? (def(id) as { face: Node[] }).face : []) ?? []);
  return out;
};
const node = (id: string, part: string) => nodes(id, part)[0]!;
const keyOf = (id: string, n: Node) => nodeKeys(def(id)).get(n) ?? '';

/* ---- a container's kids */
{
  const d = desk('ace', 'lab', ['planner']), term = nodes('planner', 'terminal')[0]!, kids = kidsOf(ix, term, d, keyOf('planner', term));
  ok('the terminal knows its bar, the plan and its lines, in order', kids[0]!.part === 'termBar' && kids[1]!.part === 'planText' && kids.slice(2).every(k => k.part === 'termLine'), kids.map(k => k.part));
  ok('the lines Lab adds are its extras; the sample line is not', kids.filter(k => k.shown && k.extra).length === 4 && !kids.find(k => k.shown && k.part === 'termLine')!.extra, kids.map(k => [k.shown, k.extra]));
  const line = nodes('planner', 'termLine')[0]!, lk = kidsOf(ix, line, d, keyOf('planner', line));
  ok('a line prints each control after its flag: the node\'s label', lk.map(k => s(k.label)).join(' ') === '--temp --cfg --top-p', lk.map(k => s(k.label)));
  ok('a kid says what it edits', J(lk.map(k => k.keys[0])) === J(['lm.temp', 'lm.cfg', 'lm.topp']));
  const studio = kidsOf(ix, term, desk('ace', 'studio', ['planner']), keyOf('planner', term));
  ok('at Studio the Lab lines are not drawn', studio.filter(k => k.extra).every(k => !k.shown));
}
{
  const d = desk('yue2', 'lab', ['sampler', 'adhere']), row = nodes('adhere', 'row')[0]!, kids = kidsOf(ix, row, d, keyOf('adhere', row));
  ok('a row of rack groups: each group, its first child the switch it hangs on', J(kids.map(k => [k.part, k.inner[0]])) === J([['group', 'row'], ['group', 'powerChip'], ['group', 'powerChip']]), kids.map(k => k.inner));
  ok('a group switched off stands back, and says why', kids[1]!.dim !== null && s(kids[1]!.dim) === 'Plan CFG · off' && kids[0]!.dim === null);
  const on = kidsOf(ix, row, desk('yue2', 'lab', ['sampler', 'adhere'], { 'yue.pcfg': 1.5 }), keyOf('adhere', row));
  ok('and stands back no more once on', on[1]!.dim === null);
}
/* ---- a guard's words, held or not (models.ts dimWords) */
{
  const line = nodes('repetition', 'termLine')[0]!, k = keyOf('repetition', line);
  const on = partModel(ix, def('repetition'), line, desk('ace', 'lab', ['planner', 'repetition']), k, 'term')!;
  const off = partModel(ix, def('repetition'), line, desk('ace', 'lab', ['planner', 'repetition'], { 'lm.rep': 1 }), k, 'term')!;
  ok('a guard says "guard off" whether or not it is off, and stands back only when it is', s(on.dimWords) === 'guard off' && on.dim === null && s(off.dim) === 'guard off');
  const term = nodes('repetition', 'terminal')[0]!, t = partModel(ix, def('repetition'), term, desk('ace', 'lab', ['planner', 'repetition'], { 'lm.on': false }), keyOf('repetition', term), 'tile')!;
  ok('with the planner off the guard stands back and says so', s(t.dim) === 'planner off');
}
/* ---- a control none of whose parameters applies is not drawn (the concept's ok()) */
{
  const kv = nodes('plans', 'ledButton').find(x => x.param === 'mm.kvq8')!;
  ok('Compact planner cache waits for an adapter and the Plan fader below 1', !nodeShown(ix, kv, desk('minimax-m3', 'lab', ['planner', 'plans'])));
  ok('and is drawn with them', nodeShown(ix, kv, desk('minimax-m3', 'lab', ['planner', 'plans', 'adapters'], { 'ad.plan': 0.8 }, {}, { slots: [{ file: 'a', type: 'LoRA' }] })));
}
/* ---- the read-outs */
{
  const p = (vals: Desk['vals'], auto: Desk['auto'] = {}) => readoutModel(ix, node('tempo', 'beatPulse'), desk('ace', 'lab', ['tempo'], vals, auto))!.pulse!;
  ok('beat lamps rest on auto, four to a bar', J(p({})) === J({ beats: 4, beat: '0s', idle: true }), p({}));
  ok('at 120 in 3/4: three, half a second apart', J(p({ 'tempo.bpm': 120, 'tempo.sig': '3/4' }, { 'tempo.bpm': false })) === J({ beats: 3, beat: '0.500s', idle: false }));
}
{
  const dz = (vals: Desk['vals']) => readoutModel(ix, node('replan', 'dice'), desk('yue2', 'lab', ['planner', 'replan'], vals))!.dice!;
  ok('a die per try', dz({}).n === 3 && dz({ 'yue.tries': 6 }).n === 6 && !dz({}).off);
  ok('off with re-planning off', dz({ 'yue.replan': false }).off);
}
{
  const f = (hop: number) => readoutModel(ix, node('flow', 'flowWindows'), desk('minimax-m3', 'lab', ['sampler', 'flow'], { 'mm.hop': hop }))!.flow!;
  ok('six-second hops: seven windows over 48 s, 30 a song, 32% fewer than the checkpoint\'s 44 (25-module-flow.js)', f(150).windows.length === 7 && f(150).count === 30 && s(f(150).compare) === "windows per 3:00 · 32% fewer than the checkpoint's 44" && s(f(150).seam) === '2.0 s shared at each seam', [s(f(150).compare), s(f(150).seam)]);
  ok('the checkpoint\'s own hop says so; the longest says to listen', s(f(100).compare) === "windows per 3:00 · as many as the checkpoint's own hop" && /listen at the joins$/.test(s(f(160).seam)));
  ok('the windows alternate lanes, each 8 s wide', J(f(150).windows[1]) === J({ x: 0.125, w: 0.1667, lane: 1 }), f(150).windows[1]);
}
{
  const g = (vals: Desk['vals']) => readoutModel(ix, node('perf', 'speedGauge'), desk('ace', 'lab', ['sampler', 'perf'], vals))!.speed!.text;
  ok('the gauge: nothing traded is 0%; each trade weighs as the concept weighs it', g({}) === '0%' && g({ 'perf.cfgcut': 0.5 }) === '+20%' && g({ 'perf.cfgcut': 0.5, 'perf.cache': 0.2, 'perf.lmcut': 0.7 }) === '+36%', [g({ 'perf.cfgcut': 0.5 }), g({ 'perf.cfgcut': 0.5, 'perf.cache': 0.2, 'perf.lmcut': 0.7 })]);
}
{
  const c = (vals: Desk['vals']) => readoutModel(ix, node('adhere', 'adhereCurve'), desk('yue2', 'lab', ['sampler', 'adhere'], vals))!.curve!;
  const flat = c({});
  ok('a flat plan strength at 1: the line on the unity mark, no band, no ramp', flat.points.startsWith('8.0,48.0') && flat.band === null && flat.cads === null && s(flat.legend[0]!.words) === 'plan strength 1.00 → 1.00');
  const on = c({ 'yue.pcfg': 2.5, 'yue.cads': 0.5, 'yue.pws': 2, 'yue.pwc': 'cosine' });
  ok('Plan CFG and CADS on: the band, the ramp, the legend; a cosine curve eases', !!on.band && on.band.opacity === 0.725 && !!on.cads && on.legend.length === 3 && s(on.legend[1]!.words) === 'plan CFG 2.5' && on.points.split(' ')[1] !== '15.1,15.3', [on.band, on.legend.map(l => s(l.words))]);
}
{
  const term = nodes('planner', 'planText')[0]!;
  const plan = (e: EngineId, vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, on = ['planner', 'tempo']) => readoutModel(ix, partModelNode(term, e), desk(e, 'lab', on, vals, auto))!.plan!;
  const text = (p: ReturnType<typeof plan>) => [s(p.command), ...p.lines.map(l => `${s(l.key)}=${s(l.value)}/${l.tone}`), p.note ? s(p.note) : ''].join(' | ');
  ok('ACE-Step, nothing set: the planner\'s picks', text(plan('ace')) === "plan --cot | caption=(none yet)/k | tempo=90 BPM (planner's pick)/y | key=D major (planner's pick)/y | form=planner decides/y | ", text(plan('ace')));
  ok('what you set is yours, a key spelled as a musician does, the sections as the form', text(plan('ace', { 'tempo.bpm': 128, 'tempo.key': 'A# minor', 'lyrics.text': '[Verse]\na\n[Chorus]\nb' }, { 'tempo.bpm': false })) === 'plan --cot | caption=(none yet)/k | tempo=128 BPM (yours)/v | key=B♭ minor (yours)/v | form=Verse · Chorus/v | ', text(plan('ace', { 'tempo.bpm': 128, 'tempo.key': 'A# minor', 'lyrics.text': '[Verse]\na\n[Chorus]\nb' }, { 'tempo.bpm': false })));
  ok('the planner off says so', text(plan('ace', { 'lm.on': false })) === 'plan --off | The planner is off. The DiT renders from your caption and fields.');
  ok('a long caption is cut at a word with an ellipsis (B-16)', s(plan('ace', { 'prompt.text': 'a '.repeat(10) + 'very long description of a song that goes on and on past sixty four characters' }).lines[0]!.value).endsWith('…'));
  ok('MiniMax-Music3 reads tempo and key from the caption', /^plan --model 8B \| caption=\(none yet\)\/k \| tempo=from the caption prose\/y/.test(text(plan('minimax-m3'))), text(plan('minimax-m3')));
  ok('YuE2 composes a score as its chain of thought says', text(plan('yue2', { 'lm.cotmode': 'melody' })) === 'compose --cot "Melody only" | caption=(none yet)/k | score=melody line, no chords/v | form=composer decides/y | ', text(plan('yue2', { 'lm.cotmode': 'melody' })));
  const hm = plan('heartmula');
  ok('HeartMuLa: tags, the lyrics, and the model ends the song', s(hm.command) === 'tags' && s(hm.after) === '(none yet)' && text(hm).includes('length=model ends it, up to 4:00/y'), text(hm));
}
{
  const l = (vals: Desk['vals']) => readoutModel(ix, node('lrc', 'lrcPreview'), desk('ace', 'lab', ['post', 'lrc'], vals))!.lrc!.lines.map(s);
  ok('the timestamps name the lyrics\' first section, a verse by default', J(l({})) === J(['[00:12.40] verse', '[00:18.95] chorus']) && l({ 'lyrics.text': '[Bridge]\nx' })[0] === '[00:12.40] bridge');
  ok('sections: a tag starts one, words before any tag are a verse', J(sectionNames('la\n[Chorus]\nx\n[Outro 2]')) === J(['Verse', 'Chorus', 'Outro 2']));
  ok('keys spelled on the flat side where a musician would', spellKey('C# major') === 'D♭ major' && spellKey('F# major') === 'F♯ major' && spellKey('D# minor') === 'E♭ minor' && spellKey('G# minor') === 'G♯ minor');
}
/* ---- a drop zone's file */
{
  ok('an audio file is taken by its type or its name; anything else is refused and said', fileRefused('kick.wav', '') === null && fileRefused('take', 'audio/flac') === null && s(fileRefused('notes.txt', 'text/plain')) === "notes.txt isn't an audio file. Use WAV, FLAC, MP3, OGG or M4A.");
  const zone = node('master', 'dropZone'), steps: Change[] = [];
  let d = desk('ace', 'lab', ['post', 'master']);
  const bound = () => bindPart(ix, def('master'), zone, () => d, c => { steps.push(c); d = { ...d, state: { ...d.state, ...c.state } }; });
  bound().state.masterRef!.apply('load', 'ref.wav', undefined, { file: 'ref.wav' });
  ok('loading names the step for the file, in the definition\'s words', s(steps[0]!.words.label) === 'Load ref.wav as the mastering reference', s(steps[0]!.words.label));
  bound().state.masterRef!.apply('load', 'other.wav', undefined, { file: 'other.wav' });
  ok('replacing names both files', s(steps[1]!.words.label) === 'the mastering reference: ref.wav → other.wav', s(steps[1]!.words.label));
  bound().state.masterRef!.apply('clear', '');
  ok('clearing is a step a file has without saying', s(steps[2]!.words.label) === 'Clear the mastering reference' && s(steps[2]!.words.undone) === 'Put other.wav back', s(steps[2]!.words.label));
  const lent = nodes('timbre', 'dropZone')[0]!, m = partModel(ix, def('timbre'), lent, desk('ace', 'lab', ['timbre', 'post', 'master'], { 'pp.master.timbre': true }, {}, { masterRef: 'm.wav' }), keyOf('timbre', lent), 'tile')!;
  ok('a file the Mastering pedal lends Timbre is shown as lent', J(m.state?.map(x => [x.key, x.value, x.lent, x.owner])) === J([['masterRef', 'm.wav', true, 'master']]), m.state);
}

/** The node as an engine's line gives it (its reads and its look per engine), for readoutModel. */
function partModelNode(n: Node, e: EngineId): Node {
  const line = (n.per ?? []).find(l => !l.engines || l.engines.includes(e));
  return line ? ({ ...n, ...line } as Node) : n;
}

console.log(fails.length ? `\n${fails.length} failed:\n  ${fails.join('\n  ')}` : `containers: ${n} checks of the containers' and read-outs' core: ok`);
process.exitCode = fails.length ? 1 : 0;
