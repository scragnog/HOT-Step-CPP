// Tests of what the React layer relies on that is not the concept's derivations already checked by test-core.ts:
// the desk's layout as a pure function against the layouts the concept page chose (tools/fixtures/pack.json, recorded
// before the cutover; its collector is in the tag rebuild-parity), and the core's selectors the view asks
// (src/derive/desk.ts and the additions to strings.ts, tiers.ts, view.ts and records.ts made for it).
//
//   node tools/test-view.ts        exit 1 on a failure

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import type { Definition, EngineId, Node, Tier } from '../src/definitions/types.ts';
import { ENGINE_IDS } from '../src/definitions/engines.ts';
import {
  addTo, alsoSet, arrivalChange, applyChange, bareText, bindPart, boardCount, boardNodes, checkText, deskModel, deskStep, emptyBook, indexOf, lcdMsg,
  meterScale, nodeKeys, nodeShown, nodeWord, optionsOf, parkedOn, pedalModel, pedalState, readDesk, removeFrom, removedWords, savedDeskOf, stringsOf,
  tileName, workflowOf, type Change, type Desk, type Msg,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';
import { deskSolve, type PackItem } from '../src/app/desk/pack.ts';
import { need, section, skips } from './lib/sections.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
const ok = (what: string, cond: boolean, detail: unknown = '') => { if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}): Desk =>
  ({ engine, tier, vals, auto: {}, state: {}, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const def = (id: string) => need(DEFINITIONS.find(d => d.id === id), id);

/* ---------------------------------------------------------------- the layout against the concept's */
const PACK = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'pack.json'), 'utf8')) as { cases: { id: string; cols: number; items: PackItem[]; out: { i: number; c: number; w: number; y: number; h: number }[]; plan: { o: number; c: number }[] }[] };
let packed = 0;
for (const c of PACK.cases) {
  const r = deskSolve(c.cols, c.items);
  const got = r.out.map(x => ({ i: x.i, c: x.c, w: x.w, y: x.y, h: x.h })).sort((a, b) => a.i - b.i), want = [...c.out].sort((a, b) => a.i - b.i);
  ok(`layout ${c.id}`, J(got) === J(want) && J(r.plan) === J(c.plan), { got, want });
  packed++;
}

/* ---------------------------------------------------------------- every word the view looks up is a key the definitions give */
const KEYS = new Set(DEFINITIONS.flatMap(stringsOf).map(s => s.key));
let looked = 0;
for (const d of DEFINITIONS) {
  const keys = nodeKeys(d);
  for (const e of d.engines) for (const tier of ['quick', 'studio', 'lab'] as Tier[]) {
    const dk = desk(e, tier, [d.id]);
    const walk = (n: Node) => {
      const k = keys.get(n); ok(`${d.id}: every node has a key`, !!k, n.part);
      for (const f of ['label', 'tip', 'text', 'between'] as const) { const m = k ? nodeWord(ix, n, dk, k, f) : undefined; if (m) { looked++; ok(`${d.id} ${e} ${tier}: ${m.id} is a key stringsOf gives`, KEYS.has(m.id)); } }
      (n.children ?? []).forEach(walk);
    };
    ('face' in d ? d.face ?? [] : []).forEach(walk);
    for (const p of d.params) if (p.type === 'choice' && (p.engines ?? d.engines).includes(e)) for (const o of optionsOf(ix, p.key, dk)) {
      for (const m of [o.label, o.lcd, o.detail]) if (m && typeof m === 'object' && 'id' in m) ok(`${p.key} ${e}: option word ${m.id} is a key`, KEYS.has(m.id));
    }
  }
}

/* ---------------------------------------------------------------- the selectors the view asks */
/* The examples below name real modules, pedals and units. When one of them has been removed (06-how-to.md, "Remove a
   module"), its examples are skipped and said, rather than failing: give them another definition when that matters. */
const EXAMPLES = ['avoid', 'gain', 'guidance', 'norm', 'post', 'ppvae', 'refiner', 'ss', 'tempo', 'tile', 'vst', 'wh'];
const gone = EXAMPLES.filter(id => !DEFINITIONS.some(d => d.id === id));
if (gone.length) console.log(`the view's examples name ${gone.join(', ')}, which the definitions no longer have: those examples are skipped`);
else {
const ace = (tier: Tier, on: string[], vals: Desk['vals'] = {}) => desk('ace', tier, on, vals);
/* a container none of whose children is drawn is not drawn; a control bound to what the engine lacks is not drawn */
section(() => {
  const g = def('guidance') as Extract<Definition, { kind: 'module' }>;
  const row = g.face.find(n => n.part === 'row')!, pad = g.face.find(n => n.part === 'xyPad')!;
  ok('the minimax-m3 row is drawn on minimax-m3 with APG', nodeShown(ix, row, desk('minimax-m3', 'lab', ['guidance', 'sampler'], { 'samp.plugins': true, 'samp.guide': 'apg' })));
  /* a control none of whose parameters applies is not drawn, as the concept's ok() leaves it out (package 2) */
  ok('the minimax-m3 row is not drawn while its knob does not apply (no APG)', !nodeShown(ix, row, desk('minimax-m3', 'lab', ['guidance'])));
  ok('the minimax-m3 row is not drawn on ace', !nodeShown(ix, row, ace('lab', ['guidance'])));
  ok('the APG pad is drawn on ace', nodeShown(ix, pad, ace('lab', ['guidance'])));
  ok('a row whose only knob is on no engine it is shown on is not drawn', !nodeShown(ix, { part: 'row', children: [{ part: 'knob', param: 'apg.mom' }] } as Node, desk('minimax-m3', 'lab', ['guidance'])));
});
/* a pedal's text that stands beside a switch is laid out once, beside it */
section(() => {
  const cells = boardNodes(ix, def('ss'), ace('lab', ['post', 'ss'])).cells.filter(c => c.param === 'pp.ss.seed');
  ok('the fixed seed is laid out once, beside its switch', cells.length === 1 && cells[0]!.beside === 'pp.ss.seedf', cells);
});
/* a side unit's setting in "Also set" carries the unit's name, as a pedal's does (the concept's alsoWords) */
section(() => {
  const a = alsoSet(ix, def('post'), ace('studio', ['post', 'wh'], { 'pp.wh.beam': 8 }));
  ok('Also set names the side unit', !!a && /Whisper lyrics beam 8/.test(say(a.text)), a && say(a.text));
});
/* a module's name at Quick is its quickName when Quick shows less of it */
ok('Tempo is "Tempo & length" at Quick on ACE-Step', say(tileName(def('tempo'), ace('quick', ['tempo']))) === 'Tempo & length');
ok('and "Tempo, key & length" at Studio', say(tileName(def('tempo'), ace('studio', ['tempo']))) === 'Tempo, key & length');
ok('and "Tempo" at Quick on MiniMax-Music3', say(tileName(def('tempo'), desk('minimax-m3', 'quick', ['tempo']))) === 'Tempo');
/* the steps that change the desk's make-up, and what a removal says */
ok('Add step', say(deskStep('add', def('avoid'), ace('studio', [])).label) === 'Add Avoid');
ok('Remove step', say(deskStep('remove', def('post'), ace('studio', [])).label) === 'Remove Pedalboard');
ok('removing the board says its pedals', say(removedWords(def('post'), ace('studio', []), 3, false)) === 'Removed Pedalboard and its 3 pedals');
ok('removing a module with no pedals says none', say(removedWords(def('avoid'), ace('studio', []), 0, false)) === 'Removed Avoid');
ok('the last pedal takes the board with it', say(removedWords(def('ss'), ace('studio', []), 0, true)) === 'Took StableStep off the board. That was the last one, so the Pedalboard left the desk too');
/* what a pedal arrives as: engaged, unless it would have nothing to run */
section(() => {
  const d0 = ace('studio', ['post']);
  ok('StableStep arrives engaged', applyChange(d0, arrivalChange(ix, def('ss'), d0)).vals['pp.ss.on'] === true);
  ok('the VST3 chain with no plugins arrives bypassed', applyChange(d0, arrivalChange(ix, def('vst'), d0)).vals['pp.vst.on'] === false);
});
/* the board's count, in the concept's words */
ok('board count', say(boardCount(ix, ace('lab', ['post', 'ss', 'refiner', 'wh'], { 'pp.ss.on': true, 'pp.wh.on': true }), 'pp.on')) === '2 pedals · 1 on · 1 side unit, 1 on');
ok('board count, bypassed', say(boardCount(ix, ace('lab', ['post', 'ss'], { 'pp.on': false, 'pp.ss.on': true }), 'pp.on')) === '1 pedal · chain bypassed: no pedal runs');
ok('board count, empty', say(boardCount(ix, ace('lab', ['post']), 'pp.on')) === 'No pedals in the chain yet');
ok('board count, an engaged pedal with nothing to run', say(boardCount(ix, ace('lab', ['post', 'vst'], { 'pp.vst.on': true }), 'pp.on')) === '1 pedal · 0 on · 1 needs setting up');
/* parked: what the engine in use does not use */
section(() => {
  const p = parkedOn(ix, desk('yue2', 'studio', ['avoid', 'post', 'ppvae', 'ss']));
  ok('parked modules and pedals', J(p.modules.map(d => d.id)) === '["avoid"]' && J(p.board.map(d => d.id)) === '["ppvae"]' && /^Parked: YuE2 doesn't use these\. Each returns/.test(say(p.says!)), { m: p.modules.map(d => d.id), b: p.board.map(d => d.id) });
});
/* words the parts print */
ok('an LCD prints an option\'s lcd words', say(lcdMsg(ix, 'pp.norm.target', ace('lab', ['post', 'norm']))) === '-14 LUFS');
ok('a pedal knob\'s number without its unit', J(bareText(ix, 'pp.norm.ceil', ace('lab', ['post', 'norm']))) === J({ text: '−1.0', unit: 'dBFS' }));
ok('a percent stays on its number', bareText(ix, 'pp.ss.str', ace('lab', ['post', 'ss'])) === null);
ok('a checked text says why it is not taken', say(checkText(ix, 'pp.ss.seed', ace('lab', ['post', 'ss']), 'not a seed')!) === 'The fixed seed is a whole number from 0 to 4294967295.');
ok('and nothing when it is', checkText(ix, 'pp.ss.seed', ace('lab', ['post', 'ss']), '1234') === null);
/* the desk's order is kept by the records */
section(() => {
  const d0 = ace('studio', ['post', 'avoid', 'ss', 'wh']);
  const saved = savedDeskOf(ix, d0, emptyBook());
  ok('the autosave keeps the modules in the desk\'s order', J(saved.modules) === '["post","avoid"]', saved.modules);
  const back = readDesk(ix, JSON.parse(J(saved)))!;
  ok('and reading it back keeps the order', J([...back.desk.on].filter(id => id === 'post' || id === 'avoid')) === '["post","avoid"]');
  ok('a workflow keeps it too', J(workflowOf(ix, d0, 'x').modules) === '["post","avoid"]');
});
/* ---------------------------------------------------------------- the parts' contract, at run time (SL-03, SL-11)
   A setter writes only what its node binds, against the desk in force when it is called: held across another change it
   names its step from the value it really changes; called once its definition has left the desk it changes nothing; a
   typed value is read, clamped and said; a key past an end stops there and says nothing. */
section(() => {
  let live = ace('lab', ['post', 'ss'], { 'pp.ss.on': true });
  const got: Change[] = [], told: Msg[] = [];
  const ss = def('ss'), knob = { part: 'knob', param: 'pp.ss.str' } as Node;
  const held = bindPart(ix, ss, knob, () => live, c => { got.push(c); live = applyChange(live, c); }, m => told.push(m));
  live = applyChange(live, { vals: { 'pp.ss.str': 50 }, auto: {}, state: {}, words: { label: '', put: '' } as never });
  held.params['pp.ss.str']!.set(20);
  ok('a setter held across a change names its step from the value in force (SL-11)', say(got.at(-1)!.words.label) === 'Refine 50% → 20%', say(got.at(-1)!.words.label));
  ok('a part is given setters for what its node binds and nothing else', J(Object.keys(held.params)) === '["pp.ss.str"]');
  let threw = false; try { held.params['pp.ss.str']!.set(30, undefined, undefined, { 'avoid.text': 'x' }); } catch { threw = true; }
  ok('a setter refuses to write another value beside its own without a named step', threw);
  held.params['pp.ss.str']!.typed('500');
  ok('a typed value past the end is put there and said (21-knob.js)', live.vals['pp.ss.str'] === 60 && /Refine: 10 to 60\. Set to 60\./.test(say(got.at(-1)!.said!)), say(got.at(-1)!.said ?? ''));
  const n = got.length; held.params['pp.ss.str']!.typed('500');
  ok('typed again at the end it changes nothing and is still said', got.length === n && /Set to 60/.test(say(told.at(-1)!)));
  held.params['pp.ss.str']!.typed('abc');
  ok('what is no number is said, and changes nothing', got.length === n && say(told.at(-1)!) === 'Type a number for Refine');
  held.params['pp.ss.str']!.press('End', false); held.params['pp.ss.str']!.press('ArrowUp', true);
  ok('a key past an end stops there and says nothing (SL-12)', got.length === n && live.vals['pp.ss.str'] === 60);
  held.params['pp.ss.str']!.press('Home', false);
  ok('a key the contract knows moves the value', live.vals['pp.ss.str'] === 10);
  held.params['pp.ss.str']!.typed('1:30'); const at130 = live.vals['pp.ss.str']; held.params['pp.ss.str']!.typed('−5');
  ok('a typed number is read as the page writes it (m:ss is 90, put at 60; a true minus is -5, put at 10)', at130 === 60 && live.vals['pp.ss.str'] === 10, [at130, live.vals['pp.ss.str']]);
  live = { ...live, on: new Set([...live.on].filter(x => x !== 'ss')) };
  const before = got.length; held.params['pp.ss.str']!.set(40);
  ok('a setter whose definition has left the desk changes nothing (an answer that came too late)', got.length === before);
});
/* the desk's make-up rules are the core's (addTo, removeFrom) */
section(() => {
  const d0 = ace('studio', ['avoid']);
  const a = addTo(ix, d0, ['avoid'], [], 'ss', undefined, {});
  ok('the first pedal brings the Pedalboard', a.did === 'added' && J(a.mods) === '["avoid","post"]' && J(a.board) === '["ss"]', a);
  ok('a Lab module outside Lab is offered Lab', addTo(ix, d0, ['avoid'], [], 'guidance', undefined, {}).did === 'lab');
  ok('what is there is there', addTo(ix, d0, ['avoid'], [], 'avoid', undefined, {}).did === 'there');
  const r = removeFrom(ix, ['avoid', 'post'], ['ss'], 'ss');
  ok('the last pedal takes the Pedalboard with it', r.last && J(r.mods) === '["avoid"]' && !r.board.length, r);
  const rb = removeFrom(ix, ['post', 'avoid'], ['ss', 'tile'], 'post');
  ok('the Pedalboard takes its pedals, and remembers where it stood', rb.pedals === 2 && !rb.board.length && rb.stood === 'avoid', rb);
  const back = addTo(ix, d0, ['tempo', 'avoid'], [], 'post', undefined, { post: 'avoid' });
  ok('a module added back goes where it stood', back.did === 'added' && J(back.mods) === '["tempo","post","avoid"]', back);
});
/* what pedals and units are, and the empty desk's offers, are the core's */
section(() => {
  const d = ace('lab', ['post', 'ss', 'gain', 'wh'], { 'pp.on': false, 'pp.ss.on': true, 'pp.wh.on': true });
  ok('a pedal on while the chain is bypassed is held', pedalState(ix, def('ss') as never, d) === 'held');
  ok('a pedal with no switch always runs, and stops with the chain', pedalState(ix, def('gain') as never, d) === 'always' && pedalModel(ix, def('gain') as never, d).cls.split(' ').includes('off'));
  ok('the empty Quick desk names what the first offer is on its engine', say(deskModel(ix, desk('sa3', 'quick', []), []).empty!.says!) === 'Add a length, a seed or pedals from Modules' && say(deskModel(ix, ace('quick', []), []).empty!.says!) === 'Add tempo, a seed or pedals from Modules');
  ok('a VU meter prints the scale of the concept', J(meterScale({}, undefined).marks.filter(m => m.words).map(m => m.words)) === '["-20","-10","-5","0","+3"]');
  ok('an own scale writes a true minus', meterScale({ min: -24, max: -6 }, -14).marks[0]!.words === '−24');
});
}
void ENGINE_IDS;

console.log(`layout: ${packed} layouts the concept chose, ${PACK.cases.length - fails.filter(f => f.startsWith('layout')).length} reproduced`);
console.log(`node words: ${looked} looked up, every one a key the definitions give`.replace('every one', fails.some(f => f.includes('is a key')) ? 'NOT every one' : 'every one'));
if (fails.length) { console.log(`\n${fails.length} failed:`); fails.slice(0, 40).forEach(f => console.log('  ' + f.slice(0, 400))); process.exit(1); }
console.log(`the view's selectors: ok${skips()}`);
