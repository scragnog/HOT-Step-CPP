// Tests of what the React layer relies on that is not the concept's derivations already checked by test-core.ts:
// the desk's layout as a pure function against the layouts the concept page chose (tools/fixtures/pack.json, recorded
// by tools/concept/collect-pack.mjs), and the core's selectors the view asks (src/derive/desk.ts and the additions to
// strings.ts, tiers.ts, view.ts and records.ts made for it).
//
//   node tools/test-view.ts        exit 1 on a failure

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import type { Definition, EngineId, Node, Tier } from '../src/definitions/types.ts';
import { ENGINE_IDS } from '../src/definitions/engines.ts';
import {
  alsoSet, arrivalChange, applyChange, bareText, boardCount, boardNodes, checkText, deskStep, emptyBook, indexOf, lcdMsg, nodeKeys,
  nodeShown, nodeWord, optionsOf, parkedOn, readDesk, removedWords, savedDeskOf, stringsOf, tileName, workflowOf, type Desk,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';
import { deskSolve, type PackItem } from '../src/app/desk/pack.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
const ok = (what: string, cond: boolean, detail: unknown = '') => { if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}): Desk =>
  ({ engine, tier, vals, auto: {}, state: {}, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const def = (id: string) => DEFINITIONS.find(d => d.id === id)!;

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
const ace = (tier: Tier, on: string[], vals: Desk['vals'] = {}) => desk('ace', tier, on, vals);
/* a container none of whose children is drawn is not drawn; a control bound to what the engine lacks is not drawn */
{
  const g = def('guidance') as Extract<Definition, { kind: 'module' }>;
  const row = g.face.find(n => n.part === 'row')!, pad = g.face.find(n => n.part === 'xyPad')!;
  ok('the minimax-m3 row is drawn on minimax-m3', nodeShown(ix, row, desk('minimax-m3', 'lab', ['guidance'])));
  ok('the minimax-m3 row is not drawn on ace', !nodeShown(ix, row, ace('lab', ['guidance'])));
  ok('the APG pad is drawn on ace', nodeShown(ix, pad, ace('lab', ['guidance'])));
  ok('a row whose only knob is on no engine it is shown on is not drawn', !nodeShown(ix, { part: 'row', children: [{ part: 'knob', param: 'apg.mom' }] } as Node, desk('minimax-m3', 'lab', ['guidance'])));
}
/* a pedal's text that stands beside a switch is laid out once, beside it */
{
  const cells = boardNodes(ix, def('ss'), ace('lab', ['post', 'ss'])).cells.filter(c => c.param === 'pp.ss.seed');
  ok('the fixed seed is laid out once, beside its switch', cells.length === 1 && cells[0]!.beside === 'pp.ss.seedf', cells);
}
/* a side unit's setting in "Also set" carries the unit's name, as a pedal's does (the concept's alsoWords) */
{
  const a = alsoSet(ix, def('post'), ace('studio', ['post', 'wh'], { 'pp.wh.beam': 8 }));
  ok('Also set names the side unit', !!a && /Whisper lyrics beam 8/.test(say(a.text)), a && say(a.text));
}
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
{
  const d0 = ace('studio', ['post']);
  ok('StableStep arrives engaged', applyChange(d0, arrivalChange(ix, def('ss'), d0)).vals['pp.ss.on'] === true);
  ok('the VST3 chain with no plugins arrives bypassed', applyChange(d0, arrivalChange(ix, def('vst'), d0)).vals['pp.vst.on'] === false);
}
/* the board's count, in the concept's words */
ok('board count', say(boardCount(ix, ace('lab', ['post', 'ss', 'refiner', 'wh'], { 'pp.ss.on': true, 'pp.wh.on': true }), 'pp.on')) === '2 pedals · 1 on · 1 side unit, 1 on');
ok('board count, bypassed', say(boardCount(ix, ace('lab', ['post', 'ss'], { 'pp.on': false, 'pp.ss.on': true }), 'pp.on')) === '1 pedal · chain bypassed: no pedal runs');
ok('board count, empty', say(boardCount(ix, ace('lab', ['post']), 'pp.on')) === 'No pedals in the chain yet');
ok('board count, an engaged pedal with nothing to run', say(boardCount(ix, ace('lab', ['post', 'vst'], { 'pp.vst.on': true }), 'pp.on')) === '1 pedal · 0 on · 1 needs setting up');
/* parked: what the engine in use does not use */
{
  const p = parkedOn(ix, desk('yue2', 'studio', ['avoid', 'post', 'ppvae', 'ss']));
  ok('parked modules and pedals', J(p.modules.map(d => d.id)) === '["avoid"]' && J(p.board.map(d => d.id)) === '["ppvae"]' && /^Parked: YuE2 doesn't use these\. Each returns/.test(say(p.says!)), { m: p.modules.map(d => d.id), b: p.board.map(d => d.id) });
}
/* words the parts print */
ok('an LCD prints an option\'s lcd words', say(lcdMsg(ix, 'pp.norm.target', ace('lab', ['post', 'norm']))) === '-14 LUFS');
ok('a pedal knob\'s number without its unit', J(bareText(ix, 'pp.norm.ceil', ace('lab', ['post', 'norm']))) === J({ text: '−1.0', unit: 'dBFS' }));
ok('a percent stays on its number', bareText(ix, 'pp.ss.str', ace('lab', ['post', 'ss'])) === null);
ok('a checked text says why it is not taken', say(checkText(ix, 'pp.ss.seed', ace('lab', ['post', 'ss']), 'not a seed')!) === 'The fixed seed is a whole number from 0 to 4294967295.');
ok('and nothing when it is', checkText(ix, 'pp.ss.seed', ace('lab', ['post', 'ss']), '1234') === null);
/* the desk's order is kept by the records */
{
  const d0 = ace('studio', ['post', 'avoid', 'ss', 'wh']);
  const saved = savedDeskOf(ix, d0, emptyBook());
  ok('the autosave keeps the modules in the desk\'s order', J(saved.modules) === '["post","avoid"]', saved.modules);
  const back = readDesk(ix, JSON.parse(J(saved)))!;
  ok('and reading it back keeps the order', J([...back.desk.on].filter(id => id === 'post' || id === 'avoid')) === '["post","avoid"]');
  ok('a workflow keeps it too', J(workflowOf(ix, d0, 'x').modules) === '["post","avoid"]');
}
void ENGINE_IDS;

console.log(`layout: ${packed} layouts the concept chose, ${PACK.cases.length - fails.filter(f => f.startsWith('layout')).length} reproduced`);
console.log(`node words: ${looked} looked up, every one a key the definitions give`.replace('every one', fails.some(f => f.includes('is a key')) ? 'NOT every one' : 'every one'));
if (fails.length) { console.log(`\n${fails.length} failed:`); fails.slice(0, 40).forEach(f => console.log('  ' + f.slice(0, 400))); process.exit(1); }
console.log('the view\'s selectors: ok');
