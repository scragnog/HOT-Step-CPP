// Tests of the core's additions for package 3b (src/derive/cartridges.ts: the cartridges' model and hands, section
// directives, phase windows, the XY pad's off rail; src/derive/strings.ts forcedOf and the models' forced choice; a
// fader's name and tape; the Adapters' narrow tile), each against what the concept does (25-module-adapters.js,
// 21-xy-pad.js, 21-fader.js, 30-tile.js tileW).
//
//   node tools/test-cartridges.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Node, Tier, Value } from '../src/definitions/types.ts';
import {
  adapterLists, adaptersAway, bindPart, directivesOf, dropDirectiveSlot, indexOf, nodeKeys, partAct, partModel, phaseMove, setDirective, tileModel,
  xyRail, xyRailKey, xyRailValue, type Change, type Desk, type Msg,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';
import type { ModuleDef } from '../src/definitions/types.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const ACE = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }, { file: 'synthwave-80s', type: 'DoRA' }];
const LISTS = { ...adapterLists('ace', [ACE, ACE, ACE]), ...adapterLists('yue2', [[{ file: 'folk-voice-ar', type: 'LoRA' }, { file: 'ar-dora', type: 'DoRA' }]]), ...adapterLists('', [], [{ file: 'folk-voice-planner', type: 'LoRA' }]) };
const desk = (engine: EngineId, tier: Tier, vals: Desk['vals'] = {}, slots: unknown[] = [], more: Partial<Desk> = {}): Desk =>
  ({ engine, tier, vals, auto: {}, state: { slots: slots as never }, lists: LISTS, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), 'adapters', 'planner']), ...more });
const def = (id: string) => DEFINITIONS.find(d => d.id === id)!;
const s = (m: unknown) => say(m as Msg);
const node = (id: string, part: string): Node => (def(id) as ModuleDef).face.find(x => x.part === part)!;
const CN = node('adapters', 'cartridges'), KEY = nodeKeys(def('adapters')).get(CN) ?? '';
const carts = (d: Desk) => partModel(ix, def('adapters'), CN, d, KEY, 'tile')!.instrument!.carts!;
function act(d0: Desk, name: string, arg?: unknown) {
  let d = d0; const steps: Change[] = [], told: string[] = [];
  const props = bindPart(ix, def('adapters'), CN, () => d, c => { steps.push(c); d = { ...d, vals: { ...Object.fromEntries(Object.entries(d.vals).filter(([k]) => c.vals[k] !== null)), ...Object.fromEntries(Object.entries(c.vals).filter((e): e is [string, Value] => e[1] !== null)) }, auto: { ...d.auto, ...c.auto }, state: { ...d.state, ...c.state } }; });
  const done = partAct(ix, props.node, props, d, name, arg, m => told.push(s(m)));
  return { steps, told, done, desk: d };
}
const TWO = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'folk-voice', type: 'LoKr' }];

/* ---- section directives (25-module-adapters.js sectionDirectives, setDirective, dropDirectiveSlot) */
{
  const t = '[Verse]\na\n[Chorus]{#1=0.5; #2=1}\nb';
  ok('a directive reads as each slot\'s weight; a section without one reads null', J(directivesOf(t)) === J([null, { 0: 0.5, 1: 1 }]));
  ok('setting a weight writes the section\'s directive', setDirective(t, 0, 1, 0, 2) === '[Verse]{#1=1; #2=0}\na\n[Chorus]{#1=0.5; #2=1}\nb');
  ok('every slot back at full takes the directive out', setDirective(t, 1, 0, 1, 2) === '[Verse]\na\n[Chorus]\nb');
  ok('removing a slot renumbers the later ones', dropDirectiveSlot('[V]{#1=0.5; #2=0}\nx', 0) === '[V]{#1=0}\nx');
  ok('and drops a directive left at full', dropDirectiveSlot('[V]{#1=0; #2=1}\nx', 0) === '[V]\nx');
  ok('nothing to renumber is null', dropDirectiveSlot('[V]\nx', 0) === null);
}

/* ---- phase windows (phaseHTML's keys and setPhase) */
{
  ok('a key steps 5%, a handle never closer than 5% to the other', J(phaseMove([0, 100], 0, { key: 'ArrowRight' })) === J([5, 100]) && J(phaseMove([90, 95], 0, { key: 'PageUp' })) === J([90, 95]));
  ok('Home and End go to the ends of the handle\'s own range', J(phaseMove([20, 80], 1, { key: 'Home' })) === J([20, 25]) && J(phaseMove([20, 80], 0, { key: 'End' })) === J([75, 80]));
  ok('a hand snaps to 5%', J(phaseMove([0, 100], 1, { f: 0.71 })) === J([0, 70]));
  ok('a key that moves nothing is null', phaseMove([0, 100], 0, { key: 'a' }) === null);
}

/* ---- the model */
{
  const q = carts(desk('ace', 'quick', { 'ad.adv': 'advanced' }, [{ file: 'synthwave-80s.safetensors', type: 'DoRA' }, TWO[0]]));
  ok('Quick lays the first flat, its name without its extension, the rest said once (Q-18)', q.look === 'flat' && q.carts.length === 1 && q.carts[0]!.name.join('') === 'synthwave-80s' && s(q.more) === '2 adapters, stacked.', q.carts[0]?.name);
  ok('a DoRA\'s art is a vector, its head from its name', q.carts[0]!.art.kind === 'vec' && !!q.carts[0]!.art.head);
  const one = carts(desk('ace', 'studio', {}, [TWO[0]]));
  ok('one cartridge at Studio: no kind, its words and the way to a second beside it, its foot there (Q-19)', one.one && one.sideFoot && s(one.carts[0]!.label) === 'DiT' && s(one.side!.second!.label) === '+ Second adapter' && one.side!.note === null && one.add === null, one);
  const st = carts(desk('ace', 'lab', { 'ad.adv': 'advanced', 'lyrics.text': '[Verse]\na\n[Chorus]\nb\n[Verse 2]\nc' }, [{ ...TWO[0], ph: [20, 80] }, TWO[1]]));
  ok('a stack at Lab: #n labels, shares, phases and a chip per section (V1, C, V2)', s(st.carts[0]!.label) === '#1 · DiT' && s(st.carts[0]!.eff) === 'sends 0.38' && st.carts[0]!.phase?.span === '20–80%' && J(st.carts[0]!.masks!.map(m => m.tag)) === J(['V1', 'C', 'V2']), st.carts[0]);
  ok('the empty slot names its kind at Lab, and the planner adapter\'s slot is offered', s(st.add!.label) === '+ DiT adapter' && !!st.planner && !('cart' in st.planner));
  ok('the list for the next slot leaves out what is loaded', J(st.list!.items.map(x => x.file)) === J(['synthwave-80s']));
  const un = carts(desk('ace', 'studio', {}, TWO));
  ok('two adapters while Stack is Single: the second is loaded but not used, with the way to stack them (A-14)', s(un.unused!.text) === 'folk-voice is loaded but not used while Stack is Single.' && !!un.unused!.stack, un.unused);
  const yue = carts(desk('yue2', 'studio', {}, [{ file: 'folk-voice-ar', type: 'LoRA' }]));
  ok('YuE2\'s AR cartridge: Strength beside Plan and Compose (E-05), the NAR slot offered beside it (E-10)', J(yue.carts[0]!.faders.map(f => s(f.label))) === J(['Strength', 'Plan', 'Compose']) && s(yue.add!.label) === '+ NAR · renderer adapter' && /NAR renderer adapter/.test(s(yue.side!.note)), yue);
  ok('a planner housing is never offered a DoRA (B-02)', carts(desk('yue2', 'lab', {}, [])).list!.items.every(x => x.type !== 'DoRA'));
  const pl = carts(desk('ace', 'lab', { 'ad.planner': 'folk-voice-planner' }, [TWO[0]]));
  ok('the planner adapter as its cartridge, its note where a fader would be', !!pl.planner && 'cart' in pl.planner && s(pl.planner.cart.note) === 'Shapes the plan, not the sound' && pl.planner.cart.housing === 'plan');
  ok('another engine\'s adapters, said from the book', adaptersAway({ stash: { ace: { state: { slots: TWO } } } } as never, 'yue2', 'slots') === ' · ACE-Step 1.5 keeps its 2 adapters for when you switch back');
}

/* ---- the hands */
{
  const l = act(desk('ace', 'studio', {}, [TWO[0]]), 'load', { file: 'folk-voice', type: 'LoKr' });
  ok('a second adapter loaded while Stack is Single stacks them, one step named for the file (L-22)', l.steps.length === 1 && s(l.steps[0]!.words.label) === 'Load folk-voice' && l.desk.vals['ad.adv'] === 'advanced', l.steps.map(x => x.words));
  const r = act(desk('ace', 'lab', { 'ad.adv': 'advanced', s1: 1.5, 'lyrics.text': '[V]{#1=0.5; #2=0}\nx' }, TWO), 'remove', 0);
  ok('removing one moves the later strength up and renumbers the directives, said with Undo', r.desk.vals.s0 === 1.5 && r.desk.vals['lyrics.text'] === '[V]{#1=0}\nx' && r.told[0] === 'Removed tape-lofi-v2. The lyric directives were renumbered to match' && s(r.steps[0]!.words.label) === 'Remove tape-lofi-v2', r);
  const p = act(desk('ace', 'lab', { 'ad.adv': 'advanced' }, TWO), 'phase', { i: 0, h: 0, key: 'ArrowRight' });
  ok('a phase key is a step named for its adapter (L-01)', s(p.steps[0]?.words.label) === 'Phase of tape-lofi-v2 0–100% → 5–100%', p.steps.map(x => x.words));
  const m = act(desk('ace', 'lab', { 'ad.adv': 'advanced', 'lyrics.text': '[Verse]\na\n[Chorus]\nb' }, TWO), 'mask', { i: 0, j: 1, w: 0.5 });
  ok('a mask chip writes the lyrics, a step named for its section and adapter (L-09)', m.desk.vals['lyrics.text'] === '[Verse]\na\n[Chorus]{#1=0.5; #2=1}\nb' && s(m.steps[0]?.words.label) === 'Chorus: tape-lofi-v2 full → half', m.steps.map(x => x.words));
  const mm = act(desk('ace', 'lab', { 'ad.adv': 'advanced', 'lyrics.text': '[Verse]{#1=1; #2=0.5}\na\n[Chorus]\nb' }, TWO), 'mask', { i: 1, j: 1, w: 0.5, from: '[Verse]\na\n[Chorus]\nb' });
  ok('a drag across chips is named for them all', s(mm.steps[0]?.words.label) === 'Section masks of folk-voice', mm.steps.map(x => x.words));
}

/* ---- a choice run at another value (A-30), a fader's name and tape, the narrow tile */
{
  const d = desk('ace', 'lab', { 'ad.adv': 'advanced', 'ad.mode': 'merge', 'lyrics.text': '[V]{#1=0.5; #2=1}\nx' }, TWO);
  const foot = ((CN.children ?? []).find(x => x.label === 'Apply as')!.children ?? [])[0]!;
  const pm = partModel(ix, def('adapters'), foot, d, 'k', 'foot')!.params[0]!;
  ok('Apply as while directives are set: Runtime pressed, the others disabled, the choice kept and said', J(pm.options!.map(o => [o.value, o.on, !!o.off])) === J([['merge', false, true], ['runtime', true, false], ['runtime_lowrank', false, true]]) && s(pm.forced!.note) === 'Runtime while section directives are set · Merge returns after' && pm.value === 'merge', pm);
  const bank = (CN.children ?? []).find(x => x.part === 'bank')!.children!.find(x => x.param === 'ag.self')!;
  const f = partModel(ix, def('adapters'), bank, d, 'k', 'foot')!.control!.fader!;
  ok('a bank fader is named in full with its strength, its short tape carrying the full name (V-01)', s(f.name) === 'Self-Attn strength' && s(f.tape) === 'Self-Attn');
  const tile = (dd: Desk) => tileModel(ix, def('adapters') as ModuleDef, dd).cls;
  ok('the tile is one column at Quick and Studio with one cartridge, two in a stack and at Lab', !/w2/.test(tile(desk('ace', 'studio', {}, [TWO[0]]))) && /w2/.test(tile(desk('ace', 'studio', { 'ad.adv': 'advanced' }, TWO))) && /w2/.test(tile(desk('ace', 'lab', {}, [TWO[0]]))));
}

/* ---- the XY pad's off rail (21-xy-pad.js yFrac, yVal and its keys) */
{
  ok('off stands on the rail; the plot starts a step above it', J(xyRail(0, 1, 0.01, 0)) === J({ off: true, at: 1 }) && J(xyRail(0, 1, 0.01, 0.01)) === J({ off: false, at: 0 }));
  ok('a hand over the rail is off, on the plot a step above it at least', xyRailValue(0, 1, 0.01, 1.2, true) === 0 && xyRailValue(0, 1, 0.01, 0, false) === 0.01 && xyRailValue(0, 1, 0.01, 1, false) === 1);
  ok('keys: off goes down to the widest, up from the widest is off, a step otherwise', xyRailKey(0, 1, 0.01, 0, 'ArrowDown', false) === 1 && xyRailKey(0, 1, 0.01, 1, 'ArrowUp', false) === 0 && xyRailKey(0, 1, 0.01, 0.5, 'ArrowDown', true) === 0.4 && xyRailKey(0, 1, 0.01, 0.5, 'ArrowLeft', false) === null);
}

console.log(fails.length ? `\n${fails.length} failed:\n  ${fails.join('\n  ')}` : `cartridges: ${n} checks of the cartridges' core: ok`);
process.exitCode = fails.length ? 1 : 0;
