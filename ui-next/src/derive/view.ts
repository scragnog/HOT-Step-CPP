// What a thin view asks the core, so that no component decides anything about a value (docs/05, "What the view may
// decide"): which choice of a Quick row is checked and what pressing one does, a pedal's or unit's laid-out nodes at a
// tier, what an action does, whether a node stands back, a pedal's idle reason, the note that names the Lab modules a
// tier hides, and loading a workflow or a draft.

import type { Action, Definition, Draft, Node, NodeOverride, Tier, Value, Workflow } from '../definitions/types.ts';
import { layOut, pedalUnits, type PartName } from '../definitions/parts.ts';
import { RETIRED } from '../definitions/retired.ts';
import { type Desk, type Index, applies, changed, has, holds, nodeAt, onDesk, resolve, valueOf } from './core.ts';
import { type Msg, type Slot, list, msg } from './messages.ts';
import { applyChange, type Change } from './parts.ts';
import { migrate } from './records.ts';
import { defName, rowWord, stepWords, type RowStep } from './steps.ts';
import { fillMsg, formatMsg, hash, nodeWord, paramWords, stripWords, textMsg } from './strings.ts';
import { plateWords } from './tiers.ts';
import { cleanValues, setValue } from './values.ts';

/* ---------------------------------------------------------------- a Quick choice row */

/** The row as steps.ts names it, with each choice's words as a message keyed by its place (strings.ts). */
export function rowOf(ix: Index, d: Definition, node: Node, desk: Desk, key: string): RowStep {
  const at = nodeAt(ix, node, desk)!, choices = at.choices ?? [];
  /* a choice's words are keyed under the line that gives the row its choices (strings.ts), so retuning one engine's
     presets keeps its words */
  const line = ((node.per || []) as NodeOverride[]).filter(l => l.choices && (!l.engines || l.engines.includes(desk.engine)) && (!l.tiers || l.tiers.includes(desk.tier)) && (!l.when || holds(ix, l.when, desk))).pop();
  const sfx = line ? `@${hash(JSON.stringify([line.engines, line.tiers, line.when]))}` : '';
  return { key: at.param ?? Object.keys(typeof choices[0]?.[1] === 'object' ? choices[0][1] as object : {})[0] ?? '', label: at.label ? nodeWord(ix, node, desk, key, 'label') ?? fillMsg(ix, `${key}.label`, at.label, desk) : at.param ? paramWords(ix, at.param, desk).label : '', choices: choices.map(([w, v], i) => [{ id: `${key}.c.${i}${sfx}`, en: w }, v]) };
  void d;
}

/** Which choice of a row is checked (the one whose values are all in force), and what the row says: the choice's words,
 *  or, when the values match none of them, the value between the choices (`between`) or each value set, so a value set
 *  at another tier is never hidden. */
export function choiceOf(ix: Index, d: Definition, node: Node, desk: Desk, key: string): { checked: number | null; says: Slot } {
  const at = nodeAt(ix, node, desk)!, row = rowOf(ix, d, node, desk, key);
  const inForce = (v: Value | { [k: string]: Value }) => (typeof v === 'object' ? Object.entries(v).every(([k, x]) => valueOf(ix, k, desk) === x) : valueOf(ix, row.key, desk) === v);
  const i = (at.choices ?? []).findIndex(c => inForce(c[1]));
  if (i >= 0) return { checked: i, says: row.choices[i]![0] };
  /* the row's words for a value between its choices name it as {v}, filled with the value as the page writes it */
  if (at.between && at.param) return { checked: null, says: { ...fillMsg(ix, `${key}.between`, at.between, desk), slots: { v: textMsg(ix, at.param, desk) } } };
  return { checked: null, says: rowWord(ix, row, desk) };
}

/** Pressing a choice: every value it sets, as one step named by the row ("Loudness: Gentle → Loud"). */
export function applyChoice(ix: Index, d: Definition, node: Node, desk: Desk, key: string, i: number): Change {
  const row = rowOf(ix, d, node, desk, key), c = (nodeAt(ix, node, desk)!.choices ?? [])[i];
  if (!c) throw new Error(`the row has no choice ${i}`);
  const sets = typeof c[1] === 'object' ? c[1] : { [row.key]: c[1] };
  const vals: Change['vals'] = {}, auto: Change['auto'] = {};
  for (const [k, v] of Object.entries(sets)) { vals[k] = setValue(ix, desk, k, v).value; if (resolve(ix.params.get(k)!.p, desk.engine).auto) auto[k] = false; }
  const words = stepWords(ix, desk, applyChange(desk, { vals, auto, state: {}, words: { label: msg(''), put: msg('') } }), { row })!;
  return { vals, auto, state: {}, words };
}

/* ---------------------------------------------------------------- pedals and side units, laid out */

/** One laid-out cell of a pedal or a unit: the part, the parameter, and how it stands. */
export interface Cell { part: PartName | 'plate' | 'node'; param?: string; node?: Node; plate?: Msg | null; beside?: string; shown: boolean }
/** A pedal's or unit's body at the desk's tier (docs/01, "How a pedal and a side unit are laid out"): at Quick its
 *  switch; at Studio its switch and main knob, or its Studio nodes; at Lab its numbers as knobs (a hidden one a blank
 *  plate with its words), its choices and text, its Lab nodes, then its switches, each with what stands beside it. */
export function boardNodes(ix: Index, d: Definition, desk: Desk): { units: 1 | 2; cells: Cell[] } {
  if (d.kind === 'module') throw new Error(`${d.id} is a module: its tile is its face`);
  const on = d.engines.includes(desk.engine) ? d.params.filter(p => has(ix, p.key, desk.engine)) : [];
  const numbers = on.filter(p => p.type === 'number'), units = pedalUnits(numbers.length);
  /* the nodes as the definition writes them (the renderer resolves each, and finds its words by it) */
  const face = (t: Tier) => (d.face || []).filter(n => !!nodeAt(ix, n, { ...desk, tier: t }));
  const sw = d.on ? [{ part: 'ledButton' as const, param: d.on, shown: true }] : [];
  if (desk.tier === 'quick') return { units, cells: d.kind === 'unit' ? [...face('quick').map(n => ({ part: 'node' as const, node: n, shown: true })), ...sw] : sw };
  if (desk.tier === 'studio') {
    const main = d.kind === 'pedal' && d.main ? [{ part: 'knob' as const, param: d.main, shown: applies(ix, d.main, desk) }] : face('studio').map(n => ({ part: 'node' as const, node: n, shown: true }));
    return { units, cells: [...sw, ...main] };
  }
  const plate = plateWords(ix, d, desk);
  const cells: Cell[] = [
    ...(d.kind === 'unit' ? face('lab').filter(n => !(n.opts as { notAtLab?: boolean } | undefined)?.notAtLab).map(n => ({ part: 'node' as const, node: n, shown: true })) : []),
    ...numbers.map(p => (applies(ix, p.key, desk) ? { part: 'knob' as const, param: p.key, shown: true } : { part: 'plate' as const, param: p.key, plate, shown: false })),
    ...on.filter(p => (p.type === 'choice' || p.type === 'text') && !p.beside).map(p => { const o = resolve(p, desk.engine).options; return { part: layOut(d.kind, p.type, Array.isArray(o) ? o.length : null), param: p.key, shown: applies(ix, p.key, desk) }; }),
    ...(d.kind === 'pedal' ? face('lab').map(n => ({ part: 'node' as const, node: n, shown: true })) : []),
    ...on.filter(p => p.type === 'boolean' && !p.beside).flatMap(p => [
      { part: 'ledButton' as const, param: p.key, shown: true },
      ...on.filter(q => q.beside === p.key).map(q => ({ part: layOut(d.kind as 'pedal' | 'unit', q.type, null), param: q.key, beside: p.key, shown: valueOf(ix, p.key, desk) === false })),
    ]),
  ];
  return { units, cells };
}

/** A pedal's idle reason at the desk's tier (engaged with nothing to run): the first that holds, or null. */
export function idleOf(ix: Index, d: Definition, desk: Desk): { says: Msg; actions: Action[] } | null {
  if (d.kind !== 'pedal' || !d.on || valueOf(ix, d.on, desk) !== true) return null;
  const s = (d.idle || []).find(x => (!x.tiers || x.tiers.includes(desk.tier)) && holds(ix, x.when, desk));
  return s ? { says: stripWords(ix, `create.${d.id}.idle`, s, desk).says, actions: s.actions ?? [] } : null;
}

/** Whether a node stands back on the desk, and why. */
export function dimOf(ix: Index, node: Node, desk: Desk, key: string): Msg | null {
  const at = nodeAt(ix, node, desk);
  return at?.dim && holds(ix, at.dim.when, desk) ? fillMsg(ix, `${key}.dim`, at.dim.says, desk) : null;
}

/* ---------------------------------------------------------------- actions */

/** What an action does: a change to the desk (named as every step is), or something only the shell can do (go to a
 *  tier and ring controls, add a module, run a named action). */
export type Did = { change: Change } | { go: Tier; ring: string[] } | { add: string } | { call: string };
export function runAction(ix: Index, d: Definition, a: Action, desk: Desk, node?: Node): Did {
  const one = (vals: Change['vals'], auto: Change['auto']): Did => {
    const words = stepWords(ix, desk, applyChange(desk, { vals, auto, state: {}, words: { label: msg(''), put: msg('') } }))!;
    return { change: { vals, auto, state: {}, words } };
  };
  switch (a.do) {
    case 'set': return one({ [a.param]: setValue(ix, desk, a.param, a.value).value }, resolve(ix.params.get(a.param)!.p, desk.engine).auto ? { [a.param]: false } : {});
    case 'auto': return one({}, { [a.param]: true });
    case 'example': return one({ [a.param]: resolve(ix.params.get(a.param)!.p, desk.engine).example ?? '' }, {});
    case 'reset': {
      const keys = new Set<string>(); const walk = (n: Node) => { [n.param, ...(n.params ?? [])].forEach(k => k && keys.add(k)); (n.children ?? []).forEach(walk); };
      if (node) walk(node);
      return one(Object.fromEntries([...keys].map(k => [k, null])), Object.fromEntries([...keys].filter(k => resolve(ix.params.get(k)!.p, desk.engine).auto).map(k => [k, true])));
    }
    case 'tag': {
      const key = d.params.find(p => p.type === 'text')?.key ?? 'lyrics.text', text = String(valueOf(ix, key, desk) ?? '');
      return one({ [key]: `${text}${text && !text.endsWith('\n') ? '\n' : ''}${a.tag}\n` }, {});
    }
    case 'tier': return { go: a.tier, ring: a.ring ?? [] };
    case 'add': return { add: a.item };
    case 'bespoke': return { call: a.name };
  }
}

/* ---------------------------------------------------------------- the desk as a whole */

/** The note above the desk while the tier hides Lab modules that are on it (and used by its engine): their names. */
export function hiddenLabModules(ix: Index, desk: Desk): Msg | null {
  if (desk.tier === 'lab') return null;
  /* in the desk's order (the order of desk.on, the user's: RC-14), as the concept names them */
  const order = [...desk.on], hid = ix.defs.filter(d => d.kind === 'module' && !!d.labUnder && onDesk(d, desk) && d.engines.includes(desk.engine)).sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  return hid.length ? msg('desk.hidden', { names: list(hid.map(d => defName(d, desk))) }) : null;
}

/** How many of the desk's settings are changed (the go bar's "N changes"): every parameter of what is on the desk and
 *  used by its engine, the pedals' foot switches left out. */
export function changesOn(ix: Index, desk: Desk): string[] {
  return ix.defs.filter(d => onDesk(d, desk) && d.engines.includes(desk.engine) && !(d.kind === 'module' && d.source))
    .flatMap(d => d.params.filter(p => !(d.kind !== 'module' && d.on === p.key) && changed(ix, p.key, desk)).map(p => p.key));
}

/** Loading a workflow: its modules and board on the desk (under their current ids; a retired id is dropped without a
 *  word, anything else unknown is counted), and its tier. Values are untouched. */
export function loadWorkflow(ix: Index, desk: Desk, raw: unknown): { desk: Desk; told: Msg | null } {
  const r = migrate(raw);
  if (!r || r.kind !== 'workflow') return { desk, told: msg('load.unreadable') };
  const w = r.record as Workflow;
  let lost = 0;
  const ids = (list0: unknown[]) => list0.flatMap(x => {
    const id = typeof x === 'string' ? ix.formerIds.get(x) ?? x : null;
    if (id && ix.defs.some(d => d.id === id)) return [id];
    if (!(typeof x === 'string' && RETIRED.ids.includes(x))) lost++;
    return [];
  });
  const on = new Set([...ix.defs.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...ids(Array.isArray(w.modules) ? w.modules : []), ...ids(Array.isArray(w.board) ? w.board : [])]);
  const tier: Tier = (['quick', 'studio', 'lab'] as const).includes(w.tier) ? w.tier : desk.tier;
  return { desk: { ...desk, on, tier }, told: lost ? msg('load.told', { n: lost }) : null };
}

/** Loading a draft: its content values, cleaned, for its engine; a draft for another engine is refused with a word. */
export function loadDraft(ix: Index, desk: Desk, raw: unknown): { desk: Desk; told: Msg | null } {
  const r = migrate(raw);
  if (!r || r.kind !== 'draft') return { desk, told: msg('load.unreadable') };
  const dr = r.record as Draft;
  if (dr.engine !== desk.engine) return { desk, told: msg('load.engine', { engine: dr.engine }) };
  const c = cleanValues(ix, dr.vals, dr.auto, RETIRED.keys);
  const n = c.dropped.length;
  return { desk: { ...desk, vals: { ...desk.vals, ...c.vals }, auto: { ...desk.auto, ...c.auto } }, told: n ? msg('load.told', { n }) : null };
}

export { formatMsg };
