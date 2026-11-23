// What each tier says about what it does not show (docs/05, "What each tier says about what is set"): a tile's
// "Also set" line and where its link goes, the parts' tips, and whether a control shows as changed. The concept's
// paintAlso and alsoWords (30-tile.js).

import type { Definition, Node, Tier, Value } from '../definitions/types.ts';
import { PARTS, type PartSpec } from '../definitions/parts.ts';
import { type Desk, type Index, applies, changed, editsAt, has, resolve, valueOf } from './core.ts';
import { type Msg, type Slot, list, msg } from './messages.ts';
import { fillMsg, formatMsg, paramWords } from './strings.ts';
import { defName } from './steps.ts';

/** An item of the line: the parameter's `short` (when it says more than two letters) or its label without a
 *  parenthesis, after the pedal's name for a pedal's; a switch is its name, or "<name> off"; else "<name> <value>". */
export function alsoWords(ix: Index, key: string, desk: Desk): Msg {
  const { p, d } = ix.params.get(key)!, w = paramWords(ix, key, desk), v = valueOf(ix, key, desk)!;
  let n: Msg = w.short && w.short.en!.length > 2 ? w.short : { ...w.label, en: w.label.en!.replace(/\s*\(.*\)$/, '') };
  if (d.kind === 'pedal') n = msg('name.owned', { owner: defName(d, desk), label: n });
  if (p.type === 'boolean') return v ? n : msg('also.off', { name: n });
  return msg('also.value', { name: n, value: formatMsg(ix, key, desk, v) });
}

/** The parameters a tile answers for: its own; the board's module's (`board`) are its own and those of every pedal
 *  and unit on the board on this engine. */
export function tileKeys(ix: Index, d: Definition, desk: Desk): string[] {
  const own = d.params.map(p => p.key);
  if (d.kind !== 'module' || !d.board) return own;
  return [...own, ...ix.defs.filter(x => x.kind !== 'module' && desk.on.has(x.id) && x.engines.includes(desk.engine)).flatMap(x => x.params.map(p => p.key))];
}

/** The keys a pedal or unit edits at a tier: at Lab every parameter, at Studio its main knob (or its Studio face) and
 *  switch, at Quick its switch. */
function boardEdits(ix: Index, d: Definition, tier: Tier, desk: Desk): Set<string> {
  if (d.kind === 'module') return editsAt(ix, d, { ...desk, tier });
  if (tier === 'lab') return new Set(d.params.map(p => p.key));
  const out = new Set<string>(d.on ? [d.on] : []);
  if (tier === 'studio' && d.kind === 'pedal') { if (d.main) out.add(d.main); editsAt(ix, d, { ...desk, tier }).forEach(k => out.add(k)); }
  if (d.kind === 'unit') editsAt(ix, d, { ...desk, tier }).forEach(k => out.add(k));
  return out;
}

export interface AlsoSet { keys: string[]; text: Msg; link: Tier }

/** The tile's "Also set" line at the desk's tier: its changed parameters no node shown at this tier edits, and the
 *  tier its link goes to: Studio from Quick only when Studio edits every one of them, else Lab. Null at Lab, or when
 *  nothing hidden is changed. */
export function alsoSet(ix: Index, d: Definition, desk: Desk): AlsoSet | null {
  if (desk.tier === 'lab' || d.kind !== 'module') return null;
  const keys = tileKeys(ix, d, desk);
  const editsHere = (t: Tier) => {
    const s = new Set(editsAt(ix, d, { ...desk, tier: t }));
    for (const x of ix.defs) if (x.kind !== 'module' && keys.some(k => ix.params.get(k)!.d === x)) boardEdits(ix, x, t, desk).forEach(k => s.add(k));
    return s;
  };
  const here = editsHere(desk.tier);
  const hidden = keys.filter(k => has(ix, k, desk.engine) && changed(ix, k, desk) && !here.has(k));
  if (!hidden.length) return null;
  const studio = editsHere('studio');
  const link: Tier = desk.tier === 'quick' && hidden.every(k => studio.has(k)) ? 'studio' : 'lab';
  return { keys: hidden, text: msg('also.set', { items: list(hidden.map(k => alsoWords(ix, k, desk)), 'dot') }), link };
}
export const alsoLinkWords = (a: AlsoSet): Msg => msg('also.link', { tier: msg(`tier.${a.link}`) });

/** A control's tip: its parameter's tip (or its node's words, `nodeTipKey` its key), then how to work its part. */
export function tipOf(ix: Index, node: Node, desk: Desk, nodeTipKey = ''): Slot {
  const spec = (PARTS as Record<string, PartSpec>)[node.part];
  const key = node.param ?? (Array.isArray(node.params) ? node.params[0] : undefined);
  const own = node.tip ? fillMsg(ix, nodeTipKey, node.tip, desk) : key && ix.params.has(key) ? paramWords(ix, key, desk).tip : undefined;
  const how = spec?.howTo ? { id: `part.${node.part}.howTo`, en: spec.howTo } : undefined;
  return list([own, how].filter((x): x is Msg => !!x), 'comma');
}

/** What a pedal's blanking plate says over the knobs its parameters' `when` hides on the desk (docs/01, "How a pedal
 *  and a side unit are laid out"): for knobs hidden by { param, is }, "with <that parameter's label>: <the labels of
 *  the values that show them, joined by or>"; for a switch, "with <its label> on". Null when nothing is hidden, or a
 *  hidden knob's condition is not of that form (the plate is then blank). */
export function plateWords(ix: Index, d: Definition, desk: Desk): Msg | null {
  const hidden = d.params.filter(p => p.type === 'number' && has(ix, p.key, desk.engine) && !applies(ix, p.key, desk));
  if (!hidden.length) return null;
  const by = new Map<string, Set<Value>>();
  for (const p of hidden) {
    const w = resolve(p, desk.engine).when as { param?: string; is?: Value | Value[] } | undefined;
    if (!w || !w.param || w.is === undefined || Object.keys(w).length !== 2) return null;
    const set = by.get(w.param) || new Set<Value>(); [w.is].flat().forEach(v => set.add(v)); by.set(w.param, set);
  }
  return msg('plate.with', { groups: list([...by].map(([key, vals]) => {
    const label = paramWords(ix, key, desk).label;
    /* a switch reads "Chain on", not "Chain on: on" (DH-24) */
    if (ix.params.get(key)!.p.type === 'boolean' && vals.size === 1) return msg('plate.switch', { label, value: formatMsg(ix, key, desk, [...vals][0]!) });
    return msg('plate.group', { label, values: list([...vals].map(v => formatMsg(ix, key, desk, v)), 'or') });
  }), 'semicolon') });
}
