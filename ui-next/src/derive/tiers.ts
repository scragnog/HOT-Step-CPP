// What each tier says about what it does not show (docs/05, "What each tier says about what is set"): a tile's
// "Also set" line and where its link goes, the parts' tips, and whether a control shows as changed. The concept's
// paintAlso and alsoWords (30-tile.js).

import type { Definition, Node, Tier } from '../definitions/types.ts';
import { PARTS, type PartSpec } from '../definitions/parts.ts';
import { type Desk, type Index, changed, editsAt, fill, formatValue, has, valueOf, wordsOf } from './core.ts';

const TIER_NAME: { [t in Tier]: string } = { quick: 'Quick', studio: 'Studio', lab: 'Lab' };

/** An item of the line: the parameter's `short` (when it says more than two letters) or its label without a
 *  parenthesis, after the pedal's name for a pedal's; a switch is its name, or "<name> off"; else "<name> <value>". */
export function alsoWords(ix: Index, key: string, desk: Desk): string {
  const { p, d } = ix.params.get(key)!, w = wordsOf(ix, p, desk), v = valueOf(ix, key, desk)!;
  let n = w.short && w.short.length > 2 ? w.short : w.label.replace(/\s*\(.*\)$/, '');
  if (d.kind === 'pedal') n = `${d.name} ${n.charAt(0).toLowerCase()}${n.slice(1)}`;
  if (p.type === 'boolean') return v ? n : `${n} off`;
  return `${n} ${formatValue(w, v, desk.lists)}`;
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

export interface AlsoSet { keys: string[]; text: string; link: Tier }

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
  return { keys: hidden, text: `Also set: ${hidden.map(k => alsoWords(ix, k, desk)).join(' · ')}.`, link };
}
export const alsoLinkWords = (a: AlsoSet) => `Edit in ${TIER_NAME[a.link]}`;

/** A control's tip: its parameter's tip (or its node's), then how to work its part, as the part says it. */
export function tipOf(ix: Index, node: Node, desk: Desk): string {
  const spec = (PARTS as Record<string, PartSpec>)[node.part];
  const key = node.param ?? (Array.isArray(node.params) ? node.params[0] : undefined);
  const own = node.tip ?? (key && ix.params.has(key) ? wordsOf(ix, ix.params.get(key)!.p, desk).tip : undefined);
  const text = [own ? fill(ix, own, desk, key ? wordsOf(ix, ix.params.get(key)!.p, desk) : undefined) : '', spec?.howTo ?? ''].filter(Boolean).join(' ');
  return text;
}
