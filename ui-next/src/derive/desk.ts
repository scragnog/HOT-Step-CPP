// What the desk's make-up says, as the page asks it: which nodes are drawn, which tiles a tier shows and which wait
// parked, the names of the steps that change the desk's make-up (add, remove, move, an engine switch), what a removal
// says, and the Pedalboard's count. The concept's renderDesk, removeMod and paintCount (30-desk.js, 27-pedalboard.js).
// Like the rest of the core, nothing here has a branch for any one module.

import type { Definition, EngineId, Node } from '../definitions/types.ts';
import { PARTS, type PartSpec } from '../definitions/parts.ts';
import { engineName } from '../definitions/engines.ts';
import { type Desk, type Index, applies, has, nodeAt, valueOf } from './core.ts';
import { type Msg, list, msg } from './messages.ts';
import { defName, type StepWords } from './steps.ts';
import { idleOf } from './view.ts';
import type { Change } from './parts.ts';

/** Whether a node is drawn on the desk: it is shown here (nodeAt); a part that edits is bound to something the engine
 *  in use has; a container has at least one child that is drawn (docs/01, "Four kinds of part"). */
export function nodeShown(ix: Index, n: Node, desk: Desk): boolean {
  const at = nodeAt(ix, n, desk);
  if (!at) return false;
  const spec = (PARTS as Record<string, PartSpec>)[at.part];
  const binds = [at.param, ...(at.params ?? [])].filter((k): k is string => !!k);
  if (spec?.edits && binds.length && !binds.some(k => has(ix, k, desk.engine))) return false;
  /* on a module's tile, a control none of whose parameters applies is not drawn, as the concept leaves out what does not
     apply (its ok()); a node that should stay and stand back says so with `dim`. A pedal or a side unit keeps every
     cell's room (a plate, or a field beside its switch: docs/01, "How a pedal and a side unit are laid out") */
  if (spec?.kind === 'control' && binds.length && binds.every(k => ix.params.get(k)?.d.kind === 'module') && !binds.some(k => applies(ix, k, desk))) return false;
  if (spec?.kind === 'container') return (at.children ?? []).some(c => nodeShown(ix, c, desk));
  return true;
}

/** A module is used by the desk's engine. One that is not waits parked. */
export const usedHere = (d: Definition, e: EngineId): boolean => d.engines.includes(e);
/** A module's tile shows at the desk's tier: a Lab module (labUnder) only at Lab. */
export const tileAtTier = (d: Definition, desk: Desk): boolean => !(d.kind === 'module' && d.labUnder) || desk.tier === 'lab';

/** What a module is called on the desk at its tier: at Quick its `quickName` when Quick shows less of it (Tempo is
 *  "Tempo & length" there), on its engine's line when that has one; else its name (defName). The tile's heading, the
 *  Modules panel's card and the steps that add, remove or move it say this. */
export function tileName(d: Definition, desk: Desk): Msg {
  if (desk.tier === 'quick' && d.kind === 'module') {
    const line = (d.per || []).find(l => l.engines.includes(desk.engine) && l.quickName);
    if (line) return { id: `create.${d.id}.quickName@${line.engines.join('+')}`, en: line.quickName! };
    if (d.quickName && !(d.per || []).some(l => l.engines.includes(desk.engine) && l.name)) return { id: `create.${d.id}.quickName`, en: d.quickName };
  }
  return defName(d, desk);
}

/** The step that changes the desk's make-up: adding, removing or moving a module, pedal or side unit, or switching
 *  the engine. Each is one undo step, named as the concept names it. */
export function deskStep(kind: 'add' | 'remove' | 'move', d: Definition, desk: Desk, ix?: Index): StepWords {
  const name = tileName(d, desk);
  /* what Undo puts back, as the concept's desk entries say it (14-undo.js namesOf): the modules, the pedals, or both. A
     pedal or unit changes the modules too when it brings the board or, the last one off, takes it away; the board
     taken off takes its pedals */
  const onBoard = ix ? [...desk.on].filter(id => ix.defs.some(x => x.id === id && x.kind !== 'module')).length : 0;
  const board = ix ? ix.defs.some(x => x.kind === 'module' && !!x.board && desk.on.has(x.id)) : true;
  const both = d.kind === 'module' ? !!d.board && kind === 'remove' && onBoard > 0 : kind === 'add' ? !board : onBoard <= 1;
  const what = both ? list([msg('path.mods'), msg('path.pedals')]) : msg(d.kind === 'module' ? 'path.mods' : 'path.pedals');
  return { label: msg(`step.${kind}`, { name }), put: msg('step.put.desk', { what }) };
}
export const engineStep = (to: EngineId): StepWords => ({ label: msg('step.engine', { engine: engineName(to) }), put: msg('step.put.desk', { what: msg('path.engine') }) });

/** What taking something off the desk says: a module (with the pedals the Pedalboard took with it), or a pedal or a
 *  side unit off the board (and whether it was the last, which takes the Pedalboard off too). */
export function removedWords(d: Definition, desk: Desk, pedals: number, last: boolean): Msg {
  const name = tileName(d, desk);
  return d.kind === 'module' ? msg('desk.removed', { name, n: pedals }) : msg('desk.tookOff', { name, last: String(last) });
}

/** The modules on the desk that the engine in use does not use, and the pedals and side units on the board it does
 *  not use: they wait parked, and come back on an engine that uses them. */
export function parkedOn(ix: Index, desk: Desk): { modules: Definition[]; board: Definition[]; says: Msg | null } {
  const on = [...desk.on].map(id => ix.defs.find(d => d.id === id)).filter((d): d is Definition => !!d);
  const modules = on.filter(d => d.kind === 'module' && !d.builtIn && !d.board && !d.source && !usedHere(d, desk.engine));
  const hasBoard = on.some(d => d.kind === 'module' && d.board);
  const board = hasBoard ? on.filter(d => d.kind !== 'module' && !usedHere(d, desk.engine)) : [];
  const n = modules.length + board.length;
  return { modules, board, says: n ? msg('desk.parked', { engine: engineName(desk.engine), n }) : null };
}

/** The Pedalboard's count: how many pedals, how many run (an always-on pedal runs; an engaged pedal with nothing to
 *  run is counted as needing setting up), or that the chain is bypassed; the side units and how many are on; what is
 *  parked. `chain` is the board module's switch. */
export function boardCount(ix: Index, desk: Desk, chain: string | null): Msg {
  const on = [...desk.on].map(id => ix.defs.find(d => d.id === id)).filter((d): d is Definition => !!d);
  const pedals = on.filter(d => d.kind === 'pedal' && usedHere(d, desk.engine)), units = on.filter(d => d.kind === 'unit' && usedHere(d, desk.engine));
  const parked = on.filter(d => d.kind !== 'module' && !usedHere(d, desk.engine)).length;
  const isOn = (k: string | null) => k === null || valueOf(ix, k, desk) === true;
  const parts: Msg[] = [];
  if (!pedals.length) parts.push(msg('board.none'));
  else {
    parts.push(msg('board.pedals', { n: pedals.length }));
    if (!isOn(chain)) parts.push(msg('board.bypassed'));
    else {
      const idle = pedals.filter(d => d.kind === 'pedal' && !!d.on && isOn(d.on) && !!idleOf(ix, d, desk)).length;
      const running = pedals.filter(d => d.kind === 'pedal' && isOn(d.on)).length - idle;
      parts.push(msg('board.on', { n: running }));
      if (idle) parts.push(msg('board.idle', { n: idle }));
    }
  }
  if (units.length) parts.push(msg('board.units', { n: units.length, on: units.filter(d => d.kind === 'unit' && isOn(d.on)).length }));
  if (parked) parts.push(msg('board.parked', { n: parked }));
  return msg('board.count', { parts: list(parts, 'dot') });
}

/** What a pedal or side unit arrives on the board as (Q-10): engaged, unless it would have nothing to run (the VST3
 *  chain with no plugins, Mastering with no reference), when it arrives bypassed. A module brings no values. */
export function arrivalChange(ix: Index, d: Definition, desk: Desk): Change {
  const vals: Change['vals'] = {};
  if (d.kind !== 'module' && d.on) {
    const engaged: Desk = { ...desk, vals: { ...desk.vals, [d.on]: true }, on: new Set([...desk.on, d.id]) };
    vals[d.on] = !(d.kind === 'pedal' && idleOf(ix, d, engaged));
  }
  return { vals, auto: {}, state: {}, words: deskStep('add', d, desk, ix) };
}
