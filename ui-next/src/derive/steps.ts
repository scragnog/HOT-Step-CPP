// Undo step names (docs/05, "Undo step names"): what a step is called, and what its Undo and Redo say, from the desk
// before and after it and the definitions alone. The concept's rules (14-value-steps.js vsWords), on the definitions'
// words. A part never names a step: it asks for one of its definition's named steps, or sets values, and this names
// what changed.

import type { Definition, Json, ParamDef, Step, Text, Value } from '../definitions/types.ts';
import { type Desk, type Index, fill, formatValue, has, isAuto, nameOn, resolve, valueOf, wordsOf } from './core.ts';

export interface StepWords {
  /** What the step is called: the Undo button's label. */
  label: string;
  /** "Undone: ..." and "Redone: ..." in the toasts, when they say more than the label. */
  undone?: string;
  redone?: string;
  /** What Undo will put back, said before it is pressed. */
  put: string;
}

/** A step a part asks for by name: one of its parameter's or its state's `steps`, with the slots the part fills. */
export interface NamedStep { key: string; step: string; slots?: { [slot: string]: string | number } }
/** A Quick choice row's words, so its step reads as the row does ("How adventurous: Even → Bold"). */
export interface RowStep { key: string; label: string; choices: [Text, Value | { [k: string]: Value }][] }

/** A named step's words, and the parameters it may change besides its own (types.ts Step). */
export const stepSays = (s: Step): Text => (typeof s === 'string' ? s : s.says);
export const stepChanges = (s: Step): string[] => (typeof s === 'string' ? [] : s.changes);

const lowFirst = (s: string) => (/^[A-Z][a-z]*(?=[\s(]|$)/.test(s) ? s[0]!.toLowerCase() + s.slice(1) : s);
const and = (a: string[]) => (a.length > 1 ? a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1] : a[0] ?? '');
const plural = (n: string) => /s$/.test(n.split(' ').pop() ?? '');

/** The definition a parameter belongs to, for naming: a pedal's or unit's name, else its module's. */
const ownerOf = (ix: Index, key: string): Definition | undefined => ix.params.get(key)?.d;

/** What a step calls a parameter: its undoName, its words on the desk, with its owner's name first when another
 *  parameter on the engine prints the same words; a pedal's or unit's switch is the pedal or unit. */
export function stepName(ix: Index, key: string, desk: Desk): string {
  const { p, d } = ix.params.get(key)!;
  const w = wordsOf(ix, p, desk);
  if (w.undoName) return fill(ix, w.undoName, desk, w);
  const label = w.label, own = nameOn(d, desk.engine);
  /* a pedal's or unit's switch labelled On is the pedal or unit */
  if (d.kind !== 'module' && d.on === key && label === 'On') return own;
  /* every parameter's words on this engine, as the concept counts them: those another engine has too */
  const same = [...ix.params.values()].filter(x => (x.p.key === key ? label : resolve(x.p, desk.engine).label).toLowerCase() === label.toLowerCase()).length;
  return same > 1 && !label.toLowerCase().startsWith(own.toLowerCase()) ? `${own} ${lowFirst(label)}` : label;
}

/** A value as a step shows it: "auto" on auto, else as it is written everywhere. */
function shown(ix: Index, key: string, desk: Desk): string {
  if (isAuto(ix, key, desk)) return 'auto';
  const { p, d } = ix.params.get(key)!;
  /* a value the engine in use does not have is said as an engine that has it says it */
  const e = has(ix, key, desk.engine) ? desk.engine : (p.engines ?? d.engines).find(x => d.engines.includes(x))!;
  const on = { ...desk, engine: e };
  return formatValue(wordsOf(ix, p, on), valueOf(ix, key, on)!, desk.lists);
}

const J = (x: unknown) => JSON.stringify(x);
/** The parameters and state a step changed: a value, an auto switch, or a state. */
export function changesOf(ix: Index, before: Desk, after: Desk): { keys: string[]; state: string[] } {
  const keys = new Set<string>();
  for (const k of new Set([...Object.keys(before.vals), ...Object.keys(after.vals)])) if (J(before.vals[k]) !== J(after.vals[k]) && ix.params.has(k)) keys.add(k);
  for (const k of new Set([...Object.keys(before.auto), ...Object.keys(after.auto)])) if ((before.auto[k] ?? true) !== (after.auto[k] ?? true) && ix.params.has(k)) keys.add(k);
  const state = [...new Set([...Object.keys(before.state), ...Object.keys(after.state)])].filter(k => J(before.state[k] ?? null) !== J(after.state[k] ?? null));
  return { keys: [...keys], state };
}

function stateNoun(ix: Index, key: string): string { return ix.states.get(key)?.s.noun ?? key; }
/** What a desk path is called in "Change ..." and "Puts back ...". */
function pathNames(ix: Index, keys: string[], state: string[], desk: Desk): string {
  const vs = keys.map(k => stepName(ix, k, desk)), rest = [...new Set(state.map(s => stateNoun(ix, s)))];
  const theVs = vs.filter(x => x.startsWith('the ')), plain = vs.filter(x => !x.startsWith('the '));
  return and([...rest, ...theVs, ...(plain.length > 2 ? [`${plain.length} settings`] : plain)]);
}

/** The words of the step from `before` to `after`. `named`: the step a part asked for; `row`: the Quick choice row the
 *  change came from. Null when nothing changed. */
export function stepWords(ix: Index, before: Desk, after: Desk, hint?: { named?: NamedStep; row?: RowStep }): StepWords | null {
  const { keys, state } = changesOf(ix, before, after);
  if (!keys.length && !state.length) return null;
  if (hint?.named) {
    const n = hint.named, own = ix.params.get(n.key)?.p ?? ix.states.get(n.key)?.s;
    const def = own && (own as { steps?: { [k: string]: Step } }).steps?.[n.step];
    if (!def) throw new Error(`${n.key} has no step "${n.step}"`);
    const slots: { [k: string]: string } = { ...Object.fromEntries(Object.entries(n.slots || {}).map(([k, v]) => [k, String(v)])) };
    if (ix.params.has(n.key)) slots.name = stepName(ix, n.key, after);
    if (ix.states.has(n.key)) { const s = ix.states.get(n.key)!.s; slots.noun = s.noun; slots.from = stateText(ix, n.key, before); slots.to = stateText(ix, n.key, after); }
    const label = fill(ix, stepSays(def), after).replace(/\{(\w+)\}/g, (m, k: string) => slots[k] ?? m);
    return { label, put: `Puts back ${pathNames(ix, keys, state, before)}` };
  }
  /* the seed: its number while it is kept, else random */
  const seedKey = [...ix.states.values()].find(x => x.s.kind === 'seed')?.s.key;
  if (seedKey && (state.includes(seedKey) || (keys.length === 1 && !state.length && lockOf(ix, seedKey) === keys[0]))) {
    const a = seedText(ix, seedKey, before), b = seedText(ix, seedKey, after);
    if (a !== b) return { label: `Seed ${a} → ${b}`, undone: `Seed back to ${a}`, redone: `Seed ${b}`, put: `Puts the seed back to ${a}` };
  }
  if (keys.length === 1 && !state.length) {
    const k = keys[0]!, p = ix.params.get(k)!.p, name = stepName(ix, k, before);
    if (p.type === 'text') {
      /* a plural is said by the parameter's own undo name (the lyrics, the tags) */
      const n = /^the /.test(name) ? name : 'the ' + lowFirst(name), many = !!wordsOf(ix, p, before).undoName && plural(n);
      return { label: `Edit ${n}`, put: `Puts back ${n} as ${many ? 'they were' : 'it was'}` };
    }
    if (hint?.row && hint.row.key === k) {
      const word = (desk: Desk) => { const v = valueOf(ix, k, desk), c = hint.row!.choices.find(x => typeof x[1] !== 'object' && (x[1] === v || (typeof v === 'number' && Math.abs((x[1] as number) - v) < 1e-9))); return c ? c[0] : shown(ix, k, desk); };
      const a = word(before), b = word(after), r = hint.row.label;
      return { label: `${r}: ${a} → ${b}`, undone: `${r} back to ${a}`, redone: `${r}: ${b}`, put: `Puts ${r} back to ${a}` };
    }
    const a = shown(ix, k, before), b = shown(ix, k, after);
    return { label: `${name} ${a} → ${b}`, undone: `${name} back to ${a}`, redone: `${name} ${b}`, put: `Puts ${name} back to ${a}` };
  }
  /* two numbers moved together (an XY pad): both, with their values; both back where they start is a reset */
  if (keys.length === 2 && !state.length && keys.every(k => ix.params.get(k)!.p.type === 'number')) {
    const n = keys.map(k => stepName(ix, k, before)), own = nameOn(ownerOf(ix, keys[0]!)!, before.engine);
    if (keys.every(k => nameOn(ownerOf(ix, k)!, before.engine) === own) && n.every(x => x.startsWith(own + ' '))) n[1] = n[1]!.slice(own.length + 1);
    if (keys.every(k => !(k in after.vals) && !(k in after.auto))) return { label: `Reset ${and(n)}`, put: `Puts back ${and(n)}` };
    return { label: keys.map((k, i) => `${n[i]} ${shown(ix, k, before)} → ${shown(ix, k, after)}`).join(', '), undone: `${and(n)} put back`, put: `Puts back ${and(n)}` };
  }
  const what = pathNames(ix, keys, state, before);
  return { label: `Change ${what}`, put: `Puts back ${what}` };
}

/** The switch that keeps the seed: the boolean the seed's part edits beside it (Keep this seed). */
function lockOf(ix: Index, seedKey: string): string | undefined {
  const d = ix.states.get(seedKey)!.d;
  return d.params.find(p => p.type === 'boolean')?.key;
}
function seedText(ix: Index, key: string, desk: Desk): string {
  const lock = lockOf(ix, key), v = desk.state[key];
  return (!lock || valueOf(ix, lock, desk) === true) && typeof v === 'number' ? String(v) : (ix.states.get(key)!.s.empty ?? 'random');
}
function stateText(ix: Index, key: string, desk: Desk): string {
  const s = ix.states.get(key)!.s, v = desk.state[key] as Json | undefined;
  if (v === undefined || v === null || v === '' || (Array.isArray(v) && !v.length)) return s.empty ?? 'none';
  if (typeof v === 'string' || typeof v === 'number') return String(v);
  return s.noun;
}

/** The parameters a named step may change: the parameter itself (or the state's part's own), and the step's `changes`. */
export function stepMayChange(p: ParamDef | undefined, def: Step): string[] {
  return [...(p ? [p.key] : []), ...stepChanges(def)];
}
