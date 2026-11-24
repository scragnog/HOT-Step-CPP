// Undo step names (docs/05, "Undo step names"): what a step is called, and what its Undo and Redo say, from the desk
// before and after it and the definitions alone. The concept's rules (14-value-steps.js vsWords), on the definitions'
// words. A part never names a step: it asks for one of its definition's named steps, or sets values, and this names
// what changed. Every name is a message (messages.ts), so it reads in any locale.

import type { Definition, Json, ParamDef, Step, Text, Value } from '../definitions/types.ts';
import { type Desk, type Index, has, isAuto, mmss, nameOn, resolve, valueOf } from './core.ts';
import { type Msg, type Slot, list, msg } from './messages.ts';
import { fillMsg, formatMsg, paramWords, stateWords, stepKey } from './strings.ts';

export interface StepWords {
  /** What the step is called: the Undo button's label. */
  label: Msg;
  /** "Undone: ..." and "Redone: ..." in the toasts, when they say more than the label. */
  undone?: Msg;
  redone?: Msg;
  /** What Undo will put back, said before it is pressed. */
  put: Msg;
}

/** A step a part asks for by name: one of its parameter's or its state's `steps`, with the slots the part fills. */
export interface NamedStep { key: string; step: string; slots?: { [slot: string]: string | number } }
/** A Quick choice row's words, so its step reads as the row does ("How adventurous: Even → Bold"). */
export interface RowStep { key: string; label: Slot; choices: [Slot, Value | { [k: string]: Value }][] }

/** A named step's words, and the parameters it may change besides its own (types.ts Step). */
export const stepSays = (s: Step): Text => (typeof s === 'string' ? s : s.says);
export const stepChanges = (s: Step): string[] => (typeof s === 'string' ? [] : s.changes);

const plural = (n: string) => /s$/.test(n.split(' ').pop() ?? '');

/** The definition a parameter belongs to, for naming: a pedal's or unit's name, else its module's. */
const ownerOf = (ix: Index, key: string): Definition | undefined => ix.params.get(key)?.d;

/** What a step calls a parameter: its undoName, its words on the desk, with its owner's name first when another
 *  parameter on the engine prints the same words; a pedal's or unit's switch is the pedal or unit. */
export function stepName(ix: Index, key: string, desk: Desk): Msg {
  const { p, d } = ix.params.get(key)!;
  const w = paramWords(ix, key, desk);
  if (w.undoName) return w.undoName;
  const label = w.label.en!, own = nameOn(d, desk.engine), ownMsg = defName(d, desk);
  /* a pedal's or unit's switch labelled On is the pedal or unit */
  if (d.kind !== 'module' && d.on === key && label === 'On') return ownMsg;
  /* the parameters that print the same words on this engine, among the definitions on the desk (a module, pedal or unit
     no one has added renames nothing: SL-13), as the concept counts them: those another engine has too */
  const same = [...ix.params.values()].filter(x => (x.d === d || (x.d.kind === 'module' && x.d.builtIn) || desk.on.has(x.d.id)) && (x.p.key === key ? label : resolve(x.p, desk.engine).label).toLowerCase() === label.toLowerCase()).length;
  return same > 1 && !label.toLowerCase().startsWith(own.toLowerCase()) ? msg('name.owned', { owner: ownMsg, label: w.label }) : w.label;
  void p;
}
/** A definition's name on the desk's engine, as a message. */
export const defName = (d: Definition, desk: Desk): Msg => {
  const line = d.kind === 'module' ? (d.per || []).find(l => l.engines.includes(desk.engine) && l.name) : undefined;
  return { id: `create.${d.id}.name${line ? `@${line.engines.join('+')}` : ''}`, en: nameOn(d, desk.engine) };
};

/** A value as a step shows it: "auto" on auto, else as it is written everywhere. */
function shown(ix: Index, key: string, desk: Desk): Slot {
  if (isAuto(ix, key, desk)) return 'auto';
  const { p, d } = ix.params.get(key)!;
  /* a value the engine in use does not have is said as an engine that has it says it */
  const e = has(ix, key, desk.engine) ? desk.engine : (p.engines ?? d.engines).find(x => d.engines.includes(x))!;
  const on = { ...desk, engine: e };
  return formatMsg(ix, key, on, valueOf(ix, key, on)!);
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

/** What a desk path is called in "Change ..." and "Puts back ...". */
function pathNames(ix: Index, keys: string[], state: string[], desk: Desk): Slot {
  const vs = keys.map(k => stepName(ix, k, desk)), rest = [...new Set(state)].filter(s => ix.states.has(s)).map(s => stateWords(ix, s).noun);
  const isThe = (m: Msg) => /^the /.test(m.en ?? '');
  const theVs = vs.filter(isThe), plain = vs.filter(x => !isThe(x));
  return list([...rest, ...theVs, ...(plain.length > 2 ? [msg('step.settings', { n: plain.length })] : plain)]);
}

/** The words of the step from `before` to `after`. `named`: the step a part asked for; `row`: the Quick choice row the
 *  change came from. Null when nothing changed. */
export function stepWords(ix: Index, before: Desk, after: Desk, hint?: { named?: NamedStep; row?: RowStep }): StepWords | null {
  const { keys, state } = changesOf(ix, before, after);
  if (!keys.length && !state.length) return null;
  const put = msg('step.put', { what: pathNames(ix, keys, state, before) });
  /* the seed's own step is named as every change of the seed is (below): random or the number kept */
  if (hint?.named && ix.states.get(hint.named.key)?.s.kind !== 'seed') {
    const n = hint.named, own = ix.params.get(n.key)?.p ?? ix.states.get(n.key)?.s;
    const file = fileStep(ix, n, before, after);
    if (file) return file;
    const def = own && (own as { steps?: { [k: string]: Step } }).steps?.[n.step];
    if (!def) throw new Error(`${n.key} has no step "${n.step}"`);
    const slots: { [k: string]: Slot } = { ...n.slots };
    if (ix.params.has(n.key)) slots.name = stepName(ix, n.key, after);
    /* a state's step says its value before and after, unless the part says them (a phase window's span) */
    if (ix.states.has(n.key)) { slots.noun = stateWords(ix, n.key).noun; if (!('from' in slots)) slots.from = stateText(ix, n.key, before); if (!('to' in slots)) slots.to = stateText(ix, n.key, after); }
    const label = fillMsg(ix, stepKey(ix, n.key, n.step), stepSays(def), after);
    return { label: { ...label, slots: { ...label.slots, ...slots } }, put };
  }
  /* the seed: its number while it is kept, else random */
  const seedKey = [...ix.states.values()].find(x => x.s.kind === 'seed')?.s.key;
  if (seedKey && (state.includes(seedKey) || (keys.length === 1 && !state.length && lockOf(ix, seedKey) === keys[0]))) {
    const a = seedText(ix, seedKey, before), b = seedText(ix, seedKey, after);
    if (J(a) !== J(b)) return { label: msg('step.seed', { from: a, to: b }), undone: msg('step.seed.undone', { from: a }), redone: msg('step.seed.redone', { to: b }), put: msg('step.seed.put', { from: a }) };
  }
  if (keys.length === 1 && !state.length) {
    const k = keys[0]!, p = ix.params.get(k)!.p, name = stepName(ix, k, before);
    if (p.type === 'text') {
      /* a plural is said by the parameter's own undo name (the lyrics, the tags) */
      const the = /^the /.test(name.en ?? '') ? name : msg('name.the', { name });
      const many = !!paramWords(ix, k, before).undoName && plural(name.en ?? '');
      return { label: msg('step.edit', { name: the }), put: msg('step.edit.put', { name: the, many: String(many) }) };
    }
    if (hint?.row && hint.row.key === k) {
      const word = (desk: Desk): Slot => { const v = valueOf(ix, k, desk), c = hint.row!.choices.find(x => typeof x[1] !== 'object' && (x[1] === v || (typeof v === 'number' && Math.abs((x[1] as number) - v) < 1e-9))); return c ? c[0] : shown(ix, k, desk); };
      const a = word(before), b = word(after), r = hint.row.label;
      return { label: msg('step.row', { row: r, from: a, to: b }), undone: msg('step.row.undone', { row: r, from: a }), redone: msg('step.row.redone', { row: r, to: b }), put: msg('step.row.put', { row: r, from: a }) };
    }
    const a = shown(ix, k, before), b = shown(ix, k, after);
    return { label: msg('step.set', { name, from: a, to: b }), undone: msg('step.set.undone', { name, from: a }), redone: msg('step.set.redone', { name, to: b }), put: msg('step.set.put', { name, from: a }) };
  }
  /* a Quick row whose choice set several values at once: the row's words, from choice to choice (DH-09) */
  if (hint?.row) {
    const row = hint.row, rowKeys = new Set(row.choices.flatMap(c => (typeof c[1] === 'object' ? Object.keys(c[1]) : [row.key])));
    if (keys.every(k => rowKeys.has(k)) && !state.length) {
      const a = rowWord(ix, row, before), b = rowWord(ix, row, after), r = row.label;
      return { label: msg('step.row', { row: r, from: a, to: b }), undone: msg('step.row.undone', { row: r, from: a }), redone: msg('step.row.redone', { row: r, to: b }), put: msg('step.row.put', { row: r, from: a }) };
    }
  }
  /* two numbers moved together (an XY pad): both, with their values; both back where they start is a reset */
  if (keys.length === 2 && !state.length && keys.every(k => ix.params.get(k)!.p.type === 'number')) {
    const n = keys.map(k => stepName(ix, k, before)), own = nameOn(ownerOf(ix, keys[0]!)!, before.engine);
    const ownedBoth = keys.every(k => nameOn(ownerOf(ix, k)!, before.engine) === own) && n.every(x => x.id === 'name.owned');
    if (ownedBoth) n[1] = n[1]!.slots!.label as Msg;
    const names = list(n);
    if (keys.every(k => !(k in after.vals) && !(k in after.auto))) return { label: msg('step.reset', { names }), put: msg('step.put', { what: names }) };
    const each = keys.map((k, i) => msg('step.set', { name: n[i]!, from: shown(ix, k, before), to: shown(ix, k, after) }));
    return { label: msg('step.pair', { a: each[0]!, b: each[1]! }), undone: msg('step.pair.undone', { names }), put: msg('step.put', { what: names }) };
  }
  const what = pathNames(ix, keys, state, before);
  return { label: msg('step.change', { what }), put: msg('step.put', { what }) };
}

/** The steps a file has without saying. */
export const FILE_STEPS = ['load', 'clear'];
/** A file's own steps, as the concept names them (13-files.js fileStep): loaded into an empty place, its definition's
 *  `load` words ("Load kick.wav as the timbre reference"); replacing one, from the one to the other; cleared, "Clear"
 *  and the state's noun. Null for any other step, or a load the definition names itself when the place was empty. */
function fileStep(ix: Index, n: NamedStep, before: Desk, after: Desk): StepWords | null {
  const hit = ix.states.get(n.key);
  if (!hit || hit.s.kind !== 'file' || !FILE_STEPS.includes(n.step)) return null;
  const x = String(before.state[n.key] ?? ''), y = String(after.state[n.key] ?? ''), noun = stateWords(ix, n.key).noun;
  if (!y) return { label: msg('step.file.clear', { noun }), undone: msg('step.file.back', { file: x }), redone: msg('step.file.cleared', { file: x }), put: msg('step.file.putBack', { file: x }) };
  if (x) return { label: msg('step.file.replace', { noun, from: x, to: y }), put: msg('step.file.putBack', { file: x }) };
  const def = hit.s.steps?.load;
  if (def) return null;
  return { label: msg('step.file.load', { file: y, noun }), undone: msg('step.file.out', { file: y }), redone: msg('step.file.loaded', { file: y }), put: msg('step.file.takeOut', { file: y }) };
}

/** The word a Quick row shows for the desk: the choice whose values are all in force, or none (choiceOf, tiers.ts). */
export function rowWord(ix: Index, row: RowStep, desk: Desk): Slot {
  const inForce = (v: Value | { [k: string]: Value }) => (typeof v === 'object' ? Object.entries(v).every(([k, x]) => valueOf(ix, k, desk) === x) : valueOf(ix, row.key, desk) === v);
  return row.choices.find(c => inForce(c[1]))?.[0] ?? list(row.choices.flatMap(c => (typeof c[1] === 'object' ? Object.keys(c[1]) : [])).filter((k, i, a) => a.indexOf(k) === i).map(k => shown(ix, k, desk)), 'comma');
}

/** The switch that keeps the seed: the boolean the seed's part edits beside it (Keep this seed). */
function lockOf(ix: Index, seedKey: string): string | undefined {
  const d = ix.states.get(seedKey)!.d;
  return d.params.find(p => p.type === 'boolean')?.key;
}
function seedText(ix: Index, key: string, desk: Desk): Slot {
  const lock = lockOf(ix, key), v = desk.state[key];
  return (!lock || valueOf(ix, lock, desk) === true) && (typeof v === 'number' || typeof v === 'string') ? String(v) : stateWords(ix, key).empty ?? 'random';
}
function stateText(ix: Index, key: string, desk: Desk): Slot {
  const v = desk.state[key] as Json | undefined, w = stateWords(ix, key);
  if (v === undefined || v === null || v === '' || (Array.isArray(v) && !v.length)) return w.empty ?? 'none';
  if (typeof v === 'string' || typeof v === 'number') return String(v);
  /* a span in seconds, in the song's times (the repaint region, Q-24) */
  const sp = v as { start?: unknown; end?: unknown };
  if (typeof sp === 'object' && !Array.isArray(sp) && typeof sp.start === 'number' && typeof sp.end === 'number') return `${mmss(Math.min(sp.start, sp.end))}–${mmss(Math.max(sp.start, sp.end))}`;
  return w.noun;
}

/** The parameters a named step may change: the parameter itself (or the state's part's own), and the step's `changes`. */
export function stepMayChange(p: ParamDef | undefined, def: Step): string[] {
  return [...(p ? [p.key] : []), ...stepChanges(def)];
}
