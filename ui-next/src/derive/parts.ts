// What a part is given at run time (parts.ts PartProps), built from its node and the desk, and nothing else. A part
// changes the desk only through these setters, which exist for exactly what its node binds; a change comes back as a
// Change whose undo step the core has already named (steps.ts). So a part cannot write a value it is not bound to,
// keep its own copy, decide a default or name a step: none of those is in what it is given.

import type { Definition, Json, Node, StateDef, Value } from '../definitions/types.ts';
import type { BoundParam, BoundState, PartProps, ReadValue } from '../definitions/parts.ts';
import { type Desk, type Index, applies, changed, editsOf, has, isAuto, nodeAt, resolve, textOf, valueOf, wordsOf } from './core.ts';
import { liveDefs } from './request.ts';
import { setValue } from './values.ts';
import { stepChanges, stepWords, type StepWords } from './steps.ts';

/** One undo step: what it changes, and its words. The store applies it and records it. */
export interface Change {
  vals: { [key: string]: Value | null };
  auto: { [key: string]: boolean };
  state: { [key: string]: Json };
  words: StepWords;
  /** The one-time sentence a typed number out of range gets ("Steps: 1 to 64. Set to 64."). */
  said?: string;
}

/** Applies a change to a desk (null removes a value: back to its default). */
export function applyChange(desk: Desk, c: Change): Desk {
  const vals = { ...desk.vals }, auto = { ...desk.auto };
  for (const [k, v] of Object.entries(c.vals)) { if (v === null) delete vals[k]; else vals[k] = v; }
  for (const [k, v] of Object.entries(c.auto)) auto[k] = v;
  return { ...desk, vals, auto, state: { ...desk.state, ...c.state } };
}

/** The props a node's part is given on the desk: setters for the parameters its node edits and the state it names,
 *  read-only values for what it reads (parameters with their auto switch and whether they apply and are on the desk,
 *  state, desk facts), and its children. `commit` receives every change, already named. */
export function bindPart(ix: Index, d: Definition, node0: Node, desk: Desk, commit: (c: Change) => void): PartProps {
  const node = nodeAt(ix, node0, desk);
  if (!node) throw new Error(`${node0.part} is not shown here`);
  const bound = editsOf(node).filter(k => has(ix, k, desk.engine));
  const live = new Set(liveDefs(ix, desk).map(x => x.id));
  /* a batch gathers every set of one gesture into one step */
  let batch: { vals: Change['vals']; auto: Change['auto']; state: Change['state']; named?: { key: string; step: string; slots?: { [k: string]: string | number } }; said?: string } | null = null;
  const finish = (b: NonNullable<typeof batch>) => {
    const after = applyChange(desk, { ...b, words: { label: '', put: '' } });
    const words = stepWords(ix, desk, after, b.named ? { named: b.named } : undefined);
    if (words) commit({ vals: b.vals, auto: b.auto, state: b.state, words, ...(b.said ? { said: b.said } : {}) });
  };
  const change = (fn: (b: NonNullable<typeof batch>) => void) => {
    if (batch) { fn(batch); return; }
    const b = { vals: {}, auto: {}, state: {} }; fn(b); finish(b);
  };
  const stepOf = (owner: { key: string; steps?: StateDef['steps'] }, step: string) => {
    const s = owner.steps?.[step];
    if (!s) throw new Error(`${owner.key} has no step "${step}": name it in the definition's steps`);
    return s;
  };
  const allowed = (keys: string[], extra: string[], k: string) => {
    if (!keys.includes(k) && !extra.includes(k)) throw new Error(`${node.part} may not change ${k}: its node does not bind it, nor does the step it is taking`);
  };
  const params: Record<string, BoundParam> = {};
  for (const k of bound) {
    const { p } = ix.params.get(k)!;
    params[k] = {
      key: k, face: wordsOf(ix, p, desk), value: valueOf(ix, k, desk)!, text: textOf(ix, k, desk), changed: changed(ix, k, desk), auto: isAuto(ix, k, desk), applies: applies(ix, k, desk),
      set(value, step, slots, also) {
        const r = setValue(ix, desk, k, value);
        change(b => {
          b.vals[k] = r.value; if (resolve(p, desk.engine).auto) b.auto[k] = false; if (r.said) b.said = r.said;
          if (step) {
            const s = stepOf(p, step);
            b.named = { key: k, step, ...(slots ? { slots } : {}) };
            for (const [x, v] of Object.entries(also || {})) { allowed(bound, stepChanges(s), x); b.vals[x] = setValue(ix, desk, x, v).value; }
          } else if (also) throw new Error(`${node.part} changes other values beside ${k}: name the step that does`);
        });
      },
      setAuto(on) { change(b => { b.auto[k] = on; }); },
      reset() { change(b => { b.vals[k] = null; delete b.auto[k]; if (resolve(p, desk.engine).auto) b.auto[k] = true; }); },
    };
  }
  const state: Record<string, BoundState> = {};
  for (const key of node.state || []) {
    const hit = ix.states.get(key)!;
    const lends = node.part === 'dropZone' && !!(node.opts as { from?: string } | undefined)?.from;
    state[key] = {
      key, kind: hit.s.kind, value: desk.state[key] ?? null,
      apply(step, next, also, slots) {
        if (lends || hit.d !== d) throw new Error(`${node.part} shows ${key} and may not change it`);
        const s = stepOf(hit.s, step);
        change(b => {
          b.state[key] = next; b.named = { key, step, ...(slots ? { slots } : {}) };
          for (const [x, v] of Object.entries(also || {})) { allowed(bound, stepChanges(s), x); b.vals[x] = setValue(ix, desk, x, v).value; }
        });
      },
    };
  }
  const reads: Record<string, ReadValue> = {};
  for (const k of node.reads || []) {
    if (ix.params.has(k)) {
      const on = has(ix, k, desk.engine);
      reads[k] = { key: k, value: on ? valueOf(ix, k, desk)! : null, text: on ? textOf(ix, k, desk) : '', auto: on && isAuto(ix, k, desk), applies: on && applies(ix, k, desk), onDesk: live.has(ix.params.get(k)!.d.id) };
    } else if (ix.states.has(k)) reads[k] = { key: k, value: desk.state[k] ?? null, text: '', auto: false, applies: true, onDesk: live.has(ix.states.get(k)!.d.id) };
    else reads[k] = { key: k, value: desk.facts[k] ?? null, text: '', auto: false, applies: true, onDesk: true };
  }
  return {
    node, engine: desk.engine, tier: desk.tier, params, state, reads,
    batch(fn) { if (batch) { fn(); return; } batch = { vals: {}, auto: {}, state: {} }; try { fn(); } finally { const b = batch; batch = null; finish(b); } },
    run() { /* actions are run by the shell, which names their steps the same way */ },
    children: [],
  };
}
