// What a part is given at run time (parts.ts PartProps), built from its node and the desk, and nothing else. A part
// changes the desk only through these setters, which exist for exactly what its node binds; a change comes back as a
// Change whose undo step the core has already named (steps.ts). So a part cannot write a value it is not bound to,
// keep its own copy, decide a default or name a step: none of those is in what it is given.

import type { Definition, Json, Node, StateDef, Value } from '../definitions/types.ts';
import type { BoundParam, BoundState, PartProps, ReadValue } from '../definitions/parts.ts';
import { type Desk, type Index, applies, changed, editsOf, factOf, has, isAuto, nodeAt, resolve, stateOf, valueOf, wordsOf } from './core.ts';
import { liveDefs } from './request.ts';
import { cleanState, setValue } from './values.ts';
import { type Msg, msg } from './messages.ts';
import { checkText, paramWords, textMsg } from './strings.ts';
import { stepChanges, stepWords, type StepWords } from './steps.ts';
import { fractionValue, keyStep, parseTyped } from './models.ts';
import { autoEstimate, rulerSnap, scrubValue } from './controls.ts';
import { FILE_STEPS } from './steps.ts';

/** One undo step: what it changes, and its words. The store applies it and records it. */
export interface Change {
  vals: { [key: string]: Value | null };
  auto: { [key: string]: boolean };
  state: { [key: string]: Json };
  words: StepWords;
  /** The one-time sentence a typed number out of range gets ("Steps: 1 to 64. Set to 64."). */
  said?: Msg;
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
 *  state, desk facts), and its children. `commit` receives every change, already named.
 *
 *  `desk` is the desk the values are read from; pass a function returning the desk in force and every setter cleans,
 *  names and checks its change against the desk as it is when the setter is called, not as it was when the part was
 *  drawn: a setter held across other changes (a file that loads, a list that arrives, a run that takes seconds) names
 *  its step from the value it really changes, and a change to what is no longer bound (another engine, the module off
 *  the desk) is dropped. */
export function bindPart(ix: Index, d: Definition, node0: Node, deskIn: Desk | (() => Desk), commit: (c: Change) => void, tell?: (m: Msg) => void): PartProps {
  const now = typeof deskIn === 'function' ? deskIn : () => deskIn, desk = now();
  const node = nodeAt(ix, node0, desk);
  if (!node) throw new Error(`${node0.part} is not shown here`);
  const bound = editsOf(node).filter(k => has(ix, k, desk.engine));
  const live = new Set(liveDefs(ix, desk).map(x => x.id));
  /* a batch gathers every set of one gesture into one step */
  let batch: { at: Desk; vals: Change['vals']; auto: Change['auto']; state: Change['state']; named?: { key: string; step: string; slots?: { [k: string]: string | number } }; said?: Msg } | null = null;
  const finish = (b: NonNullable<typeof batch>) => {
    const after = applyChange(b.at, { ...b, words: { label: msg(''), put: msg('') } });
    const words = stepWords(ix, b.at, after, b.named ? { named: b.named } : undefined);
    if (words) commit({ vals: b.vals, auto: b.auto, state: b.state, words, ...(b.said ? { said: b.said } : {}) });
    /* a typed value already at the end it was put at changes nothing, and is still said (21-knob.js) */
    else if (b.said) tell?.(b.said);
  };
  /* what the node binds is still there: its definition on the desk and the key on the engine in force */
  const bindable = (at: Desk, k: string) => {
    if (typeof deskIn !== 'function') return true;
    const owner = ix.params.get(k)?.d ?? ix.states.get(k)?.d;
    return !!owner && ((owner.kind === 'module' && !!owner.builtIn) || at.on.has(owner.id)) && (!ix.params.has(k) || has(ix, k, at.engine));
  };
  const change = (k: string, fn: (b: NonNullable<typeof batch>, at: Desk) => void) => {
    if (batch) { if (bindable(batch.at, k)) fn(batch, batch.at); return; }
    const at = now(); if (!bindable(at, k)) return;
    const b = { at, vals: {}, auto: {}, state: {} }; fn(b, at); finish(b);
  };
  const stepOf = (owner: { key: string; kind?: StateDef['kind']; steps?: StateDef['steps'] }, step: string) => {
    /* a file loads and clears without saying (docs/05, "Undo step names"): steps.ts names those */
    const s = owner.steps?.[step] ?? (owner.kind === 'file' && FILE_STEPS.includes(step) ? '' : undefined);
    if (s === undefined) throw new Error(`${owner.key} has no step "${step}": name it in the definition's steps`);
    return s;
  };
  const allowed = (keys: string[], extra: string[], k: string) => {
    if (!keys.includes(k) && !extra.includes(k)) throw new Error(`${node.part} may not change ${k}: its node does not bind it, nor does the step it is taking`);
  };
  /* one parameter set: cleaned on the engine in force; a typed value outside the range is said; a named step may set
     what it lists */
  const setOne = (k: string, value: Value, typed: boolean, step?: string, slots?: { [s: string]: string | number }, also?: { [x: string]: Value }) => {
    const { p } = ix.params.get(k)!;
    if (!step && also) throw new Error(`${node.part} changes other values beside ${k}: name the step that does`);
    change(k, (b, at) => {
      const r = setValue(ix, at, k, value, typed);
      b.vals[k] = r.value; if (resolve(p, at.engine).auto) b.auto[k] = false; if (r.said) b.said = r.said;
      if (step) {
        const s = stepOf(p, step);
        b.named = { key: k, step, ...(slots ? { slots } : {}) };
        for (const [x, v] of Object.entries(also || {})) { allowed(bound, stepChanges(s), x); b.vals[x] = setValue(ix, at, x, v).value; }
      }
    });
  };
  const params: Record<string, BoundParam> = {};
  for (const k of bound) {
    const { p } = ix.params.get(k)!;
    params[k] = {
      key: k, face: wordsOf(ix, p, desk), value: valueOf(ix, k, desk)!, text: textMsg(ix, k, desk), changed: changed(ix, k, desk), auto: isAuto(ix, k, desk), applies: applies(ix, k, desk),
      set(value, step, slots, also) {
        /* text that fails its check is refused, and said with what is kept (12-values.js): a field that keeps it keeps
           its own draft and never sets it */
        if (p.type === 'text' && typeof value === 'string') { const at = now(), bad = bindable(at, k) ? checkText(ix, k, at, value.trim()) : null; if (bad) { const v = valueOf(ix, k, at); tell?.(v === '' || v === undefined ? msg('field.refused.empty', { err: bad }) : msg('field.refused', { err: bad, v: String(v) })); return; } }
        setOne(k, value, false, step, slots, also);
      },
      press(key, shift) { const at = now(); if (!bindable(at, k)) return false; const to = keyStep(ix, k, at, key, shift); if (to === null) return false; setOne(k, to, false); return true; },
      typed(text) {
        const at = now(); if (!bindable(at, k)) return;
        /* an empty box (or "auto") puts a parameter with auto back on auto (21-numbox.js commitNum) */
        if (resolve(p, at.engine).auto && /^(auto)?$/i.test(text.trim())) { change(k, b => { b.auto[k] = true; }); return; }
        const r = parseTyped(ix, k, at, text);
        if ('error' in r) tell?.(resolve(p, at.engine).auto ? msg('box.typeNumberOrAuto', { label: paramWords(ix, k, at).label }) : r.error); else setOne(k, r.value, true);
      },
      toFraction(f) { const at = now(); if (!bindable(at, k)) return; const v = fractionValue(ix, k, at, f); if (v !== null) setOne(k, v, false); },
      scrub(from, px, shift) { const at = now(); if (!bindable(at, k)) return; const v = scrubValue(ix, k, at, from, px, shift); if (v !== null) setOne(k, v, false); },
      along(f, span, px) { const at = now(); if (!bindable(at, k)) return; setOne(k, rulerSnap(ix, k, at, f, span, px) ?? Math.min(1, Math.max(0, f)) * span, false); },
      /* turning auto off starts the value where auto was (S-17): "up to 4:00" leaves 4:00, not the default */
      setAuto(on) { change(k, (b, at) => { b.auto[k] = on; const e = on || k in at.vals ? null : autoEstimate(ix, k, at); if (e !== null && e !== resolve(p, at.engine).default) b.vals[k] = setValue(ix, at, k, e).value; }); },
      reset() { change(k, (b, at) => { b.vals[k] = null; delete b.auto[k]; if (resolve(p, at.engine).auto) b.auto[k] = true; }); },
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
        /* state is cleaned where it enters, as a value is: what its kind cannot hold is refused */
        const clean = cleanState(hit.s, next);
        if (clean === undefined) throw new Error(`${JSON.stringify(next).slice(0, 80)} is not a ${hit.s.kind} for ${key}`);
        change(key, (b, at) => {
          b.state[key] = clean; b.named = { key, step, ...(slots ? { slots } : {}) };
          for (const [x, v] of Object.entries(also || {})) { allowed(bound, stepChanges(s), x); b.vals[x] = setValue(ix, at, x, v).value; }
        });
      },
    };
  }
  const reads: Record<string, ReadValue> = {};
  for (const k of node.reads || []) {
    if (ix.params.has(k)) {
      const on = has(ix, k, desk.engine);
      reads[k] = { key: k, value: on ? valueOf(ix, k, desk)! : null, text: on ? textMsg(ix, k, desk) : '', auto: on && isAuto(ix, k, desk), applies: on && applies(ix, k, desk), onDesk: live.has(ix.params.get(k)!.d.id) };
    } else if (ix.states.has(k)) reads[k] = { key: k, value: stateOf(ix, k, desk) ?? null, text: '', auto: false, applies: true, onDesk: live.has(ix.states.get(k)!.d.id) };
    else reads[k] = { key: k, value: factOf(ix, k, desk) ?? null, text: '', auto: false, applies: true, onDesk: true };
  }
  return {
    node, engine: desk.engine, tier: desk.tier, params, state, reads,
    batch(fn) { if (batch) { fn(); return; } batch = { at: now(), vals: {}, auto: {}, state: {} }; try { fn(); } finally { const b = batch; batch = null; finish(b); } },
    run() { /* actions are run by the shell, which names their steps the same way */ },
    children: [],
  };
}
