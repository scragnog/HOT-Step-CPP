// A node bound on the desk: its model, worked out by the core and redrawn only when it changes (useModel), and the
// actions that change exactly what it binds. Every action reads the desk as it is when it is called (bindPart with the
// store's getter), so a part that acts later is cleaned and named against the desk in force; what no longer binds is
// dropped. A part's changes carry its node's name, so a gesture of one part never swallows another's.

import { useMemo } from 'react';
import type { Action, Definition, Node } from '../../definitions/types.ts';
import { actionWords, applyChoice, bindPart, checkText, fileRefused, partAct, partModel, runAction, stripOf, type Change, type CellWords, type Msg, type PartModel, type Surface } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk, type Entry } from '../store/desk.ts';
import { runNamed } from '../register/actions.ts';
import { setTierFromView } from '../shell/tier.ts';
import type { ParamView, PartView } from '../kit/types.ts';
import { useModel } from './useModel.ts';

const desk = () => useDesk.getState().desk;

/** What running an action did, applied: a change is committed, a tier is gone to and rung, a module is added, a
 *  named action is run by the feature that builds it. */
export function dispatch(d: Definition, a: Action, node?: Node, from?: string, arg?: string): void {
  const did = runAction(ix, d, a, desk(), node);
  if ('change' in did) useDesk.getState().commit(did.change, from);
  else if ('go' in did) setTierFromView(did.go, { ring: did.ring });
  else if ('add' in did) useDesk.getState().add(did.add);
  else runNamed(did.call, arg, from);
}

/** Runs action i of the strip a module's tile shows now (its definition's strip that holds). */
export function runStrip(d: Definition, i: number): void {
  const a = stripOf(ix, d, desk())?.actions?.[i];
  if (a) dispatch(d, a);
}
/** Sets a pedal's or a side unit's switch (its foot switch, its power) as any control sets it: one step, named. */
export function setSwitch(d: Definition, key: string, on: boolean): void {
  if (ix.params.get(key)?.d === d) useDesk.getState().setValue(key, on);
}

/** A node's actions: stable for the node, each acting on the desk in force. */
function actionsOf(def: Definition, node: Node, key: string) {
  const from = `${def.id}:${key || (node.param ?? '')}`;
  /* the step this node's own setters took last: what a hand's toast offers to undo (RA-09) */
  let took: Entry | null = null;
  const commit = (c: Change) => { took = useDesk.getState().commit(c, from); };
  const tell = (m: Msg) => useDesk.getState().showToast({ text: m });
  let open: ReturnType<typeof bindPart> | null = null;
  /* bound now: the node as the desk shows it now; null when it is not shown any more */
  const bound = () => { if (open) return open; try { return bindPart(ix, def, node, desk, commit, tell); } catch { return null; } };
  const param = (k: string) => bound()?.params[k];
  return {
    from,
    set: (k: string, v: string | number | boolean) => param(k)?.set(v),
    reset: (k: string) => param(k)?.reset(),
    setAuto: (k: string, on: boolean) => param(k)?.setAuto(on),
    press: (k: string, key: string, shift: boolean) => param(k)?.press(key, shift) ?? false,
    typed: (k: string, t: string) => param(k)?.typed(t),
    toFraction: (k: string, f: number) => param(k)?.toFraction(f),
    scrub: (k: string, from: number, px: number, shift: boolean) => param(k)?.scrub(from, px, shift),
    along: (k: string, f: number, span: number, px: number) => param(k)?.along(f, span, px),
    check: (k: string, t: string) => checkText(ix, k, desk(), t),
    choose: (i: number) => { try { commit(applyChoice(ix, def, node, desk(), key, i)); } catch { /* the row is not shown any more */ } },
    run: (i: number) => { const a = actionWords(ix, node, desk(), key)[i]?.action; if (a) dispatch(def, a, node, from); },
    apply: (k: string, step: string, next: unknown, slots?: { [slot: string]: string | number }) => bound()?.state[k]?.apply(step, next as never, undefined, slots),
    /* a file whose state says its load (`tells`) is said with the step's words and Undo (E-23: a recording to continue
       from); any other loads without a toast */
    load: (k: string, name: string, type: string) => {
      const no = fileRefused(name, type); if (no) { tell(no); return; }
      took = null; bound()?.state[k]?.apply('load', name, undefined, { file: name });
      const e = took as Entry | null;
      if (e && ix.states.get(k)?.s.tells?.includes('load')) useDesk.getState().showToast({ text: e.words.label, action: { label: { id: 'ui.undo' }, run: () => useDesk.getState().runUndo(e) } });
    },
    batch: (fn: () => void) => { const b = bound(); if (!b) return; open = b; try { b.batch(fn); } finally { open = null; } },
    /* a bespoke part's hand (src/derive/instruments.ts): what it says may offer to undo the step it just took. A hand
       that needs the shell (a step that brings a module, a question answered from the toast, the live region, focus in
       another part) is a named action its node lists: the feature that builds it runs it, given what the hand chose and
       this part, whose gesture its change joins (the prompt card's and the lyric sheet's tools) */
    act: (name: string, arg?: unknown) => {
      const named = actionWords(ix, node, desk(), key).find(w => w.action.do === 'bespoke' && w.action.name === name)?.action;
      if (named) { dispatch(def, named, node, from, arg === undefined ? undefined : String(arg)); return true; }
      const b = bound(); if (!b) return false;
      took = null;
      return partAct(ix, b.node, b, desk(), name, arg, (m, undo) => {
        /* said in the live region only, as the concept's say() (package 3c) */
        if (undo === 'say') { useDesk.getState().say(m); return; }
        const e = undo ? took : null, s = useDesk.getState();
        s.showToast({ text: m, ...(e ? { action: { label: { id: 'ui.undo' }, run: () => useDesk.getState().runUndo(e) } } : {}) });
      });
    },
    gesture: { begin: (sub = '') => useDesk.getState().beginGesture(`${from}|${sub}`), end: (sub = '') => useDesk.getState().endGesture(`${from}|${sub}`) },
  };
}

/** The part's props for a node at a place, or null when the core does not draw it. */
export function usePartView(def: Definition, node: Node, key: string, surface: Surface, cell?: CellWords): PartView | null {
  const ck = cell ? JSON.stringify(cell) : '';
  const model = useModel(s => partModel(ix, def, node, s.desk, key, surface, cell), `${def.id}|${key}|${node.param ?? ''}|${surface}|${ck}`);
  const acts = useMemo(() => actionsOf(def, node, key), [def, node, key]);
  return useMemo(() => (model ? assemble(model, acts) : null), [model, acts]);
}

function assemble(m: PartModel, a: ReturnType<typeof actionsOf>): PartView {
  const params: ParamView[] = m.params.map(p => ({
    ...p,
    set: v => a.set(p.key, v), reset: () => a.reset(p.key), setAuto: on => a.setAuto(p.key, on),
    press: (k, shift) => a.press(p.key, k, shift), typed: t => a.typed(p.key, t), toFraction: f => a.toFraction(p.key, f), check: t => a.check(p.key, t),
    scrub: (from, px, shift) => a.scrub(p.key, from, px, shift), along: (f, span, px) => a.along(p.key, f, span, px),
  }));
  const state = (m.state ?? []).map(st => ({ ...st, apply: (step: string, next: unknown, slots?: { [slot: string]: string | number }) => a.apply(st.key, step, next, slots), load: (name: string, type: string) => a.load(st.key, name, type) }));
  return { ...m, params, state, choose: a.choose, run: a.run, batch: a.batch, act: a.act, gesture: a.gesture };
}
