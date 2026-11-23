// An engine switch (docs/05, "Engine switches"): what each value does, what comes back, what is held, and what the
// toast says. The concept's enterEngine and engineNews (13-engine-state.js), on the definitions' owner and held.

import type { EngineId, Json, Value } from '../definitions/types.ts';
import { engineName } from '../definitions/engines.ts';
import { type Desk, type Index, applies, changed, enginesOf, everythingOn, formatValue, has, isAuto, resolve, valueOf } from './core.ts';
import { type Msg, type Slot, list, msg } from './messages.ts';
import { defName } from './steps.ts';

/** What the desk keeps for the engines not in use. */
export interface EngineBook {
  /** Each engine's own values and state, put aside when it was left. */
  stash: { [engine: string]: { vals: { [k: string]: Value }; auto: { [k: string]: boolean }; state: { [k: string]: Json } } };
  /** A song value an engine could not take: the value, the limit shown, and the engine it returns on. */
  hold: { [key: string]: { value: number; shown: number; from: EngineId } };
  /** Where each held parameter was last in use, to name the engine its value returns on. */
  lastOn: { [key: string]: EngineId };
}
export const emptyBook = (): EngineBook => ({ stash: {}, hold: {}, lastOn: {} });

const ownerOf = (ix: Index, key: string) => { const h = ix.params.get(key)!; return h.p.owner ?? h.d.owner; };
/** A model file choice: options the server lists from its model buckets. */
const isModel = (ix: Index, key: string, e: EngineId) => { const o = resolve(ix.params.get(key)!.p, e).options; return !!o && !Array.isArray(o) && o.from === 'models'; };
const plural = (s: string) => /s$/.test(s);

export interface Switched { desk: Desk; book: EngineBook; toast: Msg }

/** The desk moved from its engine to `to`, one undo step: every engine-owned value and state is put aside for the
 *  engine left and that engine's own come back (an engine reached for the first time starts on its defaults); song
 *  values stay; a held parameter reads at the new engine's limit and its value waits for an engine that can take it. */
export function switchEngine(ix: Index, deskIn: Desk, book0: EngineBook, to: EngineId): Switched {
  /* what is put aside, counted and held is about the stored values, whatever the desk shows */
  const desk0 = everythingOn(ix, deskIn), from = desk0.engine;
  const desk: Desk = { ...desk0, vals: { ...desk0.vals }, auto: { ...desk0.auto }, state: { ...desk0.state }, engine: to };
  const book: EngineBook = { stash: { ...book0.stash }, hold: { ...book0.hold }, lastOn: { ...book0.lastOn } };
  if (from === to) return { desk: deskIn, book: book0, toast: msg('switch.toast', { engine: engineName(to) }) };
  /* what stays behind: model files by name, adapters, and a count of the changed settings only the engine left uses */
  const own = [...new Set([...Object.keys(desk0.vals), ...Object.keys(desk0.auto)])].filter(k => ix.params.has(k) && ownerOf(ix, k) === 'engine');
  const mine = new Set([...own, ...[...new Set([...Object.keys(desk0.vals), ...Object.keys(desk0.auto)])].filter(k => ix.params.has(k) && has(ix, k, from) && !has(ix, k, to))]);
  const models: string[] = []; let sets = 0;
  for (const k of mine) {
    if (!applies(ix, k, desk0) || !changed(ix, k, desk0)) continue;
    if (isModel(ix, k, from)) models.push(formatValue(resolve(ix.params.get(k)!.p, from), valueOf(ix, k, desk0)!, desk0.lists)); else sets++;
  }
  const ownState = Object.keys(desk0.state).filter(k => ix.states.get(k)?.s.owner === 'engine');
  const slots = ownState.reduce((n, k) => n + (Array.isArray(desk0.state[k]) ? (desk0.state[k] as Json[]).length : 0), 0);
  const put = { vals: {} as { [k: string]: Value }, auto: {} as { [k: string]: boolean }, state: {} as { [k: string]: Json } };
  for (const k of own) { if (k in desk.vals) put.vals[k] = desk.vals[k]!; if (k in desk.auto) put.auto[k] = desk.auto[k]!; delete desk.vals[k]; delete desk.auto[k]; }
  for (const k of ownState) { put.state[k] = desk.state[k]!; delete desk.state[k]; }
  if (Object.keys(put.vals).length || Object.keys(put.auto).length || slots) book.stash[from] = put; else delete book.stash[from];
  /* where each held parameter was last in use */
  const heldKeys = [...ix.params.values()].filter(x => x.p.held).map(x => x.p.key);
  for (const k of heldKeys) book.lastOn[k] = has(ix, k, from) ? from : (book.lastOn[k] ?? from);
  const got = book.stash[to]; delete book.stash[to];
  let back = 0;
  if (got) { Object.assign(desk.vals, got.vals); Object.assign(desk.auto, got.auto); Object.assign(desk.state, got.state); back = Object.keys(got.vals).length + Object.values(got.state).reduce<number>((n, v) => n + (Array.isArray(v) ? v.length : 1), 0); }
  /* the song's limited values: a held value the new engine can take comes back; one it cannot is held */
  const held: [string, number, number, EngineId][] = [];
  for (const k of heldKeys) {
    if (!has(ix, k, to)) continue;
    const r = resolve(ix.params.get(k)!.p, to).range!;
    if (isAuto(ix, k, { ...desk, on: desk0.on })) continue;
    let h: EngineBook['hold'][string] | undefined = book.hold[k];
    if (h && desk.vals[k] !== h.shown) { delete book.hold[k]; h = undefined; } /* changed since it was held */
    const want = h ? h.value : typeof desk.vals[k] === 'number' ? desk.vals[k] as number : null;
    if (want === null) continue;
    if (want >= r[0] && want <= r[1]) { if (h) { desk.vals[k] = h.value; delete book.hold[k]; } continue; }
    const shown = Math.min(r[1], Math.max(r[0], want)), was = h ? h.from : book.lastOn[k]!;
    desk.vals[k] = shown; book.hold[k] = { value: want, shown, from: was };
    if (!h || h.shown !== shown) held.push([k, want, shown, was]);
  }
  /* a source module (Repaint) the new engine does not have leaves the desk */
  desk.on = deskIn.on;
  let removed: Msg | null = null;
  for (const d of ix.defs) if (d.kind === 'module' && d.source && desk.on.has(d.id) && !d.engines.includes(to)) { desk.on = new Set([...desk.on].filter(x => x !== d.id)); removed = msg('switch.removed', { name: defName(d, deskIn), engines: list(d.engines.map(engineName)) }); }
  /* the toast: one sentence per thing that happened, in the order they matter */
  const said: Slot[] = [];
  if (held.length) {
    const heldMsg = (k: string, field: 'toast' | 'value', slots: { [k: string]: Slot }): Msg => {
      const { p, d } = ix.params.get(k)!, text = field === 'toast' ? p.held!.toast : p.held!.value ?? '{value}';
      return { id: `create.${d.id}.p.${k.replace(/\./g, '-')}.held.${field}`, en: text, slots };
    };
    /* the first is "Length capped at 4:00", those after it say only what (the template's " capped" is the first's) */
    const capped = held.map(([k, , shown], i) => {
      const t = heldMsg(k, 'toast', { shown: formatValue(resolve(ix.params.get(k)!.p, to), shown) });
      return i ? { ...t, en: t.en!.replace(' capped', '') } : t;
    });
    const wait = new Map<EngineId, Msg[]>();
    for (const [k, v, , was] of held) {
      const p = ix.params.get(k)!.p, f = resolve(p, was) ?? resolve(p, enginesOf(p, ix.params.get(k)!.d)[0]!);
      wait.set(was, [...(wait.get(was) || []), heldMsg(k, 'value', { value: formatValue(f, v) })]);
    }
    said.push(msg('switch.capped', { capped: list(capped), waits: list([...wait].map(([e, a]) => msg('switch.waits', { values: list(a), many: String(a.length > 1 || plural(a[0]!.en!)), engine: engineName(e) })), 'comma') }));
  }
  const parts: Slot[] = [
    ...(models.length ? [msg('switch.models', { n: models.length, names: list(models, 'comma') })] : []),
    ...(slots ? [msg('switch.adapters', { n: slots })] : []),
    ...(sets ? [msg('switch.settings', { n: sets })] : []),
  ];
  if (parts.length) said.push(msg('switch.keeps', { engine: engineName(from), parts: list(parts) }));
  if (back) said.push(msg('switch.back', { engine: engineName(to) }));
  if (removed) said.push(removed);
  return { desk, book, toast: said.length ? msg('switch.toast.said', { engine: engineName(to), said: list(said, 'sentences') }) : msg('switch.toast', { engine: engineName(to) }) };
}

/** What a held parameter's control says while its value is held: its `held.note`, filled. */
export function heldNote(ix: Index, key: string, desk: Desk, book: EngineBook): Msg | null {
  const h = book.hold[key], hit = ix.params.get(key);
  if (!h || !hit?.p.held || desk.vals[key] !== h.shown) return null;
  const f = resolve(hit.p, desk.engine), from = resolve(hit.p, h.from);
  return { id: `create.${hit.d.id}.p.${key.replace(/\./g, '-')}.held.note`, en: hit.p.held.note, slots: { shown: formatValue(f, h.shown), value: formatValue(from, h.value), engine: engineName(h.from) } };
}

/** What the first switch to an engine in this browser adds to its toast (the concept's firstVisit, 13-engine-state.js):
 *  the modules only it and at most one other engine bring, none of them a source or already on the desk; past three,
 *  the first and a count. `toast` is the switch's own toast; null when there is nothing new. */
export function newOn(ix: Index, desk: Desk, to: EngineId, toast: Msg): Msg | null {
  const own = ix.defs.filter(d => d.kind === 'module' && !d.builtIn && !d.source && !d.board && d.engines.includes(to) && d.engines.length <= 2 && !desk.on.has(d.id));
  if (!own.length) return null;
  const names = own.length > 3 ? msg('switch.new.more', { first: defName(own[0]!, desk), n: own.length - 1 }) : list(own.map(d => defName(d, desk)));
  return msg('switch.new', { toast, names });
}
