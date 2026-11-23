// The test-only compatibility layer: what the concept's two tools (tools/behaviour, tools/visual-baseline) reach into
// the concept page for, provided on top of the store, so they can drive the rebuild as they drive the concept. Only
// the test build (vite build --mode test) contains it; main.tsx imports it there and nowhere else.
//
// This file is the layer's core: the state object `S` (backend, tier, mods, pedals, vals, and the rest of a scene),
// `render()` to apply a scene, `get`, `setVal`, `toast`, the catalogue (`MODS`, `PEDALS`, `EXTRAS`, `PED`, `BK`), the
// undo history (`UNDO`, `REDO`, each step's words for the suite's naming check) and `vsStray`, which detects a change
// to the desk that is not an undo step. A scene the scenario sets through `S` is applied as it is, as the concept's
// render() takes it: not an undo step. Its values are cleaned as a saved record's are (docs/05, "Cleaning"), so a scene
// can show only what the page itself could hold.
//
// What a work package's scenarios need beyond this (the concept's addItem, switchEngine, openScore, S.slots...) goes in
// a file of its own in testing/compat/ exporting `install()`, which registers what it adds (`expose`, `onMirror`,
// `onScene`); no package edits this file. testing/compat/slice.ts is the slice's.
//
// The concept names engines and choice values its own way (backend "mm3", a choice by its label "Crossover"); this
// layer translates both ways, by the definitions' labels, so the tools' scenes mean what they meant on the concept.

import { flushSync } from 'react-dom';
import type { EngineId, Node, Value } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';
import { bindPart, cleanValues, has, resolve, valueOf } from '../../derive.ts';
import { usePanelStore } from '../store/panels.ts';
import { BOARD, BUILT_IN, OFFERED, PEDALS as PEDAL_DEFS, UNITS, boardOrder, defOf, ix } from '../model/index.ts';
import { useDesk, type DeskStore } from '../store/desk.ts';
import { sayNow } from '../i18n/index.ts';
import { CONCEPT_ID } from '../shell/engines.ts';

export const TO_ENGINE: Record<string, EngineId> = Object.fromEntries(ENGINES.map(e => [CONCEPT_ID[e.id], e.id]));

const optionsOn = (key: string, e: EngineId) => {
  const hit = ix.params.get(key);
  if (!hit || hit.p.type !== 'choice') return null;
  const f = resolve(hit.p, (hit.p.engines ?? hit.d.engines).includes(e) ? e : (hit.p.engines ?? hit.d.engines)[0]!);
  return Array.isArray(f.options) ? f.options : f.options?.include ?? [];
};
/** A concept value (a choice by its label) as the definitions store it, and back. */
export function fromConcept(key: string, v: unknown, e: EngineId): Value | undefined {
  const opts = optionsOn(key, e);
  if (!opts) return v as Value;
  const o = opts.find(x => x.label === v || x.short === v || String(x.value).toLowerCase() === String(v).toLowerCase());
  return o ? o.value : (v as Value);
}
export function toConcept(key: string, v: Value, e: EngineId): unknown {
  const opts = optionsOn(key, e);
  return opts ? opts.find(x => x.value === v)?.label ?? v : v;
}

export interface Scene { backend: string; tier: string; mods: string[]; pedals: string[]; vals: Record<string, unknown>; [k: string]: unknown }

/* ---------------------------------------------------------------- registration, for testing/compat/*.ts */
const mirrors: ((S: Scene, s: DeskStore) => void)[] = [];
const scenes: ((S: Scene) => void)[] = [];
/** Puts names on window for the tools (a concept global a package's scenarios call). */
export const expose = (o: Record<string, unknown>): void => { Object.defineProperties(window, Object.getOwnPropertyDescriptors(o)); };
/** Adds to S whatever a package keeps in its own shape (the concept's S.eng, S.slots...), each time the store changes. */
export const onMirror = (fn: (S: Scene, s: DeskStore) => void): void => { mirrors.push(fn); fn(S, useDesk.getState()); };
/** Applies a package's part of a scene, inside render(). */
export const onScene = (fn: (S: Scene) => void): void => { scenes.push(fn); };

/** The concept's S, kept in step with the store: read it, change it, then render(). */
export const S: Scene = { backend: 'ace', tier: 'quick', mods: [], pedals: [], vals: {} };
function mirror(): void {
  const s = useDesk.getState(), e = s.desk.engine;
  S.backend = CONCEPT_ID[e]; S.tier = s.desk.tier;
  S.mods = s.mods.slice(); S.pedals = s.board.slice();
  const vals: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(s.desk.vals)) vals[k] = toConcept(k, v, e);
  for (const [k, v] of Object.entries(s.desk.auto)) vals[k + '.auto'] = v;
  S.vals = vals;
  const pn = usePanelStore.getState();
  S.left = pn.left; S.right = pn.right; S.rOpen = pn.rOpen; S.lOpen = pn.lOpen;
  for (const m of mirrors) m(S, s);
}
mirror();
useDesk.subscribe(mirror);
usePanelStore.subscribe(mirror);

/** True while a scene is applied: a scene is not a person's change, so it is never a stray. */
let inScene = false;
/** Runs fn as a scene (render(), or a package's scene helper). */
export const asScene = (fn: () => void): void => { inScene = true; try { flushSync(fn); } finally { inScene = false; } };

/** Applies the scene in S to the store, as the concept's render() takes it: not as a step, and cleaned as a record is. */
function render(): void {
  const s = useDesk.getState(), engine = TO_ENGINE[S.backend] ?? 'ace', tier = (['quick', 'studio', 'lab'].includes(S.tier) ? S.tier : 'quick') as 'quick';
  const vals: Record<string, Value> = {}, auto: Record<string, boolean> = {};
  for (const [k, v] of Object.entries(S.vals ?? {})) {
    if (k.endsWith('.auto')) { if (typeof v === 'boolean') auto[k.slice(0, -5)] = v; continue; }
    if (!ix.params.has(k)) continue;
    const x = fromConcept(k, v, engine);
    if (x !== undefined) vals[k] = x;
  }
  const mods = (S.mods ?? []).filter(id => { const d = defOf(id); return d?.kind === 'module' && !d.builtIn; });
  let board = boardOrder((S.pedals ?? []).filter(id => defOf(id) && defOf(id)!.kind !== 'module'));
  /* the concept's renderDesk: a board with pedals brings the Pedalboard; the Pedalboard with none leaves */
  if (board.length && !mods.includes(BOARD.id)) mods.push(BOARD.id);
  if (!mods.includes(BOARD.id)) board = [];
  asScene(() => {
    const clean = cleanValues(ix, vals, auto);
    useDesk.setState({ desk: { ...s.desk, engine, tier, vals: clean.vals, auto: clean.auto, on: new Set([...BUILT_IN, ...mods, ...board]) }, mods, board });
    usePanelStore.setState({ left: S.left !== false, right: S.right !== false, rOpen: S.rOpen === true, lOpen: S.lOpen === true });
    for (const f of scenes) f(S);
  });
}

/** A value in force, as the concept says it (a choice by its label). */
function get(key: string): unknown {
  const s = useDesk.getState(), v = valueOf(ix, key, s.desk);
  return v === undefined ? undefined : toConcept(key, v, s.desk.engine);
}

/** A toast, as the concept's toast(message, { label, fn }) shows one. */
function toast(m: string, act?: { label: string; fn: () => void }): void {
  flushSync(() => useDesk.getState().showToast({ text: m, ...(act ? { action: { label: act.label, run: act.fn } } : {}) }));
}

/** Sets a value as a control would: one step, named by the core. */
function setVal(key: string, v: unknown): void {
  const s = useDesk.getState(), x = fromConcept(key, v, s.desk.engine), hit = ix.params.get(key);
  if (x === undefined || !hit || !has(ix, key, s.desk.engine)) return;
  const bound = bindPart(ix, hit.d, { part: 'knob', param: key } as Node, () => useDesk.getState().desk, c => useDesk.getState().commit(c)).params[key];
  flushSync(() => bound?.set(x));
}

/** The catalogue as the concept lists it: modules by id (with `only`, `lab`, `source`), pedals, side units. */
const MODS: Record<string, { name: string; only?: string[]; lab?: boolean; source?: boolean }> = {};
for (const d of [...OFFERED.filter(x => x.kind === 'module'), BOARD]) {
  const all = d.engines.length === ENGINES.length;
  MODS[d.id] = { name: d.name, ...(all ? {} : { only: d.engines.map(e => CONCEPT_ID[e]) }), ...(d.kind === 'module' && d.labUnder ? { lab: true } : {}) };
}
const asItem = (d: { id: string; name: string; engines: EngineId[] }) => ({ id: d.id, name: d.name, ...(d.engines.length === ENGINES.length ? {} : { only: d.engines.map(e => CONCEPT_ID[e]) }) });

/* every undo step's words, for the behaviour suite's check that each is named in the page's words */
let seen = 0;
useDesk.subscribe(s => {
  const bh = (window as unknown as { __bh?: { labels?: string[] } }).__bh;
  if (s.undo.length > seen && bh) { (bh.labels ??= []).push(...s.undo.slice(seen).map(e => sayNow(e.words.label))); }
  seen = s.undo.length;
});

/* ---------------------------------------------------------------- vsStray: a change to the desk that is no step
   The concept's page writes down any change it finds between gestures that is not one it made as an undo step; the
   suite's closing check reads the list. Here: within one task, a change to what the desk holds (its values, auto
   switches, state, modules, board or engine; not the tier, which is never a step) with no change to the undo or redo
   history, outside a scene, is a stray. */
const vsStray: string[] = [];
const pathsOf = (s: DeskStore) => JSON.stringify([s.desk.vals, s.desk.auto, s.desk.state, s.mods, s.board, s.desk.engine]);
let taskStart: { paths: string; undo: unknown; redo: unknown } | null = null;
useDesk.subscribe((_s, prev) => {
  if (inScene) { taskStart = null; return; }
  if (!taskStart) {
    taskStart = { paths: pathsOf(prev), undo: prev.undo, redo: prev.redo };
    queueMicrotask(() => {
      const t = taskStart; taskStart = null; if (!t) return;
      const now = useDesk.getState(), p = pathsOf(now);
      if (p !== t.paths && now.undo === t.undo && now.redo === t.redo) vsStray.push(`changed without an undo step: ${p.slice(0, 160)}`);
    });
  }
});

/** The engines by the concept's ids, and pedals and units by id. */
const BK = Object.fromEntries(ENGINES.map(e => [CONCEPT_ID[e.id], { name: e.name }]));
const PED = Object.fromEntries([...PEDAL_DEFS, ...UNITS].map(d => [d.id, { ...asItem(d), on: d.on }]));

expose({
  S, render, setVal, get, toast, MODS, BK, PED, vsStray,
  PEDALS: PEDAL_DEFS.map(asItem), EXTRAS: UNITS.map(asItem),
  get UNDO() { return useDesk.getState().undo; },
  get REDO() { return useDesk.getState().redo; },
});

/* every package's additions, each a file of its own that exports install(), run once this layer is in place */
for (const m of Object.values(import.meta.glob<{ install?: () => void }>('./compat/*.ts', { eager: true }))) m.install?.();
