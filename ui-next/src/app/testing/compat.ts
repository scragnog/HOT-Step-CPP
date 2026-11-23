// The test-only compatibility layer: what the concept's two tools (tools/behaviour, tools/visual-baseline) reach into
// the concept page for, provided on top of the store, so they can drive the rebuild as they drive the concept. Only
// the test build (vite build --mode test) contains it; main.tsx imports it there and nowhere else.
//
// What the tools use (README files of both tools, and their states and scenarios): the state object `S` (backend,
// tier, mods, pedals, vals, and the rest of a scene), `render()` to apply a scene, the catalogue (`MODS`, `PEDALS`,
// `EXTRAS`), `setVal(key, value)`, the undo history's labels (`pushUndo`, read through window.__bh.labels) and its
// length (`UNDO`), and `vsStray` (changes made outside a step: none can be, every change is a step). A scene the
// scenario sets through `S` is applied as it is, as the concept's render() takes it: not an undo step.
//
// The concept names engines and choice values its own way (backend "mm3", a choice by its label "Crossover"); this
// layer translates both ways, by the definitions' labels, so the tools' scenes mean what they meant on the concept.

import { flushSync } from 'react-dom';
import type { EngineId, Value } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';
import { bindPart, has, resolve, valueOf } from '../../derive.ts';
import { usePanelStore } from '../store/panels.ts';
import type { Node } from '../../definitions/types.ts';
import { BOARD, BUILT_IN, OFFERED, PEDALS as PEDAL_DEFS, UNITS, boardOrder, defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { sayNow } from '../i18n/index.ts';
import { CONCEPT_ID } from '../shell/engines.ts';

const TO_ENGINE: Record<string, EngineId> = Object.fromEntries(ENGINES.map(e => [CONCEPT_ID[e.id], e.id]));

/** A concept value (a choice by its label) as the definitions store it, and back. */
function fromConcept(key: string, v: unknown, e: EngineId): Value | undefined {
  const hit = ix.params.get(key);
  if (!hit || hit.p.type !== 'choice') return v as Value;
  const f = resolve(hit.p, (hit.p.engines ?? hit.d.engines).includes(e) ? e : (hit.p.engines ?? hit.d.engines)[0]!);
  const opts = Array.isArray(f.options) ? f.options : f.options?.include ?? [];
  const o = opts.find(x => x.label === v || x.short === v || String(x.value).toLowerCase() === String(v).toLowerCase());
  return o ? o.value : (v as Value);
}
function toConcept(key: string, v: Value, e: EngineId): unknown {
  const hit = ix.params.get(key);
  if (!hit || hit.p.type !== 'choice') return v;
  const f = resolve(hit.p, (hit.p.engines ?? hit.d.engines).includes(e) ? e : (hit.p.engines ?? hit.d.engines)[0]!);
  const opts = Array.isArray(f.options) ? f.options : f.options?.include ?? [];
  return opts.find(x => x.value === v)?.label ?? v;
}

interface Scene { backend: string; tier: string; mods: string[]; pedals: string[]; vals: Record<string, unknown>; [k: string]: unknown }

/** The concept's S, kept in step with the store: read it, change it, then render(). */
const S: Scene = { backend: 'ace', tier: 'quick', mods: [], pedals: [], vals: {} };
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
}
mirror();
useDesk.subscribe(mirror);
usePanelStore.subscribe(mirror);

/** Applies the scene in S to the store, as the concept's render() takes it: as it is, not as a step. */
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
  flushSync(() => {
    useDesk.setState({ desk: { ...s.desk, engine, tier, vals, auto, on: new Set([...BUILT_IN, ...mods, ...board]) }, mods, board });
    usePanelStore.setState({ left: S.left !== false, right: S.right !== false, rOpen: S.rOpen === true, lOpen: S.lOpen === true });
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
  const bound = bindPart(ix, hit.d, { part: 'knob', param: key } as Node, s.desk, c => s.commit(c)).params[key];
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

/** The engines by the concept's ids, and pedals and units by id. */
const BK = Object.fromEntries(ENGINES.map(e => [CONCEPT_ID[e.id], { name: e.name }]));
const PED = Object.fromEntries([...PEDAL_DEFS, ...UNITS].map(d => [d.id, { ...asItem(d), on: d.on }]));

Object.assign(window, {
  S, render, setVal, get, toast, MODS, BK, PED,
  PEDALS: PEDAL_DEFS.map(asItem), EXTRAS: UNITS.map(asItem),
  vsStray: [] as string[],
  get UNDO() { return useDesk.getState().undo; },
  get REDO() { return useDesk.getState().redo; },
});
