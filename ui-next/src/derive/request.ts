// The request (docs/05, "What is sent"): the desk in the real application's names, then the Create intent the page
// submits, built with the real application's own code, which is handed in rather than restated.

import type { Definition, EngineId, Json, Value, Wire, WireKey, WireMap } from '../definitions/types.ts';
import { ENGINES } from '../definitions/engines.ts';
import { TRANSFORMS, type Transform } from '../definitions/transforms.ts';
import { type Desk, type Index, applies, holds, isAuto, resolve, valueOf } from './core.ts';

/** The Create request in real names, before the real application's own code finishes it: `params` for
 *  resolveGenerationParams() (server/src/services/generation/defaults.ts), `backendParams` for its extension bag,
 *  `content` for createContentParams() (ui/src/components/create/createContent.ts), `body` for fields set after both,
 *  `intent` for the Create intent's own parts (`compose`, `captionSource`: server/src/contracts/resolution.ts). */
export interface RealState {
  params: { [field: string]: Json };
  backendParams: { [field: string]: Json };
  content: { [field: string]: Json };
  body: { [field: string]: Json };
  intent: { [field: string]: Json };
  /** Which parameter or state wrote each field, by its wire's `to`. */
  from: { [to: string]: string[] };
}

const flat = (w: Wire): (WireKey | WireMap)[] => (Array.isArray(w) ? w.flatMap(flat) : typeof w === 'object' && 'none' in w ? [] : [w as WireKey | WireMap]);
export const wiresOf = flat;
export const toOf = (w: WireKey | WireMap): WireKey => (typeof w === 'string' ? w : w.to);
const pathOf = (name: string) => name.split(/\.|\[(\d+)\]\.?/).filter(x => x !== undefined && x !== '');

/** The wire of a state on an engine: its own, or the line for that engine. */
export function stateWire(s: NonNullable<Definition['state']>[number], e: EngineId): Wire | undefined {
  const w = s.wire;
  return Array.isArray(w) && w.length && typeof w[0] === 'object' && 'engines' in (w[0] as object)
    ? (w as { engines: EngineId[]; wire: Wire }[]).find(x => x.engines.includes(e))?.wire
    : (w as Wire);
}

function put(out: RealState, to: WireKey, value: Json, fieldDefault: (field: string) => Json | undefined, who?: string): void {
  if (who) (out.from[to] ||= []).push(who);
  const ch = to.slice(0, to.indexOf(':')), name = to.slice(to.indexOf(':') + 1);
  const bag = ch === 'param' ? out.params : ch === 'ext' ? out.backendParams : ch === 'content' ? out.content : ch === 'body' ? out.body : ch === 'intent' ? out.intent : null;
  if (!bag) return;
  const path = pathOf(name);
  if (path.length === 1) { bag[path[0]!] = value; return; }
  /* an entry of a list only exists when something sent the list (a slot's strength with no adapter loaded) */
  if (path.some(x => /^\d+$/.test(x)) && bag[path[0]!] === undefined) return;
  let at: { [k: string]: Json } | Json[] = bag;
  for (let i = 0; i < path.length - 1; i++) {
    const k = path[i]!, next = path[i + 1]!, isIdx = /^\d+$/.test(next);
    let child = (at as Record<string, Json>)[k];
    if (Array.isArray(at) && child === undefined) return; /* no such entry: a slot with nothing loaded */
    if (child === undefined || child === null || typeof child !== 'object') {
      const base = i === 0 && ch === 'param' ? fieldDefault(k) : undefined;
      child = base !== undefined && typeof base === 'object' ? structuredClone(base) : isIdx ? [] : {};
      (at as Record<string, Json>)[k] = child;
    }
    at = child as { [k: string]: Json } | Json[];
  }
  const last = path[path.length - 1]!;
  if (Array.isArray(at)) { const i = Number(last); if (at[i] === undefined) return; (at as Json[])[i] = value; }
  else at[last] = value;
}

/** The real application's default for a field: GENERATION_DEFAULTS for param:, the backend's declared default for
 *  ext:. Undefined when it is not known (a field a backend reads without declaring it). */
export type RealDefault = (to: WireKey) => Json | undefined;

/** The real defaults from GENERATION_DEFAULTS and one backend's declared extensions (GET /api/capabilities). */
export function realDefaultsFrom(generationDefaults: { [field: string]: Json }, extensions: readonly { key: string; default?: Json }[]): RealDefault {
  return to => {
    const ch = to.slice(0, to.indexOf(':')), name = to.slice(to.indexOf(':') + 1);
    if (ch === 'ext') { const d = extensions.find(x => x.key === name)?.default; return typeof d === 'string' && d.startsWith('<') ? undefined : d; }
    if (ch !== 'param') return undefined;
    let v: Json | undefined = generationDefaults;
    for (const part of pathOf(name)) {
      if (v === null || typeof v !== 'object') return undefined;
      v = (v as { [k: string]: Json })[part];
    }
    return v;
  };
}

/** The definitions whose values the request reads on the desk's engine: the built-in modules, and everything on the
 *  desk; never a source module (Repaint). */
export const liveDefs = (ix: Index, desk: Desk): Definition[] => ix.defs.filter(d => d.engines.includes(desk.engine)
  && !(d.kind === 'module' && d.source) && ((d.kind === 'module' && d.builtIn) || desk.on.has(d.id)));

/** Everything on the desk that applies on its engine, through its wires (docs/05, "What is sent"). A value is sent
 *  unless it is untouched and equals the field's real default; a field whose real default is unknown is sent once
 *  touched, and a body: field always. A module not on the desk sends nothing (its values are kept); a pedal or side
 *  unit not on the board is off: its switch sends its off value and nothing else of it is sent. A model choice left
 *  at "" (the engine's default) is sent as the model the server has selected for that bucket. */
export function realState(ix: Index, desk: Desk, fieldDefault: (field: string) => Json | undefined = () => undefined, realDefault: RealDefault = () => undefined): RealState {
  const out: RealState = { params: {}, backendParams: {}, content: {}, body: {}, intent: {}, from: {} };
  const packed = new Map<string, { to: WireKey; t: Transform; who: string[]; touched: boolean }>();
  const J = (x: unknown) => JSON.stringify(x);
  const emit = (to: WireKey, v: Json, touched: boolean, who: string) => {
    const ch = to.slice(0, to.indexOf(':'));
    if (!touched && (ch === 'param' || ch === 'ext')) {
      const d = realDefault(to);
      if (d === undefined ? ch === 'ext' : J(d) === J(v)) return;
    }
    put(out, to, v, fieldDefault, who);
  };
  const send = (w: WireKey | WireMap, raw: Json | undefined, auto: boolean, touched: boolean, who: string) => {
    if (typeof w === 'string') { if (raw !== undefined && !auto) emit(w, raw, touched, who); return; }
    if (w.when && !holds(ix, w.when, desk)) { if ('else' in w && w.else !== undefined) emit(w.to, w.else, touched, who); return; }
    if (auto) { if ('auto' in w && w.auto !== null && w.auto !== undefined) emit(w.to, w.auto, touched, who); return; }
    if (w.via) {
      const t = TRANSFORMS[w.via] as Transform;
      if (t.packs) { const g = packed.get(`${w.to}|${w.via}`) || { to: w.to, t, who: [], touched: false }; g.who.push(who); g.touched ||= touched; packed.set(`${w.to}|${w.via}`, g); return; }
      const v = t.send(raw ?? null); if (v !== undefined) emit(w.to, v, touched, who); return;
    }
    if (raw === undefined) return;
    let v: Json | undefined = raw;
    if (w.map) { const k = String(raw); v = k in w.map ? w.map[k] : undefined; }
    if (w.invert && typeof v === 'boolean') v = !v;
    if (w.scale !== undefined && typeof v === 'number') v = +(v * w.scale).toFixed(10);
    if (w.divide !== undefined && typeof v === 'number') v = +(v / w.divide).toFixed(10);
    if (v !== undefined) emit(w.to, v, touched, who);
  };
  const selected = (wires: (WireKey | WireMap)[], v: Value | undefined): Value | undefined => {
    if (v !== '') return v;
    const m = wires.map(toOf).find(t => t.startsWith('model:'));
    const name = m && desk.selection?.[m.slice(6)];
    return name ?? v;
  };
  for (const d of liveDefs(ix, desk)) {
    for (const s of d.state || []) {
      const ws = stateWire(s, desk.engine);
      const v = desk.state[s.key], empty = v === undefined || v === null || v === '' || (Array.isArray(v) && !v.length);
      /* an empty state sends nothing, except through a transform, which says what empty is ("no adapter") */
      if (ws) flat(ws).forEach(w => { if (!empty || (typeof w === 'object' && w.via)) send(w, empty ? null : v, false, !empty, s.key); });
    }
    for (const p of d.params) {
      if (!applies(ix, p.key, desk)) continue;
      const f = resolve(p, desk.engine), touched = p.key in desk.vals || p.key in desk.auto, ws = flat(f.wire);
      const v = selected(ws, valueOf(ix, p.key, desk));
      ws.forEach(w => send(w, v, isAuto(ix, p.key, desk), touched || v !== valueOf(ix, p.key, desk), p.key));
    }
  }
  /* a pedal or side unit not on the board is off */
  for (const d of ix.defs) {
    if (d.kind === 'module' || !d.on || !d.engines.includes(desk.engine) || desk.on.has(d.id)) continue;
    const p = ix.params.get(d.on)!.p;
    if (!applies(ix, p.key, desk)) continue;
    flat(resolve(p, desk.engine).wire).forEach(w => send(w, false, false, false, p.key));
  }
  for (const { to, t, who, touched } of packed.values()) {
    const vals: { [key: string]: Value | Json | undefined } = {};
    for (const k of t.packs!) vals[k] = ix.params.has(k) ? (applies(ix, k, desk) ? valueOf(ix, k, desk) : undefined) : desk.state[k];
    const v = t.send(vals); if (v !== undefined) emit(to, v, touched, who.join('+'));
  }
  return out;
}

/* ---------------------------------------------------------------- the Create intent */

/** The real application's code the request is finished with, and the facts it reads. The page imports the two
 *  functions, as the existing UI does; the tools load them from the repository (tools/real-app). */
export interface RealApp {
  /** server/src/services/generation/defaults.ts */
  resolveGenerationParams(params: { [field: string]: unknown }, opts: { backendId: string; settings: { [k: string]: unknown } }): { [field: string]: unknown };
  /** ui/src/components/create/createContent.ts, and its defaults (createContentFromDraft({})) */
  createContentParams(content: { [field: string]: unknown }, opts: { [k: string]: unknown }): { [field: string]: unknown };
  contentDefaults: { [field: string]: Json };
  generationDefaults: { [field: string]: Json };
  /** Each engine's declared extensions (GET /api/capabilities). */
  extensions: { [engine: string]: readonly { key: string; default?: Json }[] };
  /** Each engine's longest song (capabilities().core.duration.max). */
  durationMax: { [engine: string]: number | null };
  /** The app's settings every Create request carries (Settings panel): the trigger settings the resolver reads, and
   *  source, coResident, cacheLmCodes and the parallel switches copied onto the request. */
  settings: { trigger: { [k: string]: unknown }; request: { [k: string]: unknown } };
}

/** The Create intent (server/src/contracts/resolution.ts createIntentSchema) the page submits to
 *  /api/resolve/preview, as the existing Create page does (CreatePanel.tsx createIntent). */
export interface CreateIntent {
  kind: 'create';
  params: { [field: string]: unknown };
  compose: { [field: string]: Json };
  captionSource?: { [field: string]: Json };
}

export function createIntent(ix: Index, desk: Desk, real: RealApp): CreateIntent {
  const e = desk.engine, eng = ENGINES.find(x => x.id === e)!;
  const rs = realState(ix, desk, f => real.generationDefaults[f], realDefaultsFrom(real.generationDefaults, real.extensions[e] ?? []));
  const resolved = real.resolveGenerationParams({ ...rs.params, backendParams: rs.backendParams }, { backendId: e, settings: real.settings.trigger });
  const content = { ...real.contentDefaults, ...rs.content };
  const contentParams = real.createContentParams(content, {
    mm3Mode: 'createContent' in eng && 'mm3Mode' in eng.createContent, yue2Active: 'createContent' in eng && 'yue2Active' in eng.createContent,
    durationMax: real.durationMax[e] ?? undefined,
  });
  const compose = (rs.intent.compose ?? {}) as { [k: string]: Json };
  const source = rs.intent.captionSource as { [k: string]: Json } | undefined;
  return {
    kind: 'create',
    params: {
      ...('jobOverride' in eng ? { backendOverride: e } : {}), ...resolved, ...rs.body, ...contentParams,
      caption: content.caption, lyrics: content.lyrics, ...real.settings.request,
    },
    compose,
    ...(source ? { captionSource: source } : {}),
  };
}

/* ---------------------------------------------------------------- reading the real fields back */

/** The desk values a set of real fields stands for on an engine: a saved profile or a song's settings ("Reuse
 *  settings") read through the wires backwards. A field is found under its name or a former one (`was` on the wire);
 *  `invert`, `scale`, `divide` and `map` are undone, a transform's `read` gives its parameters back, and a value the
 *  definitions cannot use is left out (the caller cleans what comes back). */
export function fromReal(ix: Index, e: EngineId, real: { [field: string]: Json }): { [key: string]: Value | Json } {
  const out: { [key: string]: Value | Json } = {};
  const get = (w: WireMap | WireKey): Json | undefined => {
    const m = typeof w === 'string' ? { to: w } as WireMap : w;
    for (const to of [m.to, ...(m.was || [])]) {
      const name = to.slice(to.indexOf(':') + 1);
      let v: Json | undefined = real;
      for (const part of pathOf(name)) { if (v === null || typeof v !== 'object') { v = undefined; break; } v = (v as { [k: string]: Json })[part]; }
      if (v !== undefined) return v;
    }
    return undefined;
  };
  const done = new Set<string>();
  const consider = (key: string, w: WireKey | WireMap) => {
    const ch = toOf(w).split(':')[0]!;
    if (ch === 'ui' || ch === 'model' || (typeof w === 'object' && w.when)) return;
    const v = get(w); if (v === undefined) return;
    if (typeof w === 'object' && w.via) {
      const t = TRANSFORMS[w.via] as Transform;
      if (t.packs && done.has(w.via)) return;
      const r = t.read(v); if (r === undefined) return;
      if (t.packs) { done.add(w.via); Object.assign(out, r as object); } else out[key] = r as Value | Json;
      return;
    }
    let x: Json | undefined = v;
    if (typeof w === 'object') {
      if (w.map) { const hit = Object.entries(w.map).find(([, r]) => JSON.stringify(r) === JSON.stringify(v)); x = hit ? hit[0] : undefined; if (x !== undefined && ix.params.get(key)?.p.type === 'boolean') x = x === 'true'; }
      if (w.invert && typeof x === 'boolean') x = !x;
      if (w.scale !== undefined && typeof x === 'number') x = +(x / w.scale).toFixed(10);
      if (w.divide !== undefined && typeof x === 'number') x = +(x * w.divide).toFixed(10);
      if ('auto' in w && JSON.stringify(w.auto) === JSON.stringify(v)) { out[`${key}`] = null; return; }
    }
    if (x !== undefined && !(key in out)) out[key] = x;
  };
  for (const { p, d } of ix.params.values()) {
    if (!d.engines.includes(e) || !(p.engines ?? d.engines).includes(e)) continue;
    flat(resolve(p, e).wire).forEach(w => consider(p.key, w));
  }
  for (const { s, d } of ix.states.values()) {
    if (!d.engines.includes(e)) continue;
    const w = stateWire(s, e); if (w) flat(w).forEach(x => consider(s.key, x));
  }
  return out;
}
