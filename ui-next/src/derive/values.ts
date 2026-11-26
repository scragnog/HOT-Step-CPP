// Values: what a stored value may be, the one cleaning rule every path that brings values in uses (a reload,
// another tab, an import, a restore, an engine switch, a factory record), and setting a value from a control.
// docs/05, "Values: one rule for every path".

import type { Definition, Json, ParamDef, StateDef, Value } from '../definitions/types.ts';
import { CHECKS } from '../definitions/parts.ts';
import { type Desk, type Index, enginesOf, has, modelDefault, resolve } from './core.ts';
import { type Msg, msg } from './messages.ts';
import { paramWords } from './strings.ts';

/** A number whose step is whole on every engine it is on is a whole number: it is stored, set and sent whole. */
export function isWhole(p: ParamDef, d: Definition): boolean {
  return p.type === 'number' && enginesOf(p, d).filter(e => d.engines.includes(e)).every(e => Number.isInteger(resolve(p, e).range?.[2] ?? 0.5));
}

/** A parameter's domain: the union of its engines' ranges, and of their fixed options. */
export function domainOf(p: ParamDef, d: Definition): { lo: number; hi: number; options: (string | number)[] | null } {
  const engs = enginesOf(p, d).filter(e => d.engines.includes(e));
  const ranges = engs.map(e => resolve(p, e).range).filter((r): r is [number, number, number] => !!r);
  const opts = engs.map(e => resolve(p, e).options);
  const fixed = opts.every(o => Array.isArray(o)) ? [...new Set(opts.flatMap(o => (o as { value: string | number }[]).map(x => x.value)))] : null;
  return { lo: Math.min(...ranges.map(r => r[0])), hi: Math.max(...ranges.map(r => r[1])), options: fixed };
}

export const TEXT_MAX = 20000;

/** One stored value, cleaned in the parameter's domain: a number finite and clamped to the union of its engines'
 *  ranges (whole numbers rounded), a switch true or false, a choice one of its options (or, server-listed, any text
 *  or number), text cut to TEXT_MAX and passing its check on every engine that checks it. Undefined: not kept. */
export function cleanValue(p: ParamDef, d: Definition, raw: unknown): Value | undefined {
  if (p.type === 'number') {
    if (typeof raw !== 'number' || !Number.isFinite(raw)) return undefined;
    const { lo, hi } = domainOf(p, d), v = Math.min(hi, Math.max(lo, raw));
    return isWhole(p, d) ? Math.round(v) : v;
  }
  if (p.type === 'boolean') return typeof raw === 'boolean' ? raw : undefined;
  if (p.type === 'choice') {
    if (typeof raw !== 'string' && typeof raw !== 'number') return undefined;
    const { options } = domainOf(p, d);
    /* a server-listed choice keeps any value of its values' type until a list has loaded that lacks it */
    return options === null ? (typeof raw === typeof resolve(p, enginesOf(p, d)[0]!).default ? raw : undefined) : options.includes(raw) ? raw : undefined;
  }
  if (typeof raw !== 'string') return undefined;
  const text = raw.slice(0, Math.min(TEXT_MAX, ...enginesOf(p, d).filter(e => d.engines.includes(e)).map(e => resolve(p, e).maxLength ?? TEXT_MAX)));
  const engs = enginesOf(p, d).filter(e => d.engines.includes(e));
  /* text that fails its check on an engine that checks it is not kept (the field keeps what was typed only while it
     is being edited); the default is always kept, so a record holding it reads back as it was written (RC-02: Custom
     timesteps' empty default fails its own check) */
  return engs.every(e => { const r = resolve(p, e), c = r.check; return !c || text === r.default || CHECKS[c.rule].test(text); }) ? text : undefined;
}

/** The state each kind may hold. Anything else is not kept. */
export function cleanState(s: StateDef, raw: unknown): Json | undefined {
  const num = (x: unknown) => typeof x === 'number' && Number.isFinite(x);
  if (raw === null) return null;
  switch (s.kind) {
    case 'file': return typeof raw === 'string' && raw.length <= 4096 ? raw : undefined;
    /* a seed is the application's: 0 to 2^64-1, a number while it is a safe integer, its decimal text beyond
       (ui/src/utils/seed.ts); a seed that arrives as text within 2^53 is kept as the number */
    case 'seed': {
      if (num(raw)) return Number.isSafeInteger(raw) && (raw as number) >= 0 ? raw as number : undefined;
      if (typeof raw !== 'string' || !/^\d{1,20}$/.test(raw.trim()) || BigInt(raw.trim()) > 18446744073709551615n) return undefined;
      const big = BigInt(raw.trim());
      return big <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(big) : big.toString();
    }
    case 'span': {
      const r = raw as { start?: unknown; end?: unknown };
      return r && typeof r === 'object' && num(r.start) && num(r.end) && (r.start as number) >= 0 && (r.start as number) < (r.end as number) ? { start: r.start as number, end: r.end as number } : undefined;
    }
    case 'slots': {
      if (!Array.isArray(raw)) return undefined;
      const out: Json[] = [];
      for (const x of raw.slice(0, 3)) {
        const o = x as { file?: unknown; type?: unknown; ph?: unknown };
        if (!o || typeof o !== 'object' || typeof o.file !== 'string' || !o.file) continue;
        const ph = Array.isArray(o.ph) && o.ph.length === 2 && o.ph.every(num) && (o.ph[0] as number) >= 0 && (o.ph[1] as number) <= 100 && (o.ph[0] as number) < (o.ph[1] as number) ? o.ph as number[] : null;
        out.push({ file: o.file, type: typeof o.type === 'string' ? o.type : 'LoRA', ...(ph && (ph[0]! > 0 || ph[1]! < 100) ? { ph } : {}) });
      }
      return out;
    }
    case 'list': return Array.isArray(raw) ? raw as Json[] : undefined;
    case 'record': return raw && typeof raw === 'object' && !Array.isArray(raw) ? raw as Json : undefined;
  }
}

/** The pairs whose two ends never cross (staysBelow): the lower end first. */
export const pairsOf = (ix: Index): [string, string][] => [...ix.params.values()].filter(({ p }) => p.staysBelow).map(({ p }) => [p.key, p.staysBelow!]);

/** A pair whose ends arrive crossed (an imported revision with the interval's start above its end) is put in order:
 *  the two values swap, so neither is lost. */
export function orderPairs(ix: Index, vals: { [key: string]: Value }): void {
  for (const [lo, hi] of pairsOf(ix)) {
    const a = vals[lo], b = vals[hi];
    if (typeof a === 'number' && typeof b === 'number' && a > b) { vals[lo] = b; vals[hi] = a; }
  }
}

/** What cleaning a record's values said: what was read under a former name, and what could not be used. */
export interface Cleaned {
  vals: { [key: string]: Value };
  auto: { [key: string]: boolean };
  /** Keys that were read as the parameter that lists them in `was`, and option values read under their new value. */
  renamed: string[];
  /** Keys or values that could not be used; a retired key is dropped without being counted. */
  dropped: string[];
}

/** Cleans a record's values and auto switches (docs/05, "Cleaning"): every key read as its parameter (or the one that
 *  lists it in `was`), every value cleaned in its parameter's domain, every option value read through `was`, crossed
 *  pairs put in order. `retired` keys are dropped silently. */
export function cleanValues(ix: Index, vals: unknown, auto: unknown, retired: readonly string[] = []): Cleaned {
  const out: Cleaned = { vals: {}, auto: {}, renamed: [], dropped: [] };
  const obj = (x: unknown) => (x && typeof x === 'object' && !Array.isArray(x) ? x as Record<string, unknown> : {});
  for (const [k0, raw] of Object.entries(obj(vals))) {
    const k = ix.params.has(k0) ? k0 : ix.former.get(k0);
    if (!k) { if (!retired.includes(k0)) out.dropped.push(k0); continue; }
    if (k !== k0) out.renamed.push(k0);
    const { p, d } = ix.params.get(k)!;
    let x = raw;
    if (p.type === 'choice' && (typeof x === 'string' || typeof x === 'number')) {
      const now = enginesOf(p, d).map(e => resolve(p, e).options).flatMap(o => (Array.isArray(o) ? o : o?.include ?? [])).find(o => (o.was || []).includes(x as string | number));
      if (now) { x = now.value; out.renamed.push(`${k0}=${String(raw)}`); }
    }
    const v = cleanValue(p, d, x);
    if (v === undefined) out.dropped.push(k0); else out.vals[k] = v;
  }
  for (const [k0, raw] of Object.entries(obj(auto))) {
    const k = ix.params.has(k0) ? k0 : ix.former.get(k0);
    if (!k) { if (!retired.includes(k0)) out.dropped.push(`${k0} (auto)`); continue; }
    const { p, d } = ix.params.get(k)!;
    if (typeof raw === 'boolean' && enginesOf(p, d).some(e => resolve(p, e).auto)) out.auto[k] = raw; else out.dropped.push(`${k0} (auto)`);
  }
  orderPairs(ix, out.vals);
  return out;
}

/** Cleans a record's state the same way: every key its StateDef's (or a former name's), every value of its kind. */
export function cleanStates(ix: Index, state: unknown, retired: readonly string[] = []): { state: { [key: string]: Json }; dropped: string[] } {
  const out: { [key: string]: Json } = {}, dropped: string[] = [];
  for (const [k0, raw] of Object.entries(state && typeof state === 'object' && !Array.isArray(state) ? state : {})) {
    const k = ix.states.has(k0) ? k0 : ix.formerStates.get(k0);
    if (!k) { if (!retired.includes(k0)) dropped.push(k0); continue; }
    const v = cleanState(ix.states.get(k)!.s, raw);
    if (v === undefined) dropped.push(k0); else out[k] = v;
  }
  return { state: out, dropped };
}

/** Setting a value from a control, on the desk's engine: a number clamped to that engine's range and put on its step
 *  grid from the minimum (a control's own keys and drags make steps), a pair's end kept a step from the other. `said`
 *  is the one-time sentence when a typed number was out of range ("Steps: 1 to 64. Set to 64."). */
export function setValue(ix: Index, desk: Desk, key: string, raw: Value, typed = false): { value: Value; said?: Msg } {
  const hit = ix.params.get(key);
  if (!hit || !has(ix, key, desk.engine)) throw new Error(`${key} is not on ${desk.engine}`);
  const f = resolve(hit.p, desk.engine);
  /* a model file the server has selected is the engine's default, "" (the request sends it either way) */
  if (raw !== '' && raw === modelDefault(ix, key, desk)) return { value: '' };
  if (hit.p.type !== 'number' || typeof raw !== 'number' || !f.range) {
    const v = cleanValue(hit.p, hit.d, raw);
    if (v === undefined) throw new Error(`${JSON.stringify(raw)} is not a value of ${key}`);
    return { value: v };
  }
  /* a number that is no number (NaN, Infinity) is refused, as a string is */
  if (!Number.isFinite(raw)) throw new Error(`${raw} is not a value of ${key}`);
  const [lo, hi, step] = f.range;
  let v = Math.min(hi, Math.max(lo, raw));
  v = +(lo + Math.round((v - lo) / step) * step).toFixed(10);
  /* a range's two ends never cross: an end pushed past the other stops a step from it */
  for (const [a, b] of pairsOf(ix)) {
    if (key === a) { const other = desk.vals[b] ?? resolve(ix.params.get(b)!.p, desk.engine).default; if (typeof other === 'number' && v > other - step) v = +(other - step).toFixed(10); }
    if (key === b) { const other = desk.vals[a] ?? resolve(ix.params.get(a)!.p, desk.engine).default; if (typeof other === 'number' && v < other + step) v = +(other + step).toFixed(10); }
  }
  /* a typed value outside the range says where it was put; a hand or a key past an end just stops there */
  if (typed && (raw < lo || raw > hi)) {
    const w = (x: number) => (Number.isInteger(step) ? String(x) : x.toFixed(String(step).split('.')[1]?.length ?? 1));
    return { value: v, said: msg('set.said', { name: paramWords(ix, key, desk).label, lo: w(lo), hi: w(hi), v: w(v) }) };
  }
  return { value: v };
}
