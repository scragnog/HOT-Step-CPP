// What a view's model read of the desk store, so that a change to the store asks again only the views whose model read
// something that changed (docs/09, "Rendering"). A model is a pure function of the store (the core's), so what it read
// is all it depends on: the function is run on a stand-in of the store that notes every field it reads (`traced`), and
// a change is the set of what differs between two states of the store (`changes`), worked out once per change.
//
// What is noted: a field of the store (`mods`, `board`, `book`...), a field of the desk (`desk:tier`, `desk:on`), and,
// for a desk field that holds an object (the values, the auto switches, the states, the lists, what the page was told),
// each key read in it (`vals:tempo.bpm`), or all of it (`vals:*`) when its keys were listed (a spread, Object.keys,
// JSON.stringify). An object field that is replaced by another object changes only the keys that differ; one that
// appears, goes or stops being an object changes `desk:<field>` too. A spread of the desk notes `desk:*`, which changes
// when the desk gains or loses a field; a spread of the store notes `*`, which every change touches.

import type { DeskStore } from '../store/desk.ts';

export type Reads = Set<string>;

const plain = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v) && !(v instanceof Set) && !(v instanceof Map);

/** A stand-in for an object field of the desk: reading a key notes `field:key`, listing the keys notes `field:*`. */
function keyed(field: string, o: object, out: Reads): object {
  return new Proxy(o, {
    get(t, k, r) { if (typeof k === 'string') out.add(field + ':' + k); return Reflect.get(t, k, r); },
    has(t, k) { if (typeof k === 'string') out.add(field + ':' + k); return Reflect.has(t, k); },
    getOwnPropertyDescriptor(t, k) { if (typeof k === 'string') out.add(field + ':' + k); return Reflect.getOwnPropertyDescriptor(t, k); },
    ownKeys(t) { out.add(field + ':*'); return Reflect.ownKeys(t); },
  });
}

/** The store as fn reads it, noting into `out` everything read. */
function traced(s: DeskStore, out: Reads): DeskStore {
  const fields = new Map<string, object>();
  const desk = new Proxy(s.desk, {
    get(t, k, r) {
      const v = Reflect.get(t, k, r) as unknown;
      if (typeof k !== 'string') return v;
      out.add('desk:' + k);
      if (!plain(v)) return v;
      let p = fields.get(k); if (!p) { p = keyed(k, v, out); fields.set(k, p); }
      return p;
    },
    has(t, k) { if (typeof k === 'string') out.add('desk:' + k); return Reflect.has(t, k); },
    getOwnPropertyDescriptor(t, k) { if (typeof k === 'string') out.add('desk:' + k); return Reflect.getOwnPropertyDescriptor(t, k); },
    ownKeys(t) { out.add('desk:*'); return Reflect.ownKeys(t); },
  });
  return new Proxy(s, {
    get(t, k, r) { if (k === 'desk') return desk; if (typeof k === 'string') out.add(k); return Reflect.get(t, k, r); },
    ownKeys(t) { out.add('*'); return Reflect.ownKeys(t); },
  });
}

/** Runs fn on the store, and says what it read. */
export function readsOf<T>(fn: (s: DeskStore) => T, s: DeskStore): { v: T; reads: Reads } {
  const reads: Reads = new Set();
  return { v: fn(traced(s, reads)), reads };
}

const sameSet = (a: unknown, b: unknown) => a instanceof Set && b instanceof Set && a.size === b.size && [...a].every(x => b.has(x));

/** What differs between two states of the store, in the names readsOf notes. */
function diff(a: DeskStore, b: DeskStore): Set<string> {
  const d = new Set<string>();
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)]) as Set<keyof DeskStore>) if (k !== 'desk' && a[k] !== b[k]) d.add(k);
  if (a.desk === b.desk) return d;
  const da = a.desk as unknown as Record<string, unknown>, db = b.desk as unknown as Record<string, unknown>, ka = Object.keys(da), kb = Object.keys(db);
  if (ka.length !== kb.length || ka.some(k => !(k in db))) d.add('desk:*');
  for (const k of new Set([...ka, ...kb])) {
    const x = da[k], y = db[k];
    if (x === y || sameSet(x, y)) continue;
    /* an object field that stays an object changes by its keys; any other change is the field's */
    if (!plain(x) || !plain(y)) d.add('desk:' + k);
    if (!plain(x) && !plain(y)) continue;
    const ox = plain(x) ? x : {}, oy = plain(y) ? y : {};
    let any = false;
    for (const j of new Set([...Object.keys(ox), ...Object.keys(oy)])) if (ox[j] !== oy[j] || (j in ox) !== (j in oy)) { d.add(k + ':' + j); any = true; }
    if (any || Object.keys(ox).length !== Object.keys(oy).length) d.add(k + ':*');
  }
  return d;
}

/* the last few changes worked out: every view asks about the same change once the store has moved */
const seen = new WeakMap<DeskStore, { to: DeskStore; d: Set<string> }[]>();
export function changes(a: DeskStore, b: DeskStore): Set<string> {
  const l = seen.get(a) ?? [], hit = l.find(x => x.to === b);
  if (hit) return hit.d;
  const d = diff(a, b);
  l.push({ to: b, d }); if (l.length > 4) l.shift(); seen.set(a, l);
  return d;
}

/** Whether a model that read `reads` of state `a` may differ in state `b`. */
export function touched(reads: Reads, a: DeskStore, b: DeskStore): boolean {
  if (a === b) return false;
  if (reads.has('*')) return true;
  const d = changes(a, b);
  if (d.size <= reads.size) { for (const x of d) if (reads.has(x)) return true; return false; }
  for (const x of reads) if (d.has(x)) return true;
  return false;
}
