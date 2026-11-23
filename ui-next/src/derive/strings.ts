// Translation keys (docs/05, "Strings and translation"): every string a definition holds, with the key it is looked up
// under. A key is made from where the string sits and what it is bound to, never from its place in a list, so
// reordering lines, options or nodes keeps every translation; and no two strings share a key (the test checks it).

import type { Action, Definition, Node, NodeOverride, ParamDef, Strip, Value } from '../definitions/types.ts';

export interface Str { key: string; text: string }

/** A value as a key part: letters and digits kept, every other character written as its code in hex between two
 *  `_`, so two values never give one key ('F# major' is `F_23__20_major`, 'F major' `F_20_major`); "" is `_`. */
export const keyPart = (v: Value | string): string => {
  const s = String(v);
  if (s === '') return '_';
  return [...s].map(c => (/[A-Za-z0-9]/.test(c) ? c : `_${c.codePointAt(0)!.toString(16)}_`)).join('');
};
/** A short stable hash, for what has no name of its own (a condition, a node with no binding). */
export const hash = (s: string): string => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(36);
};
const J = (x: unknown) => JSON.stringify(x);
const dashed = (key: string) => key.replace(/\./g, '-');

/** Every string of a definition, keyed. */
export function stringsOf(d: Definition): Str[] {
  const out: Str[] = [], base = `create.${d.id}`;
  const add = (key: string, text: string | undefined) => { if (text !== undefined && text !== '') out.push({ key, text }); };
  add(`${base}.name`, d.name); add(`${base}.desc`, d.desc);
  if (d.kind === 'module') {
    add(`${base}.quickName`, d.quickName);
    d.tierSummary?.forEach((t, i) => add(`${base}.tier.${['quick', 'studio', 'lab'][i]}`, t));
    for (const l of d.per || []) {
      const at = `@${l.engines.join('+')}`;
      add(`${base}.name${at}`, l.name); add(`${base}.quickName${at}`, l.quickName); add(`${base}.desc${at}`, l.desc);
      l.tierSummary?.forEach((t, i) => add(`${base}.tier.${['quick', 'studio', 'lab'][i]}${at}`, t));
    }
    (d.strip || []).forEach(s => strip(`${base}.strip`, s));
  }
  if (d.kind === 'pedal') { add(`${base}.short`, d.short); (d.idle || []).forEach(s => strip(`${base}.idle`, s)); }
  for (const p of d.params) param(`${base}.p.${dashed(p.key)}`, p);
  for (const s of d.state || []) {
    const k = `${base}.s.${dashed(s.key)}`;
    add(`${k}.noun`, s.noun); add(`${k}.empty`, s.empty);
    for (const [n, st] of Object.entries(s.steps || {})) add(`${k}.step.${n}`, typeof st === 'string' ? st : st.says);
  }
  const face = 'face' in d ? d.face || [] : [];
  face.forEach(n => node(`${base}.n`, n, counts(face)));
  return out;

  function param(k: string, p: ParamDef) {
    const face = (f: Partial<ParamDef>, at: string) => {
      add(`${k}.label${at}`, f.label); add(`${k}.short${at}`, f.short); add(`${k}.undoName${at}`, f.undoName); add(`${k}.tip${at}`, f.tip);
      add(`${k}.placeholder${at}`, f.placeholder); add(`${k}.check${at}`, f.check?.message); add(`${k}.keepInvalid${at}`, f.check?.keepInvalid);
      if (f.auto) add(`${k}.auto${at}`, f.auto.says);
      if (f.format) { add(`${k}.template${at}`, f.format.template); (f.format.words || []).forEach(w => add(`${k}.w.${keyPart(w.value)}${at}`, w.text)); }
      if (Array.isArray(f.options)) for (const o of f.options) { const ok = `${k}.o.${keyPart(o.value)}`; add(`${ok}.label${at}`, o.label); add(`${ok}.short${at}`, o.short); add(`${ok}.tip${at}`, o.tip); add(`${ok}.lcd${at}`, o.lcd); add(`${ok}.detail${at}`, o.detail); }
      else for (const o of f.options?.include || []) { const ok = `${k}.o.${keyPart(o.value)}`; add(`${ok}.label${at}`, o.label); add(`${ok}.tip${at}`, o.tip); add(`${ok}.lcd${at}`, o.lcd); }
    };
    face(p, '');
    if (p.held) { add(`${k}.held.note`, p.held.note); add(`${k}.held.toast`, p.held.toast); add(`${k}.held.value`, p.held.value); }
    for (const [n, st] of Object.entries(p.steps || {})) add(`${k}.step.${n}`, typeof st === 'string' ? st : st.says);
    /* an engine line by its engines, a word line by its condition: neither moves when lines are reordered */
    for (const l of p.per || []) face(l as Partial<ParamDef>, 'engines' in l && l.engines ? `@${l.engines.join('+')}` : `@when-${hash(J(l.when))}`);
  }
  function actions(k: string, list: Action[] | undefined) {
    for (const a of list || []) {
      const id = `${a.do}-${keyPart('param' in a ? a.param : 'tier' in a ? a.tier : 'item' in a ? a.item : 'tag' in a ? a.tag : 'name' in a ? a.name : '')}${'value' in a ? '-' + keyPart(String(a.value)) : ''}`;
      add(`${k}.a.${id}.label`, a.label); if ('tip' in a) add(`${k}.a.${id}.tip`, a.tip);
    }
  }
  function strip(k: string, s: Strip) {
    const at = `${k}.${hash(J([s.when, s.tiers]))}`;
    add(`${at}.says`, s.says); actions(at, s.actions);
  }
  /** A node by what it is bound to (its part and its keys), or by its words when it is bound to nothing; a second node
   *  of the same kind adds a hash of itself. */
  /** How many siblings share each kind-and-binding, so all of a shared one are told apart by a hash of themselves. */
  function idOf(n: Node): string {
    const binds = [n.param, ...(n.params || []), ...(n.reads || []), ...(n.state || [])].filter(Boolean).join('+');
    return `${n.part}.${binds ? keyPart(binds) : keyPart((n.text ?? n.label ?? n.part).slice(0, 32))}`;
  }
  function counts(list: Node[]): Map<string, number> { const m = new Map<string, number>(); list.forEach(n => m.set(idOf(n), (m.get(idOf(n)) ?? 0) + 1)); return m; }
  function node(k: string, n: Node, seen: Map<string, number>) {
    const what = idOf(n).slice(n.part.length + 1);
    let id = `${n.part}.${what}`;
    if ((seen.get(id) ?? 0) > 1) id += `~${hash(J({ ...n, children: undefined, per: undefined }))}`;
    const at = `${k}.${id}`;
    const words = (x: Partial<Node>, sfx: string) => {
      add(`${at}.label${sfx}`, x.label); add(`${at}.tip${sfx}`, x.tip); add(`${at}.text${sfx}`, x.text); add(`${at}.dim${sfx}`, x.dim?.says); add(`${at}.between${sfx}`, x.between);
      (x.choices || []).forEach(([w, v]) => add(`${at}.c.${keyPart(typeof v === 'object' ? J(v) : String(v))}${sfx}`, w));
      actions(`${at}${sfx}`, x.actions);
      for (const [o, v] of Object.entries((x as { opts?: Record<string, unknown> }).opts || {})) {
        if (typeof v === 'string' && /[a-z] [a-z]|^[A-Z][a-z]/.test(v)) add(`${at}.opt.${o}${sfx}`, v);
        if (Array.isArray(v)) v.forEach((s, i) => typeof s === 'string' && add(`${at}.opt.${o}.${i}${sfx}`, s));
      }
    };
    words(n, '');
    for (const l of (n.per || []) as NodeOverride[]) words(l, `@${hash(J([l.engines, l.tiers, l.when]))}`);
    (n.children || []).forEach(c => node(at, c, counts(n.children!)));
  }
}
