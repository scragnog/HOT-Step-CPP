// Translation keys (docs/05, "Strings and translation"): every string a definition holds, with the key it is looked up
// under. A key is made from where the string sits and what it is bound to, never from its place in a list, so
// reordering lines, options or nodes keeps every translation; and no two strings share a key (the test checks it).

import type { Action, Definition, Node, NodeOverride, ParamDef, Strip, Value } from '../definitions/types.ts';
import { PARTS, type PartSpec } from '../definitions/parts.ts';
import { EN } from './messages.ts';

export interface Str { key: string; text: string }

/** The core's own messages (messages.ts EN) and the parts' how-to words: the strings no definition holds. */
export function coreStrings(): Str[] {
  return [...Object.entries(EN).map(([key, text]) => ({ key, text })),
    ...Object.entries(PARTS as Record<string, PartSpec>).flatMap(([name, s]) => (s.howTo ? [{ key: `part.${name}.howTo`, text: s.howTo }] : []))];
}

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
      /* a Quick choice by its place in its row (its engine line): retuning a preset's values keeps its words */
      (x.choices || []).forEach(([w], i) => add(`${at}.c.${i}${sfx}`, w));
      actions(`${at}${sfx}`, x.actions);
      /* a part's options that are words, as the inventory types them (text, texts) */
      const spec = (PARTS as Record<string, PartSpec>)[n.part]?.opts ?? {};
      for (const [o, v] of Object.entries((x as { opts?: Record<string, unknown> }).opts || {})) {
        const type = (spec as Record<string, { type: unknown }>)[o]?.type;
        if (type === 'text' && typeof v === 'string') add(`${at}.opt.${o}${sfx}`, v);
        if (type === 'texts' && Array.isArray(v)) v.forEach((s, i) => typeof s === 'string' && add(`${at}.opt.${o}.${i}${sfx}`, s));
      }
    };
    words(n, '');
    for (const l of (n.per || []) as NodeOverride[]) words(l, `@${hash(J([l.engines, l.tiers, l.when]))}`);
    (n.children || []).forEach(c => node(at, c, counts(n.children!)));
  }
}

/* ---------------------------------------------------------------- a definition's words as messages
   What the core says with a definition's words carries their translation key, so a page in another locale says them
   in its language (messages.ts). The key is the one stringsOf() gives the line the words come from on the desk. */

import type { Desk, Index } from './core.ts';
import { countOf, factOf, fill, formatValue, has, holds, isAuto, isEngineLine, valueOf, wordsOf, SLOT } from './core.ts';
import { type Msg, type Slot, msg } from './messages.ts';
import type { ParamFace } from '../definitions/types.ts';
import { engineName } from '../definitions/engines.ts';

/** The key of a parameter's word field as it reads on the desk: its base, its engine line's, or a word line's. */
export function wordKey(ix: Index, key: string, desk: Desk, field: 'label' | 'short' | 'undoName' | 'tip' | 'auto' | 'placeholder'): string {
  const { p, d } = ix.params.get(key)!;
  const base = `create.${d.id}.p.${dashed(p.key)}.${field}`;
  let at = '';
  const line = (p.per || []).find(l => isEngineLine(l) && l.engines.includes(desk.engine));
  if (line && (line as Record<string, unknown>)[field] !== undefined) at = `@${(line as { engines: string[] }).engines.join('+')}`;
  for (const l of p.per || []) if (!isEngineLine(l) && holds(ix, l.when, desk) && (l as Record<string, unknown>)[field] !== undefined) at = `@when-${hash(J(l.when))}`;
  return base + at;
}
/** The suffix of the engine line a parameter's field comes from on an engine, or "". */
const lineAt = (ix: Index, key: string, desk: Desk, field: string): string => {
  const line = (ix.params.get(key)!.p.per || []).find(l => isEngineLine(l) && l.engines.includes(desk.engine));
  return line && (line as Record<string, unknown>)[field] !== undefined ? `@${(line as { engines: string[] }).engines.join('+')}` : '';
};
/** A parameter's words on the desk, each a message with its key. */
export function paramWords(ix: Index, key: string, desk: Desk): { face: ParamFace; label: Msg; short?: Msg; undoName?: Msg; tip?: Msg } {
  const { p } = ix.params.get(key)!, face = wordsOf(ix, p, desk);
  const m = (field: 'label' | 'short' | 'undoName' | 'tip', text: string | undefined) => (text === undefined ? undefined : fillMsg(ix, wordKey(ix, key, desk, field), text, desk, face));
  return { face, label: m('label', face.label)!, ...(face.short ? { short: m('short', face.short)! } : {}), ...(face.undoName ? { undoName: m('undoName', face.undoName)! } : {}), ...(face.tip ? { tip: m('tip', face.tip)! } : {}) };
}
/** An option's label, or a format word, of a parameter on the desk, as a message. */
export const optionWord = (ix: Index, key: string, desk: Desk, value: string | number, text: string, from: 'options' | 'format'): Msg => {
  const { p, d } = ix.params.get(key)!, base = `create.${d.id}.p.${dashed(p.key)}`;
  return from === 'options' ? { id: `${base}.o.${keyPart(value)}.label${lineAt(ix, key, desk, 'options')}`, en: text } : { id: `${base}.w.${keyPart(value)}${lineAt(ix, key, desk, 'format')}`, en: text };
};
/** A state's noun and empty words. */
export function stateWords(ix: Index, key: string): { noun: Msg; empty?: Msg } {
  const { s, d } = ix.states.get(key)!, k = `create.${d.id}.s.${dashed(s.key)}`;
  return { noun: { id: `${k}.noun`, en: s.noun }, ...(s.empty ? { empty: { id: `${k}.empty`, en: s.empty } } : {}) };
}
/** The key of a parameter's or a state's named step. */
export const stepKey = (ix: Index, key: string, step: string): string => {
  const hp = ix.params.get(key);
  if (hp) return `create.${hp.d.id}.p.${dashed(key)}.step.${step}`;
  const hs = ix.states.get(key)!;
  return `create.${hs.d.id}.s.${dashed(key)}.step.${step}`;
};
/** A definition's text as a message: its template is the text, and every slot it names ({value:samp.guide},
 *  {count:adapters}, {engine}, {min} ...) is filled by the core on the desk (core.ts fill gives each one's words). */
export function fillMsg(ix: Index, id: string, text: string, desk: Desk, own?: ParamFace): Msg {
  const slots: { [name: string]: Slot } = {};
  for (const m of text.matchAll(new RegExp(SLOT.source, 'g'))) slots[m[1]!] = slotValue(ix, m[1]!, desk, own);
  return Object.keys(slots).length ? { id, en: text, slots } : { id, en: text };
}
/** The words of one slot of a definition's text, on the desk. */
function slotValue(ix: Index, slot: string, desk: Desk, own?: ParamFace): Slot {
  if (slot === 'engine') return engineName(desk.engine);
  if (own && (slot === 'min' || slot === 'max')) return own.range ? formatValue(own, own.range[slot === 'min' ? 0 : 1]) : `{${slot}}`;
  if (own && slot === 'default') return formatValue(own, own.default);
  const kind = slot.slice(0, slot.indexOf(':')), rest = slot.slice(slot.indexOf(':') + 1);
  if (kind === 'value') return ix.params.has(rest) && has(ix, rest, desk.engine) ? textMsg(ix, rest, desk) : '';
  if (kind === 'count') return countOf(ix, rest, desk);
  if (kind === 'say') return String(factOf(ix, rest, desk) ?? '');
  return fill(ix, `{${slot}}`, desk, own);
}
/** A value as it is written everywhere, as a message where it is words: an option's label, a format word, on or off. */
export function formatMsg(ix: Index, key: string, desk: Desk, v: Value): Slot {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  const word = (f.format?.words || []).find(w => w.value === v);
  if (word) return optionWord(ix, key, desk, String(v), word.text, 'format');
  if (typeof v === 'boolean') return msg(v ? 'value.on' : 'value.off');
  if (typeof v === 'string') {
    const fixed = Array.isArray(f.options) ? f.options : f.options?.include ?? [];
    const o = fixed.find(x => x.value === v);
    if (o) return optionWord(ix, key, desk, v, o.label, 'options');
  }
  return formatValue(f, v, desk.lists);
}
/** What a parameter says it is, as a message: its value written, or its auto words. */
export function textMsg(ix: Index, key: string, desk: Desk): Slot {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  if (isAuto(ix, key, desk)) return f.auto && f.auto.says ? fillMsg(ix, wordKey(ix, key, desk, 'auto'), f.auto.says, desk, f) : msg('value.about', { v: formatValue(f, f.default) });
  return formatMsg(ix, key, desk, valueOf(ix, key, desk)!);
}
