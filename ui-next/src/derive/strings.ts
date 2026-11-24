// Translation keys (docs/05, "Strings and translation"): every string a definition holds, with the key it is looked up
// under. A key is made from where the string sits and what it is bound to, never from its place in a list, so
// reordering lines, options or nodes keeps every translation; and no two strings share a key (the test checks it).

import type { Action, Definition, Node, NodeOverride, ParamDef, Strip, Value } from '../definitions/types.ts';
import { CHECKS, PARTS, UNITS, type PartSpec } from '../definitions/parts.ts';
import { EN } from './messages.ts';
import { ENGINES } from '../definitions/engines.ts';

export interface Str { key: string; text: string }

/** The core's own messages (messages.ts EN) and the parts' how-to words: the strings no definition holds. */
export function coreStrings(): Str[] {
  return [...Object.entries(EN).map(([key, text]) => ({ key, text })),
    ...Object.entries(PARTS as Record<string, PartSpec>).flatMap(([name, s]) => (s.howTo ? [{ key: `part.${name}.howTo`, text: s.howTo }] : [])),
    ...ENGINES.flatMap(e => e.tagline.map((t, i) => ({ key: `engine.${e.id}.tag${i + 1}`, text: t })))];
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
    add(`${base}.quickName`, d.quickName); add(`${base}.starter`, d.starter?.noun);
    d.tierSummary?.forEach((t, i) => add(`${base}.tier.${['quick', 'studio', 'lab'][i]}`, t));
    for (const l of d.per || []) {
      const at = `@${l.engines.join('+')}`;
      add(`${base}.name${at}`, l.name); add(`${base}.quickName${at}`, l.quickName); add(`${base}.desc${at}`, l.desc); add(`${base}.starter${at}`, l.starter?.noun);
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
      if (f.auto) { add(`${k}.auto${at}`, f.auto.says); add(`${k}.autoTip${at}`, f.auto.tip); }
      (f.forced || []).forEach((x, i) => { add(`${k}.forced.${i}.says${at}`, x.says); add(`${k}.forced.${i}.tip${at}`, x.tip); });
      if (f.format) { add(`${k}.template${at}`, f.format.template); (f.format.words || []).forEach(w => add(`${k}.w.${keyPart(w.value)}${at}`, w.text)); }
      if (Array.isArray(f.options)) for (const o of f.options) { const ok = `${k}.o.${keyPart(o.value)}`; add(`${ok}.label${at}`, o.label); add(`${ok}.short${at}`, o.short); add(`${ok}.tip${at}`, o.tip); add(`${ok}.lcd${at}`, o.lcd); add(`${ok}.pick${at}`, o.pick); add(`${ok}.detail${at}`, o.detail); }
      else for (const o of f.options?.include || []) { const ok = `${k}.o.${keyPart(o.value)}`; add(`${ok}.label${at}`, o.label); add(`${ok}.tip${at}`, o.tip); add(`${ok}.lcd${at}`, o.lcd); add(`${ok}.pick${at}`, o.pick); }
    };
    face(p, '');
    if (p.held) { add(`${k}.held.note`, p.held.note); add(`${k}.held.toast`, p.held.toast); add(`${k}.held.value`, p.held.value); }
    for (const [n, st] of Object.entries(p.steps || {})) add(`${k}.step.${n}`, typeof st === 'string' ? st : st.says);
    /* an engine line by its engines, a word line by its condition: neither moves when lines are reordered */
    for (const l of p.per || []) face(l as Partial<ParamDef>, 'engines' in l && l.engines ? `@${l.engines.join('+')}` : `@when-${hash(J(l.when))}`);
  }
  function actions(k: string, list: Action[] | undefined) {
    for (const a of list || []) {
      add(actionKey(k, a, 'label'), a.label); if ('tip' in a) add(actionKey(k, a, 'tip'), a.tip);
    }
  }
  function strip(k: string, s: Strip) {
    const at = stripKey(k, s);
    add(`${at}.says`, s.says); actions(at, s.actions);
  }
  function node(k: string, n: Node, seen: Map<string, number>) {
    const at = nodeKeyAt(k, n, seen);
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

/** The key of an action's words under its owner's key: by what it does and to what, never by its place. */
export const actionKey = (k: string, a: Action, field: 'label' | 'tip'): string =>
  `${k}.a.${a.do}-${keyPart('param' in a ? a.param : 'tier' in a ? a.tier : 'item' in a ? a.item : 'tag' in a ? a.tag : 'name' in a ? a.name : '')}${'value' in a ? '-' + keyPart(String(a.value)) : ''}.${field}`;
/** The key of a strip or an idle reason, under `create.<id>.strip` or `create.<id>.idle`: by its condition. */
export const stripKey = (k: string, s: Strip): string => `${k}.${hash(J([s.when, s.tiers]))}`;

/** A node by what it is bound to (its part and its keys), or by its words when it is bound to nothing. */
function idOf(n: Node): string {
  const binds = [n.param, ...(n.params || []), ...(n.reads || []), ...(n.state || [])].filter(Boolean).join('+');
  return `${n.part}.${binds ? keyPart(binds) : keyPart((n.text ?? n.label ?? n.part).slice(0, 32))}`;
}
/** How many siblings share each kind-and-binding, so all of a shared one are told apart by a hash of themselves. */
function counts(list: Node[]): Map<string, number> { const m = new Map<string, number>(); list.forEach(n => m.set(idOf(n), (m.get(idOf(n)) ?? 0) + 1)); return m; }
/** A node's key under its parent's: a second node of the same kind and binding adds a hash of itself. */
function nodeKeyAt(k: string, n: Node, seen: Map<string, number>): string {
  let id = idOf(n);
  if ((seen.get(id) ?? 0) > 1) id += `~${hash(J({ ...n, children: undefined, per: undefined }))}`;
  return `${k}.${id}`;
}

/** The keys of a node's children under the node's key, in their order (as nodeKeys() gives them). */
export function childKeys(key: string, children: readonly Node[]): string[] { const seen = counts([...children]); return children.map(c => nodeKeyAt(key, c, seen)); }

/** The translation key of every node of a definition's face, as stringsOf() keys its words: the view looks a node's
 *  words up under it (nodeWord), and passes it to dimOf() and choiceOf(). */
export function nodeKeys(d: Definition): Map<Node, string> {
  const out = new Map<Node, string>();
  const walk = (k: string, list: Node[]) => { const seen = counts(list); for (const n of list) { const at = nodeKeyAt(k, n, seen); out.set(n, at); walk(at, n.children || []); } };
  walk(`create.${d.id}.n`, 'face' in d ? d.face || [] : []);
  return out;
}

/* ---------------------------------------------------------------- a definition's words as messages
   What the core says with a definition's words carries their translation key, so a page in another locale says them
   in its language (messages.ts). The key is the one stringsOf() gives the line the words come from on the desk. */

import type { Desk, Index } from './core.ts';
import { choiceLists, countOf, factOf, fill, formatValue, has, holds, isAuto, isEngineLine, modelDefault, resolve, valueOf, wordsOf, SLOT } from './core.ts';
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
/** A node's own words on the desk (its label, tip, text or between), as a message keyed as stringsOf() keys it: the
 *  key of the line its words come from when a line that holds gives them, else the node's own. Undefined when it has
 *  none. `key` is the node's key (nodeKeys). */
export function nodeWord(ix: Index, n: Node, desk: Desk, key: string, field: 'label' | 'tip' | 'text' | 'between'): Msg | undefined {
  let text = n[field] as string | undefined, sfx = '';
  for (const l of (n.per || []) as NodeOverride[]) {
    if (l.engines && !l.engines.includes(desk.engine)) continue;
    if (l.tiers && !l.tiers.includes(desk.tier)) continue;
    if (l.when && !holds(ix, l.when, desk)) continue;
    if (l[field] !== undefined) { text = l[field] as string; sfx = `@${hash(J([l.engines, l.tiers, l.when]))}`; }
  }
  return text === undefined || text === '' ? undefined : fillMsg(ix, `${key}.${field}${sfx}`, text, desk);
}
/** A node's option that is words (an XY pad's axis words, a note's prefix), keyed as stringsOf() keys it. */
export function nodeOptWord(ix: Index, n: Node, desk: Desk, key: string, opt: string, i?: number): Msg | undefined {
  let v = (n.opts as Record<string, unknown> | undefined)?.[opt], sfx = '';
  for (const l of (n.per || []) as (NodeOverride & { opts?: Record<string, unknown> })[]) {
    if (l.engines && !l.engines.includes(desk.engine)) continue;
    if (l.tiers && !l.tiers.includes(desk.tier)) continue;
    if (l.when && !holds(ix, l.when, desk)) continue;
    if (l.opts?.[opt] !== undefined) { v = l.opts[opt]; sfx = `@${hash(J([l.engines, l.tiers, l.when]))}`; }
  }
  const text = i === undefined ? v : Array.isArray(v) ? v[i] : undefined;
  return typeof text === 'string' && text ? fillMsg(ix, `${key}.opt.${opt}${i === undefined ? '' : '.' + i}${sfx}`, text, desk) : undefined;
}

/** A value as it is written everywhere, as a message where it is words: an option's label, a format word, on or off. */
export function formatMsg(ix: Index, key: string, desk: Desk, v: Value): Slot {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  const word = (f.format?.words || []).find(w => w.value === v);
  if (word) return optionWord(ix, key, desk, String(v), word.text, 'format');
  if (typeof v === 'boolean') return msg(v ? 'value.on' : 'value.off');
  /* a model choice left at "" is written as the file it stands for, as the request sends it */
  const stands = v === '' ? modelDefault(ix, key, desk) : undefined;
  if (stands !== undefined) return formatValue(f, stands, desk.lists, desk.engine);
  if (typeof v === 'string') {
    const fixed = Array.isArray(f.options) ? f.options : f.options?.include ?? [];
    const o = fixed.find(x => x.value === v);
    if (o) return optionWord(ix, key, desk, v, o.label, 'options');
  }
  return formatValue(f, v, desk.lists, desk.engine);
}
/** What a parameter says it is, as a message: its value written, or its auto words. */
export function textMsg(ix: Index, key: string, desk: Desk): Slot {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  if (isAuto(ix, key, desk)) return f.auto && f.auto.says ? fillMsg(ix, wordKey(ix, key, desk, 'auto'), f.auto.says, desk, f) : msg('value.about', { v: formatValue(f, f.default) });
  return formatMsg(ix, key, desk, valueOf(ix, key, desk)!);
}
/** A number written without its unit, and the unit, for a read-out that prints the unit on its own line under the
 *  number (a pedal knob's dB, Hz, LUFS; "%" stays on the number). Null when the parameter has no such unit. */
export function bareText(ix: Index, key: string, desk: Desk): { text: Slot; unit: string } | null {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  if (!f.unit || f.unit === '%') return null;
  const t = textMsg(ix, key, desk);
  return { text: typeof t === 'string' && t.endsWith(' ' + f.unit) ? t.slice(0, -(f.unit.length + 1)) : t, unit: f.unit };
}
/** What an LCD prints for a parameter: a choice's `lcd` words when its option has them, else the value as it is
 *  written everywhere (its option's label, a number with its unit). */
export function lcdMsg(ix: Index, key: string, desk: Desk): Slot {
  const { p, d } = ix.params.get(key)!, f = wordsOf(ix, p, desk), v = valueOf(ix, key, desk);
  if (p.type === 'choice' && v !== undefined) {
    const o = (Array.isArray(f.options) ? f.options : f.options?.include ?? []).find(x => x.value === v);
    if (o?.lcd) return { id: `create.${d.id}.p.${dashed(p.key)}.o.${keyPart(o.value)}.lcd${lineAt(ix, key, desk, 'options')}`, en: o.lcd };
  }
  return textMsg(ix, key, desk);
}
/** What an LCD picker prints for its choice: the option's `pick` words, else what an LCD prints (lcdMsg). */
export function pickMsg(ix: Index, key: string, desk: Desk): Slot {
  const { p, d } = ix.params.get(key)!, f = wordsOf(ix, p, desk), v = valueOf(ix, key, desk);
  const o = p.type === 'choice' ? (Array.isArray(f.options) ? f.options : f.options?.include ?? []).find(x => x.value === v) : undefined;
  return o?.pick ? { id: `create.${d.id}.p.${dashed(p.key)}.o.${keyPart(o.value)}.pick${lineAt(ix, key, desk, 'options')}`, en: o.pick } : lcdMsg(ix, key, desk);
}
/** A choice's options on the desk's engine, each with its words as messages: the fixed list, or what the parameter
 *  includes and then the server's list for its source (desk.lists), without a value twice. */
export function optionsOf(ix: Index, key: string, desk: Desk): { value: string | number; label: Slot; short?: Msg; tip?: Msg; lcd?: Msg; pick?: Msg; detail?: Msg }[] {
  const { p, d } = ix.params.get(key)!, f = wordsOf(ix, p, desk), base = `create.${d.id}.p.${dashed(p.key)}.o`, at = lineAt(ix, key, desk, 'options');
  /* the server's list, without what the definition lists itself, and only the values it names (`only`) */
  const { fixed, listed } = choiceLists(f, desk);
  return [
    ...fixed.map(o => ({ value: o.value, label: { id: `${base}.${keyPart(o.value)}.label${at}`, en: o.label } as Msg,
      ...('short' in o && o.short ? { short: { id: `${base}.${keyPart(o.value)}.short${at}`, en: o.short } as Msg } : {}),
      ...(o.tip ? { tip: { id: `${base}.${keyPart(o.value)}.tip${at}`, en: o.tip } as Msg } : {}),
      ...(o.lcd ? { lcd: { id: `${base}.${keyPart(o.value)}.lcd${at}`, en: o.lcd } as Msg } : {}),
      ...(o.pick ? { pick: { id: `${base}.${keyPart(o.value)}.pick${at}`, en: o.pick } as Msg } : {}),
      ...('detail' in o && o.detail ? { detail: { id: `${base}.${keyPart(o.value)}.detail${at}`, en: o.detail } as Msg } : {}) })),
    ...listed.map(o => ({ value: o.value, label: o.label as Slot })),
  ];
}
/** Whether typed text passes its parameter's check on the desk's engine: null when it does (or there is none), else
 *  the check's message, which the field shows beside it while the text is kept unsent. */
export function checkText(ix: Index, key: string, desk: Desk, text: string): Msg | null {
  const { p, d } = ix.params.get(key)!, f = wordsOf(ix, p, desk);
  if (!f.check || CHECKS[f.check.rule].test(text)) return null;
  return { id: `create.${d.id}.p.${dashed(p.key)}.check${lineAt(ix, key, desk, 'check')}`, en: f.check.message };
}
/** What a field that keeps invalid text (its check's keepInvalid) says is in force meanwhile: what its words name
 *  ("Still using seed 12."), or, while nothing valid is set, that nothing is applied; `back` when there is a value to
 *  offer back (12-values.js keptMsg). Undefined for a field that does not keep. */
export function keptMsg(ix: Index, key: string, desk: Desk): { says: Msg; back: boolean } | undefined {
  const { p, d } = ix.params.get(key)!, f = wordsOf(ix, p, desk), v = valueOf(ix, key, desk);
  if (!f.check?.keepInvalid) return undefined;
  if (v === '' || v === undefined) return { says: msg('field.keptNothing'), back: false };
  return { says: msg('field.kept', { what: fillMsg(ix, `create.${d.id}.p.${dashed(p.key)}.keepInvalid${lineAt(ix, key, desk, 'check')}`, f.check.keepInvalid, desk) }), back: true };
}
/** A strip's (or an idle reason's) words and its actions' words, as messages, under its definition's key. */
export function stripWords(ix: Index, base: string, st: Strip, desk: Desk): { says: Msg; actions: { action: Action; label: Msg; tip?: Msg }[] } {
  const k = stripKey(base, st);
  return { says: fillMsg(ix, `${k}.says`, st.says, desk), actions: (st.actions ?? []).map(a => ({ action: a, label: fillMsg(ix, actionKey(k, a, 'label'), a.label, desk), ...('tip' in a && a.tip ? { tip: fillMsg(ix, actionKey(k, a, 'tip'), a.tip, desk) } : {}) })) };
}
/** A node's actions' words, as messages: an action a node's line gives is keyed under that line. */
export function actionWords(ix: Index, n: Node, desk: Desk, key: string): { action: Action; label: Msg; tip?: Msg }[] {
  let list = n.actions ?? [], k = key;
  for (const l of (n.per || []) as NodeOverride[]) {
    if (l.engines && !l.engines.includes(desk.engine)) continue;
    if (l.tiers && !l.tiers.includes(desk.tier)) continue;
    if (l.when && !holds(ix, l.when, desk)) continue;
    if (l.actions) { list = l.actions; k = `${key}@${hash(J([l.engines, l.tiers, l.when]))}`; }
  }
  return list.map(a => ({ action: a, label: fillMsg(ix, actionKey(k, a, 'label'), a.label, desk), ...('tip' in a && a.tip ? { tip: fillMsg(ix, actionKey(k, a, 'tip'), a.tip, desk) } : {}) }));
}

/** An LCD's words as printed: a number and its unit apart, the unit in small type after a thin space ("-14 LUFS"),
 *  else the words whole. The unit is one of the units the definitions use (parts.ts UNITS), which no locale
 *  translates; the number may be written with any digits and either decimal mark. */
export function splitUnit(text: string): { main: string; unit: string | null } {
  const unit = Object.keys(UNITS).find(u => u.length > 1 && text.toLocaleUpperCase().endsWith(' ' + u.toLocaleUpperCase()) && /^[−-]?[\p{Nd}.,]+$/u.test(text.slice(0, -u.length - 1)));
  return unit ? { main: text.slice(0, -unit.length - 1), unit } : { main: text, unit: null };
}

/** A choice the engine runs at another value while one of its `forced` lines holds (types.ts ParamFace.forced): that
 *  value, why the others are disabled, and what is said beside them (with what returns after, when the choice differs).
 *  Null while none holds. */
export function forcedOf(ix: Index, key: string, desk: Desk): { value: Value; tip: Msg; note: Msg } | null {
  if (!has(ix, key, desk.engine)) return null;
  const { p, d } = ix.params.get(key)!, f = resolve(p, desk.engine), at = (p.per || []).some(l => 'engines' in l && l.engines?.includes(desk.engine) && (l as Partial<ParamDef>).forced) ? `@${((p.per || []).find(l => 'engines' in l && l.engines?.includes(desk.engine)) as { engines: string[] }).engines.join('+')}` : '';
  const i = (f.forced || []).findIndex(x => holds(ix, x.when, desk));
  if (i < 0) return null;
  const x = f.forced![i]!, k = `create.${d.id}.p.${dashed(key)}.forced.${i}`, v = valueOf(ix, key, desk)!;
  const says = fillMsg(ix, `${k}.says${at}`, x.says, desk), tip = fillMsg(ix, `${k}.tip${at}`, x.tip, desk);
  return { value: x.value, tip, note: v === x.value ? says : msg('forced.returns', { says, value: formatMsg(ix, key, desk, v) }) };
}
