// Each text field's own history (F-22, the concept's 19-field-undo.js): one list of the texts a field has shown, kept by
// the page, so a field's Ctrl+Z and the page's Undo can never disagree. The browser's own history is not used: it is one
// list for the whole page, so it could undo a change in a field the user is not in, and it knows nothing of text the
// page put back.
//
// Typing adds a step (a run of keys within a second, up to a word, is one step); anything the page writes into a field
// (Write it for me, the page's Undo, a scene) is a step of its own; before every change and every Ctrl+Z the field's
// text now is compared with the last step and written down first if it differs. So any mix of typing, the field's
// Ctrl+Z and the page's Undo only ever goes back through texts the user saw. A field bound to a parameter is known by
// it, so one that is drawn again (the lyric sheet on another tier) keeps its history.
//
// A field's own Ctrl+Z is not a new step of the page's: it moves the page's step that wrote the text the field showed
// to the text it shows now, and that step goes when it is back where it started (F-06). So Ctrl+Z in a field, then
// Ctrl+Z outside it, goes on back and never puts the words back. features/field-undo.ts wires this to the page.

import { applyChange, setValue, stepWords, valueOf } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from './desk.ts';

interface Shown { v: string; s: number | null; e: number | null }
interface History { st: Shown[]; i: number; t: number; kind: string }

const BY_KEY = new Map<string, History>();
const BY_EL = new WeakMap<Element, History>();
const MAX = 200;
/** A run of typing is one step while its keys come within this long of each other. */
const RUN_MS = 1000;

/* ---------------------------------------------------------------- which field is which */

type Field = HTMLInputElement | HTMLTextAreaElement;
/** Fields whose element has an id but no data-bind, and the parameter each edits (the prompt card's box). */
const BY_ID = new Map<string, string>();
export const registerField = (id: string, param: string): void => { BY_ID.set(id, param); };
/** The text parameter a field edits, or null (a filter, a number box). */
export function paramOf(el: Element): string | null {
  const k = (el as HTMLElement).dataset?.bind ?? (el.id ? BY_ID.get(el.id) : undefined);
  return k && ix.params.get(k)?.p.type === 'text' ? k : null;
}
/** The field on the page that edits a parameter. */
export function fieldOf(param: string): Field | null {
  for (const [id, k] of BY_ID) if (k === param) { const el = document.getElementById(id); if (el) return el as Field; }
  return document.querySelector<Field>(`textarea[data-bind="${param}"], input[data-bind="${param}"]`);
}
const keyOf = (el: Element) => { const p = paramOf(el); return p ? 'b:' + p : el.id ? '#' + el.id : null; };
/** A field that keeps a history: a text area or a text input that can be typed in. */
export function isField(el: EventTarget | null): el is Field {
  if (!(el instanceof HTMLTextAreaElement) && !(el instanceof HTMLInputElement)) return false;
  if (el instanceof HTMLInputElement && /^(checkbox|radio|button|submit|range|file|color|hidden|image|reset)$/i.test(el.type)) return false;
  return !el.readOnly && !el.disabled;
}

/* ---------------------------------------------------------------- the histories */

function historyOf(el: Field): History {
  const k = keyOf(el); let h = k ? BY_KEY.get(k) : BY_EL.get(el);
  if (!h) { h = { st: [{ v: el.value, s: el.selectionStart, e: el.selectionEnd }], i: 0, t: 0, kind: '' }; if (k) BY_KEY.set(k, h); else BY_EL.set(el, h); }
  return h;
}
/** A field's history by the parameter it edits, started at `was` when it has none yet (the page wrote into a field
 *  no one had touched). */
function historyOfParam(param: string, was: string): History {
  const k = 'b:' + param; let h = BY_KEY.get(k);
  if (!h) { h = { st: [{ v: was, s: was.length, e: was.length }], i: 0, t: 0, kind: '' }; BY_KEY.set(k, h); }
  return h;
}
function add(h: History, x: Shown, kind: string): void {
  const now = performance.now(), top = h.st[h.i]!;
  if (top.v === x.v) { top.s = x.s; top.e = x.e; return; }
  h.st.length = h.i + 1;
  /* a run of typing is one step, broken by a pause, a change of kind, or the space after a word */
  const brk = kind === 'type' && /\s$/.test(x.v.slice(0, x.s ?? 0)) && !/\s$/.test(top.v.slice(0, top.s ?? 0));
  if (kind && kind === h.kind && now - h.t < RUN_MS && h.i > 0 && !brk) h.st[h.i] = x;
  else { h.st.push(x); if (h.st.length > MAX) h.st.shift(); h.i = h.st.length - 1; }
  h.kind = kind; h.t = now;
}
const shown = (el: Field): Shown => ({ v: el.value, s: el.selectionStart, e: el.selectionEnd });

/** The field's text was changed by typing (`kind` 'type' or 'del') or otherwise (''). */
export function pushField(el: Field, kind: string): void { add(historyOf(el), shown(el), kind); }
/** The field's text now, written down if something changed it without telling (a value set directly). */
export function syncField(el: Field): void {
  const h = historyOf(el); if (h.st[h.i]!.v === el.value) return;
  /* a field bound to nothing (a number box's typing, a filter) that the page drew again starts afresh, as a field drawn
     anew does: its old typing is not what it shows now */
  if (!keyOf(el)) { h.st = [shown(el)]; h.i = 0; h.kind = ''; return; }
  h.kind = ''; add(h, shown(el), '');
}
/** Whether the field has a step to go back (or forward) to. */
export function canStep(el: Field, redo: boolean): boolean {
  const h = historyOf(el), behind = h.st[h.i]!.v !== el.value;
  return redo ? !behind && h.i < h.st.length - 1 : behind || h.i > 0;
}

let applying = false;
/** True while a field is stepping through its own history: its input is not typing. */
export const stepping = (): boolean => applying;
/** The page wrote into a field it edits a parameter of (anything but typing in it): a step of the field's own. */
export function pageWrote(param: string, was: string, now: string): void {
  if (applying) return;
  const el = fieldOf(param);
  if (el && (el.value === now || el.readOnly)) return;
  const h = el ? historyOf(el) : historyOfParam(param, was);
  if (el && h.st[h.i]!.v !== el.value) add(h, shown(el), '');
  else if (!el && h.st[h.i]!.v !== was) add(h, { v: was, s: was.length, e: was.length }, '');
  h.kind = ''; add(h, { v: now, s: now.length, e: now.length }, '');
}

/* the value setter of the element's own kind, so a framework that tracks the value sees the change as typing */
const setNative = (el: Field, v: string) => Object.getOwnPropertyDescriptor(el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value')!.set!.call(el, v);

/** The field's Ctrl+Z (or, `redo`, Ctrl+Shift+Z): its text goes one step back (or forward); false when it has none. */
export function stepField(el: Field, redo: boolean): boolean {
  syncField(el);
  const h = historyOf(el), j = h.i + (redo ? 1 : -1);
  if (j < 0 || j >= h.st.length) return false;
  h.i = j; h.kind = '';
  const x = h.st[j]!, param = paramOf(el);
  applying = true;
  try {
    setNative(el, x.v);
    try { el.setSelectionRange(Math.min(x.s ?? x.v.length, x.v.length), Math.min(x.e ?? x.v.length, x.v.length)); } catch { /* an input that has no selection */ }
    /* a field bound to a value: the desk takes the text first; then the field is told, as the browser tells it of its
       own undo, so a part that keeps a draft (a checked field) holds the same text */
    if (param) retext(param, x.v);
    el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: redo ? 'historyRedo' : 'historyUndo' }));
  } finally { applying = false; }
  return true;
}

/* ---------------------------------------------------------------- the page's step that wrote the text */

/** A field's own step moves the page's newest step that wrote the text the field showed to the text it shows now
 *  (the step goes when that is where it started); when no step wrote that text, it is a step of its own. */
function retext(param: string, text: string): void {
  const s = useDesk.getState(), desk = s.desk;
  if (valueOf(ix, param, desk) === text) return;
  let v: string;
  try { v = String(setValue(ix, desk, param, text).value); } catch { return; }
  const path = 'v:' + param, was = param in desk.vals ? JSON.stringify(desk.vals[param]) : undefined, now = JSON.stringify(v);
  const next = applyChange(desk, { vals: { [param]: v }, auto: {}, state: {}, words: { label: { id: '' }, put: { id: '' } } });
  for (let i = s.undo.length - 1; i >= 0; i--) {
    const e = s.undo[i]!; if (e.before.get(path) === e.after.get(path)) continue;
    if (e.after.get(path) !== was) break;
    const after = new Map(e.after); after.set(path, now);
    /* it is gone when every path is back where it started; a text back at its default is where an absent one starts */
    const def = JSON.stringify(valueOf(ix, param, { ...desk, vals: {} }));
    const same = (p: string, a: string | undefined, b: string | undefined) => a === b || (p === path && (a ?? def) === (b ?? def));
    const gone = [...new Set([...after.keys(), ...e.before.keys()])].every(p => same(p, after.get(p), e.before.get(p)));
    s.amendStep(e, gone ? null : after, next);
    return;
  }
  const words = stepWords(ix, desk, next); if (!words) return;
  s.commit({ vals: { [param]: v }, auto: {}, state: {}, words });
}

/* what the page writes into a field it edits (a tool, the page's Undo, a scene): a step of the field's own */
const texts = (vals: Record<string, unknown>) => Object.keys(vals).filter(k => ix.params.get(k)?.p.type === 'text');
useDesk.subscribe((s, prev) => {
  if (s.desk === prev.desk || applying) return;
  for (const k of new Set([...texts(s.desk.vals), ...texts(prev.desk.vals)])) {
    const a = valueOf(ix, k, prev.desk), b = valueOf(ix, k, s.desk);
    if (a !== b && typeof b === 'string') pageWrote(k, typeof a === 'string' ? a : '', b);
  }
});
