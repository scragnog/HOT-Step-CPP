// The toasts (36-toast.js): a short stack (F-12). #toast is the newest message; a message it replaces moves up into a
// copy of its own (.toast-old) that keeps the rest of its time and its action, so a new message never takes another
// one's Undo away early. At most three show; the oldest plain message goes first, and a plain message never pushes out
// one that carries an action (F-18). The same plain message twice in a row is said once, its time started again. Each
// lasts longer for a longer message, holds while hovered or focused, and can be dismissed; one with an action shows
// its time running out along its foot. While a toast with an action shows, the first Tab after it appears goes to its
// action, Shift+Tab goes back, and Tab from its last button carries on from where focus came from (C-13, F-11), but
// only while focus is still where it was when the toast appeared and no layer has opened since (S-17). Escape on a
// toast's button dismisses that toast; focus that was in a toast goes back where it came from (C-25). And the live
// region the page speaks through.

import { useEffect, useLayoutEffect, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { create } from 'zustand';
import { dockH, placeToast, watchToast } from './placeToast.ts';
import { useDesk, type Toast as ToastData } from '../store/desk.ts';
import { sayNow, ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';

/* a toast with an action lasts as long as it takes to read (about 60ms a character, 10 to 20 seconds, S-12); one
   without, 4 to 8 seconds */
const lifeOf = (t: ToastData) => { const n = sayNow(t.text).length; return t.action ? Math.min(20000, Math.max(10000, n * 60)) : Math.min(8000, Math.max(4000, 3500 + n * 30)); };
const LAYERS = '#pop:not([hidden]), #menu:not([hidden]), #dd:not([hidden]), #pickerWrap:not([hidden]), #notes:not([hidden]), #scoreModal:not([hidden]), #engPanel:not([hidden])';
const shownEl = (el: Element | null): el is HTMLElement => !!el && el.isConnected && (el as HTMLElement).getClientRects().length > 0;
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
/** The control after `from` in the page's Tab order, the toasts left out. */
function tabNext(from: Element): HTMLElement | null {
  const all = [...document.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(x => shownEl(x) && !x.closest('.toast') && !x.closest('[inert]'));
  const i = all.indexOf(from as HTMLElement); return i < 0 ? null : all[i + 1] ?? null;
}
const hostEl = () => document.getElementById('toast');

/** A timer that holds while its toast is pointed at or focused, keeping what it had left. */
class Life {
  left: number; at = 0; id = 0; end: () => void;
  constructor(ms: number, end: () => void) { this.left = ms; this.end = end; }
  arm() { clearTimeout(this.id); this.at = performance.now(); this.id = window.setTimeout(this.end, this.left); }
  hold(floor: number) { if (!this.id) return; clearTimeout(this.id); this.id = 0; this.left = Math.max(floor, this.left - (performance.now() - this.at)); }
  rest() { return this.id ? this.left - (performance.now() - this.at) : this.left; }
  stop() { clearTimeout(this.id); this.id = 0; }
}

/* ---------------------------------------------------------------- the stack, outside React: timers, focus, places */

interface Old { key: number; t: ToastData; el: HTMLElement; life: Life; shown: number }
const stack = create<{ olds: Old[]; life: number }>(() => ({ olds: [], life: 0 }));
let cur: { t: ToastData; life: Life } | null = null, keys = 0;
/* where focus was when the toast appeared (and whether it has claimed its Tab yet), and where focus came from */
const tab: { at: Element | null; tabbed: boolean; from: Element | null; layers: number } = { at: null, tabbed: true, from: null, layers: 0 };

function backFrom(el: Element | null): void {
  const f = tab.from; (shownEl(f) && !el?.contains(f) ? f : document.querySelector<HTMLElement>(LAYERS) ? null : document.getElementById('go'))?.focus({ preventScroll: true });
}
function hideCur(): void {
  const host = hostEl(), inside = !!host?.contains(document.activeElement); cur?.life.stop(); cur = null;
  useDesk.getState().showToast(null);
  /* at once, so an action run after it (Undo, a save row) can move focus on from there (36-toast.js hideToast) */
  if (inside) backFrom(host);
}
function dropOld(o: Old): void {
  o.life.stop(); const inside = o.el.contains(document.activeElement);
  stack.setState(s => ({ olds: s.olds.filter(x => x !== o) })); o.el.remove(); placeStack();
  if (inside) backFrom(o.el);
}
/** The message on #toast moves into a copy above it, with the time it had left. */
function demote(c: { t: ToastData; life: Life }, focused: boolean): void {
  const host = hostEl()!, el = document.createElement('div'); el.className = 'toast toast-old'; el.style.animation = 'none';
  host.before(el);
  const left = Math.max(1500, c.life.rest()), o: Old = { key: ++keys, t: c.t, el, life: new Life(left, () => dropOld(o)), shown: left };
  c.life.stop();
  el.addEventListener('mouseenter', () => o.life.hold(2000)); el.addEventListener('mouseleave', () => { if (!el.contains(document.activeElement)) o.life.arm(); });
  el.addEventListener('focusin', () => o.life.hold(2000)); el.addEventListener('focusout', e => { if (!el.contains(e.relatedTarget as Node)) o.life.arm(); });
  el.addEventListener('keydown', e => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); dropOld(o); } });
  let next = [...stack.getState().olds, o];
  /* at most three show: the oldest plain message goes first; one with an action goes only when all left carry one */
  while (next.length > 2) { const i = next.findIndex(x => !x.t.action), gone = next[i >= 0 ? i : 0]!; gone.life.stop(); gone.el.remove(); next = next.filter(x => x !== gone); }
  stack.setState({ olds: next });
  if (focused) requestAnimationFrame(() => (el.querySelector<HTMLElement>('.toast-act') ?? el.querySelector<HTMLElement>('button'))?.focus({ preventScroll: true }));
  else o.life.arm();
}
/** A new message on #toast (or none). */
function shown(t: ToastData | null, was: ToastData | null): void {
  const host = hostEl(); if (!host) return;
  host.hidden = !t;
  if (!t) { cur?.life.stop(); cur = null; return; }
  if (was && !t.action && !was.action && cur && sayNow(t.text) === sayNow(was.text)) {
    /* the same plain message twice in a row: said once, its time started again */
    cur.life.stop(); cur = { t, life: new Life(lifeOf(t), hideCur) };
  } else {
    if (was && cur) demote(cur, host.contains(document.activeElement));
    cur = { t, life: new Life(lifeOf(t), hideCur) };
    Object.assign(tab, { at: document.activeElement, tabbed: !t.action, layers: document.querySelectorAll(LAYERS).length });
    if (was) { host.style.animation = 'none'; void host.offsetWidth; host.style.animation = ''; }
  }
  stack.setState({ life: cur.life.left });
  if (!host.matches(':hover') && !host.contains(document.activeElement)) cur.life.arm();
}
useDesk.subscribe((s, p) => { if (s.toast !== p.toast) shown(s.toast, p.toast); });

/** The older toasts stand above the newest, 8px apart, each centred on the desk column as the newest is. */
function placeStack(): void {
  placeToast();
  const t = hostEl(), vw = document.documentElement.clientWidth, c = document.getElementById('centre')?.getBoundingClientRect();
  const g = document.querySelector('.gobar')?.getBoundingClientRect(), stuck = !!g && g.height > 0 && Math.abs(g.bottom - (innerHeight - dockH())) < 2;
  let y = t && !t.hidden ? innerHeight - t.getBoundingClientRect().top + 8 : stuck ? innerHeight - g!.top + 16 : 24 + dockH();
  for (const o of [...stack.getState().olds].reverse()) {
    const w = o.el.offsetWidth, cx = c && c.width ? c.left + c.width / 2 : vw / 2;
    o.el.style.right = 'auto'; o.el.style.left = `${Math.max(16, Math.min(vw - w - 16, cx - w / 2))}px`; o.el.style.bottom = `calc(${Math.round(y)}px + env(safe-area-inset-bottom, 0px))`;
    y += o.el.offsetHeight + 8;
  }
}

/* the newest holds while it is pointed at or focused; Escape on its button dismisses it */
function listen(host: HTMLElement): () => void {
  const on = () => cur?.life.hold(2000), off = () => { if (!host.hidden && !host.contains(document.activeElement) && !host.matches(':hover')) cur?.life.arm(); };
  const from = (e: FocusEvent) => { on(); if (e.relatedTarget && !host.contains(e.relatedTarget as Node)) tab.from = e.relatedTarget as Element; };
  const down = () => { const a = document.activeElement; if (a && a !== document.body && !host.contains(a)) tab.from = a; };
  const esc = (e: KeyboardEvent) => { if (e.key === 'Escape' && host.contains(document.activeElement)) { e.preventDefault(); e.stopPropagation(); hideCur(); } };
  /* Tab into a toast that carries an action, and out of it again (C-13, F-11, S-17) */
  const key = (e: KeyboardEvent) => {
    if (e.key !== 'Tab' || e.ctrlKey || e.altKey || e.metaKey || e.defaultPrevented || host.hidden) return;
    const a = host.querySelector<HTMLElement>('.toast-act'), act = document.activeElement; if (!a) return;
    if (!e.shiftKey && !tab.tabbed && act === tab.at && !host.contains(act) && document.querySelectorAll(LAYERS).length <= tab.layers && !document.querySelector('[aria-modal="true"]:not([hidden])')) {
      tab.tabbed = true; tab.from = act; e.preventDefault(); e.stopImmediatePropagation(); a.focus(); return;
    }
    if (e.shiftKey && act === a && shownEl(tab.from)) { e.preventDefault(); e.stopImmediatePropagation(); tab.from.focus(); return; }
    const f = [...host.querySelectorAll<HTMLElement>('button')], fromEl = shownEl(tab.from) ? tab.from : null;
    if (!e.shiftKey && f.length && act === f[f.length - 1] && fromEl) { const nx = tabNext(fromEl); if (nx) { e.preventDefault(); e.stopImmediatePropagation(); nx.focus(); } }
  };
  host.addEventListener('pointerenter', on); host.addEventListener('pointerleave', off); host.addEventListener('focusin', from); host.addEventListener('focusout', off); host.addEventListener('pointerdown', down); host.addEventListener('keydown', esc);
  document.addEventListener('keydown', key, true); addEventListener('resize', placeStack);
  return () => {
    host.removeEventListener('pointerenter', on); host.removeEventListener('pointerleave', off); host.removeEventListener('focusin', from); host.removeEventListener('focusout', off); host.removeEventListener('pointerdown', down); host.removeEventListener('keydown', esc);
    document.removeEventListener('keydown', key, true); removeEventListener('resize', placeStack);
  };
}

/* ---------------------------------------------------------------- drawing */

function Body({ t, life, close, act }: { t: ToastData; life: number; close: () => void; act: () => void }) {
  const say = useSay();
  return (
    <>
      <span className="toast-msg">{say(t.text)}</span>
      {t.action ? <button className="toast-act" type="button" onClick={act}>{say(t.action.label)}</button> : null}
      <button className="toast-x" type="button" aria-label={say(ui('toast.close'))} title={say(ui('toast.close.title'))} onClick={close}><Icon name="x" size={14} /></button>
      {t.action ? <i className="toast-life" aria-hidden="true" style={{ '--life': `${life}ms` } as CSSProperties} /> : null}
    </>
  );
}

export function Toast() {
  const t = useDesk(s => s.toast), olds = stack(s => s.olds), life = stack(s => s.life), host = hostEl();
  useLayoutEffect(() => { placeStack(); const f = requestAnimationFrame(placeStack); return () => cancelAnimationFrame(f); }, [t, olds]);
  useEffect(watchToast, []);
  useEffect(() => { const h = hostEl(); if (!h) return; const now = useDesk.getState().toast; if (now && !cur) shown(now, null); else h.hidden = !now; return listen(h); }, []);
  return (
    <>
      {olds.map(o => createPortal(<Body t={o.t} life={o.shown} close={() => dropOld(o)} act={() => { dropOld(o); o.t.action?.run(); }} />, o.el, String(o.key)))}
      {host && t ? createPortal(<Body t={t} life={life} close={hideCur} act={() => { const a = t.action!; hideCur(); a.run(); }} />, host) : null}
    </>
  );
}

export function LiveRegion() {
  const say = useSay(), said = useDesk(s => s.said), host = document.getElementById('say');
  if (!host) return null;
  return createPortal(<>{said ? said.text.map(x => say(x)).join(' ') : ''}</>, host);
}
