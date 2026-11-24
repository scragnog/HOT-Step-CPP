// The module picker's keys and its opening and closing (the concept's 33-picker.js; its view is picker.tsx): F9 and
// every "Open the picker" button open it with focus in its filter; Escape clears the filter, then closes it, focus
// going back to what opened it; F9 closes it and, while it is closing, opens it again (F-29). While it is open its keys
// come first: Tab goes round it (and a toast that shows joins the round, F-18), a letter types into the filter, the
// arrows walk the wall by where the drawings stand and stay in their column going up and down (F-29), Home and End,
// Enter or Space adds, and Ctrl+Enter generates nothing.

import { flushSync } from 'react-dom';
import { defOf } from '../model/index.ts';
import { onKey } from '../register/keys.ts';
import { useDesk } from '../store/desk.ts';
import { usePicker, type PickerInfo } from '../store/overlays.ts';
import { usePanelStore, drawerMode } from '../store/panels.ts';
import { ring } from '../desk/ring.ts';
import { focusables, layerClosed, layerOpen, topLayer } from './layers.ts';
import { closePop } from './pop.ts';
import { sayParked } from './adding.ts';
import { reduced } from '../kit/motion.ts';

const host = () => document.getElementById('pickerWrap');
/** The definition a picker id names (a pedal is pd:<id>, a side unit ex:<id>). */
export const defOfItem = (id: string) => defOf(id.includes(':') ? id.split(':')[1]! : id);

/* ---------------------------------------------------------------- opening and closing */

let back: HTMLElement | null = null, closeT = 0;

export function openPicker(): void {
  clearTimeout(closeT);
  const s = usePicker.getState();
  if (!s.open || s.closing) back = (s.open ? back : document.activeElement as HTMLElement | null);
  flushSync(() => usePicker.setState({ open: true, closing: false, q: '', info: { kind: 'hint' }, peek: false }));
  const el = host(); if (!el) return;
  layerOpen(el, { trigger: back, modal: true, close: () => closePicker() });
  document.getElementById('pkSearch')?.focus({ preventScroll: true });
}
/** Closes the picker after its exit; `refocus`: focus goes back to what opened it. */
export function closePicker(refocus = true): void {
  const s = usePicker.getState(); if (!s.open || s.closing) return;
  const done = () => {
    flushSync(() => usePicker.setState({ open: false, closing: false, peek: false }));
    const el = host(); if (el) layerClosed(el, false);
    if (refocus && back?.isConnected) back.focus({ preventScroll: true });
    back = null;
  };
  if (reduced()) { done(); return; }
  usePicker.setState({ closing: true });
  closeT = window.setTimeout(done, 160);
}
export const pickerOpen = (): boolean => usePicker.getState().open && !usePicker.getState().closing;

/* ---------------------------------------------------------------- what a press on a drawing does */

export function activate(id: string): void {
  const s = useDesk.getState(), d = defOfItem(id); if (!d) return;
  const r = s.add(d.id);
  if (r.did === 'lab') {
    flushSync(() => usePicker.setState({ info: { kind: 'item', id } }));
    document.querySelector<HTMLElement>('#pkInfo [data-labadd]')?.focus();
    return;
  }
  if (r.did === 'there') { closePicker(); ring(d.id, d.kind === 'module' ? 'flash' : 'here'); return; }
  sayParked(d.id, r.step);
  /* the keys stay on it, and then the line says what arrived (its focus would say what it is) */
  flushSync(() => usePicker.setState({ rove: id }));
  document.querySelector<HTMLElement>(`#pickerWrap .fl-it[data-pkid="${CSS.escape(id)}"]`)?.focus({ preventScroll: true });
  usePicker.setState({ info: { kind: 'added', id } });
}

/** The drawings shown, in the order the keys walk them. */
const visible = () => [...document.querySelectorAll<HTMLElement>('#pickerWrap .fl-it')].filter(el => !el.classList.contains('gone') && !el.classList.contains('nomatch'))
  .sort((a, b) => (+a.style.order || 0) - (+b.style.order || 0));
const roveTo = (el: HTMLElement | undefined) => { if (!el) return; usePicker.setState({ rove: el.dataset.pkid! }); el.focus(); el.scrollIntoView({ block: 'nearest' }); };

/* the arrows go to the nearest drawing that way; up and down stay in their own column and stop at its foot (F-29) */
function move(cur: HTMLElement, dx: number, dy: number): void {
  const r0 = cur.getBoundingClientRect(), cx = r0.left + r0.width / 2, cy = r0.top + r0.height / 2, wall = cur.closest('.fl-wall');
  let best: HTMLElement | undefined, bs = Infinity;
  visible().forEach(el => {
    if (el === cur || (dy && el.closest('.fl-wall') !== wall)) return;
    const r = el.getBoundingClientRect(), x = r.left + r.width / 2 - cx, y = r.top + r.height / 2 - cy;
    const primary = dx ? x * dx : y * dy, secondary = dx ? Math.abs(y) : Math.abs(x); if (primary <= 4) return;
    const sc = primary + secondary * 2.5; if (sc < bs) { bs = sc; best = el; }
  });
  roveTo(best);
}

/* while a filter is typed, the info line speaks for the drawing Enter will take (C-33) and the keys start there */
export const setQuery = (q: string) => usePicker.setState({ q, ...(q.trim() ? { info: { kind: 'first' } as PickerInfo, rove: null } : {}) });

/* the picker's keys, before anything else on the page while it is open */
document.addEventListener('keydown', e => {
  if (!pickerOpen() || e.key === 'F9') return;
  const search = document.getElementById('pkSearch') as HTMLInputElement | null; if (!search) return;
  const inSearch = e.target === search, stop = () => { e.preventDefault(); e.stopPropagation(); };
  if (e.key === 'Escape') { stop(); if (search.value) { flushSync(() => setQuery('')); search.focus(); } else closePicker(); return; }
  if (e.key === 'Tab') {
    stop();
    /* a toast raised while the picker is open joins its cycle while it shows (F-18) */
    const toast = document.getElementById('toast'), f = [search, ...[...document.querySelectorAll<HTMLElement>('#pickerWrap .fl-it[tabindex="0"]')].filter(x => x.offsetParent),
      ...document.querySelectorAll<HTMLElement>('#pkInfo button'), ...document.querySelectorAll<HTMLElement>('#pkBar button'), document.getElementById('pkClose')!,
      ...(toast && !toast.hidden ? focusables(toast) : [])];
    const i = f.indexOf(document.activeElement as HTMLElement); f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length]!.focus();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { stop(); return; }
  if (!inSearch && e.key.length === 1 && e.key !== ' ' && !e.ctrlKey && !e.metaKey && !e.altKey && !(e.target as Element).closest?.('#toast')) {
    stop(); search.focus(); flushSync(() => setQuery(search.value + e.key)); return;
  }
  if (inSearch && (e.key === 'ArrowDown' || e.key === 'Enter')) {
    const v = visible(); if (!v.length) return; stop();
    if (e.key === 'Enter' && search.value) { activate(v[0]!.dataset.pkid!); return; }
    roveTo(v[0]); return;
  }
  const el = (e.target as Element).closest?.<HTMLElement>('#pickerWrap .fl-it'); if (!el) return;
  if (e.key === 'Enter' || e.key === ' ') { stop(); if (!e.repeat) activate(el.dataset.pkid!); return; }
  const mv = ({ ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] } as Record<string, [number, number]>)[e.key];
  if (mv) { stop(); move(el, ...mv); return; }
  if (e.key === 'Home' || e.key === 'End') { stop(); const v = visible(); roveTo(e.key === 'Home' ? v[0] : v[v.length - 1]); }
}, true);

/* F9 opens and closes it, and opens it again while it is closing (F-29); not over a modal layer or an open drawer;
   over a non-modal layer (a popover, a dropdown) it closes that first (C-14) */
onKey({
  id: 'picker-f9', order: 20,
  when: e => e.key === 'F9' && !e.ctrlKey && !e.altKey && !e.metaKey,
  run: e => {
    e.preventDefault();
    if (pickerOpen()) { closePicker(); return; }
    const l = topLayer();
    if (l && l.el !== host() && l.modal) return;
    if (drawerMode() && usePanelStore.getState().drawer) return;
    if (l && l.el !== host()) l.close();
    closePop(false);
    document.querySelector('#dd:not([hidden])')?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    openPicker();
  },
});
/* every "Open the picker" button (the Modules panel's head, the empty desk) */
document.addEventListener('click', e => { if ((e.target as Element).closest?.('[data-openpicker]')) openPicker(); });
