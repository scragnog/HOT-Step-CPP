// Overlays, one path in and out (the concept's 16-overlays.js): a layer that opens is registered with what opened it,
// focus goes into it and comes back to its trigger when it closes (or, if the trigger was redrawn, to the same control
// found again); Escape closes the top layer only; Tab stays inside a modal layer, and leaving a non-modal one (a
// popover) with Tab closes it and carries on from its button. While any layer is open the page's Undo and Redo say
// what to close first (14-undo.js), and the page's own shortcuts that would act under it wait. Other packages' overlays
// (the notes, the song menu, the drafts) register here too.

import { onKey, isMod } from '../register/keys.ts';
import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';

interface Layer { el: HTMLElement; trigger: HTMLElement | null; key: string | null; modal: boolean; close: () => void }
const LAYERS: Layer[] = [];

/** Shown: connected, laid out, and not inside something hidden. */
export const shown = (el: Element | null | undefined): el is HTMLElement => !!el && el.isConnected && el.getClientRects().length > 0 && !el.closest('[hidden]');
/** The controls Tab reaches inside an element, in order. */
export const focusables = (el: Element): HTMLElement[] => [...el.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([type=hidden]):not([disabled]), textarea:not([disabled]), select, [tabindex]:not([tabindex="-1"])')].filter(shown);

/* a trigger the page redrew is found again by what it is (its id, or its data attributes) */
const keyOf = (el: HTMLElement | null): string | null => {
  if (!el || el === document.body) return null;
  if (el.id) return '#' + CSS.escape(el.id);
  const a = [...el.attributes].filter(x => x.name.startsWith('data-') && x.name !== 'data-tip-held').map(x => `[${x.name}="${CSS.escape(x.value)}"]`).join('');
  return a ? el.localName + a : null;
};

export function layerOpen(el: HTMLElement, o: { trigger?: HTMLElement | null; modal?: boolean; close: () => void }): void {
  const i = LAYERS.findIndex(l => l.el === el), prev = i >= 0 ? LAYERS.splice(i, 1)[0]! : null;
  const trigger = prev ? prev.trigger : o.trigger ?? (document.activeElement as HTMLElement | null);
  LAYERS.push({ el, trigger, key: prev ? prev.key : keyOf(trigger), modal: !!o.modal, close: o.close });
}
/** A layer closed: focus goes back to what opened it, unless focus has already gone somewhere else on the page. */
export function layerClosed(el: HTMLElement, refocus = true): void {
  const i = LAYERS.findIndex(l => l.el === el); if (i < 0) return;
  const l = LAYERS.splice(i, 1)[0]!; if (!refocus) return;
  const a = document.activeElement; if (a && a !== document.body && !el.contains(a) && shown(a)) return;
  let t: HTMLElement | null = shown(l.trigger) ? l.trigger : null;
  if (!t && l.key) try { t = [...document.querySelectorAll<HTMLElement>(l.key)].find(shown) ?? null; } catch { t = null; }
  t?.focus({ preventScroll: true });
}
/** The open layer on top, if any (a layer that went away without saying so is forgotten). */
export function topLayer(): Layer | null {
  for (let i = LAYERS.length - 1; i >= 0; i--) { if (shown(LAYERS[i]!.el)) return LAYERS[i]!; LAYERS.splice(i, 1); }
  return null;
}

const ddOpen = () => !!document.querySelector('#dd:not([hidden])');

/* Escape closes the top layer only (a dropdown answers its own Escape first) */
onKey({
  id: 'layer-escape', order: 5,
  when: e => e.key === 'Escape' && !ddOpen() && !!topLayer(),
  run: e => { e.preventDefault(); topLayer()!.close(); },
});
/* the page's Undo and Redo wait under a layer, and say why (14-undo.js) */
onKey({
  id: 'layer-undo', order: 5,
  when: e => (isMod(e, 'z') || isMod(e, 'y', false)) && !!topLayer() && !(document.activeElement instanceof HTMLInputElement && topLayer()!.el.contains(document.activeElement)),
  run: e => { e.preventDefault(); useDesk.getState().showToast({ text: ui(isMod(e, 'z', true) || isMod(e, 'y', false) ? 'undo.closeFirst.redo' : 'undo.closeFirst') }); },
});

/* Tab inside a modal layer goes round it; leaving a popover with Tab closes it and carries on from its button */
document.addEventListener('keydown', e => {
  if (e.key !== 'Tab') return;
  const l = topLayer(); if (!l || l.el.id === 'pickerWrap') return;
  /* a toast raised while a modal layer is open joins its cycle while it shows (F-18; the picker's keys do the same) */
  const toast = document.getElementById('toast'), joins = l.modal && toast && !toast.hidden ? focusables(toast) : [];
  const f = [...focusables(l.el), ...joins], a = document.activeElement as HTMLElement | null, i = a ? f.indexOf(a) : -1;
  if (l.modal) {
    if (!f.length) { e.preventDefault(); return; }
    if (i < 0) { e.preventDefault(); f[0]!.focus(); } else if (e.shiftKey && i === 0) { e.preventDefault(); f[f.length - 1]!.focus(); } else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0]!.focus(); }
    return;
  }
  if (!a || !l.el.contains(a)) return;
  if ((e.shiftKey && i <= 0) || (!e.shiftKey && i === f.length - 1)) { l.close(); if (e.shiftKey) e.preventDefault(); }
}, true);
