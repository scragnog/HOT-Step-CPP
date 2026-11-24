// Tooltips (the concept's 19-tooltip.js, C-31, C-48): one styled tip for the pointer, the keyboard and a held touch,
// read from the controls' title attributes, which the page has already written in its language.
//
// - A pointer resting on a control shows its title after 400ms (at once when it moves on to a neighbour while a tip is
//   up), and it goes the moment the pointer leaves. While the pointer is over it the control's title is held aside, so
//   the browser's own tooltip never doubles it; it is put back when the pointer leaves.
// - A control reached by the keyboard shows its tip after a beat; that tip goes on the next pointer move of more than
//   8px, on any key but a modifier, and after 6 seconds (F-19).
// - Under a finger, pressing and holding shows the tip, and lifting the finger still presses the control, however long
//   it was held (S-01): a tip never costs the press it came with.
// - A tip never gets in the way of the control it explains (Q-22, L-06): none while a control is being operated, and a
//   tip due in the 1.5s after a press or a change waits for the rest of it rather than being dropped; none that only
//   repeats the label already printed on the control; and it opens on the side that covers none of the control's
//   read-outs (placeTip). In the side panels' lists it stands beside the row.
// - What it says is cut by the core's rules (src/derive/tips.ts): every word and digit kept, how to work a control said
//   once a session, a finger's words under a finger. The tip itself is hidden from screen readers, which announce a
//   title as the control's description already.

import { sameAsLabel, tipLesson, touchTip } from '../../derive.ts';
import { sayNow, ui } from '../i18n/index.ts';

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const COARSE = matchMedia('(pointer: coarse)');
const QUIET = 1500;
let timer = 0, host: HTMLElement | null = null, warm = 0, quiet = 0, byKey = false, keyAt: [number, number] | null = null, keyLife = 0;
const pointer: [number, number] = [-1, -1];
const taught = new Set<string>();

const textOf = (t: HTMLElement) => (t.title || t.dataset.tipHeld || '').trim();
function tipFor(el: EventTarget | null): HTMLElement | null { const t = (el as Element | null)?.closest?.<HTMLElement>('[title], [data-tip-held]'); return t && textOf(t) ? t : null; }
function holdTitle(t: HTMLElement): void { if (t.title) { t.dataset.tipHeld = t.title; t.removeAttribute('title'); } }
function giveTitle(t: HTMLElement | null): void { if (t && t.dataset.tipHeld != null) { if (!t.hasAttribute('title')) t.title = t.dataset.tipHeld; delete t.dataset.tipHeld; } }
const wait = () => Math.max(0, QUIET - (performance.now() - quiet));

/* the quiet period after a press or a change delays a tip, it never cancels it (L-06) */
function armTip(t: HTMLElement, ms: number, still: () => boolean): void {
  clearTimeout(timer);
  timer = window.setTimeout(function go() {
    if (!still() || !t.isConnected) return;
    const w = wait(); if (w > 0) { timer = window.setTimeout(go, w); return; }
    if (t.title) holdTitle(t);
    showTip(t);
  }, ms);
}

/** The words a tip shows now: a finger's words under a finger, how to work a control once a session. */
function words(t: HTMLElement): string {
  let text = textOf(t);
  if (COARSE.matches) text = touchTip(text, sayNow({ id: 'part.knob.howTo' }), sayNow(ui('tip.knobTouch')));
  const l = tipLesson(text, taught); l.teaches.forEach(s => taught.add(s));
  return l.text;
}
function tipEl(): HTMLElement {
  let tip = document.getElementById('tip');
  if (!tip) { tip = document.createElement('div'); tip.id = 'tip'; tip.className = 'tip'; tip.setAttribute('aria-hidden', 'true'); document.body.appendChild(tip); }
  return tip;
}
function showTip(el: HTMLElement, key = false): void {
  const t = tipFor(el); if (!t) return;
  const lines = (t.innerText || '').split('\n');
  if (sameAsLabel(lines, textOf(t))) return;
  const text = words(t); if (!text || sameAsLabel(lines, text)) return;
  const tip = tipEl(); tip.textContent = text; tip.hidden = false;
  const r = t.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight, vw = document.documentElement.clientWidth;
  const side = t.closest('#presetList, #browser'), pr = side?.getBoundingClientRect();
  if (pr && pr.right + 8 + w <= vw - 8) { tip.style.left = (pr.right + 8) + 'px'; tip.style.top = clamp(r.top + r.height / 2 - h / 2, 8, innerHeight - h - 8) + 'px'; }
  else if (pr && pr.left - 8 - w >= 8) { tip.style.left = (pr.left - 8 - w) + 'px'; tip.style.top = clamp(r.top + r.height / 2 - h / 2, 8, innerHeight - h - 8) + 'px'; }
  else placeTip(tip, t, r, w, h, vw);
  byKey = key; keyAt = key ? [...pointer] : null;
  clearTimeout(keyLife); if (key) keyLife = window.setTimeout(hideTip, 6000);
}

/* Where a tip opens (L-06): on the side that covers least of what the control reports to: the read-outs bound to the
   same setting anywhere on screen, and every read-out in the control's own tile, pedal or row. Above, below, then
   beside; a control may name its own first choice (data-tip-side); on the desk above comes first, elsewhere below. */
const READS = 'output, [data-out], .lcd, .tk-lcd, .u-lcd, .gauge, .ruler2, .rscale, .scope, .tk-cost, .kl, .sm-row, .term > div, .eff, .phase, .secband';
function avoid(t: HTMLElement): DOMRect[] {
  const el = t.closest<HTMLElement>('[data-ctl]'), k = el?.dataset.ctl || t.dataset.knob || t.dataset.scrub || t.dataset.fader || t.dataset.ruler || t.dataset.bind;
  const box = t.closest('.carts, .kn, .pedal, .sm-row, .fdr, .tile-body, .term, .lcd') || t.parentElement;
  const els = new Set<Element>(box ? [...box.querySelectorAll(READS)] : []);
  if (k) document.querySelectorAll(`[data-out="${CSS.escape(k)}"]`).forEach(e => els.add(e));
  const row = t.closest('.sm-row'); if (row) { if (row.previousElementSibling) els.add(row.previousElementSibling); if (row.nextElementSibling) els.add(row.nextElementSibling); }
  return [...els].filter(e => e !== t).map(e => e.getBoundingClientRect()).filter(q => q.width && q.height);
}
function placeTip(tip: HTMLElement, t: HTMLElement, r: DOMRect, w: number, h: number, vw: number): void {
  const G = 8, top0 = t.closest('#bento') ? 70 : 8, desk = !!t.closest('#bento, .board');
  const cx = clamp(r.left + r.width / 2 - w / 2, 8, vw - w - 8), cy = clamp(r.top + r.height / 2 - h / 2, top0, innerHeight - h - 8);
  const C: Record<string, [number, number]> = { above: [cx, r.top - h - G], below: [cx, r.bottom + G], right: [r.right + G, cy], left: [r.left - G - w, cy] };
  const pref = t.closest<HTMLElement>('[data-tip-side]')?.dataset.tipSide;
  const order = pref === 'beside' ? ['right', 'left', 'above', 'below'] : pref === 'below' ? ['below', 'above', 'right', 'left'] : desk ? ['above', 'below', 'right', 'left'] : ['below', 'above', 'right', 'left'];
  const av = avoid(t), cover = (x: number, y: number) => av.reduce((s, q) => s + Math.max(0, Math.min(x + w, q.right) - Math.max(x, q.left)) * Math.max(0, Math.min(y + h, q.bottom) - Math.max(y, q.top)), 0);
  let best: { x: number; y: number; sc: number } | null = null;
  order.forEach((n, i) => { const [x, y] = C[n]!, fits = x >= 8 && x + w <= vw - 8 && y >= top0 && y + h <= innerHeight - 8, sc = (fits ? 0 : 1e9) + cover(x, y) * 10 + i; if (!best || sc < best.sc) best = { x, y, sc }; });
  tip.style.left = best!.x + 'px'; tip.style.top = best!.y + 'px';
}
function hideTip(): void {
  clearTimeout(timer); clearTimeout(keyLife); byKey = false; keyAt = null;
  const tip = document.getElementById('tip'); if (tip && !tip.hidden) { tip.hidden = true; warm = performance.now(); }
}

/* a control being operated has no tip: a change, the keys that move a value or press a button, a mouse press */
['input', 'change'].forEach(ev => document.addEventListener(ev, () => { quiet = performance.now(); hideTip(); }, true));
document.addEventListener('keydown', e => {
  if (/^(Arrow|Page|Home|End)/.test(e.key) || ((e.key === 'Enter' || e.key === ' ') && (e.target as Element).closest?.('button'))) { quiet = performance.now(); hideTip(); return; }
  /* a tip the keyboard opened goes on any key but a modifier (F-19) */
  if (e.key === 'Escape' || (byKey && !/^(Shift|Control|Alt|Meta)$/.test(e.key))) hideTip();
}, true);
document.addEventListener('pointerdown', e => { if (e.pointerType !== 'touch') quiet = performance.now(); }, true);

/* the keyboard: a control reached by Tab or an arrow shows its tip after a beat */
document.addEventListener('focusin', e => {
  hideTip();
  const el = e.target as HTMLElement;
  if (!document.documentElement.classList.contains('kbnav') || el.matches?.('input, textarea')) return;
  clearTimeout(timer);
  timer = window.setTimeout(function go() {
    if (document.activeElement !== el || !el.isConnected) return;
    const w = wait(); if (w > 0) { timer = window.setTimeout(go, w); return; }
    showTip(el, true);
  }, 450);
});
document.addEventListener('focusout', hideTip);
addEventListener('scroll', hideTip, { passive: true });

/* the pointer */
document.addEventListener('pointermove', e => {
  if (e.pointerType === 'touch') return;
  pointer[0] = e.clientX; pointer[1] = e.clientY;
  /* a keyboard tip gives way to the pointer (F-19) */
  if (byKey && keyAt && (keyAt[0] < 0 || Math.hypot(e.clientX - keyAt[0], e.clientY - keyAt[1]) > 8)) hideTip();
});
document.addEventListener('pointerover', e => {
  if (e.pointerType === 'touch') return;
  const t = tipFor(e.target); if (t === host) return;
  giveTitle(host); hideTip(); host = t; if (!t) return; holdTitle(t);
  armTip(t, performance.now() - warm < 300 ? 0 : 400, () => host === t);
});
document.addEventListener('pointerout', e => {
  if (e.pointerType === 'touch' || !host) return;
  if (e.relatedTarget && host.contains(e.relatedTarget as Node)) return;
  giveTitle(host); host = null; hideTip();
});
/* a finger: pressing and holding shows the tip; sliding off before lifting is how a press is let go, as everywhere on a
   touch screen, and lifting presses the control as it would have anyway (S-01) */
document.addEventListener('pointerdown', e => {
  hideTip();
  /* a press gives the title back until the tip shows again (armTip holds it then): the control's title is its
     explanation, and a press is when the page or a reader looks at it again (the concept's controls are redrawn by a
     press, title and all) */
  if (e.pointerType !== 'touch') { const t = host; if (t) { giveTitle(t); armTip(t, 400, () => host === t); } return; }
  const x = e.clientX, y = e.clientY, el = e.target as HTMLElement;
  timer = window.setTimeout(() => { showTip(el); timer = window.setTimeout(hideTip, 2600); }, 500);
  const cancel = (ev: PointerEvent) => {
    if (ev.type === 'pointermove' && Math.hypot(ev.clientX - x, ev.clientY - y) <= 8) return;
    if (document.getElementById('tip')?.hidden !== false) clearTimeout(timer);
    removeEventListener('pointermove', cancel); removeEventListener('pointerup', cancel);
  };
  addEventListener('pointermove', cancel); addEventListener('pointerup', cancel);
}, true);
