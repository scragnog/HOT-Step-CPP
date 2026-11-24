// Touch on the controls a hand drags (the concept's 19-touch-gesture.js, S-29): under a finger a gesture is either a
// scroll of the page or a change to the control, never both, and a swipe that happens to start on a control leaves it
// alone. A mouse or a pen is untouched.
//
// - Controls dragged sideways (a ruler, a scrub token, the look-back strip, the CFG handles, the reference blend, an
//   adapter's phases and masks, the waves): the first 8px decide. Mostly sideways, the control takes the touch from
//   where the finger is then; mostly up or down, the page scrolls and the control is not touched.
// - Controls dragged up and down or both ways (knobs, faders, number boxes, XY pads): the page's own scroll axis, so the
//   finger says it means the control by resting on it for 200ms; then it lifts, as a mouse press lifts it. A finger
//   that moves first scrolls the page; the first time, the page says how to turn one (S-36).
// - A tap (down and up without travel) is a press: the control gets its pointerdown at the point, then the release.
//
// Every such control lets the page pan (touch-action: pan-y pinch-zoom, 35-states.css). This listener runs before the
// controls' own (the document's capture phase) and keeps the touch's pointerdown from them; once the gesture is
// decided for the control it hands them a pointerdown of its own, and cancels the touch's moves so the page stays.

import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';

const SIDEWAYS = '[data-ruler], [data-scrub], [data-codes], [data-ivl], [data-venn], [data-phase], [data-mask], [data-contwave], [data-rpwave]';
const UPRIGHT = '.knob, [data-fader], [data-num], [data-xy]';
const HOLD = 200, SLOP = 8;

interface Gesture { id: number; target: EventTarget; x: number; y: number; h: boolean; armed: boolean; scroll: boolean; timer: number }
let tg: Gesture | null = null, synth = false, taught = false;

/** The gesture is the control's: it gets a pointerdown of its own at the finger's place. */
function arm(x: number, y: number): void {
  if (!tg || tg.armed) return; tg.armed = true; clearTimeout(tg.timer);
  synth = true;
  try { tg.target.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, composed: true, pointerId: tg.id, pointerType: 'touch', isPrimary: true, clientX: x, clientY: y, button: 0, buttons: 1 })); }
  finally { synth = false; }
}

document.addEventListener('pointerdown', e => {
  if (synth || e.pointerType !== 'touch') return;
  const el = (e.target as Element).closest?.(SIDEWAYS + ', ' + UPRIGHT); if (!el) return;
  e.stopImmediatePropagation();
  const g: Gesture = { id: e.pointerId, target: e.target!, x: e.clientX, y: e.clientY, h: el.matches(SIDEWAYS), armed: false, scroll: false, timer: 0 };
  tg = g;
  if (!g.h) g.timer = window.setTimeout(() => { if (tg === g && !g.scroll) arm(g.x, g.y); }, HOLD);
}, true);
/* the moves before the gesture is decided are no one's; after, a scroll is the browser's and a control's move the control's */
document.addEventListener('pointermove', e => {
  if (!tg || e.pointerId !== tg.id || tg.armed) return;
  e.stopImmediatePropagation(); if (tg.scroll) return;
  const dx = e.clientX - tg.x, dy = e.clientY - tg.y; if (Math.hypot(dx, dy) < SLOP) return;
  if (tg.h && Math.abs(dx) > Math.abs(dy)) { arm(e.clientX, e.clientY); return; }
  tg.scroll = true; clearTimeout(tg.timer);
  /* a swipe that began on a knob scrolled the page: the first time, say how a finger turns one (S-36) */
  teach(tg);
}, true);
function teach(g: Gesture): void { if (!g.h && !taught) { taught = true; useDesk.getState().showToast({ text: ui('touch.rest') }); } }
/* a touch move the control has taken must not also move the page */
document.addEventListener('touchmove', e => { if (tg?.armed && e.cancelable) e.preventDefault(); }, { capture: true, passive: false });
document.addEventListener('pointerup', e => {
  if (!tg || e.pointerId !== tg.id) return;
  const g = tg; tg = null; clearTimeout(g.timer);
  /* a tap: the control gets its press now, at the point, then this release */
  if (!g.armed && !g.scroll) { tg = g; arm(e.clientX, e.clientY); tg = null; }
}, true);
/* the browser took the touch for a scroll of the page before it travelled far enough to say so */
document.addEventListener('pointercancel', e => { if (tg && e.pointerId === tg.id) { clearTimeout(tg.timer); if (!tg.armed) teach(tg); tg = null; } }, true);
