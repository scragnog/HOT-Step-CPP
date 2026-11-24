// Enlarged hit areas never take a click from a neighbour (the concept's 19-hit-areas.js, A-12): the pads are each small
// control's ::before (styles/concept/35-states.css), and where two pads overlap (the time signature's arrows beside its
// picker, a row of chips) the later one in the page is on top. A click that lands on a control only through its pad goes
// to the control whose own box is nearest the pointer; a click on a control's own box is never moved. Keep PADDED in
// step with the list in 35-states.css.

const PADDED = '.tile-head .grip, .tok, .tok-ro, .pill-tog, .cyc, .sm-del, .sm-star, .flag, .tchips > button, .tbtn, .tpow, .scope-btn, .instr-btn, .wild-x, .wild-auto, .x, .px, .pchev, .jump, .lcd-btns > button, .lcd-bars > button, .chipsw button, .drop-x, .pend-x, .u-x, .toast-x, [data-mask], .phase [data-ph], .seg.sm > button, .rec, .led-btn, .dst, .sbtn, .small-btn, .icon-btn, .pref, button.auto, .strip > button, .lcd-step button, .mood button, .itabs > button, .gender .seg > button, .lcd-ends .seg > button, .play, .more, .linkbtn, .tool, .qchoice button, .seg > button, .pk-plugin, .pk-lcd, .tk-reels .reel2, #ppause, #pstop, .scope-link';
const CONTROL = 'button, input, textarea, select, a[href], [role="slider"], [tabindex]:not([tabindex="-1"])';

const boxDist = (r: DOMRect, x: number, y: number) => Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));

function nearestPadded(e: MouseEvent): HTMLElement | null {
  const x = e.clientX, y = e.clientY; if (!e.isTrusted || (!x && !y)) return null;
  const hit = (e.target as Element | null)?.closest?.<HTMLElement>(PADDED);
  if (!hit || boxDist(hit.getBoundingClientRect(), x, y) === 0) return null;
  let best = hit, bd = boxDist(hit.getBoundingClientRect(), x, y);
  document.elementsFromPoint(x, y).forEach(el => {
    const c = el.closest<HTMLElement>(PADDED) ?? el.closest<HTMLElement>(CONTROL);
    if (!c || c === best || (c as HTMLButtonElement).disabled || c.contains(hit)) return;
    const d = boxDist(c.getBoundingClientRect(), x, y);
    if (d < bd) { best = c; bd = d; }
  });
  return best !== hit ? best : null;
}

document.addEventListener('click', e => {
  const to = nearestPadded(e); if (!to) return;
  e.preventDefault(); e.stopImmediatePropagation();
  to.focus?.({ preventScroll: true }); to.click();
}, true);
