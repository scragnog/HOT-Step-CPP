// The keyboard focus ring (the concept's 37-focus-ring.js): it settles in only when Tab or an arrow key moved focus
// (`kbnav` on the root, M-09); focus handed back by a closing overlay appears without the settle. A pointer never
// lights it (G-B-04, G-B-12): focus a script moves while the pointer was the last input is quiet, and the first arrow
// key on a control the pointer focused brings its ring up. Text fields keep their ring (they are being typed into).
// Its key listeners are on the capture phase, so they read focus after a control's own key handler has moved it.

const KB_NAV = /^(Tab|ArrowUp|ArrowDown|ArrowLeft|ArrowRight|Home|End|PageUp|PageDown)$/;
const TYPED = 'input, textarea, select, [contenteditable]';
let pointerLast = false;

document.addEventListener('keydown', e => { document.documentElement.classList.toggle('kbnav', KB_NAV.test(e.key)); }, true);
document.addEventListener('pointerdown', () => { document.documentElement.classList.remove('kbnav'); pointerLast = true; }, true);
document.addEventListener('keydown', e => {
  if (/^(Shift|Control|Alt|Meta)$/.test(e.key)) return;
  const a = document.activeElement as HTMLElement | null;
  if (pointerLast && e.key !== 'Tab' && KB_NAV.test(e.key) && a && a !== document.body && !a.matches(TYPED + ', [data-num]'))
    setTimeout(() => { if (document.activeElement === a && a.isConnected && !a.matches(':focus-visible')) { a.blur(); a.focus({ preventScroll: true }); } }, 0);
  pointerLast = false;
}, true);
/* focus moved by script after a pointer press asks the browser not to show it as keyboard focus */
for (const K of [HTMLElement, SVGElement]) {
  const f = K.prototype.focus;
  K.prototype.focus = function (this: HTMLElement | SVGElement, o?: FocusOptions & { focusVisible?: boolean }) {
    if (pointerLast && !(o && 'focusVisible' in o) && !this.matches?.(TYPED)) o = { ...(o ?? {}), focusVisible: false };
    return f.call(this, o);
  };
}
