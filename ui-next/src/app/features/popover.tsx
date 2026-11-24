// The popover's view: what was opened (pop.ts openPop) drawn into #pop, placed by its button, registered as a layer
// (layers.ts) and given focus: its pressed or primary button, else its first control.

import { useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { fill } from '../register/slots.ts';
import { usePop } from '../store/overlays.ts';
import { sayNow } from '../i18n/index.ts';
import { focusables, layerOpen } from './layers.ts';
import { closePop, placeFloat, popHost } from './pop.ts';

export function Popover() {
  const st = usePop();
  useLayoutEffect(() => {
    const el = popHost();
    if (!el || !st.body || !st.anchor) return;
    el.hidden = false; el.classList.remove('fixedpop', 'up');
    const label = st.label !== null ? sayNow(st.label) : st.anchor.getAttribute('aria-label') || st.anchor.textContent?.trim() || '';
    if (label) el.setAttribute('aria-label', label); else el.removeAttribute('aria-label');
    placeFloat(el, st.anchor, st.place);
    layerOpen(el, { trigger: st.anchor, close: () => closePop() });
    (el.querySelector<HTMLElement>('[aria-pressed="true"], .primary:not(:disabled)') ?? focusables(el)[0])?.focus({ preventScroll: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st.n]);
  const Body = st.body, el = popHost();
  if (!el || !Body) return null;
  return createPortal(<Body key={st.n} />, el);
}
fill('overlays', Popover);
