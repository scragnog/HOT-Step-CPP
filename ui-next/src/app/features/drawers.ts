// The side panels as drawers, at 1180px and below (31-side-panels.js, 16-overlays.js): Escape closes the open one and
// focus goes back to its button; while one is open it is a modal dialog and Tab stays inside it (C-13).

import { onKey } from '../register/keys.ts';
import { closeDrawer } from '../shell/drawers.ts';
import { usePanelStore } from '../store/panels.ts';

const FOCUSABLE = 'button:not([disabled]):not([hidden]), input:not([disabled]), [tabindex="0"]';
const open = () => { const d = usePanelStore.getState().drawer; return d ? document.getElementById(d === 'left' ? 'leftPanel' : 'rightPanel') : null; };

onKey({
  id: 'drawer-escape', order: 10, create: true,
  when: e => e.key === 'Escape' && !!usePanelStore.getState().drawer && !document.querySelector('#dd:not([hidden])'),
  run: e => { e.preventDefault(); closeDrawer(true); },
});
onKey({
  id: 'drawer-tab', order: 10, create: true,
  when: e => e.key === 'Tab' && !!open(),
  run: e => {
    const p = open()!, f = [...p.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(x => x.getClientRects().length), i = f.indexOf(document.activeElement as HTMLElement);
    if (!f.length) return;
    if (i < 0) { e.preventDefault(); f[0]!.focus(); } else if (e.shiftKey && i === 0) { e.preventDefault(); f[f.length - 1]!.focus(); } else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0]!.focus(); }
  },
});
