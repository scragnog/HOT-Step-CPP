// The page's tier keys wait under a modal layer (pending F-08 in the concept: "No page shortcut acts under a modal
// layer (the Concept notes, the score modal, a drawer): Alt+1/2/3 there leave the tier as it is"). Asked before the
// tier keys (features/tier-keys.ts), it takes Alt+1, 2 and 3 while a modal layer (features/layers.ts) or a side panel
// as a drawer is open, and does nothing with them. The page's other shortcuts already wait there: Ctrl+Z and Ctrl+Y say
// to close what is open first (layers.ts), F6 and F9 ask for no layer (tile-jump.ts, picker-keys.ts).

import { onKey } from '../register/keys.ts';
import { usePanelStore } from '../store/panels.ts';
import { topLayer } from './layers.ts';

onKey({
  id: 'modal-waits', order: 6,
  when: e => e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && /^Digit[123]$/.test(e.code) && (!!topLayer()?.modal || !!usePanelStore.getState().drawer),
  run: e => e.preventDefault(),
});
