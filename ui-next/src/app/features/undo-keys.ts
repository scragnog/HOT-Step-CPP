// Ctrl+Z, and Ctrl+Shift+Z or Ctrl+Y: the page's undo and redo outside a text field (a field keeps its own typing to
// take back). With nothing to undo they say so (A-19); while a drawer covers the desk they say why nothing happened
// (14-undo.js).

import { useDesk } from '../store/desk.ts';
import { usePanelStore } from '../store/panels.ts';
import { ui } from '../i18n/index.ts';
import { isMod, isTextField, onKey } from '../register/keys.ts';

const redoKey = (e: KeyboardEvent) => isMod(e, 'z', true) || isMod(e, 'y', false);

onKey({
  id: 'undo-redo',
  when: e => (isMod(e, 'z', false) || redoKey(e)) && !isTextField(document.activeElement),
  run: e => {
    e.preventDefault();
    const s = useDesk.getState(), redo = redoKey(e);
    if (usePanelStore.getState().drawer) { s.showToast({ text: ui(redo ? 'undo.closeFirst.redo' : 'undo.closeFirst') }); return; }
    if (!(redo ? s.redo : s.undo).length) { s.showToast({ text: ui(redo ? 'undo.nothing.redo' : 'undo.nothing') }); return; }
    if (redo) s.runRedo(); else s.runUndo();
  },
});
