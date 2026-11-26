// Ctrl+Z inside a text field (F-22, L-05, E-18): the field's own history first (store/fields.ts), then, with nothing of
// its own left to step back through, the page's Undo, so pressing the keys again always does something. Ctrl+Shift+Z
// and Ctrl+Y the same way forward. The browser's own Undo is never used, from the keys or from its context menu: it is
// one list for the whole page and would reach a field out of sight (F-04).
//
// The histories are fed from the page: typing (input events), what the page writes into a field (a change to a text
// value that the field does not already show), and the field's text whenever focus or a key arrives (a value set
// without telling is a step of its own).

import { canStep, isField, pushField, stepField, stepping, syncField } from '../store/fields.ts';
import { isMod, onKey } from '../register/keys.ts';

const redoKey = (e: KeyboardEvent) => isMod(e, 'z', true) || isMod(e, 'y', false);
const active = () => document.activeElement;

/* the field's own Ctrl+Z, asked before the page's Undo (register/keys.ts order); it declines once the field has
   nothing of its own left, and the page's Undo answers */
onKey({
  id: 'field-undo', order: 10,
  when: e => (isMod(e, 'z', false) || redoKey(e)) && isField(active()) && canStep(active() as HTMLTextAreaElement, redoKey(e)),
  run: e => { e.preventDefault(); stepField(active() as HTMLTextAreaElement, redoKey(e)); },
});

document.addEventListener('focusin', e => { if (isField(e.target)) syncField(e.target); }, true);
document.addEventListener('beforeinput', e => {
  const el = e.target; if (!isField(el)) return;
  /* the browser's own Undo (its context menu) runs the field's instead, and only on the field that has focus */
  if (e.inputType === 'historyUndo' || e.inputType === 'historyRedo') { e.preventDefault(); if (el === active()) stepField(el, e.inputType === 'historyRedo'); return; }
  syncField(el);
}, true);
document.addEventListener('input', e => {
  const el = e.target; if (!isField(el) || stepping()) return;
  const t = (e as InputEvent).inputType || '', data = (e as InputEvent).data ?? '';
  pushField(el, t === 'insertText' && data.length === 1 ? 'type' : /^delete(Content|Word)(Backward|Forward)$/.test(t) ? 'del' : '');
}, true);
/* a key in a field: its text now is written down first, so a value set without telling is a step of its own */
document.addEventListener('keydown', e => { if (isField(e.target) && (e.ctrlKey || e.metaKey)) syncField(e.target); }, true);
