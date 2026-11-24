// Adding from the view, where it says more than the store's add: a Lab module outside Lab offers Lab (in the popover
// by the card it came from, or with nothing to anchor the offer to, a toast); taking the offer switches to Lab and adds
// it, and in the picker the keys stay on it; a module that arrives but the engine does not use says so, with Undo.

import { createElement } from 'react';
import { flushSync } from 'react-dom';
import { engineName } from '../../definitions/engines.ts';
import { tileName, usedHere } from '../../derive.ts';
import { defOf } from '../model/index.ts';
import { useDesk, type Entry } from '../store/desk.ts';
import { usePicker } from '../store/overlays.ts';
import { ui } from '../i18n/index.ts';
import { setTierFromView } from '../shell/tier.ts';
import { closePop, openPop } from './pop.ts';
import { LabOffer } from './lab-offer.tsx';

/** A Lab module added outside Lab: the offer in the popover by `anchor`, or, with no anchor, a toast. */
export function offerLab(id: string, anchor?: HTMLElement | null): void {
  const d = defOf(id); if (!d) return;
  if (anchor) { openPop(anchor, () => createElement(LabOffer, { id })); return; }
  useDesk.getState().showToast({ text: ui('labOffer.toast', { name: tileName(d, useDesk.getState().desk) }) });
}

/** What a module that arrived but the engine does not use says: it is on the desk, parked, with Undo of the add
 *  that brought it (`step`, the store's add's own). */
export function sayParked(id: string, entry: Entry | null): void {
  const s = useDesk.getState(), d = defOf(id);
  if (!d || usedHere(d, s.desk.engine) || !entry) return;
  s.showToast({ text: ui('added.notUsed', { name: tileName(d, s.desk), engine: engineName(s.desk.engine) }), action: { label: ui('undo'), run: () => useDesk.getState().runUndo(entry) } });
}

/** "Switch to Lab and add it": the tier changes, the module arrives; in the picker, the keys stay on it. */
export function labAdd(id: string): void {
  closePop();
  setTierFromView('lab', { instant: true });
  const r = useDesk.getState().add(id);
  if (r.did === 'added') sayParked(id, r.step);
  if (!usePicker.getState().open) return;
  flushSync(() => usePicker.setState({ rove: id }));
  document.querySelector<HTMLElement>(`#pickerWrap .fl-it[data-pkid="${CSS.escape(id)}"]`)?.focus({ preventScroll: true });
  usePicker.setState({ info: { kind: 'added', id, lab: true } });
}
