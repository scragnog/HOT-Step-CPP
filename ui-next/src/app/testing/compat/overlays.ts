// What package 8's scenarios read on the concept page beyond the layer's core (testing/compat.ts): the picker's
// category and filter (S.pkCat, S.pkQ), the places a drag's preview chooses among (dragRects), the tooltip's sentence
// splitter (tipSentences), whether the engine uses a module (modOk), and the concept's addItem, which offers Lab in a
// toast when a Lab module is added outside Lab with nothing to anchor the offer to. The slice's addItem
// (compat/slice.ts) adds through the store and says nothing; this one replaces it once every install has run.

import { flushSync } from 'react-dom';
import { tipSentences, usedHere } from '../../../derive.ts';
import { defOf } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { usePicker } from '../../store/overlays.ts';
import { currentDragRects } from '../../features/dnd.tsx';
import { offerLab } from '../../features/adding.ts';
import { expose, S } from '../compat.ts';

const itemOf = (id: string) => defOf(id.includes(':') ? id.split(':')[1]! : id);

export function install(): void {
  Object.defineProperty(S, 'pkCat', { get: () => usePicker.getState().cat, set: v => usePicker.setState({ cat: v }), enumerable: true, configurable: true });
  Object.defineProperty(S, 'pkQ', { get: () => usePicker.getState().q, enumerable: true, configurable: true });
  expose({
    get dragRects() { return currentDragRects(); },
    tipSentences,
    modOk(id: string) { const d = itemOf(id); return !!d && usedHere(d, useDesk.getState().desk.engine); },
  });
  queueMicrotask(() => expose({
    addItem(id: string, before?: string | null) {
      const d = itemOf(id); if (!d) return;
      flushSync(() => { if (useDesk.getState().add(d.id, before ?? undefined) === 'lab') offerLab(d.id); });
    },
  }));
}
