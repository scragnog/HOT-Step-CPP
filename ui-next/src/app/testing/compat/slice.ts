// What the slice's scenarios call on the concept page beyond the layer's core (testing/compat.ts): adding an item,
// switching the engine as a step (switchEngine) or as a scene (enterEngine), choosing a tier, whether a module shows at
// the tier, removing a module, repainting Undo after a scenario empties the history, and the engines' book as S.eng.

import { flushSync } from 'react-dom';
import { switchEngine as coreSwitch } from '../../../derive.ts';
import { defOf, ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { setTierFromView } from '../../shell/tier.ts';
import { CONCEPT_ID } from '../../shell/engines.ts';
import { TO_ENGINE, asScene, expose, onMirror, type Scene } from '../compat.ts';

/** The concept's item ids: a module by its id, a pedal as pd:<id>, a side unit as ex:<id>. */
const itemOf = (id: string) => defOf(id.includes(':') ? id.split(':')[1]! : id);

export function install(): void {
  expose({
    addItem(id: string, before?: string | null) { const d = itemOf(id); if (d) asStep(() => useDesk.getState().add(d.id, before ?? undefined)); },
    removeMod(id: string) { asStep(() => useDesk.getState().remove(id)); },
    switchEngine(bk: string) { const e = TO_ENGINE[bk]; if (e) asStep(() => useDesk.getState().switchEngine(e)); },
    /* the concept's enterEngine moves the desk to an engine without a step; its callers set the rest of the scene */
    enterEngine(bk: string) {
      const e = TO_ENGINE[bk], s = useDesk.getState(); if (!e || e === s.desk.engine) return;
      const r = coreSwitch(ix, s.desk, s.book, e);
      asScene(() => useDesk.setState({ desk: { ...r.desk, on: s.desk.on }, book: r.book }));
    },
    setTier(k: string, o?: { focus?: boolean; ring?: string[] }) { if (k === 'quick' || k === 'studio' || k === 'lab') setTierFromView(k, o ?? {}); },
    shown(m: string) { const d = defOf(m); return !(d?.kind === 'module' && d.labUnder) || useDesk.getState().desk.tier === 'lab'; },
    /* a scenario that empties UNDO and REDO in place asks the page to draw it */
    paintUndo() { const s = useDesk.getState(); useDesk.setState({ undo: [...s.undo], redo: [...s.redo] }); },
  });
  onMirror((S: Scene, s) => {
    const stash = Object.fromEntries(Object.entries(s.book.stash).map(([e, x]) => [CONCEPT_ID[e as keyof typeof CONCEPT_ID] ?? e, x]));
    const hold = Object.fromEntries(Object.entries(s.book.hold).map(([k, h]) => [k, { v: h.value, shown: h.shown, from: CONCEPT_ID[h.from] }]));
    S.eng = { stash, hold };
  });
}

/** A person's action, drawn at once so the scenario reads its result. */
const asStep = (fn: () => void): void => flushSync(fn);
