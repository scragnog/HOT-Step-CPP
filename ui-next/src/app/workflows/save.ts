// Opening and closing the save row (32-workflows.js openSave, closeSave): its row is SaveRow.tsx.

import { findWorkflow, type Msg } from '../../derive.ts';
import { layoutChanged, useWf } from '../store/workflows.ts';
import { useWfView } from './state.ts';
/** The open workflow, if it is still here. */
export const openWf = (): string | null => { const s = useWf.getState(); return findWorkflow(s.saved, s.open.workflow)?.w.name ?? null; };

/** Opens the save row: as a revision of the open workflow while its layout is as saved, else as a new workflow (or as
 *  asked); `prefill` is the name it starts with (Save as…). */
export function openSave(as?: 'rev' | 'wf', prefill?: Msg): void {
  const v = useWfView.getState();
  useWfView.setState({ saveOpen: true, saveAs: as ?? (openWf() && !layoutChanged() ? 'rev' : 'wf'), confirm: null, err: null, prefill: { text: prefill ? JSON.stringify(prefill) : '', n: (v.prefill?.n ?? 0) + 1 } });
}
export function closeSave(refocus: boolean): void {
  useWfView.setState({ saveOpen: false, confirm: null, err: null });
  if (refocus) document.getElementById('saveBtn')?.focus({ preventScroll: true });
}

