// What the Workflows panel and the Drafts list show open, beyond the records (store/workflows.ts): the save row, the
// row being renamed, and the one popover they open (the row menu, the drafts). View state only: nothing here is a
// value, and none of it is kept.

import { create } from 'zustand';
import type { RevId } from '../../derive.ts';

/** A row of the list: a workflow by its name, or a revision by its identity and whether it is the user's. */
export type RowRef = { kind: 'wf'; name: string; user: boolean } | { kind: 'rev'; id: RevId; user: boolean };

export interface WfView {
  /** The save row: open, saving a revision or a workflow, and the name it asked to replace (`confirm`, C-02). */
  saveOpen: boolean;
  saveAs: 'rev' | 'wf';
  confirm: string | null;
  /** A refusal said in the row's own hint line (C-22), and its count, so the same refusal said twice is said twice. */
  err: { says: import('../../derive.ts').Msg; n: number } | null;
  /** What the save field should hold when it next shows (Save as… fills it). */
  prefill: { text: string; n: number } | null;
  /** The row whose name is a field, and why the name typed there was refused (said under the row: F-11). */
  renaming: RowRef | null;
  renameErr: import('../../derive.ts').Msg | null;
  /** The popover the panel or the go bar opened: the row menu of a row, or the drafts. */
  pop: { kind: 'menu'; row: RowRef; anchor: HTMLElement; armed?: 'delpreset' | 'rdel' } | { kind: 'drafts'; anchor: HTMLElement } | null;
  /** The row that takes Tab in the list (F-07): its selector. */
  rove: string | null;
}

export const useWfView = create<WfView>()(() => ({ saveOpen: false, saveAs: 'rev', confirm: null, err: null, prefill: null, renaming: null, renameErr: null, pop: null, rove: null }));

/** A row's selector, by the concept's data attributes. */
export const rowSel = (r: RowRef, backend: (e: RevId['engine']) => string): string => r.kind === 'wf'
  ? `.pre.wfrow[data-preset="${CSS.escape(r.name)}"]`
  : `.pre.rev[data-rev="${CSS.escape(r.id.workflow)}"][data-rn="${CSS.escape(r.id.name)}"][data-rb="${backend(r.id.engine)}"]`;
