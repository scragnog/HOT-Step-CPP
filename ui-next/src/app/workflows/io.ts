// A revision exported as a JSON file, and a file chosen to import (the concept's exportRev and the import button's
// file input): the file's shape and every word are the core's (exportFile, importRevision); this hands the file over.

import { exportFile, findRevision, msg, type RevId } from '../../derive.ts';
import { useDesk } from '../store/desk.ts';
import { importFile, useWf } from '../store/workflows.ts';
import { saveFile } from './file.ts';

export function exportRevision(id: RevId, user: boolean): void {
  const s = useWf.getState(), r = findRevision(s.saved, id, user); if (!r) return;
  const f = exportFile(s.saved, id.workflow, r, useDesk.getState().desk.tier);
  saveFile(f.name, f.text);
  useDesk.getState().showToast({ text: msg('wf.exported', { wf: id.workflow, name: r.name }) });
}

/** Reads a chosen file and imports it. */
export function importChosen(f: File): void {
  void f.text().then(importFile, () => useDesk.getState().showToast({ text: msg('wf.import.failed') }));
}
