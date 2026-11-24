// Package 3c's instruments beyond their parts: the server sessions told to the desk (store/build.ts: the build
// session's takes, a recording's length), and what loading a recording for Continue from recording says (E-23): a
// toast with the step's own words and Undo, as the file's step is named. A recording that comes back with Undo or Redo,
// or with a scene, says nothing here (Undo says what it did).

import '../store/build.ts';
import { useDesk } from '../store/desk.ts';

/** The file state whose loading is told (Continue's recording). */
const RECORDING = 'contFile';

useDesk.subscribe((s, prev) => {
  const f = s.desk.state[RECORDING];
  const e = s.undo.at(-1);
  /* a new step: one the history did not hold (Redo brings back one it held) */
  if (f === prev.desk.state[RECORDING] || typeof f !== 'string' || !f || !e || prev.undo.includes(e) || prev.redo.includes(e)) return;
  s.showToast({ text: e.words.label, action: { label: { id: 'ui.undo' }, run: () => useDesk.getState().runUndo(e) } });
});
