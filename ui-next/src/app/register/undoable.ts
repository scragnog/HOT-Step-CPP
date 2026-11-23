// State of a work package's own that Undo covers (a workflow opened, the drafts): the package keeps it in its own store
// and registers how to read and write it; the desk's history then records it with every step and puts it back as it
// does a value, path by path (store/desk.ts). A change to it is made inside `deskStep(words, fn)`, which records one
// step named by the words the core gives. State the server owns (a take that arrives, a job's progress) is not
// registered: Undo never takes back what the server has done (docs/09, "Server-owned state").

import type { Json } from '../../definitions/types.ts';

export interface Undoable { read(): Json; write(v: Json): void }
const SLICES = new Map<string, Undoable>();

export function defineUndoable(name: string, s: Undoable): void {
  if (SLICES.has(name)) throw new Error(`the undoable state "${name}" is defined twice`);
  SLICES.set(name, s);
}
export const undoables = (): ReadonlyMap<string, Undoable> => SLICES;
