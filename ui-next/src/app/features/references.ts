// Every reference track a drop zone takes is remembered as a recent reference (the concept's 13-files.js: a file
// loaded anywhere but Continue's recording joins S.refs), for the Timbre reference tile to offer again. The list is the
// server's (services/references.ts), so remembering one is no step of the desk.

import { useDesk } from '../store/desk.ts';
import { references } from '../services/references.ts';

/** The file states whose files are reference tracks. */
const REFERENCE_FILES = ['timbreFile', 'masterRef'];

useDesk.subscribe((s, prev) => {
  for (const k of REFERENCE_FILES) {
    const f = s.desk.state[k];
    if (typeof f === 'string' && f && f !== prev.desk.state[k]) references.remember(f);
  }
});
