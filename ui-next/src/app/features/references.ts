// Every reference track a drop zone takes is remembered as a recent reference (the concept's 13-files.js: a file
// loaded anywhere but Continue's recording joins S.refs), for the Timbre reference tile to offer again. The list is the
// server's (the references service), so remembering one is no step of the desk; it is told to the desk as a server list
// (`refs.recent`), which the recentRefs part draws (src/derive/instruments.ts). Which states hold reference tracks is
// their role (`reference`).

import { roleKeys } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { service } from '../register/services.ts';

const REFERENCE_FILES = roleKeys(ix, 'reference');
const refs = service('references');
const tell = () => useDesk.getState().told({ lists: { 'refs.recent': refs.list().map(f => ({ value: f, label: f })) } });
refs.subscribe(tell); tell();

useDesk.subscribe((s, prev) => {
  for (const k of REFERENCE_FILES) {
    const f = s.desk.state[k];
    if (typeof f === 'string' && f && f !== prev.desk.state[k]) refs.remember(f);
  }
});
