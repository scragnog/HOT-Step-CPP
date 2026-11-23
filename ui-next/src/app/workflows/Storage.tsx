// What the page says about storage (10-working-state.js saveState, 13-stored.js): the read-out before Undo says Saved
// only once every store took its last write, and Not saved while any refused (F-15); a refused write is said in a
// toast (the desk's autosave once, a save asked for every time, after its own toast); what boot could not read is said
// once, with the copies kept aside to download (F-16). The words are the core's.

import { useEffect } from 'react';
import { msg, saveState, storeFailed, storeTold, type StoreWhy } from '../../derive.ts';
import { useDesk, DESK_KEY } from '../store/desk.ts';
import { asideFile, hasAside, useStored } from '../store/stored.ts';
import { useSay } from '../i18n/index.ts';
import { saveFile } from './file.ts';

const fails = (f: { [key: string]: StoreWhy }) => Object.values(f);

export function SaveState() {
  const say = useSay(), f = useStored(s => s.fails), st = saveState(fails(f));
  return (
    <span className={`save-state${st.failed ? ' failed' : ''}`} id="saveState" title={say(st.tip)}>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
      <span>{say(st.text)}</span>
    </span>
  );
}

const download = () => { const f = asideFile(); saveFile(f.name, f.text); useDesk.getState().showToast({ text: msg('store.aside.done') }); };

/** The toasts storage asks for. */
export function StoreToasts() {
  /* what boot could not read, once the page is up */
  useEffect(() => {
    const t = setTimeout(() => {
      const told = useStored.getState().told; if (!told.length) return;
      useStored.setState({ told: [] });
      useDesk.getState().showToast({ text: storeTold(told), ...(hasAside() ? { action: { label: msg('store.aside'), run: download } } : {}) });
    }, 700);
    return () => clearTimeout(t);
  }, []);
  /* a refused write: the desk's autosave says so once until a write succeeds again; a save asked for, every time */
  useEffect(() => {
    let toldDesk = false;
    return useStored.subscribe((s, p) => {
      if (!Object.keys(s.fails).length) toldDesk = false;
      if (!s.failed || s.failed === p.failed) return;
      const desk = s.failed.key === DESK_KEY;
      if (desk && toldDesk) return;
      if (desk) toldDesk = true;
      const text = storeFailed(fails(s.fails), desk);
      setTimeout(() => useDesk.getState().showToast({ text }), 0);
    });
  }, []);
  return null;
}
