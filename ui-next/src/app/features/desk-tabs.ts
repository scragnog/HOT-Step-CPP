// The desk in two tabs (the concept's 10-working-state.js, C-09; RC-08): what store/desk-tabs.ts hears is said here.
// The desk is written again when the tab is hidden or left (a write the browser refused, or a change not yet written),
// and a newer desk another tab wrote is offered when this tab is looked at: "Another tab has a newer desk. This one
// still shows its own.", with Use that desk, one step that Undo takes back.

import { msg, putWords, savedDeskOf } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { DESK_KEY, deskOf, saveDesk, useDesk } from '../store/desk.ts';
import { deskRead, heard, toOffer } from '../store/desk-tabs.ts';

/** Puts the other tab's desk (storage's, as it is now) on this one, as one step. */
function useTheirs(): void {
  let text: string | null = null;
  try { text = localStorage.getItem(DESK_KEY); } catch { return; }
  const p = deskOf(text); if (!p) return;
  const s = useDesk.getState(), to = { ...s.desk, engine: p.desk.engine, tier: p.desk.tier, vals: p.desk.vals, auto: p.desk.auto, state: p.desk.state, on: p.desk.on };
  /* the desk taken is the one storage holds: it is not written back as this tab's */
  deskRead(JSON.stringify(savedDeskOf(ix, to, p.book)));
  const n = s.undo.length;
  s.deskStep({ label: msg('desk.other.step'), put: putWords(ix, s.desk, to, s.mods.join() !== p.mods.join(), s.board.join() !== p.board.join()) },
    () => useDesk.setState({ desk: to, book: p.book, mods: p.mods, board: p.board }));
  /* the step just made, if the desk changed at all */
  const u = useDesk.getState().undo, entry = u.length > n || (u.length === n && u.at(-1) !== s.undo.at(-1)) ? u.at(-1) : undefined;
  useDesk.getState().showToast({ text: msg('desk.other.taken'), ...(entry ? { action: { label: { id: 'ui.undo' }, run: () => useDesk.getState().runUndo(entry) } } : {}) });
}

/** Offers another tab's newer desk, once per tab, while this one is looked at. */
function offer(): void {
  if (document.visibilityState !== 'visible' || !toOffer()) return;
  useDesk.getState().showToast({ text: msg('desk.other'), action: { label: msg('desk.other.use'), run: useTheirs } });
}

addEventListener('storage', e => { if (e.key !== DESK_KEY || !e.newValue) return; heard(e.newValue); if (document.hasFocus()) offer(); });
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') saveDesk(); else offer(); });
addEventListener('focus', offer);
addEventListener('pagehide', () => { saveDesk(); });
/* a newer desk found at boot (this tab reloaded with its own) is offered once the page is up */
setTimeout(() => { if (document.hasFocus() || document.visibilityState === 'visible') offer(); }, 0);
