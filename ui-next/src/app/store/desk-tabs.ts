// The desk in two tabs (the concept's 10-working-state.js, C-09; RC-08). Every tab writes the desk it holds as it
// changes, marked with the tab that wrote it and when; the tab changed last wins the next visit. A tab keeps its own
// last desk for itself too (sessionStorage, which a reload keeps), so reloading a tab shows the desk it was showing even
// when another tab wrote since, and that newer desk is offered instead of lost. A tab that hears another tab write
// keeps its own desk and offers the other one once per tab, when it is next looked at (features/desk-tabs.ts says it);
// nothing is written over a newer desk unless this tab changes.

import { writeStored } from './stored.ts';

/** This page load's mark on what it writes. */
const TAB = Math.random().toString(36).slice(2, 10);
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const isRec = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && !Array.isArray(x);
const parse = (t: string | null): Record<string, unknown> | null => { try { const v = t ? JSON.parse(t) as unknown : null; return isRec(v) ? v : null; } catch { return null; } };
const own = (key: string): string | null => { try { return sessionStorage.getItem(key); } catch { return null; } };

/** The desk this tab last wrote (its record's text, without the mark) and when. */
let written = '', wroteAt = 0;
/** A newer desk another tab wrote, not yet offered: which tab, and when. */
let elsewhere: { tab: string; at: number } | null = null;
const offered = new Set<string>();

/** The text the desk is read from at boot: storage's, or this tab's own when another tab has written since this tab
 *  last did (its newer desk is then offered). A desk in storage that no tab marked (an older page, a test, a hand) is
 *  read as it is. */
export function bootDesk(key: string): string | null {
  let text: string | null = null;
  try { text = localStorage.getItem(key); } catch { return own(key); }
  const theirs = parse(text), mine = parse(own(key));
  if (mine && theirs && typeof theirs.tab === 'string' && theirs.tab !== mine.tab && isNum(theirs.at)) {
    elsewhere = { tab: theirs.tab, at: theirs.at }; wroteAt = isNum(mine.at) ? mine.at : 0;
    return own(key);
  }
  return text;
}
/** The desk was read at boot (its record's text): it counts as written, so it is not written again unchanged. */
export function deskRead(text: string): void { written = text; }

/** Writes the desk's record if it is not what this tab last wrote; false when the browser refused. */
export function writeDesk(key: string, text: string): boolean {
  if (text === written) return true;
  const at = Date.now(), marked = JSON.stringify({ ...(JSON.parse(text) as object), tab: TAB, at });
  try { sessionStorage.setItem(key, marked); } catch { /* this tab's own copy is a convenience */ }
  if (!writeStored(key, marked)) return false;
  written = text; wroteAt = at; elsewhere = null;
  return true;
}

/** Another tab wrote the desk (a storage event's new text): remembered, to be offered. */
export function heard(text: string | null): void {
  const st = parse(text);
  if (st && typeof st.tab === 'string' && st.tab !== TAB && isNum(st.at)) elsewhere = { tab: st.tab, at: st.at };
}
/** A newer desk to offer now: once per other tab, and only while it is newer than what this tab wrote. */
export function toOffer(): boolean {
  const e = elsewhere; if (!e || e.at <= wroteAt) return false;
  elsewhere = null;
  if (offered.has(e.tab)) return false;
  offered.add(e.tab);
  return true;
}
