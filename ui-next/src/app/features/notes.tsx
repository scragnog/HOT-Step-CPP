// The Concept notes (the concept's 34-notes.js, overlays/18-notes.html and notes/*.html): a modal drawer opened from
// the top bar's notes button, over a scrim. It slides in from the right (the concept's stylesheet), focus goes to its
// close button, Tab goes round it (and round a toast that shows meanwhile, F-18), the page's shortcuts wait under it,
// and Escape, its ×, or a press on the scrim close it, the drawer sliding out and focus going back to the button
// (notes-open.ts, layers.ts). Its edges fade where there is more to read; its × stays in reach as it scrolls. What it
// says is the page's words (i18n/words/notes.ts), drawn into the skeleton's #notes; the scrim is the skeleton's
// #notesScrim.

import { Fragment, useEffect, useLayoutEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { PAGE_WORDS, ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { fill } from '../register/slots.ts';
import { useEdgeFade } from '../shell/edgeFade.ts';
import { useNotesLogs } from '../store/notes-logs.ts';
import { closeNotes, notesDrawer, notesScrim } from './notes-open.ts';

/** The notes' sections, in the concept's order (notes/10-tiers.html ... 34-grounding.html). */
const SECTIONS = ['tiers', 'adding', 'workflows', 'kept', 'grounding'] as const;
/** A section's points: every `ui.notes.<section>.<n>` the words hold, from 1. */
const pointsOf = (s: string): string[] => { const out: string[] = []; for (let i = 1; PAGE_WORDS.has(`ui.notes.${s}.${i}`); i++) out.push(`notes.${s}.${i}`); return out; };

/** Words with the concept's bold marks (`<b>...</b>`) drawn in bold. */
const rich = (text: string): ReactNode[] => text.split(/<b>(.*?)<\/b>/).map((s, i) => (i % 2 ? <b key={i}>{s}</b> : <Fragment key={i}>{s}</Fragment>));

export function Notes() {
  const say = useSay(), open = useNotesLogs(s => s.notes), ref = useRef<HTMLElement | null>(notesDrawer());
  useLayoutEffect(() => {
    const n = notesDrawer(), sc = notesScrim(); if (!n || !sc) return;
    n.hidden = !open; sc.hidden = !open;
  }, [open]);
  useLayoutEffect(() => {
    const n = notesDrawer(); if (!n) return;
    n.setAttribute('aria-labelledby', 'notesTitle'); n.classList.add('fade-own');
  }, []);
  useEdgeFade(ref);
  /* a press on the scrim closes it (the top bar's notes button opens it: shell/TopBar.tsx) */
  useEffect(() => {
    const press = (e: MouseEvent) => { if (e.target === notesScrim()) closeNotes(); };
    document.addEventListener('click', press);
    return () => document.removeEventListener('click', press);
  }, []);
  const n = notesDrawer(); if (!n) return null;
  return createPortal(
    <>
      <button type="button" className="x close" id="notesClose" aria-label={say(ui('notes.close'))} onClick={closeNotes}><Icon name="x" size={15} /></button>
      <h2 id="notesTitle">{say(ui('notes.title'))}</h2>
      <p style={{ color: 'var(--muted)', margin: 0 }}>{say(ui('notes.sub'))}</p>
      {SECTIONS.map(s => (
        <Fragment key={s}>
          <h3>{say(ui(`notes.${s}.title`))}</h3>
          <ul>{pointsOf(s).map(id => <li key={id}>{rich(say(ui(id)))}</li>)}</ul>
        </Fragment>
      ))}
    </>, n);
}
fill('overlays', Notes);
