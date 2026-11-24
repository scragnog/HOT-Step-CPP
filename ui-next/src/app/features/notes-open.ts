// Opening and closing the Concept notes (the concept's 34-notes.js openNotes, closeNotes; the view is notes.tsx): the
// drawer and its scrim show, the drawer is the top layer (modal: Tab goes round it, Escape closes it, the page's
// shortcuts wait) and focus goes to its ×; closing plays its exit and gives focus back to the notes button
// (features/layers.ts).

import { flushSync } from 'react-dom';
import { useNotesLogs } from '../store/notes-logs.ts';
import { layerClosed, layerOpen } from './layers.ts';
import { ovOut } from './overlay-exit.ts';

export const notesDrawer = (): HTMLElement | null => document.getElementById('notes');
export const notesScrim = (): HTMLElement | null => document.getElementById('notesScrim');

export function openNotes(): void {
  const n = notesDrawer(); if (!n || !n.hidden) return;
  flushSync(() => useNotesLogs.setState({ notes: true }));
  layerOpen(n, { trigger: document.getElementById('notesBtn'), modal: true, close: closeNotes });
  document.getElementById('notesClose')?.focus({ preventScroll: true });
}
export function closeNotes(): void {
  const n = notesDrawer(); if (!n || n.hidden) return;
  ovOut(n, 'drawer'); ovOut(notesScrim(), 'fade');
  flushSync(() => useNotesLogs.setState({ notes: false }));
  layerClosed(n);
}
