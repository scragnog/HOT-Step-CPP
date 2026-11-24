// The lead-sheet preview's own state (features/score-preview.tsx draws it): what it shows (the sheet's own score, or
// one the planner wrote, with the core's drawing of it), where focus goes back to, opening and closing it, and saving
// its ABC. Not a step, not saved: it is a view of the planner's answer.

import { useSyncExternalStore } from 'react';
import { abcInfo, msg, scoreAsk, scoreModel, type Score, type ScoreModel, type Slot } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { scorePlanner } from '../services/score.ts';

export type View = { pasted: { abc: string; title: string; meta: Slot; file: string } } | { score: Score; m: ScoreModel } | null;
let view: View = null, back: HTMLElement | null = null;
const subs = new Set<() => void>();
const show = (v: View) => { view = v; subs.forEach(f => f()); };
export const useView = () => useSyncExternalStore(f => { subs.add(f); return () => subs.delete(f); }, () => view);

/** Opens the preview: the sheet's own score, or one planned from `seed` (a fresh one when none is given). */
export async function open(seed?: number): Promise<void> {
  const ask = scoreAsk(ix, useDesk.getState().desk); if (!ask) return;
  if (!view) back = document.activeElement as HTMLElement | null;
  if (ask.abc && seed === undefined) { show({ pasted: { abc: ask.abc, ...abcInfo(ask.abc) } }); return; }
  const score = await scorePlanner.plan(seed ?? Math.floor(Math.random() * 2147483647), ask);
  show({ score, m: scoreModel(score) });
}
export function close(): void { show(null); const b = back; back = null; if (b?.isConnected) b.focus({ preventScroll: true }); }
export function save(name: string, text: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
  useDesk.getState().showToast({ text: msg('score.saved') });
}

/** The score shown (the test layer's SC). */
export const scoreShown = (): Score | null => (view && 'score' in view ? view.score : null);
