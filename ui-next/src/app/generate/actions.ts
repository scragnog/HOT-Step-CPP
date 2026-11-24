// What the go bar, Latest and the song menu do (35-gobar.js generate, cancelAll, showChanges; 35-songs.js the song
// menu): each asks the core what to do and what to say, starts or cancels runs in the generation store, and changes
// the desk only through a part's setter or one step the core names.

import { flushSync } from 'react-dom';
import type { Node, Value } from '../../definitions/types.ts';
import { bindPart, boxHolds, changesUnmarked, refusalOf, repaintDesk, reuseWords, runOf, sentTo, type Change } from '../../derive.ts';
import { BUILT_IN, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { cancelStep, dismissCard, startRun, useGen, type Card } from '../store/generate.ts';
import { runNamed } from '../register/actions.ts';
import { setTierFromView } from '../shell/tier.ts';
import { ui } from '../i18n/index.ts';
import { inView, reduced } from './motion.ts';

/** A value set as a control sets it: one step, named by the core; `words` renames the step (Reuse). */
export function setValue(key: string, v: Value, label?: Change['words']['label']): void {
  const hit = ix.params.get(key); if (!hit) return;
  const commit = (c: Change) => useDesk.getState().commit(label ? { ...c, words: { ...c.words, label } } : c);
  bindPart(ix, hit.d, { part: 'knob', param: key } as Node, () => useDesk.getState().desk, commit).params[key]?.set(v as never);
}
const focus = (sel: string) => requestAnimationFrame(() => document.querySelector<HTMLElement>(sel)?.focus({ preventScroll: true }));
/** A link or a control pulses where it is (the concept's ping). */
export function ping(els: Element[]): void {
  for (const x of els) { x.classList.remove('ping'); void (x as HTMLElement).offsetWidth; x.classList.add('ping'); setTimeout(() => x.classList.remove('ping'), reduced() ? 1200 : 1100); }
}

/** A press of Generate (or Ctrl+Enter): a refusal is said, with its way out; otherwise the run starts and its first
 *  card comes into sight if it is not (the page moves only as far as it needs to, G-M-06). */
export function generate(o: { seed?: number } = {}): boolean {
  const s = useDesk.getState(), r = refusalOf(ix, s.desk);
  if (r?.kind === 'empty') {
    s.showToast({ text: r.says, action: { label: ui('gen.example'), run: () => { setValue('prompt.text', r.example); focus('#prompt'); } } });
    focus('#prompt'); return false;
  }
  if (r?.kind === 'score') { runNamed('scorePreview'); return false; }
  if (r?.kind === 'clash') { s.showToast({ text: r.says }); return false; }
  /* a run certain not to fit is not started: the way out pulses and the line is said (F-21) */
  if (r?.kind === 'wontFit') { ping([...document.querySelectorAll('#meter [data-takesfit]')]); s.say(r.says); return false; }
  const spec = runOf(ix, s.desk, Math.random, o.seed);
  for (const m of spec.says) s.showToast({ text: m });
  const run = flushSyncRun(() => startRun(spec));
  requestAnimationFrame(() => { const c = document.querySelector(`#latest .song[data-id="${run.ids[0]}"]`); if (c && !inView(c)) c.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' }); });
  return true;
}
const flushSyncRun = <T,>(fn: () => T): T => { let out!: T; flushSync(() => { out = fn(); }); return out; };

/** Cancel all: one step, with Undo (F-13); focus stays on Generate. */
export function cancelAll(): void {
  const q = useGen.getState().cards.filter(c => c.pending); if (!q.length) return;
  cancelStep(new Set(q.map(c => c.id)), null); focus('#go');
}
/** × on a card: that song only; the rest carry on. Focus goes to Generate, never to the next song's × (F-13). */
export function cancelOne(id: string): void {
  const c = useGen.getState().cards.find(x => x.id === id); if (!c?.pending) return;
  cancelStep(new Set([id]), c); focus('#go');
}
/** Retry a refused run with the takes that fit: the notice goes, Takes is set (a step), and Generate runs. */
export function retry(id: string): void {
  const c = useGen.getState().cards.find(x => x.id === id); if (!c?.error) return;
  dismissCard(id); setValue('takes.n', c.error.retry); generate(); focus('#go');
}
/** Dismiss a notice: focus goes to the next notice's first button, else to Generate (F-13). */
export function dismiss(id: string): void {
  dismissCard(id);
  const next = useGen.getState().cards.find(x => x.error);
  focus(next ? `#latest .song.errored[data-id="${next.id}"] button` : '#go');
}

/** "N changes": every changed control on the desk pulses; with none drawn, the core says where they are (B-14, Q-15). */
export function showChanges(): void {
  const els = [...document.querySelectorAll('#bento [data-ctl].changed, #bento .changed[data-ctl], #bento .pedal .kn.changed, #bento .cart, #bento .q-also')].filter(x => (x as HTMLElement).offsetParent);
  if (!els.length) {
    const u = changesUnmarked(ix, useDesk.getState().desk);
    useDesk.getState().showToast({ text: u.says, ...(u.lab ? { action: { label: ui('gen.showLab'), run: () => setTierFromView('lab', { focus: true }) } } : {}) });
    return;
  }
  if (!els.some(inView)) els[0]!.scrollIntoView({ block: 'center', behavior: reduced() ? 'auto' : 'smooth' });
  ping(els);
}
/** "Use N takes": Takes set to what fits, said. */
export function takesFit(n: number): void { setValue('takes.n', n); useDesk.getState().say(ui('gen.takesSet', { n })); }

/* ---------------------------------------------------------------- the song menu */

/** "Repaint a region": one step (with the switch to ACE-Step when it is elsewhere), one toast with Undo (E-14). */
export function repaint(c: Card): void {
  const s = useDesk.getState(), r = repaintDesk(ix, s.desk, s.book, s.mods, c);
  s.deskStep({ label: r.label, put: r.label }, () => useDesk.setState({ desk: { ...r.desk, on: new Set([...BUILT_IN, ...r.mods, ...s.board]) }, book: r.book, mods: r.mods }));
  const entry = useDesk.getState().undo[useDesk.getState().undo.length - 1]!;
  useDesk.getState().showToast({ text: r.says, action: { label: ui('undo'), run: () => useDesk.getState().runUndo(entry) } });
}
/** "Reuse its description": the text it was made from goes back in the box, one step; if the box held something, the
 *  toast says it was replaced, with Undo (E-20). */
export function reuse(c: Card): void {
  const s = useDesk.getState(), had = boxHolds(ix, s.desk).trim(), w = reuseWords(s.desk.engine);
  setValue('prompt.text', c.text ?? c.style, w.label);
  const entry = useDesk.getState().undo[useDesk.getState().undo.length - 1];
  if (had && entry) useDesk.getState().showToast({ text: w.replaced, action: { label: ui('undo'), run: () => useDesk.getState().runUndo(entry) } });
}
/** An item that opens another page says where it goes. */
export function sendTo(page: string, c: Card): void { useDesk.getState().showToast({ text: sentTo(page, c.title) }); }
