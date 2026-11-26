// The prompt card's and the lyric sheet's tools (33-prompt.js, 25-module-lyrics.js), each a named action their nodes
// list (parts.ts ACTIONS): what each writes, its step's words and what it says after are the core's
// (src/derive/fields.ts); this applies the step, says it with Undo where the concept offers it, and puts focus where
// the step leaves it. A step that brings a module (the tempo field, a latent's tempo) brings it in the same step.

import { flushSync } from 'react-dom';
import {
  addTo, applyChange, bindPart, captionSources, composeCaption, copyStep, datasetStep, expandStep, insertTag, latentClear, latentStep, lockedMsg, moveToLyrics, msg,
  nodeOf, plainStep, tagStep, tempoStep, writeForMe, type Change, type Msg, type ToolStep,
} from '../../derive.ts';
import type { Json } from '../../definitions/types.ts';
import { ix } from '../model/index.ts';
import { defineAction } from '../register/actions.ts';
import { useDesk, type Entry } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';

const PROMPT = nodeOf(ix, 'promptBox')!, LYRICS = nodeOf(ix, 'lyricSheet')!, KEY = PROMPT.node.param!;
const store = () => useDesk.getState();
const desk = () => store().desk;
/* the tools' steps are their own, never merged into a gesture another part has open */
const FROM = 'prompt-tools';

export const tell = (text: Msg) => store().showToast({ text });
/** Says what a step did, with Undo of that step (the one the tool took, never the newest in the history). */
function withUndo(text: Msg, e: Entry | null): void {
  store().showToast(e ? { text, action: { label: ui('undo'), run: () => store().runUndo(e) } } : { text });
}
/** Applies a tool's step and says it; false when it changed nothing. */
export function apply(r: ToolStep | null, from = FROM): boolean {
  if (!r) return false;
  let e: Entry | null = null;
  flushSync(() => { e = store().commit(r.change, from); });
  if (r.said) { if (r.undo && e) withUndo(r.said, e); else tell(r.said); }
  return true;
}
/** A step that brings the module a value lives on, when it is not on the desk: one step, as deskStep records one. */
function bringing(words: Change['words'], change: { vals: Change['vals']; auto: Change['auto']; state?: { [k: string]: Json } }, brings: string | null): Entry | null {
  let e: Entry | null = null;
  flushSync(() => { e = store().deskStep(words, () => {
    const s = store(), d = applyChange(s.desk, { vals: change.vals, auto: change.auto, state: change.state ?? {}, words });
    const r = brings ? addTo(ix, s.desk, s.mods, s.board, brings, undefined, s.gone) : null;
    store().applyDesk(r?.did === 'added' ? { desk: d, mods: r.mods, board: r.board, flash: brings! } : { desk: d });
  }); });
  return e;
}
/* focus that a tool's own button took with it when it went (the box's Expand once the wildcards are gone) goes to the box */
const keepFocus = (sel: string) => requestAnimationFrame(() => { const a = document.activeElement; if (!a || a === document.body || !(a as HTMLElement).offsetParent) document.querySelector<HTMLElement>(sel)?.focus({ preventScroll: true }); });
const field = (sel: string) => document.querySelector<HTMLTextAreaElement>(sel);
const LYRICS_FIELD = `textarea[data-bind="${LYRICS.node.param}"]`;

defineAction('promptWrite', arg => {
  const r = writeForMe(ix, KEY, desk(), arg === 'replace');
  if ('ask' in r) { store().showToast({ text: r.ask, action: { label: msg('write.replace'), run: () => { const x = writeForMe(ix, KEY, desk(), true); if ('change' in x) apply(x); } } }); return; }
  if ('says' in r) { tell(r.says); return; }
  apply(r);
});
defineAction('composeCaption', () => {
  const r = composeCaption(ix, KEY, desk());
  if ('says' in r) { tell(r.says); if (r.focus) field('#prompt')?.focus(); return; }
  apply(r);
});
defineAction('expandWildcards', arg => {
  const r = expandStep(ix, KEY, desk(), arg === 'lyrics' ? 'lyrics' : 'prompt', Math.random);
  if ('says' in r) { tell(r.says); return; }
  apply(r); if (arg !== 'lyrics') keepFocus('#prompt');
});
/* F-13: the lyrics it moved are where focus goes */
defineAction('moveToLyrics', () => { if (apply(moveToLyrics(ix, KEY, desk()))) requestAnimationFrame(() => field(LYRICS_FIELD)?.focus({ preventScroll: true })); });
defineAction('tempoFromPrompt', () => {
  const r = tempoStep(ix, KEY, desk()); if (!r) return;
  withUndo(r.said, bringing(r.words, r, r.brings));
});
defineAction('plainCaption', () => { apply(plainStep(ix, KEY, desk())); keepFocus('#prompt'); });
defineAction('editCaptionCopy', () => { apply(copyStep(ix, KEY, desk())); requestAnimationFrame(() => field('#prompt')?.focus({ preventScroll: true })); });
defineAction('datasetTag', tag => { if (tag) apply(tagStep(ix, KEY, desk(), tag)); });
defineAction('chooseDataset', ds => { if (ds !== undefined) apply(datasetStep(ix, desk(), ds)); });
defineAction('importLatent', name => {
  if (!name) return;
  const r = latentStep(ix, KEY, desk(), name);
  if ('says' in r) { tell(r.says); return; }
  withUndo(r.said, bringing(r.words, r, r.brings));
});
defineAction('clearLatent', () => { apply(latentClear(ix, desk())); keepFocus('#latBtn [data-file="latent"]'); });
defineAction('promptLocked', () => store().say(lockedMsg(ix, desk())));

/* the lyric sheet's tabs: a tag goes in at a line boundary, never inside a word, and the caret lands on the line after
   it; with no caret in the sheet yet (never clicked into since it was drawn), at the end (G-B-06). With Instrumental on
   the sheet is kept as it is, and says why. The tag is typing in the sheet: one step with the typing around it. */
const visited = new WeakSet<Element>();
document.addEventListener('focusin', e => { if ((e.target as Element).matches?.(LYRICS_FIELD)) visited.add(e.target as Element); });
defineAction('insertSectionTag', (tag, from) => {
  const ta = field(LYRICS_FIELD); if (!tag || !ta) return;
  if (ta.readOnly) { tell(msg('lyrics.instrOn')); return; }
  const caret = document.activeElement === ta || visited.has(ta) ? ta.selectionEnd : null, r = insertTag(ta.value, caret, tag);
  try { flushSync(() => bindPart(ix, LYRICS.d, LYRICS.node, desk, c => store().commit(c, from)).params[LYRICS.node.param!]?.set(r.text)); } catch { return; }
  ta.focus({ preventScroll: true }); ta.setSelectionRange(r.caret, r.caret); visited.add(ta);
  if (caret === null) ta.scrollTop = ta.scrollHeight;
});

/* the caption source lists the tracks of the dataset in force: the page asks for them when the dataset changes (the
   simulated engine holds every dataset's tracks) */
let listed: string | null = null;
useDesk.subscribe(s => {
  const c = captionSources(ix, s.desk); if (!c || c.ds === listed) return;
  listed = c.ds; s.told({ lists: { ...s.desk.lists, 'caption.sources': c.list } });
});
