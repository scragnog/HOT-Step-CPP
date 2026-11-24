// What the prompt's and the lyric sheet's scenarios call on the concept page beyond the layer's core (testing/compat.ts):
// setFieldText (text put into a field as the page puts it, an input the field takes as typing), renderSuggest and
// paintPromptTools (the concept's repaints, which the rebuild does by itself), and the latent as S.latent.
//
// The concept keeps the box's text in the box itself, so its scenarios set it by writing the field's value directly
// ("#prompt".value = ...), as a scene. Here the text lives in the desk: a field's value written directly is taken
// into the desk as a scene, at the next render() or the next press, key or focus on the page, before the page acts on
// it. Only the prompt card's and the lyric sheet's fields are read so (a checked field elsewhere keeps a draft of its
// own on purpose).

import { flushSync } from 'react-dom';
import type { Node, Value } from '../../../definitions/types.ts';
import { applyChange, bindPart, has, isAuto, setValue, valueOf } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { asScene, expose, fromConcept, onMirror, type Scene } from '../compat.ts';

const FIELDS = '#prompt, .prompt-card input[data-bind], .paper textarea[data-bind]';
const keyOf = (el: HTMLInputElement | HTMLTextAreaElement) => (el.id === 'prompt' ? 'prompt.text' : el.dataset.bind ?? '');

/** The fields whose text differs from the desk's: written directly. */
function written(): Record<string, string> {
  const out: Record<string, string> = {}, d = useDesk.getState().desk;
  for (const el of document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>(FIELDS)) {
    const k = keyOf(el);
    if (el.readOnly || ix.params.get(k)?.p.type !== 'text' || !has(ix, k, d.engine)) continue;
    if (valueOf(ix, k, d) !== el.value) out[k] = el.value;
  }
  return out;
}
function take(vals: Record<string, string>): void {
  const keys = Object.keys(vals); if (!keys.length) return;
  asScene(() => {
    const s = useDesk.getState(), v: Record<string, Value> = {};
    for (const k of keys) { try { v[k] = setValue(ix, s.desk, k, vals[k]!).value; } catch { /* not a text it takes */ } }
    useDesk.setState({ desk: applyChange(s.desk, { vals: v, auto: {}, state: {}, words: { label: { id: '' }, put: { id: '' } } }) });
  });
}

export function install(): void {
  const render = (window as unknown as { render: () => void }).render;
  expose({
    render() { const w = written(), after = sceneOf((window as unknown as { S: Scene }).S); render(); after(); take(w); },
    setFieldText(el: HTMLInputElement | HTMLTextAreaElement, text: string, start = 0, end = el.value.length) {
      take(written());
      el.setRangeText(text, start, end, 'end'); el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    renderSuggest() { /* the suggestions follow the box by themselves */ },
    isAuto(key: string) { return isAuto(ix, key, useDesk.getState().desk); },
    /* a value set by a scenario's script is a step of its own (the concept's t.do starts a gesture of its own), never
       part of the typing a field has open; the layer's core setVal commits with no part, which joins an open gesture */
    setVal(key: string, v: unknown) {
      const s = useDesk.getState(), x = fromConcept(key, v, s.desk.engine), hit = ix.params.get(key);
      if (x === undefined || !hit || !has(ix, key, s.desk.engine)) return;
      /* the values one script sets at once (one t.do) are one step, as the concept's gesture of its own merges them */
      const s0 = useDesk.getState();
      if (!s0.gesture?.startsWith('script|')) { s0.beginGesture('script|'); queueMicrotask(() => useDesk.getState().endGesture('script|')); }
      flushSync(() => bindPart(ix, hit.d, { part: 'knob', param: key } as Node, () => useDesk.getState().desk, c => useDesk.getState().commit(c, 'script')).params[key]?.set(x));
    },
    paintPromptTools() { take(written()); },
  });
  for (const t of ['pointerdown', 'keydown', 'focusin']) addEventListener(t, () => take(written()), true);
  onMirror((S, s) => { S.latent = typeof s.desk.state.latent === 'string' ? s.desk.state.latent : null; });
}

/* What a scene says beyond the layer's core, read before render() applies it (applying it mirrors S afresh): a parameter
   whose key ends in ".auto" (Expand on Generate, wild.auto) is a value, not another one's auto switch, which the core
   takes such a key for; and the latent (S.latent). */
function sceneOf(S: Scene): () => void {
  const autos = Object.entries(S.vals ?? {}).filter(([k, v]) => k.endsWith('.auto') && ix.params.has(k) && typeof v === 'boolean') as [string, Value][];
  const latent = typeof S.latent === 'string' && S.latent ? S.latent : null;
  return () => asScene(() => {
    const s = useDesk.getState();
    useDesk.setState({ desk: { ...s.desk, vals: { ...s.desk.vals, ...Object.fromEntries(autos) }, state: { ...s.desk.state, latent } } });
  });
}
