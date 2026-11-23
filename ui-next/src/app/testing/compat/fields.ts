// What the prompt's and the lyric sheet's scenarios call on the concept page beyond the layer's core (testing/compat.ts):
// setFieldText (text put into a field as the page puts it, an input the field takes as typing), renderSuggest and
// paintPromptTools (the concept's repaints, which the rebuild does by itself), and the latent as S.latent.
//
// The concept keeps the box's text in the box itself, so its scenarios set it by writing the field's value directly
// ("#prompt".value = ...), as a scene. Here the text lives in the desk: a field's value written directly is taken
// into the desk as a scene, at the next render() or the next press, key or focus on the page, before the page acts on
// it. Only the prompt card's and the lyric sheet's fields are read so (a checked field elsewhere keeps a draft of its
// own on purpose).

import type { Value } from '../../../definitions/types.ts';
import { applyChange, has, setValue, valueOf } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { useDesk } from '../../store/desk.ts';
import { asScene, expose, onMirror, onScene } from '../compat.ts';

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
    render() { const w = written(); render(); take(w); },
    setFieldText(el: HTMLInputElement | HTMLTextAreaElement, text: string, start = 0, end = el.value.length) {
      take(written());
      el.setRangeText(text, start, end, 'end'); el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    renderSuggest() { /* the suggestions follow the box by themselves */ },
    paintPromptTools() { take(written()); },
  });
  for (const t of ['pointerdown', 'keydown', 'focusin']) addEventListener(t, () => take(written()), true);
  onMirror((S, s) => { S.latent = typeof s.desk.state.latent === 'string' ? s.desk.state.latent : null; });
  onScene(S => {
    /* a parameter whose key ends in ".auto" (Expand on Generate, wild.auto) is a value, not another one's auto switch,
       which the layer's core takes such a key for */
    const vals = Object.entries(S.vals ?? {}).filter(([k, v]) => k.endsWith('.auto') && ix.params.has(k) && typeof v === 'boolean');
    if (vals.length) { const s = useDesk.getState(); useDesk.setState({ desk: { ...s.desk, vals: { ...s.desk.vals, ...Object.fromEntries(vals) as Record<string, Value> } } }); }
    const s = useDesk.getState(), want = typeof S.latent === 'string' && S.latent ? S.latent : null;
    if ((s.desk.state.latent ?? null) !== want) useDesk.setState({ desk: { ...s.desk, state: { ...s.desk.state, latent: want } } });
  });
}
