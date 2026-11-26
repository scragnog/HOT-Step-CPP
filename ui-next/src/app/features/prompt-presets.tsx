// Presets & genres (the concept's #ideasBtn, 33-prompt.js): the prompt card's named action `stylePresets` opens the
// popover (package 8's features/pop.ts) by its button with the style presets, which replace the box (one step, said
// with its Undo when it replaced words), and the genres, each pressed while the box names it and pressed again to take
// it out (C-20). What each writes is the core's (src/derive/fields.ts); a supplied caption is not written over.

import { flushSync } from 'react-dom';
import { genreChange, genresIn, GENRES, msg, nodeOf, presetStep, STYLE_PRESETS } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { defineAction } from '../register/actions.ts';
import { useDesk } from '../store/desk.ts';
import { useModel } from '../render/useModel.ts';
import { useSay } from '../i18n/index.ts';
import { closePop, openPop } from './pop.ts';
import { apply, tell } from './prompt-tools.ts';

const KEY = nodeOf(ix, 'promptBox')!.node.param!;

export function PresetsBody() {
  const say = useSay(), on = useModel(s => genresIn(ix, KEY, s.desk), 'presets-genres');
  const preset = (p: string) => { closePop(); const r = presetStep(ix, KEY, useDesk.getState().desk, p); if (r && 'says' in r) tell(r.says); else apply(r); };
  const genre = (g: string) => {
    const c = genreChange(ix, KEY, useDesk.getState().desk, g);
    if (c && 'says' in c) tell(c.says); else if (c) flushSync(() => useDesk.getState().commit(c, 'prompt-presets'));
  };
  return (
    <>
      <h4>{say(msg('presets.title'))}</h4>
      <div className="presets">{STYLE_PRESETS.map(p => <button key={p} type="button" data-preset2={p} onClick={() => preset(p)}>{p}</button>)}</div>
      <h4>{say(msg('presets.genres'))}</h4>
      <div className="chips">{GENRES.map((g, i) => <button key={g} type="button" className="chip" data-genre={g} aria-pressed={!!on[i]} onClick={() => genre(g)}>{g}</button>)}</div>
    </>
  );
}

defineAction('stylePresets', () => { const b = document.getElementById('ideasBtn'); if (b) openPop(b, () => <PresetsBody />); });
