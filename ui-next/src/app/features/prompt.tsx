// The prompt card and the lyric sheet in the shell (create/20-centre.html, 30-desk.js renderDesk, 25-module-lyrics.js
// lyricsBlock): the card under the ask, the lyric sheet as a card under it at Quick, and at Studio and Lab as the tile
// at the head of the desk; on an engine with no lyrics, the note that says so where the sheet would be. Each draws its
// definition's face with the generic renderer (the promptBox and lyricSheet parts); where it goes is the core's
// (lyricsPlace). The box is registered as the field that edits the prompt's text (store/fields.ts).

import { categoryWord, lyricsPlace, nodeOf, tileName } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { fill } from '../register/slots.ts';
import { Face } from '../render/Face.tsx';
import { useModel } from '../render/useModel.ts';
import { registerField } from '../store/fields.ts';
import { ui, useSay } from '../i18n/index.ts';

const PROMPT = nodeOf(ix, 'promptBox')!, LYRICS = nodeOf(ix, 'lyricSheet')!;
registerField('prompt', PROMPT.node.param!);

export function PromptCard() {
  return <Face def={PROMPT.d} surface="tile" only={n => n === PROMPT.node} />;
}

/** Where the sheet goes, and its name and group on the desk. */
const usePlace = () => useModel(s => ({ ...lyricsPlace(ix, s.desk), quick: s.desk.tier === 'quick', name: tileName(LYRICS.d, s.desk), group: LYRICS.d.kind === 'module' ? categoryWord(LYRICS.d.category) : '' }), 'lyrics-place');

/** Quick: the sheet as a card under the prompt card, or the note. */
export function QuickLyrics() {
  const say = useSay(), m = usePlace();
  if (!m.quick) return null;
  if (!m.here) return m.note ? <div className="cap-note">{say(m.note)}</div> : null;
  return <div className="lyr-card"><div className="head"><h3>{say(m.name)}</h3></div><Face def={LYRICS.d} surface="quick" only={n => n === LYRICS.node} /></div>;
}

/** Studio and Lab: the sheet as the tile at the head of the desk (the layout keeps it in the first column), or the note
 *  in the row the desk keeps for it. */
export function LyricsLead() {
  const say = useSay(), m = usePlace();
  if (!m.here) return m.note ? <div className="cap-note">{say(m.note)}</div> : null;
  return (
    <article className="tile w2 tall" data-m={LYRICS.d.id} data-fill="">
      <header className="tile-head"><span className="grp">{say(m.group)}</span><h3>{say(m.name)}</h3><span className="lbl">{say(ui('lyrics.builtIn'))}</span></header>
      <div className="tile-body"><Face def={LYRICS.d} surface="tile" only={n => n === LYRICS.node} /></div>
    </article>
  );
}

fill('centre.prompt', PromptCard);
fill('centre.quickLyrics', QuickLyrics);
fill('desk.lead', LyricsLead);
