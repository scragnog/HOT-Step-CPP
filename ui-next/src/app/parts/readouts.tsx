// The read-outs: they show something and edit nothing. note is a quiet line of words (on a pedal, its small print with
// its actions after it as links); lcdValue a value printed big over a caps word (a pedal's Studio face); unitLcd a
// side unit's two-line LCD; button a small button that runs its node's action. What each says, and what its actions
// do, are the core's.

import { useContext } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';
import { NoteInline } from './inline.ts';

/** A node's actions as links or buttons. */
export function ActionLinks({ v, look }: { v: PartView; look: 'link' | 'button' | 'term' }) {
  const say = useSay();
  return (
    <>
      {v.actions.map((a, i) => look === 'link'
        ? <button key={i} type="button" className="linkbtn" data-auto={a.auto} data-tier={a.tier} data-ring={a.ring} title={a.tip ? say(a.tip) : undefined} onClick={() => v.run(i)}>{say(a.label)}</button>
        : <button key={i} type="button" className={look === 'term' ? 'tbtn' : 'pbtn'} title={a.tip ? say(a.tip) : undefined} onClick={() => v.run(i)}>{say(a.label)}</button>)}
    </>
  );
}

export function Note(v: PartView) {
  const say = useSay(), tip = v.words.tip ? say(v.words.tip) : undefined, text = say(v.words.text), inline = useContext(NoteInline);
  /* one of the notes of a line that says what a choice does (the lead sheet's lyric timing): its words and links */
  if (inline) return <>{text}{v.actions.length ? ' ' : null}<ActionLinks v={v} look="link" /></>;
  if (v.surface === 'pedal') return <small className="pnote" title={tip}>{v.actions.length ? say(ui('note.then', { text })) : text}<ActionLinks v={v} look="link" /></small>;
  /* in a terminal, its quiet ink; in an LCD's row, the LCD's dimmed words */
  if (v.surface === 'term') return <span className="k" title={tip}>{text}</span>;
  if (v.surface === 'lcdrow') return <span className="lcd-dim" title={tip}>{text}</span>;
  /* the caption note under an instrument (Tempo's), with its actions after it; Quick's hint */
  if (v.opts.look === 'cap') return <div className="cap-note" title={tip}>{text}{v.actions.length ? ' ' : null}<ActionLinks v={v} look="link" /></div>;
  if (v.opts.look === 'hint') return <span className="qhint" title={tip}>{text}</span>;
  return <span className="lbl" title={tip}>{text}</span>;
}

export function LcdValue(v: PartView) {
  const say = useSay();
  /* in an LCD's row: its caps word, then the value lit (Tempo's KEY AUTO) */
  if (v.surface === 'lcdrow') {
    const r = v.reads[0];
    return <span><span className={`cdot-lcd${r?.changed ? ' changed' : ''}`} data-ctl={r?.key}>{say(v.label ?? '', { upper: true })}</span> <b data-keylcd="">{say(v.words.text ?? r?.lcd ?? '', { upper: true })}</b></span>;
  }
  return (
    <>
      <span className="bigval">{say(v.words.text ?? v.reads[0]?.lcd ?? '')}</span>
      <small>{say(v.label)}{v.actions.length ? <><br /><ActionLinks v={v} look="link" /></> : null}</small>
    </>
  );
}

export function UnitLcd(v: PartView) {
  const say = useSay();
  if (v.words.text) return <span className="pwin">{say(v.words.text)}</span>;
  return <span className="u-lcd" aria-hidden="true">{(v.lines ?? []).map((l, i) => <span key={i}>{say(l.text, l.upper ? { upper: true } : { lower: true })}</span>)}</span>;
}

export const Button = (v: PartView) => <ActionLinks v={v} look={v.surface === 'term' ? 'term' : 'button'} />;

definePart('note', Note);
definePart('lcdValue', LcdValue);
definePart('unitLcd', UnitLcd);
definePart('button', Button);
