// The read-outs: they show something and edit nothing. note is a quiet line of words (on a pedal, its small print with
// its actions after it as links); lcdValue a value printed big over a caps word (a pedal's Studio face); unitLcd a
// side unit's two-line LCD; button a small button that runs its node's action. What each says, and what its actions
// do, are the core's.

import { definePart, ui, useSay, type PartView } from '../kit/index.ts';

/** A node's actions as links or buttons. */
export function ActionLinks({ v, look }: { v: PartView; look: 'link' | 'button' }) {
  const say = useSay();
  return (
    <>
      {v.actions.map((a, i) => look === 'link'
        ? <button key={i} type="button" className="linkbtn" data-tier={a.tier} data-ring={a.ring} title={a.tip ? say(a.tip) : undefined} onClick={() => v.run(i)}>{say(a.label)}</button>
        : <button key={i} type="button" className="pbtn" title={a.tip ? say(a.tip) : undefined} onClick={() => v.run(i)}>{say(a.label)}</button>)}
    </>
  );
}

export function Note(v: PartView) {
  const say = useSay(), tip = v.words.tip ? say(v.words.tip) : undefined, text = say(v.words.text);
  if (v.surface === 'pedal') return <small className="pnote" title={tip}>{v.actions.length ? say(ui('note.then', { text })) : text}<ActionLinks v={v} look="link" /></small>;
  return <span className="lbl" title={tip}>{text}</span>;
}

export function LcdValue(v: PartView) {
  const say = useSay();
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

export const Button = (v: PartView) => <ActionLinks v={v} look="button" />;

definePart('note', Note);
definePart('lcdValue', LcdValue);
definePart('unitLcd', UnitLcd);
definePart('button', Button);
