// textField: a one-line text field. Its look: the node's, else an LCD input on hardware (a pedal, a unit, an LCD), else
// the page's plain field; `term` is an inline terminal input that grows with what is typed (the planner's --seed),
// `size` its width in characters. A checked field commits on Enter or on leaving it; while the text fails its check
// it is marked (bad, aria-invalid) with the check's words beside it, and nothing is sent. A field whose check keeps what
// was typed (keepInvalid) keeps it after leaving, says what is in force meanwhile and offers it back (E-33); Escape
// puts back what is in force. The check, its words and what is in force are the core's.

import { useId, useState } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';

export function TextField(v: PartView) {
  const say = useSay(), p = v.params[0], id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const [left, setLeft] = useState(false);
  if (!p) return null;
  const look = (v.opts.look as string | undefined) ?? (v.surface === 'pedal' || v.surface === 'unit' || v.surface === 'lcd' ? 'lcd' : 'plain');
  const text = draft ?? String(p.value ?? ''), bad = draft !== null ? p.check(draft) : null;
  /* a valid draft is set; an invalid one in a field that keeps it stays, unsent; in any other field it is refused (the
     core says so) and what is in force comes back */
  const commit = () => { if (draft === null) return; if (!p.check(draft)) { if (draft !== p.value) p.set(draft); setDraft(null); setLeft(false); } else if (p.kept) setLeft(true); else { p.set(draft); setDraft(null); } };
  const back = () => { setDraft(null); setLeft(false); };
  const cls = [look === 'lcd' ? 'lcd-input' : look === 'term' ? 'tok-in' : '', look === 'term' && p.changed ? 'changed' : '', bad ? 'bad' : ''].filter(Boolean).join(' ') || undefined;
  const size = typeof v.opts.size === 'number' ? v.opts.size : look === 'term' ? Math.max(4, text.length) : undefined;
  const kept = bad && left ? p.kept ?? null : null;
  return (
    <>
      <input className={cls} type="text" data-bind={p.key} data-ctl={look === 'term' ? p.key : undefined} value={text} size={size}
        placeholder={p.placeholder ? say(p.placeholder) : undefined} aria-label={say(p.label)} aria-invalid={bad ? true : undefined} aria-describedby={bad ? id : undefined} title={bad ? say(bad) : undefined}
        onChange={e => setDraft(e.currentTarget.value)}
        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } else if (e.key === 'Escape') { e.preventDefault(); back(); } }}
        onBlur={commit} />
      {bad ? (
        <span className={`field-err${look === 'term' ? ' term-err' : ''}`} id={id} data-err={p.key} role="status">
          {say(bad)}{kept ? <>{` ${say(kept.says)}`}{kept.back ? <>{' '}<button type="button" className="linkbtn" data-putback={p.key} onClick={back}>{say(ui('field.putBack'))}</button></> : null}</> : null}
        </span>
      ) : null}
    </>
  );
}

definePart('textField', TextField);
