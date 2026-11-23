// textField: a one-line text field (its LCD look on a pedal or a unit: StableStep's fixed seed). A checked field
// commits on Enter or on leaving it, and while the text fails its check it is kept in the field, unsent, with the
// check's words beside it (the core's); Escape puts back what is in force.

import { useState } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function TextField(v: PartView) {
  const say = useSay(), p = v.params[0];
  const [draft, setDraft] = useState<string | null>(null);
  if (!p) return null;
  /* its look: the node's, else an LCD input on hardware (a pedal, a unit, an LCD), else plain */
  const look = (v.opts.look as string | undefined) ?? (v.surface === 'pedal' || v.surface === 'unit' || v.surface === 'lcd' ? 'lcd' : 'plain');
  const text = draft ?? String(p.value ?? ''), bad = draft !== null ? p.check(draft) : null;
  const commit = () => { if (draft === null) return; if (!p.check(draft)) { if (draft !== p.value) p.set(draft); setDraft(null); } };
  const cls = look === 'lcd' ? 'lcd-input' : look === 'term' ? 't-input' : undefined;
  return (
    <>
      <input className={cls} type="text" data-bind={p.key} value={text}
        aria-label={say(p.label)} aria-invalid={bad ? true : undefined} title={bad ? say(bad) : undefined}
        onChange={e => setDraft(e.currentTarget.value)}
        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } else if (e.key === 'Escape') { e.preventDefault(); setDraft(null); } }}
        onBlur={commit} />
      {bad ? <span className="ckmsg" role="alert">{say(bad)}</span> : null}
    </>
  );
}

definePart('textField', TextField);
