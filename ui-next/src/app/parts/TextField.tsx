// textField: a one-line text field (its LCD look on a pedal: StableStep's fixed seed). A checked field commits on Enter
// or on leaving it, and while the text fails its check it is kept in the field, unsent, with the check's words beside
// it (the core's checkText); Escape puts back what is in force.

import { useState } from 'react';
import { checkText, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import type { ViewProps } from './types.ts';

export function TextField(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const p = Object.values(props.params)[0];
  const [draft, setDraft] = useState<string | null>(null);
  if (!p) return null;
  const look = (props.node.opts as { look?: string } | undefined)?.look ?? (props.cell ? 'lcd' : 'plain');
  const text = draft ?? String(p.value ?? ''), bad = draft !== null ? checkText(ix, p.key, desk, draft) : null;
  const commit = () => { if (draft === null) return; if (!checkText(ix, p.key, desk, draft)) { if (draft !== p.value) p.set(draft); setDraft(null); } };
  return (
    <>
      <input className={look === 'lcd' ? 'lcd-input' : look === 'term' ? 't-input' : undefined} type="text" data-bind={p.key} value={text}
        aria-label={say(paramWords(ix, p.key, desk).label)} aria-invalid={bad ? true : undefined} title={bad ? say(bad) : undefined}
        onChange={e => setDraft(e.currentTarget.value)}
        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } else if (e.key === 'Escape') { e.preventDefault(); setDraft(null); } }}
        onBlur={commit} />
      {bad ? <span className="ckmsg" role="alert">{say(bad)}</span> : null}
    </>
  );
}
