// textArea: a text area whose look gives it its material. Typing is one step per visit to the field ("Edit the
// avoid"); leaving it puts the text back at its first line.
//
//   plain  the page's field (the concept's .ctl textarea)
//   note   the taped sticky note (Avoid, 25-module-avoid.js): the text fades at its foot while there is more than the
//          note shows (Q-38)
//   staff  the ruled ABC sheet (the lead sheet, 25-module-score.js)
//   paper  the lyric paper (Repaint's lyrics, 25-module-repaint.js)
//   term   a quoted string in a terminal line that wraps downward (the planner's --neg, 25-module-planner.js): one line of
//          text, so Enter starts no new line and a pasted line break becomes a space

import { useLayoutEffect, useRef } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function TextArea(v: PartView) {
  const say = useSay(), ref = useRef<HTMLTextAreaElement>(null), p = v.params[0];
  const look = (v.opts.look as string | undefined) ?? 'plain';
  const over = () => { const t = ref.current; if (t) t.classList.toggle('over', t.scrollHeight > t.clientHeight + 1 && t.scrollTop < t.scrollHeight - t.clientHeight - 2); };
  /* a terminal string grows with its text, a line at a time */
  const grow = () => { const t = ref.current; if (t && look === 'term') { t.style.height = 'auto'; t.style.height = `${t.scrollHeight}px`; } };
  useLayoutEffect(() => { over(); grow(); });
  if (!p) return null;
  const term = look === 'term';
  const area = (
    <textarea ref={ref} className={term ? 'tok-in tok-area' : undefined} data-bind={p.key} aria-label={say(p.label)} placeholder={p.placeholder ? say(p.placeholder) : undefined} value={String(p.value ?? '')}
      rows={term ? 1 : undefined} spellCheck={look === 'staff' || term ? false : undefined}
      onFocus={() => v.gesture.begin('text')}
      onKeyDown={e => { if (term && e.key === 'Enter') e.preventDefault(); }}
      onChange={e => { v.gesture.begin('text'); p.set(term ? e.currentTarget.value.replace(/\r?\n/g, ' ') : e.currentTarget.value); over(); }}
      onScroll={over}
      onBlur={e => { v.gesture.end('text'); e.currentTarget.scrollTop = 0; over(); }} />
  );
  const ch = p.changed ? ' changed' : '';
  if (look === 'note') return <div className={`note${ch}`} data-ctl={p.key}><i className="note-tape" aria-hidden="true" />{area}</div>;
  if (look === 'staff') return <div className={`staffpad${ch}`} data-ctl={p.key}>{area}</div>;
  if (look === 'paper') return <div className={`paper${ch}`} data-ctl={p.key}>{area}</div>;
  if (term) return <span className={`neg-q${ch}`} data-ctl={p.key}><span className="neg-o" aria-hidden="true">{'“'}</span>{area}<span className="neg-end" aria-hidden="true">{'”'}</span></span>;
  return <div className={`ctl${ch} wide`} data-ctl={p.key}>{area}</div>;
}

definePart('textArea', TextArea);
