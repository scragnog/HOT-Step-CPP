// textArea: a text area whose look gives it its material (the taped sticky note, 25-module-avoid.js). Typing is one
// step per visit to the field ("Edit the avoid"); leaving it puts the note back at its first word (Q-03), and the
// text fades at its foot while there is more than the note shows (Q-38).

import { useLayoutEffect, useRef } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function TextArea(v: PartView) {
  const say = useSay(), ref = useRef<HTMLTextAreaElement>(null), p = v.params[0];
  const over = () => { const t = ref.current; if (t) t.classList.toggle('over', t.scrollHeight > t.clientHeight + 1 && t.scrollTop < t.scrollHeight - t.clientHeight - 2); };
  useLayoutEffect(over);
  if (!p) return null;
  const area = (
    <textarea ref={ref} data-bind={p.key} aria-label={say(p.label)} placeholder={p.placeholder ? say(p.placeholder) : undefined} value={String(p.value ?? '')}
      onFocus={() => v.gesture.begin('text')}
      onChange={e => { v.gesture.begin('text'); p.set(e.currentTarget.value); over(); }}
      onScroll={over}
      onBlur={e => { v.gesture.end('text'); e.currentTarget.scrollTop = 0; over(); }} />
  );
  if (v.opts.look === 'note') return <div className={`note${p.changed ? ' changed' : ''}`} data-ctl={p.key}><i className="note-tape" aria-hidden="true" />{area}</div>;
  return <div className={`ctl${p.changed ? ' changed' : ''} wide`} data-ctl={p.key}>{area}</div>;
}

definePart('textArea', TextArea);
