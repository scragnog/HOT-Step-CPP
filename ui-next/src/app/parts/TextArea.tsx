// textArea: a text area whose look gives it its material (the taped sticky note, 25-module-avoid.js). Typing is one
// step per visit to the field ("Edit the avoid"); leaving it puts the note back at its first word (Q-03), and the
// text fades at its foot while there is more than the note shows (Q-38).

import { useLayoutEffect, useRef } from 'react';
import { fillMsg, paramWords, wordKey } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { gesture } from './value.ts';
import type { ViewProps } from './types.ts';

export function TextArea(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk), ref = useRef<HTMLTextAreaElement>(null);
  const p = Object.values(props.params)[0];
  const look = (props.node.opts as { look?: string } | undefined)?.look ?? 'plain';
  const over = () => { const t = ref.current; if (t) t.classList.toggle('over', t.scrollHeight > t.clientHeight + 1 && t.scrollTop < t.scrollHeight - t.clientHeight - 2); };
  useLayoutEffect(over);
  if (!p) return null;
  const words = paramWords(ix, p.key, desk), tag = `text:${p.key}`;
  const placeholder = p.face.placeholder ? say(fillMsg(ix, wordKey(ix, p.key, desk, 'placeholder'), p.face.placeholder, desk)) : undefined;
  const area = (
    <textarea ref={ref} data-bind={p.key} aria-label={say(words.label)} placeholder={placeholder} value={String(p.value ?? '')}
      onFocus={() => gesture.begin(tag)}
      onChange={e => { gesture.begin(tag); p.set(e.currentTarget.value); over(); }}
      onScroll={over}
      onBlur={e => { gesture.end(tag); e.currentTarget.scrollTop = 0; over(); }} />
  );
  if (look === 'note') return <div className={`note${p.changed ? ' changed' : ''}`} data-ctl={p.key}><i className="note-tape" aria-hidden="true" />{area}</div>;
  return <div className={`ctl${p.changed ? ' changed' : ''} wide`} data-ctl={p.key}>{area}</div>;
}
