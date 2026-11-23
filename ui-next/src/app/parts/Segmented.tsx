// segmented: a row of joined buttons, one pressed (the concept's seg(), 20-controls.js); `size: 'sm'` is the compact
// row inside a rack group. The options and their words are the core's (optionsOf).

import { optionsOf, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import type { ViewProps } from './types.ts';

export function Segmented(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const p = Object.values(props.params)[0];
  if (!p) return null;
  const sm = (props.node.opts as { size?: string } | undefined)?.size === 'sm';
  return (
    <span className={`seg${sm ? ' sm' : ''}${p.changed ? ' changed' : ''}`} data-ctl={p.key} role="group" aria-label={say(paramWords(ix, p.key, desk).label)}>
      {optionsOf(ix, p.key, desk).map(o => (
        <button key={String(o.value)} type="button" data-seg={p.key} data-v={String(o.value)} aria-pressed={o.value === p.value} onClick={() => p.set(o.value)}>{say(o.label)}</button>
      ))}
    </span>
  );
}
