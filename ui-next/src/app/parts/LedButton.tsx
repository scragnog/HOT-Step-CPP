// ledButton: an LED switch, a rack button with a lamp that lights when on (the concept's rackCtl for a toggle,
// 20-controls.js). Pressing it sets its switch to the other value; the step is the core's.

import { nodeWord, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import type { ViewProps } from './types.ts';

export function LedButton(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const p = Object.values(props.params)[0];
  if (!p) return null;
  const on = p.value === true;
  const label = props.cell?.label ?? (props.nodeKey ? nodeWord(ix, props.node, desk, props.nodeKey, 'label') : undefined) ?? paramWords(ix, p.key, desk).label;
  return (
    <button type="button" className={`led-btn${on ? ' on' : ''}${p.changed ? ' changed' : ''}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on} onClick={() => p.set(!on)}>
      <i />{say(label)}
    </button>
  );
}
