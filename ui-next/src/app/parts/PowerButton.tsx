// powerButton: the switch a whole instrument hangs on (the Pedalboard's chain switch, 27-pedalboard.js), drawn as the
// board head's master switch.

import { nodeWord, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import type { ViewProps } from './types.ts';

export function PowerButton(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const p = Object.values(props.params)[0];
  if (!p) return null;
  const on = p.value === true;
  const label = (props.nodeKey ? nodeWord(ix, props.node, desk, props.nodeKey, 'label') : undefined) ?? paramWords(ix, p.key, desk).label;
  return (
    <button type="button" className={`master${on ? ' on' : ''}${p.changed ? ' changed' : ''}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on} aria-label={say(ui('board.chain'))} onClick={() => p.set(!on)}>
      <i />{say(label)} <b />
    </button>
  );
}
