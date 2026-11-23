// powerButton: the switch a whole instrument hangs on (the Pedalboard's chain switch, 27-pedalboard.js), drawn as the
// board head's master switch. Its accessible name is its node's `name` where its label alone does not say what it
// switches ("Post-processing chain" for "Chain").

import { definePart, useSay, type PartView } from '../kit/index.ts';

export function PowerButton(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  const on = p.value === true;
  return (
    <button type="button" className={`master${on ? ' on' : ''}${p.changed ? ' changed' : ''}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on} aria-label={say(v.words.name ?? v.label)} onClick={() => p.set(!on)}>
      <i />{say(v.label)} <b />
    </button>
  );
}

definePart('powerButton', PowerButton);
