// ledButton: an LED switch, a rack button with a lamp that lights when on (the concept's rackCtl for a toggle,
// 20-controls.js). Pressing it sets its switch to the other side; the step is the core's. Its look on each surface is
// the rebuild's stylesheet's (styles/rebuild/15-parts.css): a rack's slab, a pedal's, or a plain switch on a tile.

import { definePart, useSay, type PartView } from '../kit/index.ts';

export function LedButton(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  const on = p.value === true;
  return (
    <button type="button" className={`led-btn${on ? ' on' : ''}${p.changed ? ' changed' : ''}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on} onClick={() => p.set(!on)}>
      <i />{say(v.label)}
    </button>
  );
}

definePart('ledButton', LedButton);
