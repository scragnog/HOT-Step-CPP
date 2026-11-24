// fader: a vertical fader with a cap, tick marks and a unity line at 1.00 (the concept's fader, 21-fader.js). A press
// on the slot travels there and the cap then follows the hand; the value contract's keys step it; a double-click or
// Delete resets it. Where the cap stands, the unity line and where a range that stops short of the printed travel
// stops are the core's (ControlModel.fader); its tape label is its node's label.

import { useState, type CSSProperties, type PointerEvent } from 'react';
import { definePart, onValueKey, onValueKeyUp, useSay, type PartView } from '../kit/index.ts';

export function Fader(v: PartView) {
  const say = useSay(), p = v.params[0], f = v.control?.fader;
  const [sliding, setSliding] = useState(false);
  if (!p || !p.range || !f) return null;
  /* the cap is 22px: a hand at the slot's top sets the top of the travel */
  const at = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), cap = 22, y = Math.min(1, Math.max(0, 1 - (e.clientY - r.top - cap / 2) / (r.height - cap)));
    p.toFraction(y * f.span);
  };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.focus({ preventScroll: true });
    v.gesture.begin('drag'); at(e);
  };
  const move = (e: PointerEvent<HTMLDivElement>) => { if (!e.currentTarget.hasPointerCapture(e.pointerId)) return; setSliding(true); at(e); };
  const up = () => { setSliding(false); v.gesture.end('drag'); };
  return (
    <div className={`fdr${p.changed ? ' changed' : ''}${f.stops ? ' stops' : ''}`} data-ctl={p.key}>
      <div className={`fdr-slot${sliding ? ' sliding' : ''}`} data-fader={p.key} role="slider" tabIndex={0} aria-label={say(f.name)} title={v.title ? say(v.title) : undefined}
        aria-valuemin={p.range[0]} aria-valuemax={p.range[1]} aria-valuenow={p.value as number} aria-valuetext={say(p.text)}
        style={{ '--p': String(f.at), '--u': String(f.unity), '--top': String(f.top) } as CSSProperties}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
        onKeyDown={e => onValueKey(e, p, v.gesture)} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={() => p.reset()}>
        <i className="fdr-ticks" /><i className="fdr-unity" /><b className="fdr-cap" />
      </div>
      <output data-out={p.key}>{say(p.text)}</output>
      {/* its tape, said short, its full name as its tip; a cartridge's own fader has none (package 3b) */}
      {v.label !== undefined ? <span className="fdr-l" title={f.tape !== null ? say(f.tape) : undefined}>{say(v.label)}</span> : null}
    </div>
  );
}

definePart('fader', Fader);
