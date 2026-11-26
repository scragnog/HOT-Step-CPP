// token: a scrub token, a number in running text or a terminal line (the concept's tok, 20-controls.js). It is dragged
// sideways (a wide range moves with the value: the core's scrub), stepped with the value contract's keys, reset by a
// double-click or Delete, and typed on Enter or a tap with a finger (S-16): Enter or leaving sets it, the core reads,
// clamps and says it, Escape leaves it. In a terminal line its label is its flag, printed by the line.

import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, onValueKey, onValueKeyUp, useSay, type ParamView, type PartView } from '../kit/index.ts';

export function Token(v: PartView) {
  const say = useSay(), p = v.params[0], el = useRef<HTMLSpanElement>(null);
  const [typing, setTyping] = useState(false);
  const drag = useRef<{ x: number; from: number; shift: boolean; moved: number; touch: boolean } | null>(null);
  if (!p || !p.range) return null;
  const start = (e: PointerEvent<HTMLSpanElement>) => ({ x: e.clientX, from: p.value as number, shift: e.shiftKey });
  const down = (e: PointerEvent<HTMLSpanElement>) => {
    e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.focus({ preventScroll: true });
    drag.current = { ...start(e), moved: 0, touch: e.pointerType === 'touch' }; v.gesture.begin('drag');
  };
  const move = (e: PointerEvent<HTMLSpanElement>) => {
    const d = drag.current; if (!d) return;
    /* Shift changes the rate from where the token is, so the value never jumps */
    if (e.shiftKey !== d.shift) Object.assign(d, start(e));
    d.moved = Math.max(d.moved, Math.abs(e.clientX - d.x));
    if (e.clientX !== d.x) p.scrub(d.from, e.clientX - d.x, e.shiftKey);
  };
  const up = () => { const d = drag.current; drag.current = null; v.gesture.end('drag'); if (d?.touch && d.moved < 4) setTyping(true); };
  const key = (e: KeyboardEvent<HTMLSpanElement>) => {
    if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); e.stopPropagation(); setTyping(true); return; }
    onValueKey(e, p, v.gesture);
  };
  return (
    <>
      <span ref={el} className={`tok${p.changed ? ' changed' : ''}`} data-scrub={p.key} data-out={p.key} data-ctl={p.key} tabIndex={0} role="slider" hidden={typing}
        aria-label={say(p.label)} aria-valuemin={p.range[0]} aria-valuemax={p.range[1]} aria-valuenow={p.value as number} aria-valuetext={say(p.text)} title={say(v.title)}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={() => p.reset()}>
        {say(p.text)}
      </span>
      {typing ? <TypeIn p={p} done={() => { setTyping(false); requestAnimationFrame(() => el.current?.focus({ preventScroll: true })); }} /> : null}
    </>
  );
}

/** The field a token turns into while it is typed (20-controls.js tokType). */
export function TypeIn({ p, done, className = 'kn-in tok-type' }: { p: ParamView; done: () => void; className?: string }) {
  const say = useSay(), v0 = String(p.value), ended = useRef(false);
  const end = (commit: boolean, raw: string) => {
    if (ended.current) return; ended.current = true;
    if (commit && raw.trim() !== v0) p.typed(raw);
    done();
  };
  return (
    <input className={className} type="text" inputMode="decimal" defaultValue={v0} autoFocus aria-label={say(p.typeName)}
      style={{ width: `${Math.max(4, String(p.range?.[1] ?? '').length + 2)}ch` }}
      onFocus={e => e.currentTarget.select()}
      onKeyDown={e => { e.stopPropagation(); if (e.key === 'Enter') { e.preventDefault(); end(true, e.currentTarget.value); } else if (e.key === 'Escape') { e.preventDefault(); end(false, ''); } }}
      onBlur={e => end(true, e.currentTarget.value)} />
  );
}

definePart('token', Token);
