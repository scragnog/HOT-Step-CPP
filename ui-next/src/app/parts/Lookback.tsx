// lookback: a look-back window drawn as a strip of codes that grows leftward into the past, with a time scale under it
// (the concept's codes strip, 25-module-repetition.js). A hand along the strip sets how far back it reaches (its left
// end is the window's whole length); the value contract's keys step it, Left reaching further back and Right less, as
// the picture does (E-29). How many bars are lit, the scale's words and the tip are the core's (ControlModel.lookback).

import { useMemo, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, onValueKeyUp, useSay, type PartView } from '../kit/index.ts';

/* the bars' heights: the concept's seeded generator (00-util.js rng(7)), so the strip is the same picture every time */
const heights = (n: number): number[] => {
  let seed = 7;
  const r = () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  return Array.from({ length: n }, () => 25 + Math.round(r() * 75));
};
const SWAP: Record<string, string> = { ArrowLeft: 'ArrowRight', ArrowRight: 'ArrowLeft' };

export function Lookback(v: PartView) {
  const say = useSay(), p = v.params[0], lb = v.control?.lookback;
  const hs = useMemo(() => heights(lb?.bars ?? 0), [lb?.bars]);
  if (!p || !p.range || !lb) return null;
  const [lo, hi] = p.range;
  /* the strip's left end is the window's whole length, its right end none */
  const at = (e: PointerEvent<HTMLDivElement>) => { const b = e.currentTarget.getBoundingClientRect(), f = Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)); p.toFraction(((1 - f) * hi - lo) / (hi - lo)); };
  const down = (e: PointerEvent<HTMLDivElement>) => { e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.focus({ preventScroll: true }); v.gesture.begin('drag'); at(e); };
  const move = (e: PointerEvent<HTMLDivElement>) => { if (e.currentTarget.hasPointerCapture(e.pointerId)) at(e); };
  const up = () => v.gesture.end('drag');
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); v.gesture.end('key'); p.reset(); return; }
    if (!/^(Arrow(Up|Down|Left|Right)|Page(Up|Down)|Home|End)$/.test(e.key)) return;
    e.preventDefault(); if (!e.repeat) v.gesture.end('key'); v.gesture.begin('key');
    p.press(SWAP[e.key] ?? e.key, e.shiftKey);
  };
  return (
    <>
      <div className={`codes${p.changed ? ' changed' : ''}`} data-codes={p.key} data-ctl={p.key} role="slider" tabIndex={0} aria-valuemin={p.range[0]} aria-valuemax={p.range[1]} aria-valuenow={p.value as number}
        aria-valuetext={say(p.text)} aria-label={say(v.label)} title={say(lb.tip)}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={() => p.reset()}>
        {hs.map((h, i) => <i key={i} className={i >= lb.bars - lb.lit ? 'w' : ''} style={{ '--h': `${h}%` } as CSSProperties} />)}
      </div>
      <div className="codes-cap"><span>{say(lb.far)}</span>{lb.mid ? <span className="cc-mid">{say(lb.mid)}</span> : null}<span>{say(lb.now)}</span></div>
    </>
  );
}

definePart('lookback', Lookback);
