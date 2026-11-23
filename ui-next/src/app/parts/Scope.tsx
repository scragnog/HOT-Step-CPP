// The two scopes of package 3a:
//
//   samplerScope  the Sampler's scope (the concept's 25-module-sampler.js): the noise schedule drawn exactly as the
//                 controls under it say (the steps, shift, schedule and its shape, or the custom timesteps), a dot a step
//                 while there are 32 or fewer, the CFG interval's band and Performance trades' hatched cutoff; its read-out
//                 says only what the controls do not (V-07), with a way to what replaces or ends what it draws; at Lab
//                 the interval's two handles, which never pass each other (the core keeps them a step apart), their
//                 labels flipping inside near the bezel (A-28, A-31). Its child is the scope's power switch
//   latentCurve   the latent distribution before (grey) and after (green) the shift and rescale, with the XY pad that
//                 sets them laid over it (its child)
//
// Every number drawn is the core's (PartModel.instrument.scope, .dist); the handles hand back where the hand is.

import { useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { definePart, onValueKey, onValueKeyUp, ui, useSay, type ParamView, type PartView } from '../kit/index.ts';

/** Words with their numbers set in bold (each between two U+0001 marks, as the core writes them). */
const bolded = (t: string): ReactNode => t.split('\u0001').map((x, i) => (i % 2 ? <b key={i}>{x}</b> : x));

export function SamplerScope(v: PartView) {
  const say = useSay(), s = v.instrument?.scope, wrap = useRef<HTMLDivElement>(null);
  const [flip, setFlip] = useState<boolean[]>([]);
  /* a handle's label sits at the top, outside the window it bounds; within a label's width of the edge it flips inside */
  useLayoutEffect(() => {
    const w = wrap.current; if (!w) return;
    const place = () => {
      const W = w.clientWidth, next = [...w.querySelectorAll<HTMLElement>('[data-ivl]')].map((h, i) => {
        const l = h.querySelector<HTMLElement>('.ivl-l'), x = (s?.handles[i]?.at ?? 0) / 100 * W, lw = (l?.offsetWidth ?? 0) + 8;
        return h.classList.contains('end') ? x + lw > W : x - lw < 0;
      });
      setFlip(f => (f.length === next.length && f.every((x, i) => x === next[i]) ? f : next));
    };
    place(); addEventListener('resize', place); return () => removeEventListener('resize', place);
  }, [s?.handles]);
  if (!s) return null;
  return (
    <div className="scope">
      <div ref={wrap} className="ivl-wrap">
        <div data-scope="">
          <svg viewBox="0 0 300 110" preserveAspectRatio="none" aria-label={say(ui('scope.label'))}>
            <defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff22" strokeWidth="2" /></pattern></defs>
            {Array.from({ length: 11 }, (_, i) => <line key={`v${i}`} x1={6 + 288 * i / 10} y1="8" x2={6 + 288 * i / 10} y2="100" stroke="#262131" />)}
            {Array.from({ length: 5 }, (_, i) => <line key={`h${i}`} x1="6" y1={8 + 92 * i / 4} x2="294" y2={8 + 92 * i / 4} stroke="#262131" />)}
            {s.band ? <rect x={s.band.x} y="8" width={s.band.w} height="92" fill="#f5b54422" /> : null}
            {s.cut ? <rect x={s.cut.x} y="8" width={s.cut.w} height="92" fill="url(#hatch)" /> : null}
            <polyline points={s.points} fill="none" stroke="#ff5b8a" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          </svg>
          {s.dots.map((d, i) => <i key={i} className="sdot" style={{ left: `${d.x}%`, top: `${d.y}%` }} />)}
        </div>
        {s.handles.map((h, i) => { const p = v.params.find(x => x.key === h.key); return p ? <Handle key={h.key} p={p} v={v} at={h.at} end={i === 1} flip={!!flip[i]} label={say(ui(i ? 'scope.to' : 'scope.from'))} name={say(p.label)} /> : null; })}
      </div>
      <div className="scope-read">
        <span data-scoperead="" style={{ display: 'contents' }}>
          {s.read.map((r, i) => {
            const t = bolded(say(r.text)), a = r.action !== null ? v.actions[r.action] : undefined;
            const link = a ? <> <button type="button" className="scope-link" data-tier={a.tier} data-ring={a.ring} onClick={() => v.run(r.action!)}>{say(a.label)}</button></> : null;
            return r.tone === 'warn' ? <span key={i} className="sr-warn">{t}{link}</span>
              : r.tone === 'ts' ? <span key={i} className="sr-ts">{t}{link}</span>
              : r.tone === 'amber' ? <span key={i} style={{ color: '#f5c26b' }}>{t}{link}</span> : <span key={i}>{t}{link}</span>;
          })}
        </span>
        {v.children}
      </div>
    </div>
  );
}

/** One of the CFG interval's handles: dragged along the scope (one step a drag), stepped by the value contract's keys. */
function Handle({ p, v, at, end, flip, label, name }: { p: ParamView; v: PartView; at: number; end: boolean; flip: boolean; label: string; name: string }) {
  const drag = useRef<HTMLElement | null>(null);
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const wrap = drag.current; if (!wrap) return;
    const r = wrap.getBoundingClientRect();
    p.toFraction(((e.clientX - r.left) / r.width * 300 - 6) / 288);
  };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); drag.current = e.currentTarget.parentElement; e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.focus({ preventScroll: true }); v.gesture.begin(`ivl-${p.key}`);
  };
  const up = () => { if (!drag.current) return; drag.current = null; v.gesture.end(`ivl-${p.key}`); };
  return (
    <div className={`ivl-h${end ? ' end' : ''}${flip ? ' flip' : ''}`} data-ivl={p.key} role="slider" tabIndex={0} aria-valuemin={p.range?.[0]} aria-valuemax={p.range?.[1]} aria-valuenow={p.value as number}
      aria-label={name} style={{ left: `${at}%` } as CSSProperties}
      onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
      onKeyDown={(e: KeyboardEvent<HTMLDivElement>) => onValueKey(e, p, v.gesture)} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={() => p.reset()}>
      <span className={`ivl-l${p.changed ? ' changed' : ''}`} data-ctl={p.key}>{label}</span>
    </div>
  );
}

export function LatentCurve(v: PartView) {
  const say = useSay(), d = v.instrument?.dist;
  if (!d) return null;
  return (
    <div className="scope">
      <div className="xy-wrap">
        <svg className="dist" viewBox="0 0 300 80" preserveAspectRatio="none" data-dist="" aria-label={say(ui('dist.label'))}>
          {d.grid.map((x, i) => <line key={i} x1={x} y1="6" x2={x} y2="74" stroke="#262131" />)}
          <line x1={d.zero} y1="6" x2={d.zero} y2="74" stroke="#4a4256" />
          <polyline points={d.before} fill="none" stroke="#8a819a" strokeWidth="1.5" strokeDasharray="3 3" />
          <polyline points={d.after} fill="none" stroke="#a6f5bd" strokeWidth="2" />
        </svg>
        {v.children}
      </div>
      <div className="scope-read"><span>{say(ui('dist.grey'))}</span><span style={{ color: '#a6f5bd' }}>{say(ui('dist.green'))}</span></div>
    </div>
  );
}

definePart('samplerScope', SamplerScope);
definePart('latentCurve', LatentCurve);
