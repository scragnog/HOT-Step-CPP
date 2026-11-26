// meter: a printed meter with a needle (the concept's vuHTML, 27-pedals.js). Its scale, marks, red zone, caption and
// where the needle points are the core's (meterScale); this draws them on the concept's arc. Its look on a tile is the
// rebuild's stylesheet's (styles/rebuild/15-parts.css), on the board the concept's.

import { Fragment, type CSSProperties } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

/* the arc's geometry: 96 degrees about (66, 78) */
const pt = (f: number, r: number) => { const a = (-48 + f * 96) * Math.PI / 180; return [(66 + r * Math.sin(a)).toFixed(1), (78 - r * Math.cos(a)).toFixed(1)] as const; };
const arc = (f0: number, f1: number, r: number) => { const [x0, y0] = pt(f0, r), [x1, y1] = pt(f1, r); return `M${x0} ${y0}A${r} ${r} 0 0 1 ${x1} ${y1}`; };

export function Meter(v: PartView) {
  const say = useSay(), m = v.meter;
  if (!m) return null;
  return (
    <span className="vu" aria-hidden="true">
      <svg viewBox="0 0 132 60" width="132" height="60">
        <path className="vu-arc" d={arc(0, m.red, 58)} /><path className="vu-arc r" d={arc(m.red, 1, 58)} /><path className="vu-red" d={arc(m.red, 1, 61)} />
        {m.marks.map((k, i) => {
          const [x0, y0] = pt(k.at, 56), [x1, y1] = pt(k.at, k.long ? 63 : 61), [tx, ty] = pt(k.at, 69), r = k.red ? 'r' : '';
          return <Fragment key={i}><path d={`M${x0} ${y0}L${x1} ${y1}`} className={r} />{k.words !== null ? <text x={tx} y={ty} className={r}>{k.words}</text> : null}</Fragment>;
        })}
        {m.caption ? <text x="8" y="55" className="vu-cap">{say(m.caption)}</text> : null}
      </svg>
      <i style={m.needle === null ? undefined : { transform: `rotate(${m.needle.toFixed(1)}deg)` } as CSSProperties} />
    </span>
  );
}

definePart('meter', Meter);
