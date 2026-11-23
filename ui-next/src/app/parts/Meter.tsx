// meter: a printed meter with a needle (the concept's vuHTML, 27-pedals.js): the first of what it reads on its scale
// (opts min, max, and where it turns red), or, reading nothing, the level the engine reports, at rest. Its look is the
// concept's, which styles it on the board (.board .vu in 27-pedalboard.css).

import type { CSSProperties } from 'react';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { valueOf } from '../../derive.ts';
import type { ViewProps } from './types.ts';

const pt = (f: number, r: number) => { const a = (-48 + f * 96) * Math.PI / 180; return [(66 + r * Math.sin(a)).toFixed(1), (78 - r * Math.cos(a)).toFixed(1)] as const; };
const arc = (f0: number, f1: number, r: number) => { const [x0, y0] = pt(f0, r), [x1, y1] = pt(f1, r); return `M${x0} ${y0}A${r} ${r} 0 0 1 ${x1} ${y1}`; };
/* the concept's VU scale, -20 to +3, red from 0 (the Mastering pedal's) */
const VU: [number, number][] = [[-20, 0], [-10, 0.26], [-7, 0.38], [-5, 0.48], [-3, 0.58], [-1, 0.67], [0, 0.72], [1, 0.79], [2, 0.88], [3, 1]];

export function Meter({ node, reads }: ViewProps) {
  const desk = useDesk(s => s.desk);
  const o = (node.opts ?? {}) as { min?: number; max?: number; red?: number };
  const first = Object.keys(reads)[0], v = first ? valueOf(ix, first, desk) : undefined;
  /* a scale of its own when it reads a value: five marks from min to max; else the VU scale */
  const own = typeof v === 'number' && o.min !== undefined && o.max !== undefined;
  const lo = o.min ?? -20, hi = o.max ?? 3, red = o.red ?? 0, at = (x: number) => (x - lo) / (hi - lo);
  const marks: [number, number][] = own ? Array.from({ length: 5 }, (_, i) => { const x = lo + (hi - lo) * i / 4; return [Math.round(x * 10) / 10, i / 4]; }) : VU;
  const redF = own ? at(red) : 0.72;
  const needle = own ? -48 + Math.min(1, Math.max(0, at(v as number))) * 96 : null;
  return (
    <span className="vu" aria-hidden="true">
      <svg viewBox="0 0 132 60" width="132" height="60">
        <path className="vu-arc" d={arc(0, redF, 58)} /><path className="vu-arc r" d={arc(redF, 1, 58)} /><path className="vu-red" d={arc(redF, 1, 61)} />
        {marks.map(([db, f], i) => { const [x0, y0] = pt(f, 56), [x1, y1] = pt(f, i % 2 === 0 ? 63 : 61), [tx, ty] = pt(f, 69), r = f >= redF ? 'r' : '';
          return <g key={i}><path d={`M${x0} ${y0}L${x1} ${y1}`} className={r} />{own || [-20, -10, -5, 0, 3].includes(db) ? <text x={tx} y={ty} className={r}>{db > 0 ? '+' + db : db}</text> : null}</g>; })}
      </svg>
      <i style={needle === null ? undefined : { transform: `rotate(${needle.toFixed(1)}deg)` } as CSSProperties} />
    </span>
  );
}
