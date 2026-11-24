// An adapter's label art (the concept's adArt, 25-module-adapters.js): the type picks the object, a tape for a LoRA,
// a chip for a LoKr, a vector for a DoRA. Drawing only: where each cell, the vector's head and the strength's share
// stand are the core's (derive/cartridges.ts CartArt); --p on the art's box moves them. Not a part: the cartridges and
// the adapter list (features/adapters.tsx) draw it.

import type { CSSProperties } from 'react';
import { ui, useSay } from '../kit/index.ts';

export interface ArtModel { kind: 'tape' | 'chip' | 'vec'; cells: { x: string; y: string; v: string }[]; head: { x: string; y: string } | null }

const reel = (cx: number) => (
  <g className="reel" style={{ transformOrigin: `${cx}px 24px` }}>
    <circle cx={cx} cy="24" r="7.5" className="hub" />
    <path d={`M${cx} 18.5v11M${cx - 4.8} 21.2l9.6 5.6M${cx - 4.8} 26.8l9.6-5.6`} className="spoke" />
  </g>
);

export function CartArt({ a }: { a: ArtModel }) {
  const say = useSay();
  if (a.kind === 'chip') return (
    <svg viewBox="0 0 84 50" className="art chip-art" aria-hidden="true">
      {[0, 1, 2, 3, 4].map(i => <g key={i}><line className="pin" x1={22 + i * 10} y1="2" x2={22 + i * 10} y2="6" /><line className="pin" x1={22 + i * 10} y1="44" x2={22 + i * 10} y2="48" /></g>)}
      <rect className="chip-body" x="14" y="6" width="56" height="38" rx="4" />
      {a.cells.map((c, i) => <rect key={i} className="chip-cell" x={c.x} y={c.y} width="7" height="7" rx="1" style={{ opacity: `calc(${c.v} * (0.18 + var(--p, .5) * 0.82))` } as CSSProperties} />)}
    </svg>
  );
  if (a.kind === 'vec' && a.head) return (
    <svg viewBox="0 0 84 50" className="art vec-art" aria-hidden="true">
      <path className="track" d="M20 36 A22 22 0 0 1 64 36" pathLength={1} /><path className="mag" d="M20 36 A22 22 0 0 1 64 36" pathLength={1} />
      <line className="arrow" x1="42" y1="36" x2={a.head.x} y2={a.head.y} /><circle className="head" cx={a.head.x} cy={a.head.y} r="2.4" /><circle className="pivot" cx="42" cy="36" r="3" />
      <text x="10" y="46">{say(ui('carts.art.mag'))}</text><text x="60" y="46">{say(ui('carts.art.dir'))}</text>
    </svg>
  );
  return (
    <svg viewBox="0 0 84 50" className="art tape-art" aria-hidden="true">
      <rect x="2" y="3" width="80" height="44" rx="6" className="body" /><rect x="13" y="12" width="58" height="24" rx="12" className="win" />
      <path d="M28 32 Q42 37 56 32" className="tapeline" />{reel(28)}{reel(56)}<path d="M22 47 L26 40 H58 L62 47" className="slotb" />
    </svg>
  );
}
