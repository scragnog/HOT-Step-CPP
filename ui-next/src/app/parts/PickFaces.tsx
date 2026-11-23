// The drawings a picker's faces share with its list (22-pick.js elTile and schedSpark): a plugin's periodic-table tile
// and a schedule's curve. Drawing only: the symbol, family, hue and curve are the
// core's (src/derive/controls.ts optionFaces). Not a part of the inventory.

import type { CSSProperties } from 'react';

/** An option's face as the core gives it (OptionFace). */
export interface Face { n?: number; sym?: string; family?: string; hue?: number; file?: string; curve?: number[]; name?: string; quant?: string; qc?: string; gb?: number; gbText?: string; mem?: number; native?: string; ink?: string }

/** A plugin's tile: its symbol on its family's colour, its place in the list in the corner. */
export const El = ({ f }: { f: Face }) => <span className="el" style={{ '--hue': f.hue ?? 200 } as CSSProperties} aria-hidden="true"><small>{f.n}</small>{f.sym}</span>;

/** A schedule's curve, sigma from the start of the run to its end. */
export function Spark({ curve, w = 34, h = 34, pad = 5 }: { curve: number[]; w?: number; h?: number; pad?: number }) {
  const last = Math.max(1, curve.length - 1);
  const pts = curve.map((y, i) => `${(pad + (w - 2 * pad) * i / last).toFixed(1)},${(pad + (h - 2 * pad) * (1 - y)).toFixed(1)}`).join(' ');
  return <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true"><polyline points={pts} fill="none" stroke="#ff5b8a" strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>;
}

