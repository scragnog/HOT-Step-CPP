// A part drawn inside another as one of its pieces (the prompt card's caption-source chips, the lyric sheet's stamp
// and its Expand on Generate pill): the outer part's view with the piece's own parameter, options and words; a field
// given as undefined is left out.

import type { PartView } from '../kit/index.ts';

type Piece = { [K in keyof PartView]?: PartView[K] | undefined };

export function subView(v: PartView, o: Piece): PartView {
  const x = { ...v, words: {}, opts: {}, ...o } as Record<string, unknown>;
  for (const k of Object.keys(o)) if (o[k as keyof Piece] === undefined) delete x[k];
  if (!('control' in o)) delete x.control;
  return x as unknown as PartView;
}
