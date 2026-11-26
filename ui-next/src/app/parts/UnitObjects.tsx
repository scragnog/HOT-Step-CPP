// The objects a side unit shows in its middle (the concept's unit obj, 28-side-units.js), and the text link. Each
// shows what the core says and edits nothing.
//
//   lrcPreview   two timed lyric lines on the unit's small LCD, the first named after the lyrics' first section
//   coverSwatch  a swatch in the newest song's colour, standing for its cover art
//   unitStatus   the unit's status words (After the render · needs MMS)
//   unitBadge    the unit's round score badge
//   link         a text link that runs its node's action (Insert example, Show in Lab)

import type { CSSProperties } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';
import { ActionLinks } from './readouts.tsx';

export function LrcPreview(v: PartView) {
  const say = useSay(), l = v.readout?.lrc;
  if (!l) return null;
  return <span className="u-lcd" aria-hidden="true">{l.lines.map((x, i) => <span key={i}>{say(x)}</span>)}</span>;
}

export function CoverSwatch(v: PartView) {
  const c = v.readout?.cover;
  return c ? <span className="u-art" aria-hidden="true" style={{ '--h': c.hue } as CSSProperties} /> : null;
}

export function UnitStatus(v: PartView) {
  const say = useSay();
  return <span className="u-status" title={v.words.tip ? say(v.words.tip) : undefined}>{v.words.text ? say(v.words.text) : null}</span>;
}

export function UnitBadge(v: PartView) {
  const say = useSay();
  return <span className="u-badge" aria-hidden="true" title={v.words.tip ? say(v.words.tip) : undefined}>{v.words.text ? say(v.words.text) : null}</span>;
}

export const Link = (v: PartView) => <ActionLinks v={v} look="link" />;

definePart('lrcPreview', LrcPreview);
definePart('coverSwatch', CoverSwatch);
definePart('unitStatus', UnitStatus);
definePart('unitBadge', UnitBadge);
definePart('link', Link);
