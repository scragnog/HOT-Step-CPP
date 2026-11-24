// segmented: a row of joined buttons, one pressed (the concept's seg(), 20-controls.js); `size: 'sm'` is the compact
// row inside a rack group. The options, their words and which is pressed are the core's.

import { definePart, useSay, type PartView } from '../kit/index.ts';

export function Segmented(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p?.options) return null;
  /* the compact row says each option short, its full words in its tip (25-module-latent.js) */
  const sm = v.opts.size === 'sm';
  return (
    <span className={`seg${sm ? ' sm' : ''}${p.changed ? ' changed' : ''}`} data-ctl={p.key} role="group" aria-label={say(p.label)}>
      {p.options.map(o => (
        <button key={String(o.value)} type="button" data-seg={p.key} data-v={String(o.value)} aria-pressed={o.on} title={sm && o.short ? say(o.label) : undefined} onClick={() => p.set(o.value)}>{say(sm ? o.short ?? o.label : o.label)}</button>
      ))}
    </span>
  );
}

definePart('segmented', Segmented);
