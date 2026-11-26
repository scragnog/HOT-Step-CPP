// choiceRow: Quick's plain choice row (the concept's qchoice and paintChoiceRow, 20-controls.js; look "mood" is the
// Planner's How adventurous, 25-module-planner.js): a radio group of words that each set a detailed value, or several.
// One Tab stop; the arrows move the choice and choose it. Which choice is checked, what the row says when the values
// match none of them, and what pressing one sets are the core's. A value between the choices (set at another tier)
// checks none: a small marker stands on the track at the value's place between them (the core's `between`), the Tab
// stop is the nearest choice, and an arrow takes the next choice that way from the value, never back past it (Q-08).

import { useLayoutEffect, useRef, type CSSProperties, type KeyboardEvent } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function ChoiceRow(v: PartView) {
  const say = useSay(), row = v.row, between = v.control?.between, track = useRef<HTMLDivElement>(null), mark = useRef<HTMLElement>(null);
  /* the marker: where the value falls between the choices' centres, or beyond an end (20-controls.js paintChoiceRow) */
  useLayoutEffect(() => {
    const t = track.current, m = mark.current; if (!t || !m || !between) return;
    const bs = [...t.children] as HTMLElement[], cx = bs.map(b => b.offsetLeft + b.offsetWidth / 2);
    if (!bs.length || !bs[0]!.offsetWidth) return;
    const off = Math.min(14, bs[0]!.offsetWidth / 2 - 4), lo = Math.floor(between.pos), hi = Math.ceil(between.pos);
    const x = between.beyond < 0 ? cx[0]! - off : between.beyond > 0 ? cx[cx.length - 1]! + off : cx[lo]! + (cx[hi]! - cx[lo]!) * (between.pos - lo);
    m.style.left = `${t.offsetLeft + x}px`; m.style.top = `${t.offsetTop + t.offsetHeight - 4}px`;
  });
  if (!row?.choices.length) return null;
  const { checked } = row, n = row.choices.length, stop = checked ?? (between ? Math.round(between.pos) : 0);
  const choose = (i: number) => { if (i !== checked) v.choose(i); };
  const key = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const d = ({ ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -Infinity, End: Infinity } as Record<string, number>)[e.key];
    if (d === undefined) return; e.preventDefault();
    /* from a value between two choices, the arrow takes the next one that way from the value */
    const from = between && checked === null && Number.isFinite(d) ? (d > 0 ? Math.ceil(between.pos) - 1 : Math.floor(between.pos) + 1) : i;
    const j = Math.max(0, Math.min(n - 1, Number.isFinite(d) ? from + d : d < 0 ? 0 : n - 1));
    choose(j); (e.currentTarget.parentElement?.children[j] as HTMLElement | undefined)?.focus();
  };
  const ctl = v.params[0]?.key, mood = v.opts.look === 'mood', changed = !!v.params[0]?.changed;
  /* standing back (the planner off), it fades, stays operable, and says why under it (Q-17; package 3b) */
  const row0 = (
    <div className={`${mood ? 'mood' : 'qchoice'}${checked === null ? ' custom' : ''}${changed ? ' changed' : ''}`} data-q={mood ? undefined : ctl} data-ctl={ctl} {...(v.dim ? { 'data-dim': '' } : {})} role="radiogroup" aria-label={say(row.label)}>
      <span className="kl">{say(row.label)}{checked === null ? <span className="q-real">{` · ${say(row.says)}`}</span> : null}</span>
      <div ref={track}>
        {row.choices.map((w, i) => (
          <button key={i} type="button" role="radio" aria-checked={i === checked} tabIndex={i === stop ? 0 : -1} title={say(row.sets[i])} onClick={() => choose(i)} onKeyDown={e => key(e, i)}
            style={mood ? { '--c': `hsl(${222 + i * 27} 70% 46%)` } as CSSProperties : undefined}>
            {say(w)}
          </button>
        ))}
      </div>
      <i ref={mark} className="q-mark" aria-hidden="true" hidden={!between} />
    </div>
  );
  return v.dim ? <>{row0}<p className="qhint lm-note">{say(v.dim)}</p></> : row0;
}

definePart('choiceRow', ChoiceRow, { dims: true });
