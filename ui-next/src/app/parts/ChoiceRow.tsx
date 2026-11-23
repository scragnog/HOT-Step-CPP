// choiceRow: Quick's plain choice row (the concept's qchoice and paintChoiceRow, 20-controls.js): a radio group of words
// that each set a detailed value, or several. One Tab stop; the arrows move the choice and choose it. Which choice is
// checked, what the row says when the values match none of them, and what pressing one sets are the core's.

import type { KeyboardEvent } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function ChoiceRow(v: PartView) {
  const say = useSay(), row = v.row;
  if (!row?.choices.length) return null;
  const { checked } = row, stop = checked ?? 0, n = row.choices.length;
  const choose = (i: number) => { if (i !== checked) v.choose(i); };
  const key = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const d = ({ ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -Infinity, End: Infinity } as Record<string, number>)[e.key];
    if (d === undefined) return; e.preventDefault();
    const j = Math.max(0, Math.min(n - 1, Number.isFinite(d) ? i + d : d < 0 ? 0 : n - 1));
    choose(j); (e.currentTarget.parentElement?.children[j] as HTMLElement | undefined)?.focus();
  };
  const ctl = v.params[0]?.key;
  return (
    <div className={`qchoice${checked === null ? ' custom' : ''}`} data-q={ctl} data-ctl={ctl} role="radiogroup" aria-label={say(row.label)}>
      <span className="kl">{say(row.label)}{checked === null ? <span className="q-real">{` · ${say(row.says)}`}</span> : null}</span>
      <div>
        {row.choices.map((w, i) => (
          <button key={i} type="button" role="radio" aria-checked={i === checked} tabIndex={i === stop ? 0 : -1} title={say(row.sets[i])} onClick={() => choose(i)} onKeyDown={e => key(e, i)}>
            {say(w)}
          </button>
        ))}
      </div>
    </div>
  );
}

definePart('choiceRow', ChoiceRow);
