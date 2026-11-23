// choiceRow: Quick's plain choice row (the concept's qchoice and paintChoiceRow, 20-controls.js): a radio group of words
// that each set a detailed value, or several. One Tab stop; the arrows move the choice and choose it. Which choice is
// checked, what the row says when the values match none of them, and what pressing one sets are the core's (choiceOf,
// applyChoice): a value set at another tier checks none, and the label says it.

import type { KeyboardEvent } from 'react';
import { applyChoice, choiceOf, rowOf } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import type { ViewProps } from './types.ts';

export function ChoiceRow({ def, node, nodeKey }: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const choices = node.choices ?? [];
  if (!choices.length) return null;
  const { checked, says } = choiceOf(ix, def, node, desk, nodeKey);
  const row = rowOf(ix, def, node, desk, nodeKey), label = row.label;
  const choose = (i: number) => { if (i !== checked) useDesk.getState().commit(applyChoice(ix, def, node, useDesk.getState().desk, nodeKey, i)); };
  const stop = checked ?? 0;
  const key = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const d = ({ ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -Infinity, End: Infinity } as Record<string, number>)[e.key];
    if (d === undefined) return; e.preventDefault();
    const j = Math.max(0, Math.min(choices.length - 1, Number.isFinite(d) ? i + d : d < 0 ? 0 : choices.length - 1));
    choose(j); (e.currentTarget.parentElement?.children[j] as HTMLElement | undefined)?.focus();
  };
  return (
    <div className={`qchoice${checked === null ? ' custom' : ''}`} data-q={node.param} data-ctl={node.param} role="radiogroup" aria-label={say(label)}>
      <span className="kl">{say(label)}{checked === null ? <span className="q-real">{` · ${say(says)}`}</span> : null}</span>
      <div>
        {row.choices.map(([w], i) => (
          <button key={i} type="button" role="radio" aria-checked={i === checked} tabIndex={i === stop ? 0 : -1} onClick={() => choose(i)} onKeyDown={e => key(e, i)}>
            {say(w)}
          </button>
        ))}
      </div>
    </div>
  );
}
