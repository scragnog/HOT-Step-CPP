// The read-outs the slice uses: they show something and edit nothing. note is a quiet line of words; lcdValue a value
// printed big over a caps word (a pedal's Studio face); unitLcd a side unit's two-line LCD; button a small button
// that runs its node's action. What each says is the core's (nodeWord, lcdMsg); an action is run by the core
// (runAction) and its result applied by the shell (actions.ts).

import { lcdMsg, nodeOptWord, nodeWord } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { ActionLinks } from './actions.tsx';
import type { ViewProps } from './types.ts';

export function Note(props: ViewProps) {
  const { node, nodeKey, def } = props;
  const say = useSay(), desk = useDesk(s => s.desk);
  const tip = nodeWord(ix, node, desk, nodeKey, 'tip'), text = say(nodeWord(ix, node, desk, nodeKey, 'text'));
  /* on a pedal a note is its small print, its actions after it as links (27-pedals.js) */
  if (def.kind === 'pedal') return <small className="pnote" title={tip ? say(tip) : undefined}>{node.actions?.length ? say(ui('note.then', { text })) : text}<ActionLinks {...props} look="link" /></small>;
  return <span className="lbl" title={tip ? say(tip) : undefined}>{text}</span>;
}

export function LcdValue(props: ViewProps) {
  const { node, nodeKey, reads } = props;
  const say = useSay(), desk = useDesk(s => s.desk);
  const first = Object.keys(reads)[0];
  const value = nodeWord(ix, node, desk, nodeKey, 'text') ?? (first ? lcdMsg(ix, first, desk) : '');
  return (
    <>
      <span className="bigval">{say(value)}</span>
      <small>{say(nodeWord(ix, node, desk, nodeKey, 'label'))}{node.actions?.length ? <><br /><ActionLinks {...props} look="link" /></> : null}</small>
    </>
  );
}

export function UnitLcd({ node, nodeKey, reads, tier }: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  if ((node.opts as { notAtLab?: boolean } | undefined)?.notAtLab && tier === 'lab') return null;
  const text = nodeWord(ix, node, desk, nodeKey, 'text');
  if (text) return <span className="pwin">{say(text)}</span>;
  const prefix = nodeOptWord(ix, node, desk, nodeKey, 'prefix');
  /* an LCD prints its first line in capitals and the next in small letters */
  const lines = Object.keys(reads).map((k, i) => (i === 0 ? (say(prefix) + say(lcdMsg(ix, k, desk))).toLocaleUpperCase() : say(lcdMsg(ix, k, desk)).toLocaleLowerCase()));
  return <span className="u-lcd" aria-hidden="true">{lines.map((l, i) => <span key={i}>{l}</span>)}</span>;
}

export function Button(props: ViewProps) {
  return <ActionLinks {...props} look="button" />;
}
