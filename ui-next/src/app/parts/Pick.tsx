// pick: the button that shows a choice and opens its dropdown (the concept's pick(), 22-pick.js), in its LCD variant
// with previous and next arrows (a pedal's or a unit's choice at Lab). The options and their words are the core's
// (optionsOf); the dropdown is the shared one (Dropdown.tsx, 30-dropdown.css).

import { useRef, useState } from 'react';
import { lcdMsg, optionsOf, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from './icons.tsx';
import { Dropdown } from './Dropdown.tsx';
import { LcdFace } from './LcdFace.tsx';
import type { ViewProps } from './types.ts';

/** A name inside a sentence ("Previous source blend") is lower case, unless it is an initialism (L-20). */
const lcFirst = (s: string) => (/^[A-Z][a-z]/.test(s) ? s[0]!.toLowerCase() + s.slice(1) : s);

export function Pick(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk), btn = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const p = Object.values(props.params)[0];
  if (!p) return null;
  const variant = (props.node.opts as { variant?: string } | undefined)?.variant ?? 'lcd';
  const label = say(paramWords(ix, p.key, desk).label), opts = optionsOf(ix, p.key, desk);
  const i = opts.findIndex(o => o.value === p.value);
  const cycle = (d: number) => { const o = opts[(i + d + opts.length) % opts.length]; if (o) p.set(o.value); };
  const face = <LcdFace text={say(lcdMsg(ix, p.key, desk))} />;
  const button = (
    <button ref={btn} type="button" className={`pk pk-${variant} `} data-pick={p.key} data-variant={variant} aria-haspopup="listbox" aria-expanded={open}
      aria-label={say(ui('pick.label', { label, value: p.text }))}
      onClick={() => setOpen(o => !o)} onKeyDown={e => { if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && !open) { e.preventDefault(); setOpen(true); } }}>
      {face}
    </button>
  );
  return (
    <span className={`pk-lcdwrap${p.changed ? ' changed' : ''}`} data-ctl={p.key}>
      <button type="button" className="cyc" data-cycle={`${p.key}|-1`} aria-label={say(ui('pick.prev', { label: lcFirst(label) }))} onClick={() => cycle(-1)}><Icon name="chevL" size={12} /></button>
      {button}
      <button type="button" className="cyc" data-cycle={`${p.key}|1`} aria-label={say(ui('pick.next', { label: lcFirst(label) }))} onClick={() => cycle(1)}><Icon name="chevR" size={12} /></button>
      {open ? <Dropdown anchor={btn} param={p} variant={variant} label={label} close={(refocus: boolean) => { setOpen(false); if (refocus) btn.current?.focus({ preventScroll: true }); }} /> : null}
    </span>
  );
}
