// pick: the button that shows a choice and opens its dropdown (the concept's pick(), 22-pick.js), in its LCD variant
// with previous and next arrows (a pedal's or a unit's choice at Lab). The options, their words and what the face
// prints are the core's; the dropdown is the shared one (Dropdown.tsx, 30-dropdown.css).

import { useCallback, useRef, useState } from 'react';
import { definePart, Icon, LcdFace, useSay, type PartView } from '../kit/index.ts';
import { Dropdown } from './Dropdown.tsx';

export function Pick(v: PartView) {
  const say = useSay(), btn = useRef<HTMLButtonElement>(null), p = v.params[0];
  const [open, setOpen] = useState(false);
  const close = useCallback((refocus: boolean) => { setOpen(false); if (refocus) btn.current?.focus({ preventScroll: true }); }, []);
  if (!p?.options) return null;
  const variant = (v.opts.variant as string | undefined) ?? 'lcd', opts = p.options;
  const i = opts.findIndex(o => o.on);
  const cycle = (d: number) => { const o = opts[(i + d + opts.length) % opts.length]; if (o) p.set(o.value); };
  return (
    <span className={`pk-lcdwrap${p.changed ? ' changed' : ''}`} data-ctl={p.key}>
      <button type="button" className="cyc" data-cycle={`${p.key}|-1`} aria-label={say({ id: 'ui.pick.prev', slots: { label: p.label } })} onClick={() => cycle(-1)}><Icon name="chevL" size={12} /></button>
      <button ref={btn} type="button" className={`pk pk-${variant} `} data-pick={p.key} data-variant={variant} aria-haspopup="listbox" aria-expanded={open}
        aria-label={say({ id: 'ui.pick.label', slots: { label: p.label, value: p.text } })}
        onClick={() => setOpen(o => !o)} onKeyDown={e => { if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && !open) { e.preventDefault(); setOpen(true); } }}>
        <LcdFace text={p.pick} />
      </button>
      <button type="button" className="cyc" data-cycle={`${p.key}|1`} aria-label={say({ id: 'ui.pick.next', slots: { label: p.label } })} onClick={() => cycle(1)}><Icon name="chevR" size={12} /></button>
      {open ? <Dropdown anchor={btn} p={p} variant={variant} close={close} /> : null}
    </span>
  );
}

definePart('pick', Pick);
