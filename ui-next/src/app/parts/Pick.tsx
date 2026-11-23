// pick: the button that shows a choice and opens its dropdown (the concept's pick(), 22-pick.js), in each variant:
//   lcd     an LCD with previous and next arrows (a pedal's or a unit's choice, a stage of a schedule)
//   plugin  a periodic-table tile (or a schedule's curve) and the kind above the value (solver, schedule, guidance)
//   chip    a pill with its label and the value (the prompt's caption source)
//   model   the role, the name, the quantisation badge, the size and a bar of the card it takes
//   stamp   an ink stamp in the language's own name (the lyric language)
// The options, their words, what each face shows and the list's heading (the node's `title`) are the core's; the
// dropdown is the shared one (Dropdown.tsx, 30-dropdown.css). ArrowDown or ArrowUp on the button opens it.

import { useCallback, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { definePart, Icon, LcdFace, useSay, type PartView } from '../kit/index.ts';
import { Dropdown, type Variant } from './Dropdown.tsx';
import { El, Spark, type Face } from './PickFaces.tsx';
import { nbHyphen } from './pickText.ts';

export function Pick(v: PartView) {
  const say = useSay(), btn = useRef<HTMLButtonElement>(null), p = v.params[0];
  const [open, setOpen] = useState(false);
  const close = useCallback((refocus: boolean) => { setOpen(false); if (refocus) btn.current?.focus({ preventScroll: true }); }, []);
  if (!p?.options) return null;
  const variant = ((v.opts.variant as string | undefined) ?? 'lcd') as Variant, opts = p.options;
  const i = opts.findIndex(o => o.on), faces = (v.control?.faces ?? []) as Face[], f: Face = faces[i] ?? {};
  const ch = p.changed ? ' changed' : '';
  const keys = (e: KeyboardEvent<HTMLButtonElement>) => { if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && !open) { e.preventDefault(); setOpen(true); } };
  const list = open ? <Dropdown anchor={btn} p={p} variant={variant} faces={faces} title={v.words.title} close={close} /> : null;
  const button = (cls: string, face: ReactNode, extra: Record<string, unknown> = {}) => (
    <button ref={btn} type="button" className={`pk pk-${variant} ${cls}`} data-pick={p.key} data-variant={variant} aria-haspopup="listbox" aria-expanded={open}
      aria-label={say({ id: 'ui.pick.label', slots: { label: p.label, value: p.text } })} onClick={() => setOpen(o => !o)} onKeyDown={keys} {...extra}>
      {face}
    </button>
  );
  const caret = <span className="pk-caret" aria-hidden="true"><Icon name="caret" size={12} /></span>;
  if (variant === 'plugin') {
    const value = say(opts[i]?.label ?? p.text);
    return <>{button(ch, <>{f.curve ? <Spark curve={f.curve} /> : <El f={f} />}<span className="pk-txt"><span className="pk-kind">{say(p.short ?? p.label)}</span><span className="pk-val" title={value}>{nbHyphen(value)}</span></span>{caret}</>, { 'data-ctl': p.key })}{list}</>;
  }
  if (variant === 'chip') return <>{button(`pchip${ch}`, <><span className="k">{say(v.label)}</span><span className="pk-val">{say(p.text)}</span>{caret}</>, { 'data-ctl': p.key })}{list}</>;
  if (variant === 'model') {
    return <>{button(ch, <>
      <span className="pk-kind">{say(v.label)}</span><span className="pk-val">{f.name ?? say(p.text)}</span>
      <span className="pk-right">{f.quant ? <span className={`qb q-${f.qc}`}>{f.quant}</span> : null}{f.gbText ? <span className="gb">{f.gbText}</span> : null}{caret}</span>
      {f.mem !== undefined ? <span className="mem"><i style={{ '--w': `${f.mem * 100}%` } as CSSProperties} /></span> : null}
    </>, { 'data-ctl': p.key })}{list}</>;
  }
  if (variant === 'stamp') return <span className={`stamp-wrap${ch}`} data-ctl={p.key} style={{ '--ink': f.ink } as CSSProperties}>{button('', f.native ?? say(p.text))}{list}</span>;
  const cycle = (d: number) => { const o = opts[(i + d + opts.length) % opts.length]; if (o) p.set(o.value); };
  return (
    <span className={`pk-lcdwrap${ch}`} data-ctl={p.key}>
      <button type="button" className="cyc" data-cycle={`${p.key}|-1`} aria-label={say({ id: 'ui.pick.prev', slots: { label: p.label } })} onClick={() => cycle(-1)}><Icon name="chevL" size={12} /></button>
      {button('', <LcdFace text={p.pick} />)}
      <button type="button" className="cyc" data-cycle={`${p.key}|1`} aria-label={say({ id: 'ui.pick.next', slots: { label: p.label } })} onClick={() => cycle(1)}><Icon name="chevR" size={12} /></button>
      {list}
    </span>
  );
}

definePart('pick', Pick);
