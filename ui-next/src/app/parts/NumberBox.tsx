// The typed number boxes (the concept's numBox, 21-numbox.js, and the BPM's LCD number, 25-module-tempo.js): type a
// number, drag up and down, or step it with the arrow keys. What the box shows (the number as typed, empty on auto),
// its words while empty, and the seconds a step count takes are the core's (ControlModel.box); what typing means
// (a number, a word refused, out of range clamped and said, empty back to auto) is the core's too (p.typed).
//
//   numberBox  a box with its caps label under it (the Sampler's Steps), and its seconds under that
//   lcdNumber  a big seven-segment number on an LCD, its label beside it, and AUTO when the parameter has auto; in an
//              LCD its pieces stand in the LCD's row (display: contents), elsewhere it brings a small LCD of its own

import { useContext, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { LcdSlots } from './kids.ts';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';

/** The box's behaviour: a draft while typing, committed on Enter or leaving (only when it changed), Escape puts the value
 *  back; the arrows step it; a drag up and down past 4px scrubs it and is one step. */
function useBox(v: PartView) {
  const p = v.params[0], box = v.control?.box;
  const [draft, setDraft] = useState<string | null>(null);
  const was = useRef<string | null>(null), drag = useRef<{ y: number; at: number; moved: boolean; id: number; el: HTMLInputElement } | null>(null);
  const shown = draft ?? box?.text ?? '';
  const commit = (text: string) => { if (text !== box?.text) p?.typed(text); setDraft(null); };
  const select = (el: HTMLInputElement) => requestAnimationFrame(() => { if (document.activeElement === el) el.select(); });
  return {
    value: shown,
    onChange: (e: { currentTarget: HTMLInputElement }) => setDraft(e.currentTarget.value),
    onFocus: (e: { currentTarget: HTMLInputElement }) => { if (drag.current?.moved) return; was.current = e.currentTarget.value; select(e.currentTarget); },
    onBlur: (e: { currentTarget: HTMLInputElement }) => { const t = e.currentTarget.value; if (was.current !== null && t !== was.current) commit(t); else setDraft(null); was.current = null; },
    onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => {
      const el = e.currentTarget;
      if (e.key === 'Enter') { e.preventDefault(); commit(el.value); was.current = box?.text ?? ''; select(el); return; }
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); setDraft(null); select(el); return; }
      if (p && /^(ArrowUp|ArrowDown|PageUp|PageDown)$/.test(e.key)) {
        e.preventDefault(); if (!e.repeat) v.gesture.end('key'); v.gesture.begin('key');
        p.press(e.key, e.shiftKey); setDraft(null); select(el);
      }
    },
    onKeyUp: () => v.gesture.end('key'),
    onPointerDown: (e: PointerEvent<HTMLInputElement>) => { drag.current = { y: e.clientY, at: p?.at ?? 0, moved: false, id: e.pointerId, el: e.currentTarget }; },
    onPointerMove: (e: PointerEvent<HTMLInputElement>) => {
      const d = drag.current; if (!d || !p?.range) return;
      const dy = d.y - e.clientY;
      if (!d.moved && Math.abs(dy) > 4) { d.moved = true; d.el.setPointerCapture(d.id); d.el.blur(); v.gesture.begin('drag'); }
      if (!d.moved) return;
      e.preventDefault();
      /* three pixels a step, and never fewer than 400 steps end to end (21-numbox.js) */
      const [lo, hi, step] = p.range, per = Math.max(step, (hi - lo) / 400);
      setDraft(null); p.toFraction(d.at + Math.round(dy / 3) * per / (hi - lo));
    },
    onPointerUp: () => { const d = drag.current; drag.current = null; if (d?.moved) v.gesture.end('drag'); },
  };
}

export function NumberBox(v: PartView) {
  const say = useSay(), p = v.params[0], box = v.control?.box;
  const h = useBox(v);
  if (!p || !box) return null;
  return (
    <label className={`numbox${p.changed ? ' changed' : ''}`} data-ctl={p.key}>
      <input className="num" type="text" data-num={p.key} inputMode="numeric" placeholder={say(box.empty)} aria-label={say(ui('num.label', { label: p.label }))} title={say(ui('num.title'))}
        autoComplete="off" spellCheck={false} {...h} />
      <span className="kl">{say(v.label)}</span>
      {box.seconds ? <output data-stepsec="">{say(box.seconds)}</output> : null}
    </label>
  );
}

export function LcdNumber(v: PartView) {
  const say = useSay(), p = v.params[0], box = v.control?.box, lent = useContext(LcdSlots);
  const h = useBox(v);
  if (!p || !box) return null;
  return (
    <span className="lcd-numw">
      <input className="lcd-num num" type="text" data-num={p.key} inputMode="text" maxLength={4} placeholder={say(box.empty)} aria-label={say(ui('num.lcd.label', { label: p.label }))} title={say(ui('num.lcd.title'))}
        autoComplete="off" spellCheck={false} {...h} />
      <span className="lcd-unit"><small className={`cdot-lcd${p.changed ? ' changed' : ''}`} data-ctl={p.key}>{say(v.label, { upper: true })}</small>{lent.unit}</span>
      {box.auto !== null || lent.btns ? (
        <span className="lcd-btns">
          {lent.btns}
          {box.auto !== null ? <button type="button" data-auto={p.key} className={box.auto ? 'on' : ''} aria-pressed={box.auto} aria-label={say(ui('auto.label', { label: p.label }))} onClick={() => p.setAuto(!box.auto)}>{say(ui('auto.lcd'))}</button> : null}
        </span>
      ) : null}
    </span>
  );
}

definePart('numberBox', NumberBox);
definePart('lcdNumber', LcdNumber);
