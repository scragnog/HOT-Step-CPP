// knob: a rotary knob with a detent tick at the default, its label under it and a read-out that can be clicked to
// type a value (the concept's knob(), 21-knob.js). It draws its parameter and moves it; the value, the words, the
// changed marker and the step are the parameter's (BoundParam).

import { useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type WheelEvent } from 'react';
import { PARTS } from '../../definitions/parts.ts';
import { bareText, nodeWord, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { gesture, onValueKey, onValueKeyUp, wheelStep } from './value.ts';
import type { ViewProps } from './types.ts';

const HOW = { id: 'part.knob.howTo', en: PARTS.knob.howTo };

export function Knob(props: ViewProps) {
  const { node, nodeKey, cell } = props;
  const say = useSay(), desk = useDesk(s => s.desk);
  const p = Object.values(props.params)[0];
  const [typing, setTyping] = useState(false);
  const drag = useRef<{ y: number; raw: number } | null>(null);
  const knobEl = useRef<HTMLDivElement>(null);
  if (!p) return null;
  const [min, max, step] = p.face.range ?? [0, 1, 0.01];
  const v = typeof p.value === 'number' ? p.value : 0, def = typeof p.face.default === 'number' ? p.face.default : min;
  const words = paramWords(ix, p.key, desk);
  const nodeLabel = nodeKey ? nodeWord(ix, node, desk, nodeKey, 'label') : undefined;
  const label = cell?.label ?? nodeLabel ?? words.label;
  const tip = nodeKey ? nodeWord(ix, node, desk, nodeKey, 'tip') ?? words.tip : words.tip;
  const title = cell?.title ?? (tip ? ui('knob.tip', { tip, how: HOW }) : HOW);
  const bare = cell?.bare ? bareText(ix, p.key, desk) : null;
  const at = (v - min) / (max - min), d = Math.min(1, Math.max(0, (def - min) / (max - min)));
  const tag = `knob:${p.key}`;

  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    drag.current = { y: e.clientY, raw: v };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.focus({ preventScroll: true });
    gesture.begin(tag);
    e.currentTarget.parentElement?.classList.add('turning');
  };
  const moveTo = (e: PointerEvent<HTMLDivElement>) => {
    const g = drag.current; if (!g) return;
    const dy = g.y - e.clientY; g.y = e.clientY; if (!dy) return;
    /* the whole range is 160px of travel; Shift moves a tenth as far, from where the knob is (A-34) */
    g.raw = Math.min(max, Math.max(min, g.raw + dy / 160 * (max - min) * (e.shiftKey ? 0.1 : 1)));
    const to = +(Math.round(g.raw / step) * step).toFixed(10);
    if (to !== p.value) p.set(to);
  };
  const up = (e: PointerEvent<HTMLDivElement>) => { drag.current = null; gesture.end(tag); e.currentTarget.parentElement?.classList.remove('turning'); };
  const wheel = (e: WheelEvent<HTMLDivElement>) => {
    if (document.activeElement !== e.currentTarget || !e.deltaY) return;
    e.preventDefault();
    wheelStep(tag, () => p.set(+(v + Math.sign(-e.deltaY) * (e.shiftKey ? 10 : 1) * step).toFixed(10)));
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); e.stopPropagation(); setTyping(true); return; }
    onValueKey(e, p);
  };
  const reset = (e: React.MouseEvent<HTMLDivElement>) => {
    const kn = e.currentTarget.parentElement!;
    kn.classList.add('resetting'); window.setTimeout(() => kn.classList.remove('resetting'), 300);
    p.reset();
  };

  return (
    <div className={`kn${p.changed ? ' changed' : ''}`} data-ctl={p.key} title={say(title)}>
      <div ref={knobEl} className={`knob${cell?.small || (node.opts as { small?: boolean } | undefined)?.small ? ' sm' : ''}`} role="slider" tabIndex={0} data-knob={p.key}
        aria-label={say(cell?.aria ?? label)} aria-valuemin={min} aria-valuemax={max} aria-valuenow={v} aria-valuetext={say(p.text)}
        style={{ '--p': String(at) } as CSSProperties}
        onPointerDown={down} onPointerMove={moveTo} onPointerUp={up} onPointerCancel={up} onWheel={wheel} onKeyDown={key} onKeyUp={() => onValueKeyUp(p)} onDoubleClick={reset}>
        {Number.isFinite(d) ? <span className="det" style={{ '--d': d.toFixed(4) } as CSSProperties} /> : null}<i />
      </div>
      <span className="kl">{say(label)}</span>
      {typing ? <TypeIn p={p} label={say(words.label)} done={() => { setTyping(false); knobEl.current?.focus({ preventScroll: true }); }} />
        : bare ? <output data-out={p.key} data-bare="" data-unit={bare.unit} onClick={() => setTyping(true)}>{say(bare.text)}</output>
          : <output data-out={p.key} onClick={() => setTyping(true)}>{say(p.text)}</output>}
    </div>
  );
}

/** A typed value (L-40): Enter or leaving sets it, clamped and said by the core; Escape leaves it as it was. */
export function TypeIn({ p, label, done }: { p: ViewProps['params'][string]; label: string; done: () => void }) {
  const say = useSay();
  const [min, max] = p.face.range ?? [0, 1];
  const v0 = String(p.value), ended = useRef(false);
  const end = (commit: boolean, raw: string) => {
    if (ended.current) return; ended.current = true;
    const text = raw.trim().replace(/−/g, '-'), n = parseFloat(text);
    if (commit && text !== v0) {
      if (Number.isNaN(n)) useDesk.getState().showToast({ text: ui('knob.typeNumber', { label }) });
      else p.set(n);
    }
    done();
  };
  return (
    <input className="kn-in" type="text" inputMode="decimal" defaultValue={v0} autoFocus aria-label={say(ui('knob.type', { label, min, max }))}
      style={{ width: `${Math.max(4, String(max).length + 2)}ch` }}
      onFocus={e => e.currentTarget.select()}
      onKeyDown={e => { e.stopPropagation(); if (e.key === 'Enter') { e.preventDefault(); end(true, e.currentTarget.value); } else if (e.key === 'Escape') { e.preventDefault(); end(false, ''); } }}
      onBlur={e => end(true, e.currentTarget.value)} />
  );
}
