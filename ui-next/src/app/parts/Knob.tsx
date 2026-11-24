// knob: a rotary knob with a detent tick at the default, its label under it and a read-out that can be clicked to
// type a value (the concept's knob(), 21-knob.js). It draws what the core says (its place, its words, its changed
// mark) and hands back where the hand or the key took it; the value, its cleaning and the step are the core's.

import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, onValueKey, onValueKeyUp, onWheelStep, ui, useSay, type ParamView, type PartView } from '../kit/index.ts';

/* a label that is one symbol (α, β) is printed as it is written, not in capitals (20-controls.js SYM) */
const SYM = /^[Ͱ-Ͽ∀-⋿]$/;

export function Knob(v: PartView) {
  const say = useSay(), p = v.params[0];
  const [typing, setTyping] = useState(false);
  const drag = useRef<{ y: number; f: number } | null>(null), knobEl = useRef<HTMLDivElement>(null);
  /* the wheel steps a focused knob and the page does not scroll under it: a listener that may prevent the scroll,
     acting on the knob's parameter as it is now */
  const now = useRef({ p, g: v.gesture });
  useEffect(() => { now.current = { p, g: v.gesture }; });
  useEffect(() => {
    const el = knobEl.current; if (!el) return;
    const f = (e: WheelEvent) => { const { p: q, g } = now.current; if (!q || document.activeElement !== el || !e.deltaY) return; e.preventDefault(); onWheelStep(e.deltaY < 0, e.shiftKey, q, g); };
    el.addEventListener('wheel', f, { passive: false }); return () => el.removeEventListener('wheel', f);
  }, []);
  if (!p || !p.range || p.at === undefined) return null;

  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    drag.current = { y: e.clientY, f: p.at! };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.focus({ preventScroll: true });
    v.gesture.begin('drag');
    e.currentTarget.parentElement?.classList.add('turning');
  };
  const moveTo = (e: PointerEvent<HTMLDivElement>) => {
    const g = drag.current; if (!g) return;
    const dy = g.y - e.clientY; g.y = e.clientY; if (!dy) return;
    /* the whole range is 160px of travel; Shift moves a tenth as far, from where the knob is (A-34) */
    g.f = Math.min(1, Math.max(0, g.f + dy / 160 * (e.shiftKey ? 0.1 : 1)));
    p.toFraction(g.f);
  };
  const up = (e: PointerEvent<HTMLDivElement>) => { drag.current = null; v.gesture.end('drag'); e.currentTarget.parentElement?.classList.remove('turning'); };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); e.stopPropagation(); setTyping(true); return; }
    onValueKey(e, p, v.gesture);
  };
  const reset = (e: React.MouseEvent<HTMLDivElement>) => {
    const kn = e.currentTarget.parentElement!;
    kn.classList.add('resetting'); setTimeout(() => kn.classList.remove('resetting'), 300);
    p.reset();
  };
  const out = p.bare
    ? <output data-out={p.key} data-bare="" data-unit={p.bare.unit} hidden={typing} onClick={() => setTyping(true)}>{say(p.bare.text)}</output>
    : <output data-out={p.key} hidden={typing} onClick={() => setTyping(true)}>{say(p.text)}</output>;
  const label = say(v.label), sym = SYM.test(label);
  /* a value that can be automatic shows the auto pill beside its label, lit while automatic (A-11), and while automatic
     the knob stands at the value auto stands for, in the auto colour (L-11) */
  const auto = p.canAuto ? <button type="button" className={`auto kn-auto${p.auto ? ' on' : ''}`} data-auto={p.key} aria-pressed={p.auto} aria-label={say(ui('auto.label', { label: p.label }))} title={say(ui('auto.knob', { label: p.label }))} onClick={() => p.setAuto(!p.auto)}>{say(ui('auto'))}</button> : null;
  return (
    <div className={`kn${p.changed ? ' changed' : ''}${p.canAuto ? ' has-auto' : ''}`} data-ctl={p.key} title={say(v.title)}>
      <div ref={knobEl} className={`knob${v.small ? ' sm' : ''}${p.canAuto && p.auto ? ' is-auto' : ''}`} role="slider" tabIndex={0} data-knob={p.key}
        aria-label={sym ? say(p.label) : say(v.aria)} aria-valuemin={p.range[0]} aria-valuemax={p.range[1]} aria-valuenow={p.value as number} aria-valuetext={say(p.text)}
        style={{ '--p': String(p.at) } as CSSProperties}
        onPointerDown={down} onPointerMove={moveTo} onPointerUp={up} onPointerCancel={up} onKeyDown={key} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={reset}>
        {p.detent !== undefined ? <span className="det" style={{ '--d': p.detent.toFixed(4) } as CSSProperties} /> : null}<i />
      </div>
      <span className={`kl${sym ? ' sym' : ''}`}>{label}{auto}</span>
      {out}
      {typing ? <TypeIn p={p} done={() => { setTyping(false); knobEl.current?.focus({ preventScroll: true }); }} /> : null}
    </div>
  );
}

/** A typed value (L-40): Enter or leaving sets it, the core reads, clamps and says it; Escape leaves it as it was.
 *  The read-out is hidden while it is typed into, and comes back (21-knob.js knobType). */
function TypeIn({ p, done }: { p: ParamView; done: () => void }) {
  const say = useSay(), v0 = String(p.value), ended = useRef(false);
  const end = (commit: boolean, raw: string) => {
    if (ended.current) return; ended.current = true;
    if (commit && raw.trim() !== v0) p.typed(raw);
    done();
  };
  return (
    <input className="kn-in" type="text" inputMode="decimal" defaultValue={v0} autoFocus aria-label={say(p.typeName)}
      style={{ width: `${Math.max(4, String(p.range?.[1] ?? '').length + 2)}ch` }}
      onFocus={e => e.currentTarget.select()}
      onKeyDown={e => { e.stopPropagation(); if (e.key === 'Enter') { e.preventDefault(); end(true, e.currentTarget.value); } else if (e.key === 'Escape') { e.preventDefault(); end(false, ''); } }}
      onBlur={e => end(true, e.currentTarget.value)} />
  );
}

definePart('knob', Knob);
