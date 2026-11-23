// ruler: a length on a minutes timeline (the concept's rulerHTML, 21-ruler.js, and Tempo's length, 25-module-tempo.js).
// A press on the track travels there and the cap then follows the hand (a length snaps to round values: the core's
// `along`); the value contract's keys step it; a double-click or Delete resets it. Its head is its label and, when the
// parameter has auto, the auto button; on auto it shows the estimate. Its scale, fill, caption, Beat I/O and trim
// markers and the line under it are the core's (ControlModel.ruler).

import { useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent } from 'react';
import { definePart, onValueKey, onValueKeyUp, ui, useSay, type PartView } from '../kit/index.ts';

export function Ruler(v: PartView) {
  const say = useSay(), p = v.params[0], r = v.control?.ruler, el = useRef<HTMLDivElement>(null);
  const [sliding, setSliding] = useState(false);
  /* near the end the trim tag stands before the time cap, 8px clear of it (L-15) */
  const flip = r?.markers?.trim?.flip ?? false;
  useLayoutEffect(() => {
    const ru = el.current, tag = ru?.querySelector<HTMLElement>('.trim'), cap = ru?.querySelector<HTMLElement>('.rcap');
    if (!ru || !tag || !cap) return;
    if (!flip) { tag.style.left = ''; return; }
    const W = ru.clientWidth, cw = cap.offsetWidth, f = r!.markers!.trim!.at;
    const capL = r!.capAt === 'end' ? W - cw : Math.min(W - 22, Math.max(22, f * W)) - cw / 2;
    tag.style.left = `${capL - 8}px`;
  });
  if (!p || !p.range || !r) return null;
  const at = (e: PointerEvent<HTMLDivElement>) => { const b = e.currentTarget.getBoundingClientRect(); p.along((e.clientX - b.left) / b.width, r.span, b.width); };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); e.currentTarget.focus({ preventScroll: true });
    v.gesture.begin('drag'); at(e);
  };
  const move = (e: PointerEvent<HTMLDivElement>) => { if (!e.currentTarget.hasPointerCapture(e.pointerId)) return; setSliding(true); at(e); };
  const up = () => { setSliding(false); v.gesture.end('drag'); };
  const label = say(v.label), mark = `lbl cdot${p.changed ? ' changed' : ''}`;
  const m = r.markers;
  return (
    <div className="dur">
      {r.hasAuto ? (
        <div className="ctl-head">
          <span className={mark} data-ctl={p.key} style={{ flex: 1 }}>{label}</span>
          <button type="button" className={`auto${r.auto ? ' on' : ''}`} data-auto={p.key} aria-pressed={r.auto} aria-label={say(ui('auto.label', { label: p.label }))} title={p.autoTip ? say(p.autoTip) : undefined} onClick={() => p.setAuto(!r.auto)}>{say(ui('auto'))}</button>
        </div>
      ) : <span className={mark} data-ctl={p.key} title={v.words.tip ? say(v.words.tip) : undefined}>{label}</span>}
      <div ref={el} className={`ruler2${r.auto ? ' auto' : ''}${sliding ? ' sliding' : ''}`} style={{ '--maj': `${r.maj}%`, '--min': `${r.min}%`, '--w': `${r.at * 100}%` } as CSSProperties}
        data-ruler={p.key} data-max={r.span} role="slider" tabIndex={0} aria-label={say(p.label)} aria-valuemin={p.range[0]} aria-valuemax={p.range[1]}
        aria-valuenow={p.value as number} aria-valuetext={r.auto ? say(ui('ruler.auto')) : say(r.caption)} title={say(ui('ruler.title'))}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
        onKeyDown={e => onValueKey(e, p, v.gesture)} onKeyUp={() => onValueKeyUp(v.gesture)} onDoubleClick={() => p.reset()}>
        <i className="rbar" />
        {m?.ioHere ? ['l', 'r'].map((s, i) => { const x = m.io?.[i]; return <span key={s} className="io" data-io={s} hidden={!x} title={x ? say(m.ioTip) : undefined} style={x ? { left: `${x.left * 100}%`, width: `${x.width * 100}%` } : undefined} />; }) : null}
        {m?.trimHere ? <span className={`trim${m.trim?.flip ? ' flip' : ''}`} data-trim="" hidden={!m.trim} style={{ left: `${(m.trim?.at ?? r.at) * 100}%` }}>{say(ui('ruler.trim'))}</span> : null}
        <b className={`rcap${r.capAt ? ` at-${r.capAt}` : ''}`} data-rcap="">{say(r.caption)}</b>
      </div>
      <div className="ruler-labels" aria-hidden="true">{r.labels.map(l => <span key={l.at} style={{ left: `${l.at}%` }}>{l.text}</span>)}</div>
      {r.held || m ? <p className="held-note" data-ionote="" hidden={!m?.note}>{m?.note ? say(m.note) : null}</p> : null}
      {r.held ? <p className="held-note" data-held={p.key} hidden /> : null}
    </div>
  );
}

definePart('ruler', Ruler);
