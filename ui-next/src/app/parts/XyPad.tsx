// xyPad: two settings that belong together on one plane (the concept's xyPad, 21-xy-pad.js). The puck goes where the
// hand is and springs toward its place (xyMotion.ts); each axis read-out carries its own setting, so a changed value is
// marked on its own axis; a drag, a held key or a double-click reset is one step (the core names a pair
// "a x → y, b x → y"). Where a hand or a key takes each value is the core's (toFraction, press).

import { useEffect, useLayoutEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, ui, useSay, type ParamView, type PartView } from '../kit/index.ts';
import { newMotion, stopMotion, toTarget, xyBox, XY_IN, type Motion } from './xyMotion.ts';
import { reduced } from '../kit/motion.ts';

/* which axis a key moves: left and right and the ends move X, up and down and Page move Y */
const AXIS: Record<string, 0 | 1> = { ArrowLeft: 0, ArrowRight: 0, Home: 0, End: 0, ArrowUp: 1, ArrowDown: 1, PageUp: 1, PageDown: 1 };

export function XyPad(v: PartView) {
  const say = useSay();
  const el = useRef<HTMLDivElement>(null), motion = useRef<Motion>(newMotion());
  const [px, py] = v.params as [ParamView | undefined, ParamView | undefined];
  /* a pad whose Y minimum means off draws off as a rail along its top, past the plot's widest end: the core says where
     the puck stands on the plot, or that it is on the rail (E-28, E-32; package 3b) */
  const rail = v.instrument?.rail, fx = px?.at ?? 0, fy = rail ? rail.at : py?.at ?? 0;
  const plotH = (t: HTMLElement) => Math.max(1, t.clientHeight - xyBox(t)[1] - 2 * XY_IN);
  useLayoutEffect(() => { const t = el.current; if (t) toTarget(t, motion.current, fx, 1 - (rail?.off ? 1 + 6 / plotH(t) : fy)); });
  useEffect(() => { const m = motion.current; return () => stopMotion(m); }, []);
  if (!px || !py || !px.range || !py.range) return null;

  const xl = v.words.xl ?? px.label, yl = v.words.yl ?? py.label;
  /* while a hand or a key works the pad, and a moment after, its changes animate */
  const working = (on: boolean) => { const m = motion.current; clearTimeout(m.liveT); if (on) m.live = true; else m.liveT = setTimeout(() => { m.live = false; }, 600); };
  const setAt = (e: PointerEvent<HTMLDivElement>) => {
    const t = el.current!, r = t.getBoundingClientRect(), [SL, SB] = xyBox(t), W = r.width - SL - 2 * XY_IN, H = r.height - SB - 2 * XY_IN, m = motion.current;
    let ax = (e.clientX - r.left - SL - XY_IN) / W, ay = 1 - (e.clientY - r.top - XY_IN) / H;
    if (e.shiftKey && m.fine) { ax = m.fine.fx + (e.clientX - m.fine.x) / W / 10; ay = m.fine.fy - (e.clientY - m.fine.y) / H / 10; }
    m.fine = e.shiftKey ? { x: e.clientX, y: e.clientY, fx: Math.min(1, Math.max(0, ax)), fy: Math.min(1, Math.max(0, ay)) } : null;
    v.batch(() => { px.toFraction(ax); if (rail) v.act('y', { f: ay, over: ay > 1 + 3 / plotH(t) }); else py.toFraction(ay); });
  };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); const t = e.currentTarget, m = motion.current;
    t.setPointerCapture(e.pointerId); t.focus({ preventScroll: true }); t.classList.remove('release'); t.classList.add('grab');
    m.dragging = true; working(true); v.gesture.begin('drag');
    if (e.shiftKey) { m.fine = { x: e.clientX, y: e.clientY, fx, fy }; return; }
    m.fine = null;
    if (!reduced()) { const r = t.getBoundingClientRect(), rip = document.createElement('span'); rip.className = 'xy-rip'; rip.style.left = e.clientX - r.left + 'px'; rip.style.top = e.clientY - r.top + 'px'; t.appendChild(rip); rip.addEventListener('animationend', () => rip.remove()); }
    setAt(e);
  };
  const up = (e: PointerEvent<HTMLDivElement>) => {
    const m = motion.current; if (!m.dragging) return; m.dragging = false; working(false); v.gesture.end('drag');
    const t = e.currentTarget; t.classList.remove('grab'); t.classList.add('release'); setTimeout(() => t.classList.remove('release'), 420);
  };
  const nudge = () => { const t = el.current!, m = motion.current; t.classList.add('nudge'); clearTimeout(m.nudge); m.nudge = setTimeout(() => t.classList.remove('nudge'), 900); };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const axis = AXIS[e.key]; if (axis === undefined) return;
    e.preventDefault();
    const p = axis ? py : px, ends = e.key === 'Home' || e.key === 'End';
    if (ends) v.gesture.end('key'); else { if (!e.repeat) v.gesture.end('key'); v.gesture.begin('key'); }
    /* Y on a pad with an off rail moves by the rail's rule (the core's) */
    working(true); if (axis && rail) v.act('yKey', { key: e.key, shift: e.shiftKey }); else p.press(e.key, e.shiftKey); nudge();
  };
  const reset = () => { working(true); v.batch(() => { px.reset(); py.reset(); }); working(false); };

  /* the corner words are printed just outside the pad, above and below its corners, so the puck reaches every corner
     without covering one (A-07) */
  const corner = (i: number) => (v.words[`corners.${i}`] ? say(v.words[`corners.${i}`]) : '');
  const cap = (a: string, b: string, w: string) => (a || b ? <div className={`xy-cap ${w}`} aria-hidden="true"><span>{a}</span><span>{b}</span></div> : null);
  const corners = [0, 1, 2, 3].map(corner), look = v.opts.look ? `xy-${v.opts.look as string}` : '';
  /* with what it moves not used (the planner off) it stands back and says so, and stays operable (Q-17; package 3b) */
  const pad = (
    <div ref={el} className={`xy ${look} ${rail ? 'has-yoff' : ''}${rail?.off ? ' y-off' : ''}`} data-xy={`${px.key}|${py.key}`} {...(rail ? { 'data-yoff': '' } : {})} {...(v.dim ? { 'data-dim': '' } : {})} tabIndex={0} role="slider" aria-roledescription={say(ui('xy.role'))}
      aria-label={say(ui('xy.label', { x: xl, y: yl }))} title={say(v.dim ?? ui('xy.title', { x: xl, y: yl }))}
      aria-valuetext={say(ui('xy.text', { x: px.label, xv: px.text, y: py.label, yv: py.text }))} aria-valuemin={px.range[0]} aria-valuemax={px.range[1]} aria-valuenow={px.value as number}
      onPointerDown={down} onPointerMove={e => { if (motion.current.dragging) setAt(e); }} onPointerUp={up} onPointerCancel={up}
      onKeyDown={key} onKeyUp={() => { v.gesture.end('key'); working(false); }} onDoubleClick={reset}>
      <canvas className="xy-trail" aria-hidden="true" /><i className="xy-grid" aria-hidden="true" />
      <i className="xy-h" /><i className="xy-v" /><i className="xy-tx" /><i className="xy-ty" />
      <b className="xy-dot"><i className="ring" /><i className="core" /></b><span className="xy-tip" aria-hidden="true"><b>{say(px.text)}</b><span>·</span><b>{say(py.text)}</b></span>
      {rail ? <i className="xy-rail" aria-hidden="true"><span>{v.words.yOff ? say(v.words.yOff) : null}</span></i> : null}
      <span className={`xy-ax x${px.changed ? ' changed' : ''}`} data-ctl={px.key}>{`${say(xl)} `}<output data-xyx="">{say(px.text)}</output></span>
      <span className={`xy-ax y${py.changed ? ' changed' : ''}`} data-ctl={py.key}>{`${say(yl)} `}<output data-xyy="">{say(py.text)}</output></span>
    </div>
  );
  return corners.some(Boolean) ? <div className={`xy-frame ${look ? look + '-frame' : ''}`}>{cap(corners[0]!, corners[1]!, 'cap-t')}{pad}{cap(corners[2]!, corners[3]!, 'cap-b')}</div> : pad;
}

definePart('xyPad', XyPad, { dims: true });
