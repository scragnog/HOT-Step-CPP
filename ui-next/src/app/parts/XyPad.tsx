// xyPad: two settings that belong together on one plane (the concept's xyPad, 21-xy-pad.js). The puck goes where the
// hand is and springs toward its place (xyMotion.ts); each axis read-out carries its own setting, so a changed value is
// marked on its own axis; a drag, a held key or a double-click reset is one step (the core names a pair
// "a x → y, b x → y").

import { useEffect, useLayoutEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react';
import { nodeOptWord, paramWords } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { gesture } from './value.ts';
import { newMotion, stopMotion, toTarget, xyBox, XY_IN, type Motion } from './xyMotion.ts';
import type { ViewProps } from './types.ts';

export function XyPad(props: ViewProps) {
  const { node, nodeKey, params } = props;
  const say = useSay(), desk = useDesk(s => s.desk);
  const el = useRef<HTMLDivElement>(null), motion = useRef<Motion>(newMotion());
  const [px, py] = Object.values(params);
  const frac = (p: typeof px) => { if (!p) return 0; const [min, max] = p.face.range ?? [0, 1]; return ((p.value as number) - min) / (max - min); };
  const fx = frac(px), fy = frac(py);
  useLayoutEffect(() => { if (el.current) toTarget(el.current, motion.current, fx, 1 - fy); });
  useEffect(() => { const m = motion.current; return () => stopMotion(m); }, []);
  if (!px || !py) return null;

  const o = (node.opts ?? {}) as { look?: string };
  const xw = paramWords(ix, px.key, desk), yw = paramWords(ix, py.key, desk);
  const xl = nodeOptWord(ix, node, desk, nodeKey, 'xl') ?? xw.label, yl = nodeOptWord(ix, node, desk, nodeKey, 'yl') ?? yw.label;
  const tag = `xy:${px.key}`, ktag = `xykey:${px.key}`;
  /* while a hand or a key works the pad, and a moment after, its changes animate */
  const working = (on: boolean) => { const m = motion.current; clearTimeout(m.liveT); if (on) m.live = true; else m.liveT = window.setTimeout(() => { m.live = false; }, 600); };
  const setAt = (e: PointerEvent<HTMLDivElement>) => {
    const t = el.current!, r = t.getBoundingClientRect(), [SL, SB] = xyBox(t), W = r.width - SL - 2 * XY_IN, H = r.height - SB - 2 * XY_IN, m = motion.current;
    let ax = (e.clientX - r.left - SL - XY_IN) / W, ay = 1 - (e.clientY - r.top - XY_IN) / H;
    if (e.shiftKey && m.fine) { ax = m.fine.fx + (e.clientX - m.fine.x) / W / 10; ay = m.fine.fy - (e.clientY - m.fine.y) / H / 10; }
    m.fine = e.shiftKey ? { x: e.clientX, y: e.clientY, fx: Math.min(1, Math.max(0, ax)), fy: Math.min(1, Math.max(0, ay)) } : null;
    const at = (p: typeof px, f: number) => { const [min, max] = p!.face.range ?? [0, 1]; return min + Math.min(1, Math.max(0, f)) * (max - min); };
    props.batch(() => { px.set(at(px, ax)); py.set(at(py, ay)); });
  };
  const down = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault(); const t = e.currentTarget, m = motion.current;
    t.setPointerCapture(e.pointerId); t.focus({ preventScroll: true }); t.classList.remove('release'); t.classList.add('grab');
    m.dragging = true; working(true); gesture.begin(tag);
    if (e.shiftKey) { m.fine = { x: e.clientX, y: e.clientY, fx, fy }; return; }
    m.fine = null;
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) { const r = t.getBoundingClientRect(), rip = document.createElement('span'); rip.className = 'xy-rip'; rip.style.left = e.clientX - r.left + 'px'; rip.style.top = e.clientY - r.top + 'px'; t.appendChild(rip); rip.addEventListener('animationend', () => rip.remove()); }
    setAt(e);
  };
  const up = (e: PointerEvent<HTMLDivElement>) => {
    const m = motion.current; if (!m.dragging) return; m.dragging = false; working(false); gesture.end(tag);
    const t = e.currentTarget; t.classList.remove('grab'); t.classList.add('release'); window.setTimeout(() => t.classList.remove('release'), 420);
  };
  const nudge = () => { const t = el.current!, m = motion.current; t.classList.add('nudge'); clearTimeout(m.nudge); m.nudge = window.setTimeout(() => t.classList.remove('nudge'), 900); };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const [xmin, xmax] = px.face.range ?? [0, 1];
    if (e.key === 'Home' || e.key === 'End') { e.preventDefault(); gesture.end(ktag); working(true); px.set(e.key === 'Home' ? xmin : xmax); nudge(); return; }
    const mv = ({ ArrowLeft: [px, -1], ArrowRight: [px, 1], ArrowDown: [py, -1], ArrowUp: [py, 1], PageUp: [py, 10], PageDown: [py, -10] } as Record<string, [typeof px, number]>)[e.key];
    if (!mv) return;
    e.preventDefault();
    const [p, d] = mv, st = p.face.range?.[2] ?? 0.01;
    if (!e.repeat) gesture.end(ktag); gesture.begin(ktag); working(true);
    p.set(+((p.value as number) + d * (/^Page/.test(e.key) ? 1 : e.shiftKey ? 10 : 1) * st).toFixed(10)); nudge();
  };
  const reset = () => { working(true); props.batch(() => { px.reset(); py.reset(); }); working(false); };

  return (
    <div ref={el} className={`xy ${o.look ? `xy-${o.look}` : ''} `} data-xy={`${px.key}|${py.key}`} tabIndex={0} role="slider" aria-roledescription={say(ui('xy.role'))}
      aria-label={say(ui('xy.label', { x: xl, y: yl }))} title={say(ui('xy.title', { x: xl, y: yl }))}
      aria-valuetext={say(ui('xy.text', { x: xw.label, xv: px.text, y: yw.label, yv: py.text }))} aria-valuemin={px.face.range?.[0]} aria-valuemax={px.face.range?.[1]} aria-valuenow={px.value as number}
      onPointerDown={down} onPointerMove={e => { if (motion.current.dragging) setAt(e); }} onPointerUp={up} onPointerCancel={up}
      onKeyDown={key} onKeyUp={() => { gesture.end(ktag); working(false); }} onDoubleClick={reset}>
      <canvas className="xy-trail" aria-hidden="true" /><i className="xy-grid" aria-hidden="true" />
      <i className="xy-h" /><i className="xy-v" /><i className="xy-tx" /><i className="xy-ty" />
      <b className="xy-dot"><i className="ring" /><i className="core" /></b><span className="xy-tip" aria-hidden="true"><b>{say(px.text)}</b><span>·</span><b>{say(py.text)}</b></span>
      <span className={`xy-ax x${px.changed ? ' changed' : ''}`} data-ctl={px.key}>{`${say(xl)} `}<output data-xyx="">{say(px.text)}</output></span>
      <span className={`xy-ax y${py.changed ? ' changed' : ''}`} data-ctl={py.key}>{`${say(yl)} `}<output data-xyy="">{say(py.text)}</output></span>
    </div>
  );
}
