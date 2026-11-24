// takesReels: the takes count as an LCD and a set of reels, one per take the engine allows, and under them the VRAM
// the count costs (the concept's 25-module-takes.js). The reels are one slider and one tab stop (A-23): arrows, Page,
// Home and End move it, each reel is a press target, a second press on one is the same as the first (L-22). Pointing
// at a reel previews its count (G-B-07); a change from the keyboard ends the preview until the pointer moves (L-32).
// Choosing kicks each newly lit reel round in turn, 60ms apart, and the LCD digit drops into place (G-M-12). How many
// reels, the grid, the count's words and the cost are the core's (PartModel.instrument.takes).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';
import { motionVar, mtime, reduced } from '../kit/motion.ts';

const KEYS: Record<string, (n: number, max: number) => number> = {
  ArrowRight: n => n + 1, ArrowUp: n => n + 1, ArrowLeft: n => n - 1, ArrowDown: n => n - 1, Home: () => 1, End: (_, m) => m, PageUp: (_, m) => m, PageDown: () => 1,
};

export function TakesReels(v: PartView) {
  const say = useSay(), t = v.instrument?.takes, p = v.params[0];
  const box = useRef<HTMLDivElement>(null), was = useRef<number | null>(null), at = useRef<[number, number] | null>(null), last = useRef<[number, number]>([0, 0]);
  const [preview, setPreview] = useState<number | null>(null), [kbd, setKbd] = useState(false);
  const n = t?.n ?? 0;
  /* the digit drops in from the side the count moved to, and the newly lit reels kick round */
  useLayoutEffect(() => {
    const el = box.current, n0 = was.current; was.current = n;
    if (!el || n0 === null || n0 === n || reduced()) return;
    el.querySelector('.tk-lcd b')?.animate([{ opacity: 0, transform: `translateY(${n > n0 ? -6 : 6}px)` }, { opacity: 1, transform: 'none' }], { duration: mtime('--t-base'), easing: motionVar('--e-out') });
    [...el.querySelectorAll('.reel2')].slice(n0, n).forEach((r, i) => r.animate([{ rotate: '-120deg', scale: '.9' }, { rotate: '0deg', scale: '1' }], { duration: mtime('--t-slow'), delay: i * 60, easing: motionVar('--e-out'), fill: 'backwards' }));
  }, [n]);
  /* only a real move counts: the browser sends one of its own when the reels are redrawn under a still pointer */
  useEffect(() => {
    if (!kbd) return;
    const f = (e: { clientX: number; clientY: number }) => { const a = at.current; if (!a || Math.hypot(e.clientX - a[0], e.clientY - a[1]) > 2) { setKbd(false); at.current = null; } };
    document.addEventListener('pointermove', f, { passive: true }); return () => document.removeEventListener('pointermove', f);
  }, [kbd]);
  useEffect(() => { const f = (e: { clientX: number; clientY: number }) => { if (!at.current) last.current = [e.clientX, e.clientY]; }; document.addEventListener('pointermove', f, { passive: true }); return () => document.removeEventListener('pointermove', f); }, []);
  if (!t || !p) return null;
  const set = (to: number) => { p.set(to); requestAnimationFrame(() => box.current?.querySelector<HTMLElement>('[data-takes]')?.focus({ preventScroll: true })); };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const f = KEYS[e.key]; if (!f) return;
    e.preventDefault(); const to = Math.min(t.max, Math.max(1, f(n, t.max)));
    if (to === n) return;
    set(to); setKbd(true); setPreview(null); at.current = last.current;
  };
  const over = (e: PointerEvent<HTMLDivElement>) => { if (kbd) return; const r = (e.target as Element).closest?.('[data-take]'); setPreview(r ? +(r.getAttribute('data-take') ?? 0) || null : null); };
  const shown = preview ?? n, pv = preview !== null && preview !== n;
  const c = t.cost;
  return (
    <>
      <div ref={box} className="takes2">
        <div className={`tk-lcd${pv ? ' pv' : ''}`} title={p.tip ? say(p.tip) : undefined}>
          <b>{shown}</b><small className={`cdot-lcd${p.changed ? ' changed' : ''}`} data-ctl={p.key}>{say(shown === 1 ? t.labels[0] : t.labels[1])}</small>
        </div>
        <div className={`tk-reels${kbd ? ' kbd' : ''}`} style={{ '--cols': t.cols } as CSSProperties} data-takes="" role="slider" tabIndex={0}
          aria-label={say(p.label)} aria-valuemin={1} aria-valuemax={t.max} aria-valuenow={n} aria-valuetext={say(t.valuetext)} title={say(ui('takes.title'))}
          onKeyDown={key} onPointerOver={over} onPointerLeave={() => setPreview(null)}>
          {Array.from({ length: t.max }, (_, i) => (
            <button key={i} type="button" className={`reel2${i < n ? ' on' : ''}`} data-take={i + 1} tabIndex={-1} aria-hidden="true" style={{ '--i': i } as CSSProperties} onClick={() => set(i + 1)}><i /></button>
          ))}
        </div>
      </div>
      <div className={`tk-cost${c.cls ? ' ' + c.cls : ''}`}>
        <span className="tk-meter"><span className="tk-bar" title={say(c.tip)}><i style={{ width: `${c.bar * 100}%` }} /></span><span className="tk-gb">{say(c.text)}</span></span>
        {c.msg ? <span>{say(c.msg)}</span> : null}
        {c.each ? <span>{say(c.each)}</span> : null}
      </div>
      {p.held ? <p className="tk-held" title={say(ui('takes.held.title'))}>{say(p.held)}</p> : null}
    </>
  );
}

definePart('takesReels', TakesReels);
