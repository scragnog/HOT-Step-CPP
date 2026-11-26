// repaintRegion: the song to repaint, its waveform and the region to regenerate (the concept's 25-module-repaint.js).
// The region's edges are handles and its body can be carried (A-32): a press within 8px of an edge drags that edge, a
// press inside drags the whole region, a press outside starts a new one only once the pointer has moved 4px, so a stray
// click never loses a region set with care; each edge carries a grab tab at mid-height (Q-31). Arrows move it, Shift
// moves its end, Alt moves further, Home and End take it to the song's ends (L-36). A drag is one step, a key press
// another, each named in the song's times (Q-24). Where a hand takes the region is the core's (act('move')); its
// children stand on the wave's caption line (the first two, at Lab) or beside the wave (the rest).

import { Children, useEffect, useLayoutEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react';
import { definePart, ui, useSay, type PartView } from '../kit/index.ts';

/* the concept's seeded generator (00-util.js rng), so a song's waveform and art are the same on every visit */
const rng = (seed: number) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
/** A song's art (35-songs.js artBg): two soft lights over a conic ground, in its hue. */
function artBg(id: number, hue: number): string {
  const r = rng(id * 977 + hue), h2 = (hue + 40 + r() * 80) % 360;
  return `radial-gradient(circle at ${20 + r() * 60}% ${20 + r() * 60}%, hsl(${hue} 92% 64%) 0, transparent ${40 + r() * 20}%),radial-gradient(circle at ${r() * 100}% ${r() * 100}%, hsl(${h2} 85% 55%) 0, transparent ${45 + r() * 20}%),conic-gradient(from ${r() * 360}deg at ${30 + r() * 40}% ${30 + r() * 40}%, hsl(${hue} 50% 22%), hsl(${h2} 60% 14%), hsl(${hue} 50% 22%))`;
}
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

function draw(cv: HTMLCanvasElement, id: number, a: number, b: number): void {
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; if (!w) return;
  cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext('2d'); if (!g) return; g.scale(dpr, dpr);
  const cs = getComputedStyle(document.documentElement), hot = cs.getPropertyValue('--hot').trim(), faint = cs.getPropertyValue('--faint').trim();
  const r = rng(id * 31), n = Math.floor(w / 3), lo = Math.min(a, b), hi = Math.max(a, b);
  g.fillStyle = hot + '22'; g.fillRect(lo * w, 0, (hi - lo) * w, h);
  for (let i = 0; i < n; i++) { const t = i / n, env = 0.35 + 0.65 * Math.abs(Math.sin(t * 9 + r())) * (0.6 + 0.4 * r()), amp = env * (h / 2 - 6); g.fillStyle = t >= lo && t <= hi ? hot : faint; g.fillRect(i * 3, h / 2 - amp, 2, amp * 2); }
  g.fillStyle = hot; [a, b].forEach(p => g.fillRect(p * w - 1, 0, 2, h));
  [a, b].forEach(p => { const x = clamp(p * w - 5, 0, w - 10); g.beginPath(); if (g.roundRect) g.roundRect(x, h / 2 - 12, 10, 24, 3); else g.rect(x, h / 2 - 12, 10, 24); g.fill(); g.fillStyle = '#fff'; for (let k = -1; k <= 1; k++) g.fillRect(x + 4, h / 2 + k * 5 - 1, 2, 2); g.fillStyle = hot; });
}

type Kind = 'a' | 'b' | 'body' | 'out';
export function RepaintRegion(v: PartView) {
  const say = useSay(), m = v.instrument?.region, cv = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ kind: Kind; x0: number; a0: number; b0: number; moved: boolean } | null>(null);
  useLayoutEffect(() => { if (m && cv.current) draw(cv.current, m.id, m.a, m.b); }, [m]);
  useEffect(() => { const f = () => { if (m && cv.current) draw(cv.current, m.id, m.a, m.b); }; addEventListener('resize', f); return () => removeEventListener('resize', f); }, [m]);
  if (!m) return null;
  const kids = Children.toArray(v.children), ctl = kids.slice(0, 2), side = kids.slice(2);
  const hit = (x: number, w: number): Kind => (Math.abs(x - m.a * w) <= 8 ? 'a' : Math.abs(x - m.b * w) <= 8 ? 'b' : x > m.a * w && x < m.b * w ? 'body' : 'out');
  const down = (e: PointerEvent<HTMLCanvasElement>) => {
    const c = e.currentTarget, r = c.getBoundingClientRect(), x = e.clientX - r.left;
    e.preventDefault(); c.focus({ preventScroll: true }); c.setPointerCapture(e.pointerId);
    drag.current = { kind: hit(x, r.width), x0: x / r.width, a0: m.a, b0: m.b, moved: false };
  };
  const move = (e: PointerEvent<HTMLCanvasElement>) => {
    const c = e.currentTarget, r = c.getBoundingClientRect(), x = e.clientX - r.left, d = drag.current;
    if (!d) { c.style.cursor = { a: 'ew-resize', b: 'ew-resize', body: 'grab', out: 'crosshair' }[hit(x, r.width)]; return; }
    if (!d.moved && Math.abs(x - d.x0 * r.width) < 4) return;
    if (!d.moved) { d.moved = true; v.gesture.begin('drag'); }
    if (d.kind === 'body') c.style.cursor = 'grabbing';
    v.act('move', { kind: d.kind, x0: d.x0, x: x / r.width, a0: d.a0, b0: d.b0 });
  };
  const up = () => { const d = drag.current; drag.current = null; if (d?.moved) v.gesture.end('drag'); };
  const key = (e: KeyboardEvent<HTMLCanvasElement>) => {
    if (!/^(Arrow(Left|Right|Up|Down)|Home|End)$/.test(e.key)) return;
    e.preventDefault(); v.act('move', { key: e.key, shift: e.shiftKey, alt: e.altKey });
  };
  return (
    <div className={`rp-grid${m.one ? ' rp-one' : ''}`}>
      <div className="wave">
        <div className="songref"><div className="art" style={{ background: artBg(m.id, m.hue) }} /><div><b>{m.title}</b><span>{say(m.style)}</span></div></div>
        <canvas ref={cv} data-rpwave="" tabIndex={0} role="slider" aria-valuemin={0} aria-valuemax={Math.round(m.dur)} aria-valuenow={Math.round(m.a * m.dur)} aria-valuetext={say(m.text)}
          aria-label={say(ui('region.label'))} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key} />
        <div className="cap"><span data-rpcap="">{say(m.cap)}</span>{m.ctl ? <span className="rp-ctl">{ctl}</span> : null}<span>{m.durText}</span></div>
      </div>
      <div style={{ display: 'grid', gap: '10px', alignContent: 'start' }}>{side}</div>
    </div>
  );
}

definePart('repaintRegion', RepaintRegion);
