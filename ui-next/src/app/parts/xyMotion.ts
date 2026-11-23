// How an XY pad's puck moves (the concept's applyXY, stepXY and trailDraw, 21-xy-pad.js): it springs toward its target,
// stretched along its direction of travel, leaving a short faint trail; the read-outs it comes near fade back. Drawing
// and motion only; the pad's values are its parameters'.

const XYIN = 18;
export interface Motion {
  cur: { x: number; y: number } | null;
  tgt: { x: number; y: number };
  anim: number; last: number; raf: number;
  pts: { x: number; y: number; t: number }[];
  fine: { x: number; y: number; fx: number; fy: number } | null;
  dragging: boolean; nudge: number; live: boolean; liveT: number;
}
export const newMotion = (): Motion => ({ cur: null, tgt: { x: 0, y: 0 }, anim: 0, last: 0, raf: 0, pts: [], fine: null, dragging: false, nudge: 0, live: false, liveT: 0 });

/** The plot's read-out strips: --sl at the left for Y, --sb along the foot for X. */
export const xyBox = (el: HTMLElement) => { const cs = getComputedStyle(el); return [parseFloat(cs.getPropertyValue('--sl')) || 0, parseFloat(cs.getPropertyValue('--sb')) || 0] as const; };
export const XY_IN = XYIN;

/** Puts the puck at the motion's current place, stretched by its speed. */
export function applyXY(el: HTMLElement, m: Motion, vx: number, vy: number): void {
  const c = m.cur; if (!c) return;
  const sp = Math.hypot(vx, vy), st = 1 + Math.min(sp * 0.45, 0.55), [SL, SB] = xyBox(el);
  el.style.setProperty('--dx', `calc(${SL + XYIN}px + ${c.x.toFixed(4)} * (100% - ${SL + 2 * XYIN}px))`);
  el.style.setProperty('--dy', `calc(${XYIN}px + ${c.y.toFixed(4)} * (100% - ${SB + 2 * XYIN}px))`);
  el.style.setProperty('--sx', st.toFixed(3)); el.style.setProperty('--sy', (1 / Math.sqrt(st)).toFixed(3));
  if (sp > 0.03) el.style.setProperty('--ang', Math.atan2(vy, vx).toFixed(3) + 'rad');
  el.classList.toggle('tip-left', c.x > 0.6); el.classList.toggle('tip-down', c.y < 0.28);
  const W = el.clientWidth, H = el.clientHeight, x = SL + XYIN + c.x * (W - SL - 2 * XYIN), y = XYIN + c.y * (H - SB - 2 * XYIN);
  el.querySelectorAll<HTMLElement>('.xy-c, .xy-ax').forEach(n => { const dx = Math.max(n.offsetLeft - x, 0, x - n.offsetLeft - n.offsetWidth), dy = Math.max(n.offsetTop - y, 0, y - n.offsetTop - n.offsetHeight), d = Math.hypot(dx, dy); n.style.setProperty('--near', d < 46 ? (0.12 + 0.88 * d / 46).toFixed(2) : '1'); });
}

function drawTrail(el: HTMLElement, m: Motion): void {
  m.raf = 0; const cv = el.querySelector<HTMLCanvasElement>('.xy-trail'); if (!cv || !el.isConnected) return;
  const dpr = devicePixelRatio || 1, w = el.clientWidth, h = el.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d'); if (!g) return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const now = performance.now(), life = 340; m.pts = m.pts.filter(p => now - p.t < life);
  g.strokeStyle = getComputedStyle(el).getPropertyValue('--xyc').trim() || '#a6f5bd'; g.lineCap = 'round'; g.lineJoin = 'round';
  for (let i = 1; i < m.pts.length; i++) { const a = m.pts[i - 1]!, b = m.pts[i]!, age = 1 - (now - b.t) / life; g.globalAlpha = age * age * 0.32; g.lineWidth = 0.8 + age * 1.6; g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); }
  g.globalAlpha = 1; if (m.pts.length) m.raf = requestAnimationFrame(() => drawTrail(el, m));
}

function step(el: HTMLElement, m: Motion, t: number): void {
  m.anim = 0; if (!el.isConnected || !m.cur) return;
  const dt = Math.min(48, Math.max(1, t - m.last)); m.last = t;
  const k = 1 - Math.exp(-dt / 55), c = m.cur, g = m.tgt, x0 = c.x, y0 = c.y;
  c.x += (g.x - c.x) * k; c.y += (g.y - c.y) * k;
  const w = el.clientWidth, h = el.clientHeight, vx = (c.x - x0) * w / dt, vy = (c.y - y0) * h / dt, [SL, SB] = xyBox(el);
  applyXY(el, m, vx, vy);
  if (Math.hypot(vx, vy) > 0.02) { m.pts.push({ x: SL + XYIN + c.x * (w - SL - 2 * XYIN), y: XYIN + c.y * (h - SB - 2 * XYIN), t: performance.now() }); if (!m.raf) m.raf = requestAnimationFrame(() => drawTrail(el, m)); }
  if (Math.abs(g.x - c.x) + Math.abs(g.y - c.y) > 0.0006) m.anim = requestAnimationFrame(tt => step(el, m, tt));
  else { c.x = g.x; c.y = g.y; applyXY(el, m, 0, 0); }
}

/** Moves the puck to its target: springing there while the pad is being worked, else at once (a value that changed
 *  elsewhere is drawn where it is, as the concept draws a pad it has just rebuilt). */
export function toTarget(el: HTMLElement, m: Motion, x: number, y: number): void {
  m.tgt = { x, y };
  if (!m.cur || matchMedia('(prefers-reduced-motion: reduce)').matches || !m.live) {
    cancelAnimationFrame(m.anim); m.anim = 0; m.cur = { ...m.tgt }; el.style.removeProperty('--ang'); applyXY(el, m, 0, 0);
  } else if (!m.anim) { m.last = performance.now(); m.anim = requestAnimationFrame(t => step(el, m, t)); }
}
export const stopMotion = (m: Motion) => { cancelAnimationFrame(m.anim); cancelAnimationFrame(m.raf); };
