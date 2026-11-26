// Where the toast stands (36-toast.js placeToast): bottom-centre of the desk column, 16px above the go bar while the
// go bar is stuck to the window's bottom edge, else 24px from the window's bottom edge; it steps up out of the way of
// an open popover, menu or dropdown (D-27, B-07). It does not follow the page as it scrolls: it moves only when the go
// bar sticks or unsticks. Drawing only.

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
/* the engine log docked at the window's foot (features/logs.tsx): the go bar stands on it, and so does the toast */
export const dockH = () => { const l = document.getElementById('logs'); return !l || l.hidden ? 0 : l.getBoundingClientRect().height; };

export function placeToast(): void {
  const t = document.getElementById('toast'); if (!t || t.hidden) return;
  const g = document.querySelector('.gobar')?.getBoundingClientRect(), cr = document.getElementById('centre')?.getBoundingClientRect(), vw = document.documentElement.clientWidth;
  const onCreate = !document.getElementById('app')?.hidden && !!cr && cr.width > 0;
  const stuck = onCreate && !!g && g.height > 0 && Math.abs(g.bottom - (innerHeight - dockH())) < 2;
  let bottom = stuck ? innerHeight - g!.top + 16 : 24 + dockH();
  const w = t.offsetWidth, cx = onCreate ? cr!.left + cr!.width / 2 : vw / 2, left = clamp(cx - w / 2, 16, vw - w - 16);
  const tr = { l: left, r: left + w, b: innerHeight - bottom, t: innerHeight - bottom - t.offsetHeight };
  for (const s of ['#pop', '#menu', '#dd']) {
    const o = document.querySelector<HTMLElement>(s); if (!o || o.hidden) continue;
    const q = o.getBoundingClientRect(); if (q.left < tr.r && q.right > tr.l && q.top < tr.b && q.bottom > tr.t) bottom = Math.max(bottom, innerHeight - q.top + 12);
  }
  t.style.right = 'auto'; t.style.left = left + 'px'; t.style.bottom = `calc(${Math.round(bottom)}px + env(safe-area-inset-bottom, 0px))`;
}

/** Re-places the toast on resize, and on scroll when the go bar sticks or unsticks; returns the cleanup. */
export function watchToast(): () => void {
  let stuck: boolean | null = null;
  const scroll = () => { const g = document.querySelector('.gobar')?.getBoundingClientRect(), s = !!g && Math.abs(g.bottom - (innerHeight - dockH())) < 2; if (s !== stuck) { stuck = s; placeToast(); } };
  addEventListener('resize', placeToast); addEventListener('scroll', scroll, { passive: true });
  return () => { removeEventListener('resize', placeToast); removeEventListener('scroll', scroll); };
}
