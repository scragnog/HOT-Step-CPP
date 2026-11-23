// The ring that says "it was already here" when something already on the desk is added again (the concept's flash and
// flashPedal, 30-desk.js): the tile, or the pedal or side unit on the board, is brought into view and rung. Drawing
// only: what to ring is the store's (flash).

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const slow = () => (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--t-slow')) || 600) * 2;

export function ring(id: string, kind: 'flash' | 'here'): void {
  requestAnimationFrame(() => {
    let el: HTMLElement | null = null, cls: string = kind;
    if (kind === 'here') el = document.querySelector<HTMLElement>(`.pedal[data-pid="${id}"], .pedal[data-xid="${id}"]`);
    if (!el) { el = document.querySelector<HTMLElement>(kind === 'here' ? '.tile[data-m="post"]' : `.tile[data-m="${id}"], .pchip[data-pm="${id}"]`); cls = 'flash'; }
    if (!el) return;
    el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls);
    window.setTimeout(() => el.classList.remove(cls), reduced() ? 1200 : slow());
    el.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
  });
}
