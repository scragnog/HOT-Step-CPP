// The page's motion, stated once for every part and shell file: whether motion is reduced, and the page's motion tokens
// (00-tokens.css: --t-base, --e-out ...) as the stylesheet sets them (the concept's 18-motion.js mtime, motionVar). Its
// numbers are times and curves read from the page, never values of the desk.

const root = () => getComputedStyle(document.documentElement);
/** Motion is reduced: each animation draws its designed alternative (M-08). */
export const reduced = (): boolean => matchMedia('(prefers-reduced-motion: reduce)').matches;
/* the tokens are the stylesheet's constants (00-tokens.css sets each once, on :root): each is read once, not on every
   motion, where reading it would make the browser work out the page's style in the middle of a change */
const tokens = new Map<string, string>();
/** A motion token's value as the page's stylesheet sets it ("cubic-bezier(...)", "200ms"). */
export const motionVar = (name: string): string => { let v = tokens.get(name); if (v === undefined) { v = root().getPropertyValue(name).trim(); if (v) tokens.set(name, v); } return v; };
/** A motion token's time, in milliseconds ("180ms", ".3s"); 0 for a token the page does not set. */
export const mtime = (name: string): number => { const v = motionVar(name); return v.endsWith('ms') ? parseFloat(v) || 0 : v.endsWith('s') ? (parseFloat(v) || 0) * 1000 : 0; };
/** A motion token's curve; the browser's ease-out for one the page does not set. */
export const easing = (name: string): string => motionVar(name) || 'ease-out';
let boot = false;
const settled = () => { if (!document.getElementById('bento')) { requestAnimationFrame(settled); return; } requestAnimationFrame(() => requestAnimationFrame(() => { boot = true; })); };
requestAnimationFrame(settled);
/** The page has drawn its first desk: what it does from then on moves (the concept's booted); what it draws as it loads
 *  just appears. */
export const booted = (): boolean => boot;
