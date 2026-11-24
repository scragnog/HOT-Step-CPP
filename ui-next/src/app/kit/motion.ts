// The page's motion, stated once for every part and shell file: whether motion is reduced, and the page's motion tokens
// (00-tokens.css: --t-base, --e-out ...) as the stylesheet sets them (the concept's 18-motion.js mtime, motionVar). Its
// numbers are times and curves read from the page, never values of the desk.

const root = () => getComputedStyle(document.documentElement);
/** Motion is reduced: each animation draws its designed alternative (M-08). */
export const reduced = (): boolean => matchMedia('(prefers-reduced-motion: reduce)').matches;
/** A motion token's value as the page's stylesheet sets it ("cubic-bezier(...)", "200ms"). */
export const motionVar = (name: string): string => root().getPropertyValue(name).trim();
/** A motion token's time, in milliseconds ("180ms", ".3s"); 0 for a token the page does not set. */
export const mtime = (name: string): number => { const v = motionVar(name); return v.endsWith('ms') ? parseFloat(v) || 0 : v.endsWith('s') ? (parseFloat(v) || 0) * 1000 : 0; };
/** A motion token's curve; the browser's ease-out for one the page does not set. */
export const easing = (name: string): string => motionVar(name) || 'ease-out';
