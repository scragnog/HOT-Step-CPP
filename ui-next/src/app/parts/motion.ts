// The instruments' motion (package 3a): the page's motion tokens (00-tokens.css: --t-base, --e-out ...) read from the
// page, and whether motion is reduced, so a part's animation keeps the concept's timing (18-motion.js mtime,
// motionVar). Not a part: the parts beside it import it. Its numbers are times read from the page's stylesheet, not
// values of the desk.

/** A motion token's value as the page's stylesheet sets it ("cubic-bezier(...)", "200ms"). */
export const motionVar = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
/** A motion token's time, in milliseconds. */
export const mtime = (name: string): number => { const v = motionVar(name); return v.endsWith('ms') ? parseFloat(v) : v.endsWith('s') ? parseFloat(v) * 1000 : 0; };
/** Motion is reduced: each instrument draws its designed alternative (M-08). */
export const reduced = (): boolean => matchMedia('(prefers-reduced-motion: reduce)').matches;
