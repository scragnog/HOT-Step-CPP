// How each engine looks: its mark and its colour (the concept's LOGO and ENG, 01-engines.js). Visual identity only;
// what an engine has is the definitions' (src/definitions/engines.ts and each definition's `engines`).

import type { EngineId } from '../../definitions/types.ts';

export const ENGINE_COLOUR: { [e in EngineId]: string } = { ace: '#ff5b8a', 'minimax-m3': '#ff6a3d', yue2: '#f5a623', sa3: '#9b85ff', heartmula: '#3fb6ff' };

const LOGO: { [e in EngineId]: string } = {
  ace: `<svg class="lg lg-ace" viewBox="0 0 48 48" aria-hidden="true"><g class="lg-spark"><path d="M24 6 L26.6 15.6 C27.9 20.4 31.6 24.1 36.4 25.4 L46 28 L36.4 30.6 C31.6 31.9 27.9 35.6 26.6 40.4 L24 50 L21.4 40.4 C20.1 35.6 16.4 31.9 11.6 30.6 L2 28 L11.6 25.4 C16.4 24.1 20.1 20.4 21.4 15.6 Z" transform="translate(24 24) scale(.86) translate(-24 -28)" fill="currentColor"/></g><path class="lg-glint" d="M38 7 l1.1 3.9 3.9 1.1 -3.9 1.1 -1.1 3.9 -1.1 -3.9 -3.9 -1.1 3.9 -1.1z" fill="currentColor"/></svg>`,
  'minimax-m3': `<svg class="lg lg-mm3" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.32" stroke-linecap="round" stroke-linejoin="round">
    <path class="lg-base" d="M.66 11.5V12.9a1.42 1.42 0 0 0 2.84 0V9.1a1.42 1.42 0 0 1 2.84 0v9a1.42 1.42 0 0 0 2.84 0V4.1a1.42 1.42 0 0 1 2.84 0v11.5M12.02 18.04v1.95a1.42 1.42 0 0 0 2.84 0V4.1a1.42 1.42 0 0 1 2.84 0v12.5a1.42 1.42 0 0 0 2.84 0V9.1a1.42 1.42 0 0 1 2.84 0v6.55"/>
    <path class="lg-run" pathLength="100" d="M.66 11.5V12.9a1.42 1.42 0 0 0 2.84 0V9.1a1.42 1.42 0 0 1 2.84 0v9a1.42 1.42 0 0 0 2.84 0V4.1a1.42 1.42 0 0 1 2.84 0v11.5M12.02 18.04v1.95a1.42 1.42 0 0 0 2.84 0V4.1a1.42 1.42 0 0 1 2.84 0v12.5a1.42 1.42 0 0 0 2.84 0V9.1a1.42 1.42 0 0 1 2.84 0v6.55"/></svg>`,
  yue2: `<svg class="lg lg-yue" viewBox="0 0 48 48" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle class="lg-ring" cx="24" cy="24" r="19.6"/>
    <path class="lg-line" pathLength="100" d="M27.6 4.6V11.2C27.6 13.6 26.2 15.2 23.8 15.2H18.4C16.2 15.2 14.8 16.7 14.8 18.6S16.2 22 18.4 22H33.4"/>
    <g class="lg-n n1"><path d="M18.6 22V32.6"/><circle cx="17.3" cy="33.4" r="1.9" fill="currentColor" stroke="none"/></g>
    <g class="lg-n n2"><path d="M24.6 22V34.8"/></g>
    <g class="lg-n n3"><path d="M30.4 22V32.6"/><circle cx="29.1" cy="33.4" r="1.9" fill="currentColor" stroke="none"/></g></svg>`,
  sa3: `<svg class="lg lg-sa3" viewBox="0 0 48 48" aria-hidden="true">
    <path class="lg-dome" d="M4 40A20 20 0 0 1 44 40Z" fill="currentColor"/>
    <g fill="none" stroke="#fff" stroke-width="1.9" stroke-linecap="round"><path class="lg-b1" d="M4.5 40C10 40 13.5 31 18.5 31S27 40 34 40"/><path class="lg-b2" d="M6 40C14 40 22.5 26.5 30 26.5S38 40 43.5 40"/></g></svg>`,
  heartmula: `<svg class="lg lg-hm" viewBox="0 0 48 48" aria-hidden="true" fill="none" stroke-linecap="round" stroke-linejoin="round"><defs><linearGradient id="hmRainbow" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff4d4d"/><stop offset=".25" stop-color="#ffb020"/><stop offset=".5" stop-color="#36d96b"/><stop offset=".75" stop-color="#3fb6ff"/><stop offset="1" stop-color="#b35cff"/></linearGradient></defs>
    <path class="lg-base" stroke="currentColor" stroke-width="3.6" d="M37.2 15.2V18.6A5.15 5.15 0 0 1 26.9 18.6V11.6L18.8 18.2L11.2 11.3V34.8C11.2 36.3 12.2 37 13.7 37H24.4L31.9 29.7L39.3 37.5"/>
    <path class="lg-run" pathLength="100" stroke="url(#hmRainbow)" stroke-width="3.6" d="M37.2 15.2V18.6A5.15 5.15 0 0 1 26.9 18.6V11.6L18.8 18.2L11.2 11.3V34.8C11.2 36.3 12.2 37 13.7 37H24.4L31.9 29.7L39.3 37.5"/></svg>`,
};
/** An engine's mark at a size, as markup. */
export const logoOf = (e: EngineId, size: number): string => LOGO[e].replace('<svg ', `<svg width="${size}" height="${size}" `);
/** The concept's short engine ids, which its markup and its saved shapes use (data-backend="mm3"). */
export const CONCEPT_ID: { [e in EngineId]: string } = { ace: 'ace', 'minimax-m3': 'mm3', yue2: 'yue', sa3: 'sa3', heartmula: 'hm' };
