// Thumbnails: each definition's drawing (src/definitions/thumbs/*.svg, SVG fragments in the thumbnail kit's classes)
// in the frame the concept draws around it (24-thumbnails.js thSvg and thModule, 27-pedalboard.js thPedal and
// thPedalMounted, 28-side-units.js thUnit). The frame of a pedal and a side unit is drawn from the definition; a
// module's file is the whole drawing. The result is markup, set as the inner HTML of the card's thumbnail box.

import type { Definition, EngineId } from '../../definitions/types.ts';
import { pedalUnits } from '../../definitions/parts.ts';

const FILES = import.meta.glob('../../definitions/thumbs/*.svg', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
const fragment = (file: string): string => {
  const raw = FILES[`../../definitions/thumbs/${file}.svg`] ?? '';
  return raw.replace(/<!--[\s\S]*?-->/g, '').replace(/\r?\n/g, '').trim();
};

const TH_W = 176, TH_H = 112, TH_IN = 15, TH_PH = 110;
let uid = 0;
const r1 = (n: number) => Math.round(n * 10) / 10;
const rect = (cls: string, x: number, y: number, w: number, h: number, r = 8) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}"/>`;
const bar = (cls: string, x: number, y: number, w: number, h = 3) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${h / 2}"/>`;
const led = (cx: number, cy: number) => `<circle class="led-on" cx="${cx}" cy="${cy}" r="2.4"/>`;
/* inner class names are prefixed t-, so no page rule can reach into a drawing */
const svg = (inner: string, cls = '', vb = `0 0 ${TH_W} ${TH_H}`, style = '') =>
  `<svg class="th ${cls}" viewBox="${vb}" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false"${style ? ` style="${style}"` : ''}>${inner.replace(/class="([^"]+)"/g, (_m, c: string) => `class="${c.trim().split(/\s+/).map(x => 't-' + x).join(' ')}"`)}</svg>`;

const fileOn = (d: Definition, e: EngineId): string => d.thumb?.per?.find(l => l.engines.includes(e))?.file ?? d.thumb?.file ?? '';
const pedalW = (d: Definition) => (pedalUnits(d.params.filter(p => p.type === 'number').length) === 2 ? 184 : 80);

function pedal(d: Extract<Definition, { kind: 'pedal' }>, e: EngineId, name: string): string {
  const W = pedalW(d), H = TH_PH, c = W / 2, g = 'thg' + (++uid);
  const foot = d.on ? `${led(c, 80)}<path class="ft-n" d="M${c - 9} 95l4.5-7.8h9l4.5 7.8-4.5 7.8h-9z"/><circle class="ft-c" cx="${c}" cy="95" r="5.8"/>` : `${rect('plate', c - 24, 87, 48, 13, 3)}${bar('eng', c - 15, 92, 30)}`;
  const screws = [[6, 6], [W - 6, 6], [6, H - 9], [W - 6, H - 9]].map(([x, y]) => `<circle class="screw" cx="${x}" cy="${y}" r="2.2"/>`).join('');
  const nw = Math.min(W - 30, 12 + name.length * 1.8);
  return `<defs><linearGradient id="${g}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" class="pf-1"/><stop offset="1" class="pf-2"/></linearGradient></defs><g class="ped">${rect('jack', -4, 13, W + 8, 10, 2)}${rect('p-lip', 0, 3, W, H - 3, 7)}<rect class="p-face" x="0" y="0" width="${W}" height="${H - 3}" rx="7" fill="url(#${g})"/>${screws}${bar('eng name', r1(c - nw / 2), 10, r1(nw), 3)}${fragment(fileOn(d, e))}${foot}</g>`;
}

/** The drawing the Modules panel shows for a definition: a module on its canvas, a pedal mounted on a piece of board,
 *  a side unit as its rack strip. `name` is its name as the page says it (for the engraved name bar's length). */
export function thumbOf(d: Definition, e: EngineId, name: string): string {
  /* one drawing per definition, engine and name: a redraw gives the same markup, so a card's drawing is never replaced
     under a press (a press that starts on a node a redraw removes is no click) */
  const k = `${d.id}|${e}|${name}`;
  let t = CACHE.get(k);
  if (t === undefined) { t = draw(d, e, name); CACHE.set(k, t); }
  return t;
}
const CACHE = new Map<string, string>();
function draw(d: Definition, e: EngineId, name: string): string {
  if (d.kind === 'pedal') {
    const W = pedalW(d), s = 0.745, w = W * s, x = (TH_W - w) / 2, y = (TH_H - TH_PH * s) / 2, jy = r1(y + 18 * s);
    const neutral = d.on ? '' : ' neutral';
    return svg(`${rect('board', 0, 0, TH_W, TH_H, 0)}<path class="cable" d="M-2 ${jy}H${r1(x - 3)}"/><path class="cable" d="M${r1(x + w + 3)} ${jy}H${TH_W + 2}"/><g transform="translate(${r1(x)} ${r1(y)}) scale(${s})">${pedal(d, e, name)}</g>`, `ped-th${neutral}`, undefined, `--h:${d.hue}`);
  }
  if (d.kind === 'unit') {
    const y = 32, h = 48, mid = y + h / 2;
    return svg(`${rect('board', 0, 0, TH_W, TH_H, 0)}${rect('u-body', TH_IN, y, TH_W - 2 * TH_IN, h, 4)}${rect('u-rail', TH_IN, y, 9, h, 4)}<circle class="screw" cx="${TH_IN + 4.5}" cy="${y + 8}" r="2"/><circle class="screw" cx="${TH_IN + 4.5}" cy="${y + h - 8}" r="2"/>${bar('eng', 30, mid - 1.5, 26)}${fragment(fileOn(d, e))}${rect('u-lip', TH_W - TH_IN - 14, y, 14, h, 4)}<path class="u-pow" d="M${TH_W - TH_IN - 7} ${mid - 9}v4M${TH_W - TH_IN - 10} ${mid - 6.5}a4.2 4.2 0 1 0 6 0"/>`, 'unit-th');
  }
  return svg(`${rect('frame', 0, 0, TH_W, TH_H, 0)}${fragment(fileOn(d, e))}`, 'mod-th');
}
