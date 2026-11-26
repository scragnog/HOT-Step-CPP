// Draws a pedal's thumbnail middle from its definition, in the concept's thumbnail kit (src/js/24-thumbnails.js and
// 27-pedalboard.js: thKnob, knobRow, sel, key): a knob per number at its default, an LCD selector per choice, a key
// per switch other than the foot switch. The frame (body in its hue, screws, name, LED and foot switch) is drawn
// from the definition by the shell, so the file holds only this middle.
//
//   node tools/thumb.ts <pedal id>            writes src/definitions/thumbs/pd-<id>.svg (not over an existing file)
//   node tools/thumb.ts <pedal id> --print    prints it instead
//
// It is a starting drawing: the concept's own pedals were composed by hand from the same pieces, and a pedal whose
// look is not its knobs (the VST3 chain's LCD, Mastering's meter) is better drawn by hand from these.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import { pedalUnits } from '../src/definitions/parts.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const id = process.argv[2];
const d = DEFINITIONS.find(x => x.kind === 'pedal' && x.id === id);
if (!d || d.kind !== 'pedal') { console.log(`usage: node tools/thumb.ts <pedal id>; pedals: ${DEFINITIONS.filter(x => x.kind === 'pedal').map(x => x.id).join(', ')}`); process.exit(1); }

const r1 = (n: number) => Math.round(n * 10) / 10;
const pt = (cx: number, cy: number, r: number, p: number) => { const a = (-135 + p * 270) * Math.PI / 180; return [r1(cx + r * Math.sin(a)), r1(cy - r * Math.cos(a))]; };
const arc = (cx: number, cy: number, r: number, p0: number, p1: number) => { const [x0, y0] = pt(cx, cy, r, p0), [x1, y1] = pt(cx, cy, r, p1); return `M${x0} ${y0}A${r} ${r} 0 ${(p1 - p0) * 270 > 180 ? 1 : 0} 1 ${x1} ${y1}`; };
const bar = (cls: string, x: number, y: number, w: number, h = 3) => `<rect class="${cls}" x="${r1(x)}" y="${r1(y)}" width="${r1(w)}" height="${h}" rx="${h / 2}"/>`;
function knob(cx: number, cy: number, r: number, p: number): string {
  const [px, py] = pt(cx, cy, r - 3.2, p);
  return [`<g class="k pk">`, `<path class="k-t" d="${arc(cx, cy, r, 0, 1)}"/>`, ...(p > 0 ? [`<path class="k-a" d="${arc(cx, cy, r, 0, p)}"/>`] : []),
    `<circle class="k-c" cx="${cx}" cy="${cy}" r="${r1(r - 2.6)}"/>`, ...(r >= 9 ? [`<circle class="k-r" cx="${cx}" cy="${cy}" r="${r1(r - 3.4)}"/>`] : []),
    `<path class="k-p" d="M${cx} ${cy}L${px} ${py}"/>`, `</g>`].join('\n');
}

const numbers = d.params.filter(p => p.type === 'number'), choices = d.params.filter(p => p.type === 'choice'), switches = d.params.filter(p => p.type === 'boolean' && p.key !== d.on);
const W = pedalUnits(numbers.length) === 2 ? 184 : 80, c = W / 2;
const at = (p: (typeof numbers)[number]) => { const [lo, hi] = p.range!; return hi > lo ? Math.max(0, Math.min(1, ((p.default as number) - lo) / (hi - lo))) : 0; };
const parts: string[] = [];
let y = numbers.length > 1 ? 34 : 40;
for (let i = 0; i < numbers.length; i += 5) {
  const row = numbers.slice(i, i + 5), r = row.length > 1 || W > 80 ? 9 : 12, step = W > 80 ? 32 : 30, x0 = c - (row.length - 1) * step / 2;
  row.forEach((p, j) => { const x = r1(x0 + j * step); parts.push(knob(x, y, r, at(p)), bar('eng', x - 7, y + r + 4, 14, 2.5)); });
  y += r * 2 + 12;
}
for (const _ of choices) { const w = W - 20; parts.push(`<rect class="lcd" x="10" y="${y}" width="${w}" height="12" rx="3"/>`, bar('lcd-t', 10 + w * 0.22, y + 4.5, w * 0.56)); y += 16; }
if (switches.length) { const kw = W > 80 ? 40 : 34; switches.slice(0, W > 80 ? 2 : 1).forEach((_, i) => { const x = 14 + i * (kw + 22); parts.push(`<rect class="pkey" x="${x}" y="${Math.min(y, 68)}" width="7" height="7" rx="1.5"/>`, bar('eng', x + 11, Math.min(y, 68) + 2.25, kw)); }); }
const svg = [`<!-- Thumbnail pd-${d.id}: inner drawing in the thumbnail kit's classes, on the pedal canvas. Drawn by tools/thumb.ts from the definition. -->`, ...parts].join('\n') + '\n';
if (process.argv.includes('--print')) { process.stdout.write(svg); process.exit(0); }
const file = path.join(ROOT, 'src', 'definitions', 'thumbs', `pd-${d.id}.svg`);
if (fs.existsSync(file)) { console.log(`thumbs/pd-${d.id}.svg exists; --print shows the drawing without writing it`); process.exit(1); }
fs.writeFileSync(file, svg.replace(/\n/g, '\r\n'));
console.log(`wrote thumbs/pd-${d.id}.svg (${numbers.length} knobs, ${choices.length} selectors, ${switches.length} keys); add thumb: { file: 'pd-${d.id}' } to the pedal`);
