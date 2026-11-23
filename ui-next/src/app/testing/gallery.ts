// The parts gallery (test build only): definitions that put every part the page has built on every surface a
// definition can put it on (a tile's body at each tier, a row, a rack, a well, an LCD, a tile's foot, a labelled
// pairing, a pedal, a side unit), so tools/visual-baseline/gallery.mjs can draw each one and check it looks right
// there, in both themes and at every tier. They are indexed with the real definitions in the test build only
// (model/index.ts) and offered nowhere; their ids start with "zz-g".

import type { EngineId, ModuleDef, Node, ParamDef, PedalDef, UnitDef } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';
import { PARTS, type PartSpec } from '../../definitions/parts.ts';

const all = (): EngineId[] => ENGINES.map(e => e.id);
const params = (p: string): ParamDef[] => [
  { key: `${p}.n1`, type: 'number', label: 'Amount', short: 'Amt', range: [0, 10, 0.5], default: 5, unit: 'dB', wire: { none: 'gallery' } },
  { key: `${p}.n2`, type: 'number', label: 'Depth', range: [0, 100, 1], default: 30, unit: '%', wire: { none: 'gallery' } },
  { key: `${p}.c`, type: 'choice', label: 'Mode', default: 'soft', options: [{ value: 'soft', label: 'Soft', lcd: 'soft' }, { value: 'hard', label: 'Hard', lcd: 'hard' }, { value: 'wide', label: 'Wide', lcd: 'wide' }], wire: { none: 'gallery' } },
  { key: `${p}.b`, type: 'boolean', label: 'Boost', default: false, wire: { none: 'gallery' } },
  { key: `${p}.t`, type: 'text', label: 'Name', default: '', placeholder: 'a name', wire: { none: 'gallery' } },
] as ParamDef[];
/** Every part of the inventory that a definition can place with a value or a read (controls and read-outs), bound to the
 *  gallery's parameters (p their prefix): a part a package adds to the inventory and builds is in the gallery with no
 *  line here; one not built yet draws as its seam. The few that need more than a value (a choice row's choices, a
 *  button's action, an XY pad's pair, a meter's scale) get it below. */
const parts = (p: string): Node[] => {
  const of: Record<string, string> = { number: `${p}.n1`, choice: `${p}.c`, boolean: `${p}.b`, text: `${p}.t` };
  return (Object.entries(PARTS) as [string, PartSpec][]).flatMap(([name, spec]): Node[] => {
    if (spec.kind === 'container' || spec.kind === 'bespoke') return [];
    const extra: Record<string, unknown> = name === 'choiceRow' ? { param: `${p}.c`, label: 'Feel', choices: [['Soft', 'soft'], ['Hard', 'hard'], ['Wide', 'wide']] }
      : name === 'meter' ? { reads: [`${p}.n1`], opts: { min: 0, max: 10, red: 8 } }
      : name === 'textArea' ? { opts: { look: 'note' } } : name === 'powerButton' ? { label: 'Power' } : {};
    if (spec.edits?.includes('pair')) return [{ part: name, params: [`${p}.n1`, `${p}.n2`], ...extra } as Node];
    /* a part with variants (a pick) is drawn once in each */
    const variants = (spec.opts as { variant?: { type: unknown } } | undefined)?.variant?.type;
    if (Array.isArray(variants) && spec.edits?.includes('choice')) return variants.map(variant => ({ part: name, param: of.choice!, opts: { variant } }) as Node);
    const t = spec.edits?.find(e => e in of);
    if (t) return [{ part: name, param: of[t]!, ...extra } as Node];
    if (name === 'button' || name === 'link') return [{ part: name, actions: [{ do: 'set', param: `${p}.b`, value: true, label: 'Boost it' }] } as Node];
    if (spec.reads) return [{ part: name, reads: [`${p}.n1`], label: 'Amount', ...extra } as Node];
    return [{ part: name, text: 'A quiet line of words', ...extra } as Node];
  }).concat([{ part: 'labelled', label: 'Paired', children: [{ part: 'segmented', param: `${p}.c` }] } as Node]);
};
/* the containers a definition can put a part in; the tile's own body is the definition's face itself */
export const GALLERY_SURFACES = ['tile', 'row', 'rack', 'well', 'lcd', 'foot', 'labelled'] as const;
const module = (c: (typeof GALLERY_SURFACES)[number], i: number): ModuleDef => {
  const p = `zzg.${c}`, kids = parts(p);
  return {
    kind: 'module', id: `zz-g-${c}`, name: `Gallery: ${c}`, desc: 'Every part.', at: 900 + i, category: 'Compose', engines: all(), owner: 'song', stored: 'settings',
    size: { w: 4, quickW: 4 }, params: params(p),
    face: c === 'tile' ? kids : c === 'labelled' ? kids.filter(n => n.part !== 'labelled').map((n, j) => ({ part: 'labelled', label: `Label ${j + 1}`, children: [n] }) as Node) : [{ part: c, children: kids } as Node],
  };
};
/** The gallery's definitions: a function, so that no other build keeps any of it. */
export const gallery = (): (ModuleDef | PedalDef | UnitDef)[] => [
  ...GALLERY_SURFACES.map(module),
  { kind: 'pedal', id: 'zz-g-pedal', name: 'Gallery pedal', desc: 'Every part on a pedal.', at: 95, engines: all(), hue: 200, on: 'zzg.pedal.b', main: 'zzg.pedal.n1', owner: 'song', stored: 'settings',
    params: params('zzg.pedal'), face: parts('zzg.pedal').map(n => ({ ...n, tiers: ['studio', 'lab'] }) as Node) },
  { kind: 'unit', id: 'zz-g-unit', name: 'Gallery unit', desc: 'Every part on a side unit.', at: 95, engines: all(), on: 'zzg.unit.b', owner: 'song', stored: 'settings',
    params: params('zzg.unit'), face: parts('zzg.unit') },
];
