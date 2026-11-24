// The module picker (F9, the concept's 33-picker.js): the wall of every module, pedal and side unit, its groups, its
// category bar, its filter and the count that names what it shows. What each snapshot is (locked, on the desk, not
// used here, a Lab module), how it matches the filter, what the bar hides or dims and what the count says are decided
// here, from the definitions and the desk; the view draws them (src/app/features/picker.tsx).

import type { Definition } from '../definitions/types.ts';
import type { Desk } from './core.ts';
import { cardModel, type CardModel } from './models.ts';
import { defDesc } from './steps.ts';
import { tileName, usedHere } from './desk.ts';
import { engineName } from '../definitions/engines.ts';
import { list, msg, type Msg, type Slot } from './messages.ts';

/** The picker's groups: the modules in two columns' worth of groups (Compose, Engine, and the Lab modules apart), and
 *  the board's (pedals in chain order, side units). */
export const PICKER_LEFT = ['Compose', 'Engine', 'Lab'] as const;
export const PICKER_RIGHT = ['Pedals', 'Side units'] as const;
export type PickerGroup = (typeof PICKER_LEFT)[number] | (typeof PICKER_RIGHT)[number];
/** The category bar: the left column's groups, All, the right column's. */
export const PICKER_CATS = [...PICKER_LEFT, 'All', ...PICKER_RIGHT] as const;
export type PickerCat = (typeof PICKER_CATS)[number];

/** Where a definition stands on the wall: a Lab module in Lab, any other module in its category, a pedal or a side
 *  unit in its own group. */
export const pickerGroupOf = (d: Definition): PickerGroup =>
  d.kind === 'pedal' ? 'Pedals' : d.kind === 'unit' ? 'Side units' : d.labUnder ? 'Lab' : d.category === 'Compose' ? 'Compose' : 'Engine';

export interface PickerItem extends CardModel {
  pgroup: PickerGroup;
  /** In the left column (a module) or the right (a pedal or side unit). */
  left: boolean;
  /** A pedal: it stands free on the shelf at its unit width. */
  pedal: boolean;
  /** Hidden by the category bar (a module of another left group while a left group is chosen; the same on the right). */
  gone: boolean;
  /** Dimmed by the category bar: the other column, while a group of one column is chosen. */
  dim: boolean;
  /** The filter matches nothing of it. */
  nomatch: boolean;
  /** What it is, said after its name (its aria-label and the info line): not used here, Lab only, already on the desk, or
   *  its description. */
  says: Slot;
}
export interface PickerModel {
  items: PickerItem[];
  /** Each group with how many of its snapshots the filter and the bar leave (dimmed ones count); a group with none is
   *  not drawn. */
  groups: { group: PickerGroup; n: number }[];
  /** What the count says: how many matches, or how many the bar shows, by what it shows. */
  count: { kind: 'match' | 'all' | 'pedals' | 'units' | 'modules'; n: number };
  /** The snapshots shown, in the order the keys walk them: by rank while a filter is typed (the first is what Enter
   *  adds), else as drawn. */
  order: string[];
  /** What Enter takes from the filter: the first shown; null with no filter. */
  target: string | null;
  /** What the hint names as what Enter adds: the first shown that is not dimmed and not locked; null with no filter. */
  fit: string | null;
}

/** The picker: every offered definition as a snapshot, filtered by `query` (matched against `words`, the name,
 *  description and group as the page shows them, as the Modules panel matches them) and by the category `cat`. */
export function pickerModel(defs: readonly Definition[], desk: Desk, query: string, cat: PickerCat, words: (d: Definition) => { name: string; desc: string; group: string }): PickerModel {
  const q = query.trim(), leftOn = (PICKER_LEFT as readonly string[]).includes(cat), rightOn = (PICKER_RIGHT as readonly string[]).includes(cat);
  const items = defs.map((d): PickerItem => {
    const c = cardModel(d, desk, q, words(d)), pgroup = pickerGroupOf(d), left = d.kind === 'module';
    const inCat = pgroup === cat;
    const gone = (left && leftOn && !inCat) || (!left && rightOn && !inCat), dim = cat !== 'All' && (left ? rightOn : leftOn);
    const says: Slot = c.notHere ? msg('card.notUsed', { engine: engineName(desk.engine), engines: list(d.engines.map(engineName)) }) : c.locked ? msg('ui.picker.lab') : c.on ? msg('ui.picker.on') : defDesc(d, desk);
    return { ...c, rank: q ? c.rank : 0, pgroup, left, pedal: d.kind === 'pedal', gone, dim, nomatch: !!q && !c.rank, says };
  });
  const shown = items.filter(x => !x.gone && !x.nomatch);
  const groups = [...PICKER_LEFT, ...PICKER_RIGHT].map(group => ({ group, n: shown.filter(x => x.pgroup === group).length }));
  const n = shown.filter(x => !x.dim).length;
  const kind = q ? 'match' : cat === 'All' ? 'all' : cat === 'Pedals' ? 'pedals' : cat === 'Side units' ? 'units' : 'modules';
  /* the walking order: in rank order while a filter is typed (a rank above 1 is drawn after the rest of its row) */
  const order = [...PICKER_LEFT, ...PICKER_RIGHT].flatMap(g => shown.filter(x => x.pgroup === g)).sort((a, b) => (a.rank > 1 ? a.rank : 0) - (b.rank > 1 ? b.rank : 0)).map(x => x.id);
  const target = q ? order[0] ?? null : null;
  const fit = q ? order.map(id => items.find(x => x.id === id)!).find(x => !x.dim && !x.locked)?.id ?? null : null;
  return { items, groups, count: { kind, n }, order, target, fit };
}

/** The words of the count, by its kind (ui.picker.count.*). */
export const pickerCountMsg = (c: PickerModel['count']): Msg => msg(`ui.picker.count.${c.kind}`, { n: c.n });

/** The Lab offer (30-desk.js labOfferHTML): a Lab module added outside Lab names itself, and, when the engine does not
 *  use it, the engines it works on and that it would wait parked here. */
export const labOffer = (d: Definition, desk: Desk): { name: Msg; parked: { engines: Slot; engine: string } | null } =>
  ({ name: tileName(d, desk), parked: usedHere(d, desk.engine) ? null : { engines: list(d.engines.map(engineName)), engine: engineName(desk.engine) } });
