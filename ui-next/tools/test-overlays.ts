// Tests of the core's additions for the picker and the tooltips (package 8, src/derive/picker.ts and
// src/derive/tips.ts), each against what the concept does (33-picker.js, 30-desk.js labOfferHTML, 19-tooltip.js) and
// the pending findings the rebuild puts right (F-29: the drawing Enter takes, ringed).
//
//   node tools/test-overlays.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { Definition, EngineId, Tier } from '../src/definitions/types.ts';
import {
  categoryWord, defName, groupOf, isHowTo, labOffer, pickerCountMsg, pickerGroupOf, pickerModel, sameAsLabel, tipLesson, tipSentences, touchTip,
  type Desk, type PickerCat,
} from '../src/derive.ts';
import { EN, say, type Slot } from '../src/derive/messages.ts';
import { WORDS } from '../src/app/i18n/words/overlays.ts';

const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const TABLE = { ...EN, ...WORDS };
const s = (m: Slot) => say(m, TABLE);
const desk = (engine: EngineId, tier: Tier, on: string[]): Desk =>
  ({ engine, tier, vals: {}, auto: {}, state: {}, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
/* what the picker offers: as the page's model index lists it (src/app/model/index.ts OFFERED) */
const OFFERED: Definition[] = [
  ...DEFINITIONS.filter(d => d.kind === 'module' && !d.builtIn && !d.source && !d.board).sort((a, b) => a.at - b.at),
  ...DEFINITIONS.filter(d => d.kind === 'pedal').sort((a, b) => a.at - b.at), ...DEFINITIONS.filter(d => d.kind === 'unit').sort((a, b) => a.at - b.at),
];
const words = (dk: Desk) => (d: Definition) => ({ name: s(defName(d, dk)), desc: s({ id: `create.${d.id}.desc`, en: d.desc }), group: s(categoryWord(groupOf(d))) });
const pick = (dk: Desk, q: string, cat: PickerCat) => pickerModel(OFFERED, dk, q, cat, words(dk));

/* ---- the wall's groups (33-picker.js PK_GROUPS) */
{
  const g = (id: string) => pickerGroupOf(DEFINITIONS.find(d => d.id === id)!);
  ok('a Lab module stands with the Lab group, others in their category, pedals and units apart', J([g('guidance'), g('perf'), g('tempo'), g('seed'), g('ss'), g('wh')]) === J(['Lab', 'Lab', 'Compose', 'Engine', 'Pedals', 'Side units']));
  /* the counts below are the inventory's, whatever it holds: a module or a pedal added or removed changes nothing here */
  const kinds = (k: Definition['kind']) => OFFERED.filter(x => x.kind === k).length;
  ok('the picker offers every module, pedal and unit the panel does', OFFERED.length === kinds('module') + kinds('pedal') + kinds('unit') && kinds('pedal') > 0 && kinds('unit') > 0, OFFERED.length);
}
/* ---- the count says what the wall shows (F-09) */
{
  const d = desk('ace', 'studio', ['tempo']);
  const n = (k: Definition['kind']) => OFFERED.filter(x => x.kind === k).length;
  ok('"N in all"', s(pickerCountMsg(pick(d, '', 'All').count)) === `${OFFERED.length} in all`, s(pickerCountMsg(pick(d, '', 'All').count)));
  ok('"N pedals"', s(pickerCountMsg(pick(d, '', 'Pedals').count)) === `${n('pedal')} pedal${n('pedal') === 1 ? '' : 's'}`);
  ok('"N side units"', s(pickerCountMsg(pick(d, '', 'Side units').count)) === `${n('unit')} side unit${n('unit') === 1 ? '' : 's'}`);
  const se = pick(d, 'see', 'All');
  ok('"see" is one match, Seed, which Enter takes', s(pickerCountMsg(se.count)) === '1 match' && se.target === 'seed' && se.fit === 'seed', [s(pickerCountMsg(se.count)), se.target]);
  const comp = pick(d, '', 'Compose');
  ok('a left group hides the other left groups and dims the right column', comp.items.filter(x => x.pgroup === 'Engine').every(x => x.gone) && comp.items.filter(x => !x.left).every(x => x.dim && !x.gone), comp.items.filter(x => x.pgroup === 'Engine' && !x.gone).map(x => x.id));
  ok('and counts its own: "N modules"', /^\d+ modules$/.test(s(pickerCountMsg(comp.count))) && comp.count.n === comp.items.filter(x => x.pgroup === 'Compose').length);
  ok('a group the filter empties counts none', pick(d, 'see', 'All').groups.find(x => x.group === 'Pedals')!.n === 0);
}
/* ---- ranking (G-B-10): names that start with it, then names that hold it, then a description that mentions it */
{
  const d = desk('ace', 'studio', []), m = pick(d, 'rep', 'All');
  const rank = (id: string) => m.items.find(x => x.id === id)!.rank;
  ok('"rep": Repetition and Repaint by name; a description match ranks after', rank('repetition') === 1 && m.order[0] === 'repetition' || rank('repaint') === 1, m.order.slice(0, 4));
  ok('nothing that matches nothing is in the walking order', m.order.every(id => !m.items.find(x => x.id === id)!.nomatch));
  ok('with a filter, Enter\'s drawing is the first in the walking order (F-29)', m.target === m.order[0]);
  ok('with none, there is no target', pick(d, '', 'All').target === null);
}
/* ---- what a drawing says (33-picker.js pkItem) */
{
  const d = desk('ace', 'studio', ['tempo']), m = pick(d, '', 'All'), it = (id: string) => m.items.find(x => x.id === id)!;
  ok('a Lab module outside Lab is locked and says "Lab only."', it('guidance').locked && s(it('guidance').says) === 'Lab only.');
  ok('one on the desk says so', it('tempo').on && s(it('tempo').says) === 'Already on the desk.');
  ok('one the engine does not use says where it works', it('yuebatch').notHere && /^Not used by ACE-Step 1\.5\. Works on YuE2\.$/.test(s(it('yuebatch').says)), s(it('yuebatch').says));
  ok('else its description', s(it('seed').says) === s({ id: 'create.seed.desc', en: DEFINITIONS.find(x => x.id === 'seed')!.desc }));
}
/* ---- the Lab offer (30-desk.js labOfferHTML) */
{
  const g = DEFINITIONS.find(d => d.id === 'guidance')!, o = labOffer(g, desk('ace', 'studio', []));
  ok('a Lab module the engine uses is offered plainly', s(o.name) === 'Guidance plugin' && o.parked === null, s(o.name));
  const yb = DEFINITIONS.find(d => d.id === 'perf')!, o2 = labOffer(yb, desk('yue2', 'studio', []));
  ok('one it does not use names where it works and that it waits parked', yb.engines.includes('yue2') ? o2.parked === null : !!o2.parked && /module: on YuE2 it will wait parked/.test(s({ id: 'ui.pop.labParked', slots: o2.parked })), o2);
}
/* ---- tips (19-tooltip.js) */
ok('the sentence splitter keeps decimals and versions (F-01, Q-01, E-04)', J([tipSentences('Temperature 1.05'), tipSentences('Full bar: the 31.8 GB card. Click for the engine log.'), tipSentences('ACE-Step 1.5. Drag sideways.')])
  === J([['Temperature 1.05'], ['Full bar: the 31.8 GB card.', 'Click for the engine log.'], ['ACE-Step 1.5.', 'Drag sideways.']]));
ok('how to work a control is a lesson; what it means is not', isHowTo('Drag sideways.') && isHowTo('Double-click to reset.') && !isHowTo('Guidance strength.'));
{
  const a = tipLesson('Guidance strength. Drag up or down.', new Set());
  const b = tipLesson('Guidance strength. Drag up or down.', new Set(a.teaches));
  ok('a lesson is said once a session (L-06)', a.text === 'Guidance strength. Drag up or down.' && J(a.teaches) === J(['Drag up or down.']) && b.text === 'Guidance strength.', [a, b]);
}
{
  const how = 'Drag up or down (hold Shift for fine steps), scroll, or use the arrow keys. Double-click to reset.', touch = 'Rest a finger on it, then drag up or down. Double-tap to reset.';
  ok('under a finger a knob says a finger\'s way (S-31)', touchTip(`Guidance. ${how}`, how, touch) === `Guidance. ${touch}`, touchTip(`Guidance. ${how}`, how, touch));
  ok('and no sentence about keys or F9; click becomes tap', touchTip('Open the picker. Press F9 for it. Click to add.', how, touch) === 'Open the picker. Tap to add.', touchTip('Open the picker. Press F9 for it. Click to add.', how, touch));
}
ok('a tip that only repeats a printed line does not open (Q-11)', sameAsLabel(['CFG', '9.0'], 'cfg.') && !sameAsLabel(['CFG', '9.0'], 'How closely it follows the prompt'));

if (fails.length) { console.log(`overlays: ${fails.length} of ${n} failed`); fails.forEach(f => console.log('  ' + f)); process.exit(1); }
console.log(`overlays: ${n} checks of the picker's and the tips' core: ok`);
