// Tests of the core's rule for Quick's fold (src/derive/fold.ts, deskModel's fold), against what the concept's
// renderDesk does (30-desk.js; its suite's 01-desk/quick-more-parked-empty): which modules fold, what the button says,
// what is never folded, and what the tier's other words say of folded modules. Two stand-in modules (the test build's,
// src/app/testing/standins.ts) make a desk the fold takes three of, as the concept's suite does.
//
//   node tools/test-fold.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import { ENGINE_IDS } from '../src/definitions/engines.ts';
import { FACTORY_WORKFLOWS } from '../src/definitions/factory.ts';
import type { EngineId, Tier, Value } from '../src/definitions/types.ts';
import { FOLD, deskModel, foldedOf, indexOf, neverFolded, onOf, type Desk } from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';
import { standIns } from '../src/app/testing/standins.ts';

const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const ix = indexOf([...DEFINITIONS, ...standIns()]);
const BOARD = DEFINITIONS.find(d => d.kind === 'module' && d.board)!.id;
const desk = (mods: string[], tier: Tier = 'quick', vals: Desk['vals'] = {}, engine: EngineId = 'ace', board: string[] = []): Desk =>
  ({ engine, tier, vals, auto: {}, state: {}, on: onOf(ix, mods, board) });
const model = (mods: string[], o: { tier?: Tier; vals?: { [k: string]: Value }; open?: boolean; engine?: EngineId } = {}) => deskModel(ix, desk(mods, o.tier, o.vals, o.engine, mods.includes(BOARD) ? ['lrc'] : []), mods, o.open);
const EIGHT = ['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes'];
const TEN = [...EIGHT, 'tx1', 'tx2', BOARD];

ok('the rule\'s numbers are the concept\'s: six shown, from nine tiles, three or more folded (A-28)', J(FOLD) === J({ shows: 6, from: 9, least: 3 }));
{
  const m = model(EIGHT);
  ok('eight modules show all eight, with no fold', m.tiles.length === 8 && m.fold === null, m.tiles);
  const b = model([...EIGHT, BOARD]);
  ok('eight modules and the Pedalboard all show: two would fold, less than the button costs (Q-05)', b.tiles.length === 9 && b.fold === null);
}
{
  const m = model(TEN);
  ok('ten modules and the board: Quick shows six and the Pedalboard', J(m.tiles) === J(['tempo', 'avoid', 'timbre', 'adapters', 'planner', 'sampler', BOARD]), m.tiles);
  ok('the button names the rest, in the desk\'s order', m.fold !== null && say(m.fold.says) === 'Show 4 more: Seed, Takes, Stand-in 1 and Stand-in 2', m.fold && say(m.fold.says));
  ok('the fold holds them, closed', J(m.fold?.ids) === J(['seed', 'takes', 'tx1', 'tx2']) && m.fold?.open === false);
  const o = model(TEN, { open: true });
  ok('open, every tile shows and the button folds them again', o.tiles.length === 11 && o.fold?.open === true && say(o.fold.says) === 'Show fewer modules', o.fold && say(o.fold.says));
  ok('open and closed are two make-ups of the desk (the layout chooses afresh, as the concept\'s deskMake)', o.make !== m.make);
  ok('the board is never folded, wherever it stands', J(model([BOARD, ...TEN.filter(x => x !== BOARD)]).fold?.ids) === J(['seed', 'takes', 'tx1', 'tx2']));
}
{
  const m = model(TEN, { vals: { 'takes.n': 3 } });
  ok('a module holding a changed value is never folded (Q-05)', m.tiles.includes('takes') && J(m.fold?.ids) === J(['seed', 'tx1', 'tx2']), m.fold?.ids);
  ok('and the button names only what it hides', say(m.fold!.says) === 'Show 3 more: Seed, Stand-in 1 and Stand-in 2');
  const two = model(TEN, { vals: { 'takes.n': 3, 'seed.lock': true } });
  ok('with two of them set, two would fold: nothing folds', two.fold === null && two.tiles.length === 11, two.fold?.ids);
}
{
  const src = DEFINITIONS.find(d => d.kind === 'module' && d.source)!;
  ok('a source module is never folded', neverFolded(ix, src as never, desk([src.id])));
  ok('a module at its defaults may fold', !neverFolded(ix, DEFINITIONS.find(d => d.id === 'seed') as never, desk(['seed'])));
}
{
  ok('Studio and Lab never fold', model(TEN, { tier: 'studio' }).fold === null && model(TEN, { tier: 'lab' }).fold === null);
  const late = model(['avoid', 'timbre', 'adapters', 'planner', 'sampler', 'seed', 'takes', 'tx1', 'tempo', BOARD]);
  ok('a folded module is named as its tile is at Quick', !!late.fold && say(late.fold.says) === 'Show 3 more: Takes, Stand-in 1 and Tempo & length', late.fold && say(late.fold.says));
}
{
  /* what the tier says of folded modules: they are on the desk and in force, so the note of modules hidden at this tier
     (Lab's) names none of them, and a Lab module, hidden at Quick, is no tile the fold counts */
  const lab = DEFINITIONS.find(d => d.kind === 'module' && d.labUnder && d.engines.includes('ace'))!;
  const m = model([...TEN, lab.id]);
  ok('a Lab module on the desk is not counted among Quick\'s tiles', J(m.fold?.ids) === J(['seed', 'takes', 'tx1', 'tx2']));
  ok('the hidden note names the Lab module alone, never a folded one', !!m.hidden && !/Seed|Takes|Stand-in/.test(say(m.hidden)) && say(m.hidden).includes(say({ id: `create.${lab.id}.name`, en: lab.name })), m.hidden && say(m.hidden));
  ok('with nothing hidden by the tier, the fold says nothing there', model(TEN).hidden === null);
  const parked = model([...TEN.slice(0, 7), 'tx1', 'tx2', BOARD], { engine: 'yue2' });
  ok('modules the engine does not use wait parked and are not counted', parked.fold === null && parked.parked !== null, parked.tiles);
}
{
  /* the factory desks: no engine has enough modules Quick shows for the fold to take three, so none folds */
  const folds: string[] = [];
  for (const w of FACTORY_WORKFLOWS) for (const e of ENGINE_IDS) {
    const mods = [...w.modules, ...(w.board.length && !w.modules.includes(BOARD) ? [BOARD] : [])];
    const d = { ...desk(mods, 'quick', {}, e, w.board) };
    const live = deskModel(ix, d, mods, true).tiles;
    if (foldedOf(ix, d, live).length) folds.push(`${w.name} on ${e}`);
  }
  ok('no factory desk folds at Quick on any engine', !folds.length, folds);
}

if (fails.length) { console.log(`fold: ${fails.length} of ${n} failed`); fails.forEach(f => console.log('  ' + f)); process.exit(1); }
console.log(`fold: ${n} checks of Quick's fold: ok`);
