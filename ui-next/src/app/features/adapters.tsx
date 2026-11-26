// The adapter lists (the concept's [data-cartadd] and [data-plannerpick] popovers, 25-module-adapters.js): the named
// actions loadAdapter and loadPlannerAdapter open, under the button that ran them, the installed adapters of the next
// slot's kind or the planner LM adapters, as a grid of cartridges the arrow keys move through (L-22). Choosing one
// loads it through the cartridges' own hand (derive/cartridges.ts: the slots' `load` step, or the planner adapter set)
// and puts focus on its fader or its ×  (L-07). And what the page is told about the adapters installed, from the
// adapters service (services/adapters.ts), once when the page loads.
//
// The list is drawn in the page's popover (features/pop.ts openPop, package 8): placed under its button, named for it,
// focus on its first file, closed by Escape, a press outside or Tab, focus back to the button.

import type { KeyboardEvent } from 'react';
import type { Definition, Node } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';
import { adapterLists, adaptersAway, roleKey, bindPart, instrumentModel, nodeAt, nodeKeys, partAct, type Desk } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { defineAction } from '../register/actions.ts';
import { openPop, closePop } from './pop.ts';
import { useModel } from '../render/useModel.ts';
import { CartArt } from '../parts/CartArt.tsx';
import { service } from '../register/services.ts';

/** The adapters loaded, by their role (definitions/roles.ts). */
const SLOTS = roleKey(ix, 'adapters') ?? '';

/** The definition and node that draw cartridges: the one whose face holds the part. */
function findCarts(): { def: Definition; node: Node; key: string } | null {
  for (const d of ix.defs) {
    const n = ('face' in d ? d.face ?? [] : []).find(x => x.part === 'cartridges');
    if (n) return { def: d, node: n, key: nodeKeys(d).get(n) ?? '' };
  }
  return null;
}
const CARTS = findCarts();
/** What the cartridges draw on a desk, their node as that desk shows it. */
const cartsOn = (desk: Desk) => { const n = CARTS ? nodeAt(ix, CARTS.node, desk) : null; return n && CARTS ? instrumentModel(ix, n, desk, CARTS.key)?.carts ?? null : null; };

/* the button a hand pressed last: the list opens under it (a click does not focus a button everywhere) */
let pressed: HTMLElement | null = null;
document.addEventListener('click', e => { pressed = (e.target as HTMLElement | null)?.closest?.('button') ?? null; }, true);
const toggle = (which: 'slot' | 'planner') => () => {
  const f = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
  const a = pressed?.isConnected ? pressed : f;
  if (a) openPop(a, () => <AdapterList which={which} />);
};
defineAction('loadAdapter', toggle('slot'));
defineAction('loadPlannerAdapter', toggle('planner'));

/** A hand on the cartridges, from here: bound on the desk in force, as the part's own would be. */
function act(name: string, arg: unknown): void {
  if (!CARTS) return;
  const desk = () => useDesk.getState().desk, commit = useDesk.getState().commit;
  try {
    const b = bindPart(ix, CARTS.def, CARTS.node, desk, c => commit(c, `${CARTS.def.id}:${CARTS.key}`), m => useDesk.getState().showToast({ text: m }));
    partAct(ix, b.node, b, desk(), name, arg, m => useDesk.getState().showToast({ text: m }));
  } catch { /* the cartridges are not on the desk any more */ }
}

/** The list in the popover: the next slot's files, or the planner LM adapters. */
export function AdapterList({ which }: { which: 'slot' | 'planner' }) {
  const say = useSay(), m = useModel(s => cartsOn(s.desk), `adapter-list|${which}`);
  const list = which === 'planner' ? m?.plannerList : m?.list;
  if (!list) return null;
  /* the list is a grid of choices: the arrow keys move through it, Home and End go to its ends (L-22) */
  const keys = (e: KeyboardEvent<HTMLDivElement>) => {
    const all = [...e.currentTarget.querySelectorAll<HTMLElement>('.cartpick')], i = all.indexOf(document.activeElement as HTMLElement);
    if (i < 0 || !all.length) return;
    const cols = getComputedStyle(e.currentTarget).gridTemplateColumns.split(' ').length;
    const j = ({ ArrowRight: i + 1, ArrowLeft: i - 1, ArrowDown: i + cols, ArrowUp: i - cols, Home: 0, End: all.length - 1 } as { [k: string]: number })[e.key];
    if (j === undefined) return;
    e.preventDefault(); all[Math.min(all.length - 1, Math.max(0, j))]!.focus();
  };
  const choose = (x: (typeof list.items)[number]) => {
    const before = useDesk.getState().desk;
    closePop(false);
    if (which === 'planner') {
      act('planner', x.value);
      useDesk.getState().setFocus(`[data-m="${CARTS!.def.id}"] [data-setv]`);
    } else {
      act('load', { file: x.file, type: x.type });
      /* focus on the fader of the cartridge just loaded (E-09, L-07) */
      const after = cartsOn(useDesk.getState().desk);
      const c = after?.carts.find(k => k.file === x.file), f = c?.faders[0]?.key;
      if (useDesk.getState().desk !== before) useDesk.getState().setFocus(f ? `[data-m="${CARTS!.def.id}"] [data-fader="${f}"]` : `[data-m="${CARTS!.def.id}"] [data-cartx]`);
    }
  };
  return (
    <>
      <h4>{say(list.head)}</h4>
      <div className="cartlist" onKeyDown={keys}>
        {list.items.map(x => (
          <button key={x.value} type="button" className={`cartpick t-${x.cls}`} {...(which === 'planner' ? { 'data-pickplanner': x.label } : { 'data-loadad': `${x.file}|${x.type}` })} title={say(x.title)} onClick={() => choose(x)}>
            <span className="mini-art"><CartArt a={x.art} /></span><b>{x.file}</b><em>{x.type}</em>
          </button>
        ))}
      </div>
      {list.note !== null ? <span className="lbl">{say(list.note)}</span> : null}
    </>
  );
}

/** The installed adapters of every engine, and the planner LM adapters, told to the page once it loads. */
async function tell(): Promise<void> {
  await Promise.all([
    service('adapters').planner().then(p => useDesk.getState().told({ lists: adapterLists('', [], p) })),
    ...ENGINES.map(e => service('adapters').slots(e.id).then(s => useDesk.getState().told({ lists: adapterLists(e.id, s) }))),
  ])
    .catch(() => { /* the lists stay as they were told; the real adapter says why it failed */ });
}
void tell();

/* the words for the adapters other engines keep (the desk fact adaptersAway), told whenever the engines' book or the
   engine changes (B-02) */
useDesk.subscribe((st, prev) => {
  if (st.book === prev.book && st.desk.engine === prev.desk.engine) return;
  st.told({ adaptersAway: adaptersAway(st.book, st.desk.engine, SLOTS) });
});
useDesk.getState().told({ adaptersAway: adaptersAway(useDesk.getState().book, useDesk.getState().desk.engine, SLOTS) });
