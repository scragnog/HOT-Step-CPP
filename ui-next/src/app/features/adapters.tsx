// The adapter lists (the concept's [data-cartadd] and [data-plannerpick] popovers, 25-module-adapters.js): the named
// actions loadAdapter and loadPlannerAdapter open, under the button that ran them, the installed adapters of the next
// slot's kind or the planner LM adapters, as a grid of cartridges the arrow keys move through (L-22). Choosing one
// loads it through the cartridges' own hand (derive/cartridges.ts: the slots' `load` step, or the planner adapter set)
// and puts focus on its fader or its ×  (L-07). And what the page is told about the adapters installed, from the
// adapters service (services/adapters.ts), once when the page loads.
//
// The list is drawn into the concept's #pop. Package 8 builds the page's popover; until then this is the only thing
// that opens #pop, and it shows, places and hides it itself (the concept's openPop and placeFloat, 17-popover.js).

import { useEffect, useLayoutEffect, useSyncExternalStore } from 'react';
import { createPortal } from 'react-dom';
import type { Definition, Node } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';
import { adapterLists, bindPart, instrumentModel, nodeKeys, partAct } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { defineAction } from '../register/actions.ts';
import { fill } from '../register/slots.ts';
import { useModel } from '../render/useModel.ts';
import { CartArt } from '../parts/CartArt.tsx';
import { adapters } from '../services/adapters.ts';

/** The definition and node that draw cartridges: the one whose face holds the part. */
function findCarts(): { def: Definition; node: Node; key: string } | null {
  for (const d of ix.defs) {
    const n = ('face' in d ? d.face ?? [] : []).find(x => x.part === 'cartridges');
    if (n) return { def: d, node: n, key: nodeKeys(d).get(n) ?? '' };
  }
  return null;
}
const CARTS = findCarts();

/* which list is open, and the button it opened from */
let open: { which: 'slot' | 'planner'; anchor: HTMLElement | null } | null = null;
const subs = new Set<() => void>();
const setOpen = (o: typeof open) => { open = o; subs.forEach(f => f()); };
/** Closes the open list; focus goes back to its button. */
function close(refocus = true) { const a = open?.anchor; a?.setAttribute('aria-expanded', 'false'); setOpen(null); if (refocus && a?.isConnected) a.focus({ preventScroll: true }); }
const useOpen = () => useSyncExternalStore(f => { subs.add(f); return () => subs.delete(f); }, () => open);
const toggle = (which: 'slot' | 'planner') => () => {
  const a = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
  setOpen(open && open.which === which && open.anchor === a ? null : { which, anchor: a });
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

export function AdapterList() {
  const say = useSay(), o = useOpen(), pop = document.getElementById('pop');
  const m = useModel(s => (CARTS && o ? instrumentModel(ix, CARTS.node, s.desk, CARTS.key)?.carts ?? null : null), `adapter-list|${o?.which ?? ''}`);
  const list = o?.which === 'planner' ? m?.plannerList : m?.list;
  /* shown under its button (above it when there is more room there), within the window; focus on its first file */
  useLayoutEffect(() => {
    const pop = document.getElementById('pop'); if (!pop) return;
    pop.hidden = !list; if (!list || !o) { pop.removeAttribute('aria-label'); return; }
    pop.setAttribute('aria-label', say(list.head));
    const a = o.anchor?.isConnected ? o.anchor.getBoundingClientRect() : null, vw = document.documentElement.clientWidth, w = pop.offsetWidth, h = pop.offsetHeight, gap = 6, edge = 12;
    if (a) {
      const below = innerHeight - edge - a.bottom - gap, above = a.top - gap - edge, up = h > below && above > below;
      pop.style.top = `${Math.round((up ? a.top - gap - h : a.bottom + gap) + scrollY)}px`;
      pop.style.left = `${Math.round(Math.min(Math.max(edge, a.left), Math.max(edge, vw - w - edge)) + scrollX)}px`; pop.style.right = 'auto';
      pop.classList.toggle('up', up);
    }
    o.anchor?.setAttribute('aria-expanded', 'true');
    pop.querySelector<HTMLElement>('.cartpick')?.focus({ preventScroll: true });
  }, [list, o, say]);
  /* a press outside it, or on another button, closes it */
  useEffect(() => {
    const pop = document.getElementById('pop'); if (!o) return;
    const out = (e: PointerEvent) => { const t = e.target as HTMLElement | null; if (t && !t.closest('#pop, #toast') && !(o.anchor && o.anchor.contains(t))) close(false); };
    /* the list is a grid of choices: the arrow keys move through it, Home and End go to its ends; Escape closes it (L-22) */
    const keys = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); return; }
      const all = [...(pop?.querySelectorAll<HTMLElement>('.cartlist > .cartpick') ?? [])], i = all.indexOf(document.activeElement as HTMLElement);
      if (i < 0) return;
      const cols = getComputedStyle(all[0]!.parentElement!).gridTemplateColumns.split(' ').length;
      const j = ({ ArrowRight: i + 1, ArrowLeft: i - 1, ArrowDown: i + cols, ArrowUp: i - cols, Home: 0, End: all.length - 1 } as { [k: string]: number })[e.key];
      if (j === undefined) return;
      e.preventDefault(); all[Math.min(all.length - 1, Math.max(0, j))]!.focus();
    };
    document.addEventListener('pointerdown', out, true); pop?.addEventListener('keydown', keys);
    return () => { document.removeEventListener('pointerdown', out, true); pop?.removeEventListener('keydown', keys); };
  }, [o]);
  if (!pop || !o || !list) return null;
  const choose = (x: (typeof list.items)[number]) => {
    const which = o.which, before = useDesk.getState().desk;
    close(false);
    if (which === 'planner') {
      act('planner', x.value);
      useDesk.getState().setFocus(`[data-m="${CARTS!.def.id}"] [data-setv]`);
    } else {
      act('load', { file: x.file, type: x.type });
      /* focus on the fader of the cartridge just loaded (E-09, L-07) */
      const after = CARTS ? instrumentModel(ix, CARTS.node, useDesk.getState().desk, CARTS.key)?.carts : null;
      const c = after?.carts.find(k => k.file === x.file), f = c?.faders[0]?.key;
      if (useDesk.getState().desk !== before) useDesk.getState().setFocus(f ? `[data-m="${CARTS!.def.id}"] [data-fader="${f}"]` : `[data-m="${CARTS!.def.id}"] [data-cartx]`);
    }
  };
  return createPortal(
    <>
      <h4>{say(list.head)}</h4>
      <div className="cartlist">
        {list.items.map(x => (
          <button key={x.value} type="button" className={`cartpick t-${x.cls}`} {...(o.which === 'planner' ? { 'data-pickplanner': x.label } : { 'data-loadad': `${x.file}|${x.type}` })} title={say(x.title)} onClick={() => choose(x)}>
            <span className="mini-art"><CartArt a={x.art} /></span><b>{x.file}</b><em>{x.type}</em>
          </button>
        ))}
      </div>
      {list.note !== null ? <span className="lbl">{say(list.note)}</span> : null}
    </>,
    pop,
  );
}
fill('overlays', AdapterList);

/** The installed adapters of every engine, and the planner LM adapters, told to the page once it loads. */
async function tell(): Promise<void> {
  await Promise.all([
    adapters.planner().then(p => useDesk.getState().told({ lists: adapterLists('', [], p) })),
    ...ENGINES.map(e => adapters.slots(e.id).then(s => useDesk.getState().told({ lists: adapterLists(e.id, s) }))),
  ])
    .catch(() => { /* the lists stay as they were told; the real adapter says why it failed */ });
}
void tell();
