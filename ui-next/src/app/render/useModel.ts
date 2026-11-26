// A view's subscription to the desk: it asks the core for its model (plain data) when the store changes something the
// model read (reads.ts), and redraws only when the model is not what it was. So a knob that turns asks again only the
// models that read its value, and redraws that knob and whatever shows its value, and nothing else (docs/09,
// "Rendering"). `key` names what the model depends on besides the store (a definition, a node, a surface): a new key
// asks afresh.
//
// Inside a tile, the model is asked at the tile's tier (render/tileTier.ts): the store's desk is seen with the tile's
// tier, and reading it notes nothing, so a tile held at its tier during a staged tier change asks nothing again. Each
// view's last answer is kept in its tile's scope, so the layout can tell later whether what the tile drew at a tier can
// have changed since (what its models read).

import { useCallback, useContext, useEffect, useRef, useSyncExternalStore } from 'react';
import type { Tier } from '../../definitions/types.ts';
import { useDesk, type DeskStore } from '../store/desk.ts';
import { readsOf, touched, type Reads } from './reads.ts';
import { TierScope, type Scope } from './tileTier.ts';
import { tally } from '../tally.ts';

interface Last<T> { s: DeskStore; key: string; json: string; v: T; reads: Reads }

/** The store as a view drawn at `tier` sees it: the desk's tier is that one, and reading it notes nothing. (The desk
 *  stands in front of an empty object, not of the store's own stand-in: a proxy's answer is checked against its
 *  target's own property, which the stand-in would note as read.) */
function atTier(s: DeskStore, tier: Tier): DeskStore {
  const d = s.desk as unknown as Record<string | symbol, unknown>, own = { value: tier, writable: true, enumerable: true, configurable: true };
  const desk = new Proxy({}, {
    get: (_, k) => (k === 'tier' ? tier : d[k]),
    has: (_, k) => k === 'tier' || k in d,
    ownKeys: () => Reflect.ownKeys(d),
    getOwnPropertyDescriptor: (_, k) => (k === 'tier' ? own : Reflect.getOwnPropertyDescriptor(d, k)),
  });
  return new Proxy(s, { get: (t, k, r) => (k === 'desk' ? desk : Reflect.get(t, k, r)) });
}

/** The view's model, asked again only when what it read changes. `scope`: the tile it is drawn in, for the tile's own
 *  models (a view inside the tile has it from TierScope). */
export function useModel<T>(fn: (s: DeskStore) => T, key: string, scope?: Scope): T {
  const last = useRef<Last<T> | null>(null), inTile = useContext(TierScope), sc = scope ?? inTile, tier = sc?.tier;
  const k = tier ? `${key}@${tier}` : key;
  useEffect(() => { if (!sc) return; sc.models.add(last); return () => { sc.models.delete(last); }; }, [sc]);
  const get = useCallback(() => {
    const s = useDesk.getState(), l = last.current;
    if (l && l.key === k) {
      if (l.s === s) return l.v;
      if (!touched(l.reads, l.s, s)) { l.s = s; return l.v; }
    }
    tally('model');
    const { v, reads } = readsOf(tier ? x => fn(atTier(x, tier)) : fn, s), json = JSON.stringify(v);
    if (l && l.key === k && l.json === json) { last.current = { ...l, s, reads }; return l.v; }
    tally('model.changed');
    last.current = { s, key: k, json, v, reads };
    return v;
  }, [k]); // eslint-disable-line react-hooks/exhaustive-deps -- fn is the same function of the store for one key
  return useSyncExternalStore(useDesk.subscribe, get);
}
