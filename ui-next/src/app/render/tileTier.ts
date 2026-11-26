// The tier each tile is drawn at, and what its models read there. A tile is drawn at the desk's tier, except while a
// tier change is staged (desk/stage.ts): a tile out of view is held at the tier it was drawn at, or (one the change
// brings) drawn as its shell alone, until its turn comes a moment later. Everything inside a tile asks its models at
// the tile's tier (TierScope, which useModel reads), so a held tile asks nothing again and is not drawn again when the
// desk's tier changes. Drawing only: the tier a tile is drawn at never changes a value or what the core says of one.

import { createContext, useSyncExternalStore } from 'react';
import type { Tier } from '../../definitions/types.ts';

/** A tile held by a staged tier change: at the tier it is drawn at, or `later`, its shell only (a tile the change
 *  brings). */
export type Hold = Tier | 'later';

const holds = new Map<string, Hold>(), subs = new Set<() => void>();
const subscribe = (f: () => void) => { subs.add(f); return () => { subs.delete(f); }; };
const tell = () => subs.forEach(f => f());

/** Holds tiles (by module id) until each is released. */
export function holdTiles(h: Map<string, Hold>): void { if (!h.size) return; h.forEach((v, k) => holds.set(k, v)); tell(); }
/** Draws a held tile at the desk's tier. */
export function releaseTile(id: string): void { if (holds.delete(id)) tell(); }
/** The tiles held now. */
export const heldTiles = (): string[] => [...holds.keys()];
/** What holds a tile (null: it is drawn at the desk's tier). */
export const useHold = (id: string): Hold | null => useSyncExternalStore(subscribe, () => holds.get(id) ?? null);

/** The models drawn in one tile at one tier: each view's last answer, whose `reads` say what it read of the store. */
export interface Scope { tier: Tier; models: Set<{ current: { reads: Set<string> } | null }> }
/** The scope of the tile a view is drawn in (none outside the tiles). */
export const TierScope = createContext<Scope | null>(null);

const scopes = new Map<string, Scope>();
/** The scope a tile is drawn in now (by module id). */
export const scopeOf = (id: string): Scope | undefined => scopes.get(id);
export const setScope = (id: string, s: Scope): void => { scopes.set(id, s); };
export const dropScope = (id: string, s: Scope): void => { if (scopes.get(id) === s) scopes.delete(id); };
/** Everything a tile's models read of the store, in reads.ts's names. */
export function readsOfTile(id: string): Set<string> | null {
  const s = scopes.get(id); if (!s) return null;
  const out = new Set<string>();
  s.models.forEach(r => r.current?.reads.forEach(x => out.add(x)));
  return out;
}
