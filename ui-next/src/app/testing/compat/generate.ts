// What generation's scenarios need of the concept's globals beyond the layer's core (testing/compat.ts): the songs
// Latest holds in the concept's shape (S.songs: id, title, take, of, style, text, dur, hue, bk, pending, batch, error,
// retryN, stream, ready, fresh, unheard), the queue (S.batches), S.playing, S.nextId, S.menuFor, S.qOpen (the repaint's
// source song, S.source, is the instruments' layer's), the player (PL, paintPlayer, togglePlay), and the
// concept's generate, cancelAll, cancelSongs and renderSongs. And `mockCard`, which makes the simulated server's card
// smaller than the page's figures, so a scenario can see a refusal the server makes.

import { flushSync } from 'react-dom';
import { mmss } from '../../../derive.ts';
import { useDesk } from '../../store/desk.ts';
import { CONCEPT_ID } from '../../shell/engines.ts';
import { cancelCards, seekPlayer, togglePlay as toggle, useGen, type Card } from '../../store/generate.ts';
import { cancelAll as cancelAllNow, generate as press } from '../../generate/actions.ts';
import { mockCard, mockFail } from '../../services/generate.ts';
import { expose, onScene, S } from '../compat.ts';

const num = (id: string | number): string | number => (typeof id === 'string' && /^\d+$/.test(id) ? +id : id);
const str = (id: unknown): string => String(id);
const concept = (c: Card) => ({
  id: num(c.id), title: c.title, ...(c.of ? { take: c.take, of: c.of } : {}), style: c.style, text: c.text, dur: c.dur, hue: c.hue, bk: CONCEPT_ID[c.engine],
  ...(c.pending ? { pending: true, batch: num(c.pending.run), ...(c.pending.stream ? { stream: true, ready: c.pending.ready } : {}) } : {}),
  ...(c.error ? { error: c.error.why, retryN: c.error.retry } : {}), ...(c.fresh ? { fresh: true } : {}), ...(c.unheard ? { unheard: true } : {}), ...(c.seed !== undefined ? { seed: c.seed } : {}),
});
const step = <T,>(fn: () => T): T => { let out!: T; flushSync(() => { out = fn(); }); return out; };

export function install(): void {
  Object.defineProperties(S, {
    songs: { configurable: true, enumerable: true, get: () => useGen.getState().cards.map(concept), set: (v: { id: unknown }[]) => { const keep = new Set(v.map(x => str(x.id))); useGen.setState(s => ({ cards: s.cards.filter(c => keep.has(c.id)) })); } },
    batches: { configurable: true, enumerable: true, get: () => useGen.getState().runs.map(r => ({ id: num(r.id), songs: useGen.getState().cards.filter(c => c.pending?.run === r.id).map(concept) })) },
    playing: { configurable: true, enumerable: true, get: () => { const p = useGen.getState().playing; return p === null ? null : num(p); } },
    nextId: { configurable: true, enumerable: true, get: () => useGen.getState().nextId },
    menuFor: { configurable: true, enumerable: true, get: () => { const m = useGen.getState().menuFor; return m === null ? null : num(m); } },
    qOpen: { configurable: true, enumerable: true, get: () => useGen.getState().qOpen, set: (v: boolean) => useGen.setState({ qOpen: !!v }) },
  });
  /* a scene that names the repaint's song by its id (S.source = 1, or { song: 1 }) gets the song's record, as the song
     menu writes it, before the instruments' layer applies it */
  onScene(sc => {
    const v = sc.source as { song?: unknown; dur?: unknown } | number | string | null | undefined, id = typeof v === 'object' && v ? (v.dur === undefined ? v.song : undefined) : v;
    const c = id === undefined || id === null ? undefined : useGen.getState().cards.find(x => x.id === str(id) && !x.pending && !x.error);
    if (c) sc.source = { song: num(c.id), title: c.title, style: c.style, dur: c.dur, hue: c.hue };
  });
  const PL = {
    get pos() { return useGen.getState().pos; }, set pos(v: number) { seekPlayer(v); },
    get paused() { return useGen.getState().paused; }, set paused(v: boolean) { useGen.setState({ paused: !!v }); },
    get s() { const g = useGen.getState(), c = g.cards.find(x => x.id === g.playing); return c ? concept(c) : null; },
  };
  expose({
    PL,
    paintPlayer() { flushSync(() => useGen.setState(s => ({ ...s }))); },
    renderSongs() { flushSync(() => useGen.setState(s => ({ ...s }))); },
    togglePlay(id: unknown) { step(() => toggle(str(id))); },
    generate(o?: { seed?: number }) { return step(() => press(o ?? {})); },
    cancelAll() { step(() => cancelAllNow()); },
    cancelSongs(ids: Set<unknown>) { return step(() => cancelCards(new Set([...ids].map(str)))); },
    mockCard(gb?: number) { mockCard.gb = gb; },
    mockFail() { mockFail.next = true; },
    /* the concept's helpers the repaint's scenarios read: a time as m:ss, and the song a repaint works on */
    mmss,
    curSong() { const src = useDesk.getState().desk.state.source as { song?: unknown } | undefined, c = src ? useGen.getState().cards.find(x => x.id === str(src.song)) : undefined; return c ? concept(c) : null; },
  });
}
