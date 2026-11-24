// Generation's state (docs/09, package 6): the songs Latest shows (the server's, newest first, with the runs still
// rendering and the runs the server refused in among them, where they were started), the queue as the page started
// it, the player, and the few things the view keeps open (the song menu, the folded queue). All of it is the
// server's or the view's: none is undoable desk state. Cancelling is the one step here Undo covers, as a step that
// undoes itself (store/desk.ts recordStep): Undo puts the runs back in the queue as they were started (F-13).

import { create } from 'zustand';
import { cancelBack, cancelWords, lostSays, type Msg, type RunSpec, type Slot } from '../../derive.ts';
import { service } from '../register/services.ts';
import type { GenerateService, Job, Song } from '../services/generate.ts';
import '../services/generate.ts';
import { useDesk } from './desk.ts';

/** A card in Latest: a song, a song still rendering (`pending`), or a run the server refused (`error`). */
export interface Card extends Omit<Song, 'createdAt'> {
  pending?: { run: string; f: number; stage: Slot | null; ready: number; stream: boolean };
  error?: { why: Msg; more: Msg; retry: number; n: number; spec: RunSpec };
  /** It has just finished (it resolves in its card), and it has not been played yet (C-45). */
  fresh?: boolean; unheard?: boolean;
}
/** A run as the page started it: what it renders with (its engine and settings, kept with it), the ids of its songs. */
export interface Run { id: string; jobId: string | null; spec: RunSpec; ids: string[]; /** How far it has got, while it renders. */ f?: number }

export interface GenStore {
  cards: Card[]; runs: Run[];
  /** The page's next id for a run or a song it starts (the concept's S.nextId). */
  nextId: number;
  playing: string | null; pos: number; paused: boolean;
  qOpen: boolean; menuFor: string | null;
  /** How far the head of the queue got: Generate's sweep stays there once the queue is empty. */
  lastF: number | null;
  /** The last song played: the player keeps drawing it while it folds away. */
  shown: string | null;
}

const gen = () => service<GenerateService>('generate');
const LIVE_KEY = 'hotstep.create.rendering';

export const useGen = create<GenStore>()(() => ({ cards: [], runs: [], nextId: 100, playing: null, pos: 0, paused: false, qOpen: false, menuFor: null, lastF: null, shown: null }));
const set = (p: Partial<GenStore> | ((s: GenStore) => Partial<GenStore>)) => useGen.setState(p);
const get = () => useGen.getState();

/* ---------------------------------------------------------------- what the view is told */

type Ev = { kind: 'done'; run: Run; ids: string[] } | { kind: 'refused'; run: Run; id: string } | { kind: 'ready'; id: string };
const heard = new Set<(e: Ev) => void>();
/** The view hears a run end (to say it when its card is out of sight) and a song become playable while it renders. */
export const onRun = (fn: (e: Ev) => void): (() => void) => { heard.add(fn); return () => { heard.delete(fn); }; };
const tell = (e: Ev) => { for (const f of heard) f(e); };

/* ---------------------------------------------------------------- starting and cancelling */

/** Starts a run: its cards go to the front of Latest at once, and the run to the back of the queue. */
export function startRun(spec: RunSpec): Run {
  const s = get(), id = String(s.nextId), ids = spec.songs.map((_, i) => String(s.nextId + 1 + i)), run: Run = { id, jobId: null, spec, ids };
  const cards: Card[] = spec.songs.map((x, i) => ({ id: ids[i]!, title: x.title, ...(x.take ? { take: x.take, of: x.of! } : {}), style: x.style, text: x.text, dur: x.dur, hue: x.hue, engine: spec.engine, ...(x.seed !== undefined ? { seed: x.seed } : {}), pending: { run: id, f: 0, stage: null, ready: 0, stream: spec.stream } }));
  set({ nextId: s.nextId + 1 + ids.length, runs: [...s.runs, run], cards: [...cards, ...s.cards] });
  void gen().submit(spec, ids).then(r => {
    const now = get().runs.find(x => x.id === id);
    if (!now) { void gen().cancel(r.jobId, ids); return; }
    set(st => ({ runs: st.runs.map(x => (x.id === id ? { ...x, jobId: r.jobId } : x)) }));
    const left = ids.filter(i => !get().cards.some(c => c.id === i && c.pending)); if (left.length) void gen().cancel(r.jobId, left);
  });
  return run;
}

/** Cancels songs still rendering or waiting: each run keeps what is left of it. Returns each run as it was started,
 *  holding only the songs cancelled, and whether the one rendering was among them. */
export function cancelCards(ids: Set<string>): { jobs: RunSpec[]; restarts: boolean } {
  const s = get(), head = s.runs[0], jobs: RunSpec[] = []; let restarts = false;
  if (s.playing && ids.has(s.playing)) stopPlayer();
  const runs: Run[] = [];
  for (const r of s.runs) {
    const gone = r.ids.filter(i => ids.has(i) && s.cards.some(c => c.id === i && c.pending));
    if (!gone.length) { runs.push(r); continue; }
    if (r === head) restarts = true;
    jobs.push({ ...r.spec, n: gone.length, songs: gone.map(i => r.spec.songs[r.ids.indexOf(i)]!) });
    if (r.jobId) void gen().cancel(r.jobId, gone);
    const left = r.ids.filter(i => !ids.has(i) && s.cards.some(c => c.id === i && c.pending));
    if (left.length) runs.push(r);
  }
  set({ runs, cards: get().cards.filter(c => !(c.pending && ids.has(c.id))) });
  return { jobs, restarts };
}

/** Cancel as one undo step (F-13): Undo queues the cancelled songs again with the settings they started with (the one
 *  that was rendering starts again, and says so); Redo cancels them again. */
export function cancelStep(ids: Set<string>, one: Card | null): void {
  let { jobs, restarts } = cancelCards(ids), back: Run[] = [];
  const n = jobs.reduce((a, j) => a + j.songs.length, 0); if (!n) return;
  const w = cancelWords(n, one);
  const entry = useDesk.getState().recordStep({ label: w.label, put: w.put }, {
    undo: () => { const said = cancelBack(n, restarts); back = jobs.map(startRun); return said; },
    redo: () => { const again = new Set(back.flatMap(r => r.ids)); ({ jobs, restarts } = cancelCards(again)); },
  });
  useDesk.getState().showToast({ text: w.says, action: { label: { id: 'ui.undo' }, run: () => useDesk.getState().runUndo(entry) } });
}

export const dismissCard = (id: string): void => set(s => ({ cards: s.cards.filter(c => c.id !== id) }));
export const toggleFold = (): void => set(s => ({ qOpen: !s.qOpen }));
export const setMenu = (id: string | null): void => set({ menuFor: id });

/* ---------------------------------------------------------------- the queue as the server reports it */

function heardJobs(jobs: Job[]): void {
  for (const j of jobs) {
    const run = get().runs.find(r => r.jobId === j.id); if (!run) continue;
    if (j.status === 'done') {
      const res = j.result ?? [], byId = new Map(res.map(x => [x.id, x]));
      set(s => ({ runs: s.runs.filter(r => r !== run), cards: s.cards.map(c => { const r = byId.get(c.id); if (r) { const { createdAt, ...song } = r; void createdAt; return { ...song, fresh: true, unheard: true }; } return c.unheard ? { ...c, unheard: false } : c; }) }));
      tell({ kind: 'done', run, ids: res.map(x => x.id) });
    } else if (j.status === 'refused' && j.refused) {
      const first = run.ids.find(i => get().cards.some(c => c.id === i && c.pending))!, n = run.ids.length, rf = j.refused;
      set(s => {
        const e0 = s.cards.find(c => c.id === first)!, title = n > 1 ? `${e0.title} · ${n} takes` : e0.title;
        const err: Card = { ...e0, title, take: 0, of: 0, error: { why: rf.why, more: rf.more, retry: rf.retry, n, spec: run.spec } };
        delete err.pending;
        /* a refusal never stacks a second identical notice (F-21) */
        const same = (c: Card) => !!c.error && c.title === title && JSON.stringify(c.error.why) === JSON.stringify(rf.why);
        return { runs: s.runs.filter(r => r !== run), cards: s.cards.filter(c => !same(c) && !(run.ids.includes(c.id) && c.id !== first)).map(c => (c.id === first ? err : c)) };
      });
      tell({ kind: 'refused', run, id: first });
    } else if (j.status === 'cancelled') set(s => ({ runs: s.runs.filter(r => r !== run) }));
    else {
      const by = new Map(j.songs.map(x => [x.id, x]));
      if (j.f !== run.f) set(s => ({ lastF: s.runs[0] === run ? j.f : s.lastF, runs: s.runs.map(r => (r === run ? { ...r, f: j.f } : r)) }));
      set(s => ({ cards: s.cards.map(c => { const x = c.pending && by.get(c.id); if (!x) return c; if (x.ready > 0 && !c.pending!.ready) queueMicrotask(() => tell({ kind: 'ready', id: c.id })); return { ...c, pending: { ...c.pending!, f: x.f, stage: x.stage, ready: x.ready } }; }) }));
      if (j.engine === 'yue2' && j.attempts) tellDesk({ attempts: j.attempts });
    }
  }
}

/* the desk is told what the cover swatch and the dice show: the newest card's colour, the tries used */
let told: { hue?: number; attempts?: number } = {};
function tellDesk(o: { hue?: number; attempts?: number }): void {
  if (Object.entries(o).every(([k, v]) => told[k as keyof typeof told] === v)) return;
  told = { ...told, ...o }; useDesk.getState().told({ songs: o });
}
useGen.subscribe((s, p) => { if (s.cards !== p.cards && s.cards[0]) tellDesk({ hue: s.cards[0].hue }); });

/* ---------------------------------------------------------------- what is rendering, kept across a reload */

/* the songs rendering are written down as they change, so the next visit can say which the server did not keep (F-27) */
useGen.subscribe((s, p) => {
  if (s.runs === p.runs && s.cards === p.cards) return;
  const live = s.runs.filter(r => r.jobId).map(r => ({ job: r.jobId!, n: r.ids.filter(i => s.cards.some(c => c.id === i && c.pending)).length }));
  try { if (live.length) localStorage.setItem(LIVE_KEY, JSON.stringify(live)); else localStorage.removeItem(LIVE_KEY); } catch { /* not kept */ }
});

/** Loads the server's songs and starts listening to its queue; says what a reload lost. */
export async function bootGenerate(): Promise<void> {
  gen().watch(heardJobs);
  const [songs, live] = await Promise.all([gen().songs(), gen().live()]);
  const max = songs.reduce((a, x) => (/^\d+$/.test(x.id) ? Math.max(a, +x.id) : a), 99);
  set(s => ({ cards: [...s.cards.filter(c => c.pending || c.error), ...songs.map(({ createdAt, ...x }) => { void createdAt; return x; })], nextId: Math.max(s.nextId, max + 1) }));
  let was: { job: string; n: number }[] = [];
  try { const a = JSON.parse(localStorage.getItem(LIVE_KEY) ?? '[]'); if (Array.isArray(a)) was = a.filter(x => x && typeof x.job === 'string' && typeof x.n === 'number'); } catch { was = []; }
  const lost = was.filter(x => !live.includes(x.job)).reduce((a, x) => a + x.n, 0);
  try { localStorage.removeItem(LIVE_KEY); } catch { /* not kept */ }
  if (lost) useDesk.getState().showToast({ text: lostSays(lost) });
}

/* ---------------------------------------------------------------- the player */

let tick = 0;
/** Plays a song from its start, or stops it when it is the one playing. */
export function togglePlay(id: string): void {
  if (get().playing === id) { stopPlayer(); return; }
  clearInterval(tick);
  set(s => ({ playing: id, shown: id, pos: 0, paused: false, cards: s.cards.map(c => (c.id === id && c.unheard ? { ...c, unheard: false } : c)) }));
  /* at the end it stops, and rests paused at the start (C-27); a song still rendering plays only as far as it is
     rendered, and waits there for more (E-40) */
  tick = window.setInterval(() => {
    const s = get(), c = s.cards.find(x => x.id === s.playing); if (!c || s.paused) return;
    if (c.pending && s.pos + 1 > Math.floor(c.dur * c.pending.ready)) return;
    const pos = Math.min(c.dur, s.pos + 1);
    set(pos >= c.dur ? { pos: 0, paused: true } : { pos });
  }, 1000);
}
export function stopPlayer(): void { clearInterval(tick); tick = 0; set({ playing: null, pos: 0, paused: false }); }
export function pausePlayer(): void { const s = get(), c = s.cards.find(x => x.id === s.playing); if (!c) return; set({ paused: !s.paused, ...(s.pos >= c.dur ? { pos: 0 } : {}) }); }
export function seekPlayer(pos: number): void { set({ pos }); }
/** The card being played. */
export const playingCard = (s: GenStore): Card | undefined => s.cards.find(c => c.id === s.playing);
