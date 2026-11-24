// The build session (Build by section, YuE2): the server's side of a build, as the real application keeps it in a
// session of its own (yue2CoverApi build*; the browser keeps the session's id). The page asks it to plan a score for
// the sections it gives, to make takes of one section, and to render a song from the kept takes; the session keeps the
// takes as they land and the section being looked at. Server-owned: no step names what it does and Undo never takes it
// back (an arrival is no step); what the user decides about the build (its keeps, discards, re-plans, renders) is the
// desk's record, through the board's own setters (src/derive/build.ts).
//
// This is the simulated server, the concept's timings: a score lands in 0.7 s, takes one by one from 0.7 s, 0.6 s
// apart. It keeps its sessions under its own key, so a build survives a reload as the server's would; a take still
// rendering when the page closed is lost, as the concept's. The real adapter is package 10's.

import type { EngineId } from '../../../definitions/types.ts';
import type { BuildService, BuildSessionState as Session, BuildState, Take } from '../build.ts';


const KEY = 'hotstep.create.buildSessions', PLAN_MS = 700, TAKE_MS = 700, TAKE_GAP = 600;
const rid0 = () => Math.random().toString(36).slice(2, 8);
/** An id no session, take or song holds yet (the same random draws can come again after a reload). */
const rid = (): string => { const used = JSON.stringify(state); let id = rid0(); while (used.includes(`"${id}"`)) id = rid0(); return id; };
const read = (): BuildState => {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? 'null') as BuildState | null;
    if (!v || typeof v !== 'object' || !v.sessions) return { planning: false, sessions: {}, songs: [] };
    /* a take still rendering when the page closed is lost */
    for (const s of Object.values(v.sessions)) for (const k of Object.keys(s.takes ?? {})) s.takes[k] = (s.takes[k] ?? []).filter(t => t.ready);
    return { planning: false, sessions: v.sessions, songs: Array.isArray(v.songs) ? v.songs : [] };
  } catch { return { planning: false, sessions: {}, songs: [] }; }
};
let state: BuildState = { planning: false, sessions: {}, songs: [] };
state = read();
const subs = new Set<() => void>();
const put = (next: BuildState) => {
  state = next;
  try { localStorage.setItem(KEY, JSON.stringify({ sessions: state.sessions, songs: state.songs })); } catch { /* storage refused: kept for this visit */ }
  subs.forEach(f => f());
};
const withSession = (id: string, fn: (s: Session) => Session) => { const s = state.sessions[id]; if (s) put({ ...state, sessions: { ...state.sessions, [id]: fn(s) } }); };
let planRun = 0;

export const buildSession: BuildService & { replace(next: BuildState): void } = {
  state: (): BuildState => state,
  subscribe(f: () => void): () => void { subs.add(f); return () => subs.delete(f); },
  /** Plans a score for these sections: a new session with its seed, once the planner has written it. A plan asked for
   *  again before the first lands replaces it. */
  plan(sections: { id: string; words: string }[]): Promise<{ session: string; seed: number }> {
    const run = ++planRun;
    put({ ...state, planning: true });
    return new Promise(done => setTimeout(() => {
      if (run !== planRun) return;
      const id = rid(), seed = Math.floor(Math.random() * 2147483647);
      put({ ...state, planning: false, sessions: { ...state.sessions, [id]: { seed, takes: {} } } });
      void sections;
      done({ session: id, seed });
    }, PLAN_MS));
  },
  /** Makes `n` takes of a section, sung to its words: each is there at once, rendering, and lands in turn. In "new
   *  chords" mode a take now and then follows the score anyway, and says so. */
  takes(session: string, section: string, words: string, n: number, ownChords: boolean): void {
    const fresh: Take[] = Array.from({ length: n }, () => ({ id: rid(), ready: false, hue: Math.round(Math.random() * 360), dur: 16 + Math.round(Math.random() * 14), w: Math.random() * 1e9 | 0, words, note: ownChords && Math.random() < 0.12 ? 'followed the score' : '' }));
    withSession(session, s => ({ ...s, takes: { ...s.takes, [section]: [...(s.takes[section] ?? []), ...fresh] } }));
    fresh.forEach((t, i) => setTimeout(() => withSession(session, s => ({ ...s, takes: { ...s.takes, [section]: (s.takes[section] ?? []).map(x => (x.id === t.id ? { ...x, ready: true } : x)) } })), TAKE_MS + i * TAKE_GAP));
  },
  /** The section the page is looking at in a session. */
  look(session: string, section: string): void { if (state.sessions[session]?.cur !== section) withSession(session, s => ({ ...s, cur: section })); },
  /** Renders a song from the kept takes, in order. */
  render(session: string, song: { title: string; style: string; dur: number; takes: string[]; engine: EngineId }): Promise<{ song: string }> {
    const id = 'b' + rid(), n = song.takes.length;
    return new Promise(done => setTimeout(() => {
      if (!state.sessions[session]) return;
      put({ ...state, songs: [{ id, title: song.title, style: song.style, dur: song.dur, hue: Math.round(Math.random() * 360), sections: n, withdrawn: false, engine: song.engine }, ...state.songs].slice(0, 64) });
      done({ song: id });
    }, 0));
  },
  /** A rendered song taken out of the Library (its render undone), or put back (redone). */
  withdraw(song: string, out: boolean): void { if (state.songs.some(s => s.id === song && s.withdrawn !== out)) put({ ...state, songs: state.songs.map(s => (s.id === song ? { ...s, withdrawn: out } : s)) }); },
  /** The whole state (the test layer's scenes). */
  replace(next: BuildState): void { planRun++; put(next); },
};
