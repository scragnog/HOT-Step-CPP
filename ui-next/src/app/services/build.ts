// The build session (Build by section, YuE2): the server's side of a build, as the real application keeps it in a
// session of its own (yue2CoverApi build*; the browser keeps the session's id). The page asks it to plan a score for
// the sections it gives, to make takes of one section, and to render a song from the kept takes; the session keeps the
// takes as they land and the section being looked at. Server-owned: no step names what it does and Undo never takes it
// back (an arrival is no step); what the user decides about the build (its keeps, discards, re-plans, renders) is the
// desk's record, through the board's own setters (src/derive/build.ts). The simulated server's: services/mock/build.ts.

import type { EngineId } from '../../definitions/types.ts';

export interface Take { id: string; ready: boolean; hue: number; dur: number; w: number; words: string; note: string }
export interface BuildSessionState { seed: number; takes: { [section: string]: Take[] }; cur?: string }
/** A song the session rendered, and whether the page has withdrawn it (its render undone). */
export interface BuiltSong { id: string; title: string; style: string; dur: number; hue: number; sections: number; withdrawn: boolean; engine: EngineId }
export interface BuildState { planning: boolean; sessions: { [id: string]: BuildSessionState }; songs: BuiltSong[] }

export interface BuildService {
  state(): BuildState;
  subscribe(f: () => void): () => void;
  /** Plans a score for these sections: a new session with its seed, once the planner has written it. A plan asked for
   *  again before the first lands replaces it. */
  plan(sections: { id: string; words: string }[]): Promise<{ session: string; seed: number }>;
  /** Makes `n` takes of a section, sung to its words: each is there at once, rendering, and lands in turn. */
  takes(session: string, section: string, words: string, n: number, ownChords: boolean): void;
  /** The section the page is looking at in a session. */
  look(session: string, section: string): void;
  /** Renders a song from the kept takes, in order, named and described as the page names it to the user. */
  render(session: string, song: { title: string; style: string; dur: number; takes: string[]; engine: EngineId }): Promise<{ song: string }>;
  /** A rendered song taken out of the Library (its render undone), or put back (redone). */
  withdraw(song: string, out: boolean): void;
}
