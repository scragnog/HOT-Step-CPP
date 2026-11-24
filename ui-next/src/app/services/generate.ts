// The generation service (docs/09, package 6): the songs the server holds, and the runs it renders. The page asks for a
// run as the core works it out (derive/generate.ts runOf: its intent and what it knows of each song), the server says
// what the songs will be (`draft`: a title from the words, a length where the engine decides it, the art's colour),
// and the page watches the queue: each run's place, how far each of its songs has got and what it is doing, and, when
// it ends, its songs or the server's refusal. Songs and runs are the server's: the page shows them, and Undo never takes
// back what the server has done. The simulated server's: services/mock/generate.ts.

import type { EngineId } from '../../definitions/types.ts';
import type { Msg, RunAsk, RunSpec, Slot } from '../../derive.ts';

/** A song the server holds, as Latest, the player and the song menu show it. */
export interface Song {
  id: string; title: string; take?: number; of?: number;
  style: string;
  /** The text it was made from (Reuse puts it back in the box, E-20). */
  text: string;
  /** Seconds. */
  dur: number;
  hue: number;
  engine: EngineId;
  createdAt: number;
  seed?: number;
}
/** One song of a run, as the server reports it while it renders. */
export interface JobSong { id: string; f: number; stage: Slot; ready: number }
/** A run in the queue, as the server reports it. */
export interface Job {
  id: string; engine: EngineId;
  status: 'queued' | 'running' | 'done' | 'refused' | 'failed' | 'cancelled';
  /** How many runs are ahead of it. */
  ahead: number;
  /** How far the run is, 0 to 1; each song's own progress, stage and how much of it can be played. */
  f: number; songs: JobSong[];
  /** Re-planning tries it has used (YuE2). */
  attempts: number;
  /** Its songs, once done. */
  result?: Song[];
  /** Why the server refused it, and how many takes would fit. */
  refused?: { why: Msg; more: Msg; retry: number };
  /** Why it stopped part-way (status failed), in the server's words. */
  failed?: { why: Slot };
}

export interface GenerateService {
  /** Every song the server holds, newest first. */
  songs(): Promise<Song[]>;
  /** What the songs of a run will be, as the server takes it: each one's title, length and colour, where the page did
   *  not set them. A run already drafted (one queued again) is given back as it is. */
  draft(run: RunAsk): RunSpec;
  /** Queues a run: the run as it was drafted, and the ids its songs will have. The request is the core's
   *  `createIntent(ix, spec.desk, real)` with the application's resolver and content builder (`RealApp`, what the
   *  request comparison runs it with), never one the adapter assembles; the simulated server sends none. */
  submit(spec: RunSpec, ids: string[]): Promise<{ jobId: string }>;
  /** Cancels some songs of a run (the run ends when none is left). */
  cancel(jobId: string, songIds: string[]): Promise<void>;
  /** The runs the server still has (after a reload, which of the page's own it kept). */
  live(): Promise<string[]>;
  /** Reports the queue whenever it moves; returns the way to stop. */
  watch(fn: (jobs: Job[]) => void): () => void;
}
