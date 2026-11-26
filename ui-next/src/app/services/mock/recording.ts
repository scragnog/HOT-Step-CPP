// A recording's length (Continue from recording): what the page is told of a file it was given, so the part to
// continue from is chosen inside it. Server-owned (the real application reads it from the uploaded file), so it is no
// step and Undo never takes it back.
//
// This is the simulated server: a file is two to four minutes long, the same length every time it is given (the
// concept's 120 to 240 s), unless a scene says otherwise. The real adapter is package 10's.

import type { RecordingService } from '../recording.ts';

const told = new Map<string, number>(), subs = new Set<() => void>();
const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7);

export const recordings: RecordingService & { tell(file: string, dur: number): void } = {
  /** The length of a file, in seconds, once known. */
  length(file: string): number { if (!told.has(file)) told.set(file, 120 + Math.abs(hash(file)) % 121); return told.get(file)!; },
  /** Every length known so far. */
  known(): { [file: string]: number } { return Object.fromEntries(told); },
  /** A length as a scene sets it (the test layer). */
  tell(file: string, dur: number): void { told.set(file, dur); subs.forEach(f => f()); },
  subscribe(f: () => void): () => void { subs.add(f); return () => subs.delete(f); },
};
