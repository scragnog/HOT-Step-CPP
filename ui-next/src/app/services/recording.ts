// A recording's length (Continue from recording): what the page is told of a file it was given, so the part to
// continue from is chosen inside it. Server-owned (the real application reads it from the uploaded file), so it is no
// step and Undo never takes it back. The simulated server's: services/mock/recording.ts.

export interface RecordingService {
  /** The length of a file, in seconds, once known. */
  length(file: string): number;
  /** Every length known so far. */
  known(): { [file: string]: number };
  /** Reports a length learnt; returns the way to stop. */
  subscribe(f: () => void): () => void;
}
