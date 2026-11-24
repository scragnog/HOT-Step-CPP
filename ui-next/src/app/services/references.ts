// The reference tracks the page has been given (the timbre reference, Mastering's), newest first: what the Timbre
// reference tile offers as recent references (the concept's S.refs, 25-module-timbre.js; the existing frontend keeps
// them on the server: ui/src/queries/global-models useReferenceTracks). Server-owned, so no step names them and Undo
// never takes one back: a file loaded on the desk is remembered (features/references.ts), and stays remembered when the
// load is undone. The simulated server's: services/mock/references.ts.

export interface ReferencesService {
  /** The recent reference files, newest first. */
  list(): readonly string[];
  /** A file the page was given: it comes first, once. */
  remember(name: string): void;
  /** Reports the list whenever it changes; returns the way to stop. */
  subscribe(f: () => void): () => void;
}
