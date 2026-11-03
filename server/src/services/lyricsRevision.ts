/**
 * lyricsRevision.ts — the version counter on `.lyrics.json`.
 *
 * Four things write that file: the editor's save, a Whisper retranscription, a
 * crop, and the API-only shift/split routes. Before this existed, an editor that
 * had been open for ten minutes would happily autosave its pre-crop timeline
 * over a cropped file, and a second window would do the same to the first.
 *
 * A writer bumps `revision`; the editor sends the revision it loaded and the
 * save is refused if the file moved on. `editedByHand` is a separate marker:
 * it says a person placed these words, so a retranscription must ask first.
 */

export function bumpRevision(data: any): void {
  if (!data || typeof data !== 'object') return;
  data.revision = revisionOf(data) + 1;
  data.editedAt = new Date().toISOString();
}

export function revisionOf(data: any): number {
  const value = Number(data?.revision);
  return Number.isFinite(value) && value > 0 ? value : 0;
}
