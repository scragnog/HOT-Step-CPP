/**
 * captionForBackend.ts — pick the caption the ACTIVE backend was trained on.
 *
 * A Lyric Studio generation carries two captions, and they are NOT two renderings
 * of the same text:
 *
 *   caption      ACE-Step 1.5 — 2-4 sentences of flowing description.
 *   caption_mm3  MiniMax-Music3 — a three-heading Structured Caption with thirteen
 *                fixed labels (Global Metadata / Vocal Details / Arrangement).
 *
 * Handing one model the other's caption is a measured quality loss, not a
 * cosmetic mismatch: in a controlled A/B (2026-08-14, one track, 5 seeds per arm,
 * no adapter) a rich ACE-style caption fed to MM3 produced the right genre in 1
 * take of 5, while the same content in MM3's format was on-genre throughout.
 * See server/src/services/lireek/prompts.ts for the full write-up.
 *
 * Fallback is deliberate and one-way: an MM3 run with no MM3 caption gets the
 * ACE caption, because degraded conditioning beats no conditioning — every
 * generation written before this field existed is in that state. The reverse
 * never happens; an ACE run is never handed a Structured Caption.
 *
 * On the MM3 side `caption_mm3` is no longer automatically the one that renders.
 * A written song's MM3 caption has a SOURCE — by default one of the artist's own
 * training tracks, picked by tempo — because reusing a training caption verbatim
 * is what reliably lands in the band's style and reaches a natural ending. See
 * utils/mm3CaptionSource.ts. Pass the album's `lyricsSetId` so that choice can be
 * resolved; without one this degrades to the song's own caption, i.e. the
 * behaviour above.
 *
 * YuE2 has the same choice for the same reason — its AR adapters are trained on
 * whole songs under their own captions, so a training caption is an
 * in-distribution prompt — but no caption column of its own: there is one
 * caption and the choice is which dataset track, if any, replaces it. That
 * choice belongs to the training DATASET, resolved from `lyricsSetId` the same
 * way as MM3's; see utils/yue2CaptionSource.ts.
 */

import { resolveMm3CaptionForGeneration } from './mm3CaptionSource';
import { resolveYue2CaptionForGeneration, YUE2_BACKEND_ID } from './yue2CaptionSource';

/** The registered id of the MiniMax-Music3 backend (server/src/services/backends/registry.ts). */
export const MM3_BACKEND_ID = 'minimax-m3';

export function captionForBackend(
  gen: { id?: number; bpm?: number; caption?: string | null; caption_mm3?: string | null; caption_yue2?: string | null },
  backendId: string | undefined,
  lyricsSetId?: number,
): string {
  if (backendId === MM3_BACKEND_ID) {
    const resolved = resolveMm3CaptionForGeneration(gen, lyricsSetId);
    if (resolved.caption.trim()) return resolved.caption;
  }
  if (backendId === YUE2_BACKEND_ID) {
    // A song that carries its own YuE2 caption (2026-09-20) is prompted with
    // it: that sentence is the shape the planner's training captions take, so
    // it no longer needs to borrow a dataset track's caption to stay in
    // distribution. The dataset-track pick remains the fallback for songs
    // written before the field existed.
    if ((gen.caption_yue2 || '').trim()) return (gen.caption_yue2 || '').trim();
    const resolved = resolveYue2CaptionForGeneration(gen, lyricsSetId);
    if (resolved.caption.trim()) return resolved.caption;
  }
  return gen.caption || '';
}
