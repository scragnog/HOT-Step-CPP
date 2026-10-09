// contracts/stemSeparation.ts — wire shapes for stem extraction/separation:
// routes/stemStudio.ts (Stem Studio's own extract + SuperSep pipeline,
// persisted under data/stems/<jobId>/) and routes/supersep.ts (a thin
// SuperSep-only proxy to ace-server, used by Cover Studio's splitter).
// Neither route requires auth — this is a local single-user app.
//
// routes/stemStudio.ts and routes/supersep.ts import the request schemas
// below instead of their own ad hoc checks, so the published shape is the
// one actually validated.

import { z } from 'zod/v4';

/** Both Stem Studio handlers check `sourceAudioUrl` with a bare `if
 *  (!sourceAudioUrl)` — any truthy value passes, including a number or an
 *  object, not just a non-empty string. `supersepSeparateRequestSchema`
 *  below is different: that route's own guard is `!audioUrl || typeof
 *  audioUrl !== 'string'`, so it genuinely requires a string. */
const truthyRequired = z.unknown().refine(Boolean, { message: 'required' });

/** The twelve DiT extraction tracks `/extract` and SuperSep's in-app track
 *  list are drawn from (also used by the `layer-render` workflow kind in
 *  studioWorkflows.ts). */
export const STEM_TRACK_NAMES = [
  'vocals', 'backing_vocals', 'drums', 'bass', 'guitar', 'keyboard',
  'percussion', 'strings', 'synth', 'fx', 'brass', 'woodwinds',
] as const;

// ── routes/stemStudio.ts ────────────────────────────────────────────────────

/** POST /api/stem-studio/extract. Runs each track as a sequential DiT
 *  generation against the source; 400 `Invalid track names: ...` lists every
 *  name outside STEM_TRACK_NAMES, not just the first — that check runs
 *  after this schema, against whatever `tracks` contains, so a non-string
 *  entry fails there (`Invalid track names: 5`), not here. Returns `{ id }`
 *  (202 semantics: the pipeline runs after the response, tracked via
 *  `/progress`). `sourceFileName`/`style`/`lyrics`/`ditSettings` are only
 *  ever used behind `|| <default>` (`'unknown'`/`''`/`''`/`{}`), so `null`
 *  and every other falsy value are accepted exactly like an absent key —
 *  this schema does not type-check them beyond that. */
export const stemExtractRequestSchema = z.object({
  sourceAudioUrl: truthyRequired,
  sourceFileName: z.unknown(),
  tracks: z.array(z.unknown()).min(1),
  style: z.unknown(),
  lyrics: z.unknown(),
  ditSettings: z.unknown(),
});
export type StemExtractRequest = z.infer<typeof stemExtractRequestSchema>;

/** POST /api/stem-studio/supersep. Runs the neural separator in-process and
 *  saves every returned stem to disk, including hidden debug stems (not
 *  listed in the job's `tracks`). `level` is parsed with
 *  `parseInt(String(level ?? '0'), 10)` — any value, including `null`,
 *  flows through `String()`; a non-numeric result becomes `NaN`, not a
 *  validation error. Returns `{ id }`, same polling contract as `/extract`. */
export const stemSupersepRequestSchema = z.object({
  sourceAudioUrl: truthyRequired,
  sourceFileName: z.unknown(),
  level: z.unknown(),
});
export type StemSupersepRequest = z.infer<typeof stemSupersepRequestSchema>;

/** GET /api/stem-studio/:jobId/progress. Progress is phase-weighted for
 *  SuperSep (separation 0-80%, saving 80-100%) and per-track for Extract.
 *  A job the process restarted while running is not resumed — once its
 *  `_meta.json` no longer exists in memory, a completed-on-disk job still
 *  reports `done`, but an interrupted one 404s rather than reporting
 *  `extracting`/`separating` forever. */
export interface StemProgressResponse {
  status: 'pending' | 'extracting' | 'separating' | 'saving' | 'done' | 'failed' | 'cancelled';
  progress: number; currentTrack: string; completedStems: string[]; totalTracks: number;
  warning?: string; error?: string; sepMessage?: string;
}

/** GET /api/stem-studio/:jobId/result. 404 `Job not found or not complete`
 *  until `_meta.json` exists. `category`/`stage` are present only for
 *  SuperSep jobs (from the engine's stem metadata); Extract jobs' stems have
 *  neither. `audioUrl` always points back at this same API. */
export interface StemResultItem {
  trackName: string; category?: string; audioUrl: string; durationSec: number; index: number;
  sizeBytes: number; stage?: number;
}
export interface StemResultResponse { id: string; type: 'extract' | 'supersep'; stems: StemResultItem[] }

/** GET /api/stem-studio/jobs. One entry per `data/stems/*\/_meta.json` on
 *  disk, newest first — not the in-memory job map, so this survives a
 *  restart. */
export interface StemJobSummary {
  id: string; type: 'extract' | 'supersep'; sourceFileName: string;
  tracks: string[]; completedStems: string[]; createdAt: string; sepLevel?: number;
}

/** GET /api/stem-studio/stats. */
export interface StemStorageStats { totalBytes: number; jobCount: number; stemCount: number }

// ── routes/supersep.ts (ace-server proxy) ──────────────────────────────────

/** POST /api/supersep/separate?level=0..5. `audioUrl` must resolve to a file
 *  already on disk under `/references/` or `/audio/`; 400 `audioUrl required
 *  in request body` if missing, 400 `Audio conversion failed: ...` if it
 *  cannot be read into an engine-compatible format. On success, returns
 *  ace-server's own `/supersep/separate` response verbatim (`{ id }` at the
 *  time of writing, but this route does not parse or constrain it). */
export const supersepSeparateRequestSchema = z.object({ audioUrl: z.string().min(1) });
export type SupersepSeparateRequest = z.infer<typeof supersepSeparateRequestSchema>;

/** GET /api/supersep/:jobId/progress. Node does not check ace-server's HTTP
 *  status here — it decodes the upstream JSON and always replies 200 with
 *  that body, even if ace-server itself replied with an error status. A
 *  body ace-server sent that isn't valid JSON throws and becomes this
 *  route's own 500. */
export interface SupersepProgressResponse { status: string; progress: number; message: string; error?: string }

/** POST /api/supersep/:jobId/release. Unlike the other proxies, Node
 *  discards ace-server's response body (`{"released":true}` or a JSON
 *  error) entirely and always replies with its own `{ ok }`, at status 200
 *  if ace-server's reply was 2xx, or ace-server's own status otherwise. */
export interface SupersepReleaseResponse { ok: boolean }

/** GET /api/supersep/:jobId/result. On a 2xx this forwards ace-server's
 *  JSON body unchanged at status 200; on a non-2xx it forwards ace-server's
 *  status and JSON error body unchanged (both legs pass through as-is). */
export interface SupersepStemMeta { name: string; category: string; index: number; stage?: number; hidden?: boolean }
export interface SupersepResultResponse { stems: SupersepStemMeta[] }

/** GET /api/supersep/:jobId/stem/:index. On success, streams ace-server's
 *  WAV bytes with `Content-Type: audio/wav` and a Node-generated
 *  `Content-Disposition` naming the file `stem_<index>.wav` — ace-server's
 *  own headers are not forwarded. On failure this does NOT pass through
 *  ace-server's error body: it replies with ace-server's status and a fixed
 *  `{ error: 'Failed to fetch stem' }`, discarding whatever ace-server
 *  actually said. */

/** POST /api/supersep/recombine. Forwarded to ace-server's
 *  `/supersep/recombine` verbatim — Node applies no schema to the request.
 *  The real wire shape ace-server requires (engine/tools/hot-step-server.cpp
 *  `svr.Post("/supersep/recombine", ...)`) is `{ id, stems: [{ index,
 *  volume?, muted? }] }`: 400 `Missing id` without `id`; a `stems` entry
 *  with no `index` is silently skipped, and an out-of-range `index` is
 *  silently ignored (clamped to the job's real stem count, not rejected);
 *  404 `Job not found`; 409 `Job not complete` if the separation job this
 *  `id` names hasn't finished. On a non-2xx, Node forwards ace-server's
 *  status and JSON body unchanged; on success it streams the resulting WAV
 *  with `Content-Type: audio/wav` and no `Content-Disposition`. */
export const supersepRecombineRequestSchema = z.object({
  id: z.string().min(1),
  stems: z.array(z.object({
    index: z.number().int().min(0),
    volume: z.number().finite().optional(),
    muted: z.boolean().optional(),
  })).optional(),
});
export type SupersepRecombineRequest = z.infer<typeof supersepRecombineRequestSchema>;
