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
 *  name outside STEM_TRACK_NAMES, not just the first. Returns `{ id }` (202
 *  semantics: the pipeline runs after the response, tracked via `/progress`). */
export const stemExtractRequestSchema = z.object({
  sourceAudioUrl: z.string().min(1),
  sourceFileName: z.string().optional(),
  tracks: z.array(z.string()).min(1),
  style: z.string().optional(),
  lyrics: z.string().optional(),
  ditSettings: z.record(z.string(), z.unknown()).optional(),
});
export type StemExtractRequest = z.infer<typeof stemExtractRequestSchema>;

/** POST /api/stem-studio/supersep. Runs the neural separator in-process and
 *  saves every returned stem to disk, including hidden debug stems (not
 *  listed in the job's `tracks`). `level` is parsed with `parseInt(...,
 *  10)`; a non-numeric value becomes `NaN`, not a validation error — the
 *  route does not reject it today. Returns `{ id }`, same polling contract
 *  as `/extract`. */
export const stemSupersepRequestSchema = z.object({
  sourceAudioUrl: z.string().min(1),
  sourceFileName: z.string().optional(),
  level: z.union([z.string(), z.number()]).optional(),
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

/** GET /api/supersep/:jobId/progress, GET .../result, POST .../release, and
 *  POST /api/supersep/recombine are unvalidated proxies: Node forwards the
 *  path param or raw JSON body to the matching ace-server `/supersep/*`
 *  endpoint and relays its status/body (or binary WAV, for `/stem/:index`
 *  and `/recombine`) back unchanged. There is no Node-side schema for
 *  ace-server's own request/response shapes; see engine/src/supersep.h. */
export interface SupersepProgressResponse { status: string; progress: number; message: string; error?: string }
export interface SupersepStemMeta { name: string; category: string; index: number; stage?: number; hidden?: boolean }
export interface SupersepResultResponse { stems: SupersepStemMeta[] }
