// contracts/songBuilder.ts — Song Builder projects and section commands
// (/api/builder; docs/dev/frontend-library.md).
//
// A project is one song built from an ordered chain of sections. Each section
// renders several candidate songs; choosing one makes it the source the next
// section extends. Every section command names the project revision it was
// based on (`expectedRevision`): a stale one is 409 with `currentRevision`,
// and every accepted command moves the revision on.

import { z } from 'zod/v4';
import type { Song } from './songs.js';

/** POST /api/builder/projects/:id/sections/generate. */
export const generateSectionSchema = z.object({
  idempotencyKey: z.string().min(1).max(200),
  expectedRevision: z.number().int().min(0),
  direction: z.enum(['first', 'append', 'prepend']),
  label: z.string().max(200).default(''),
  lyrics: z.string().max(20_000).default(''),
  /** Bars need the project's BPM; seconds are used as given. */
  length: z.union([
    z.object({ bars: z.number().int().min(1).max(512) }),
    z.object({ seconds: z.number().positive().max(600) }),
  ]),
  /** Seconds of existing audio the extension overwrites at the seam. */
  overlap: z.number().min(0).max(120).default(4),
  /** Extend-from (append) or connect-at (prepend) point on the song so far;
   *  null = the very end (append) or start (prepend). */
  clipPoint: z.number().min(0).nullable().default(null),
  seedSectionId: z.string().nullable().default(null),
  seedStrength: z.number().min(0).max(1).default(0.4),
  previewMastering: z.boolean().default(false),
  /** The browser's "Keep Models in VRAM" setting at submit. */
  coResident: z.boolean().default(false),
  /** The head's measured length, used only when its song row stores 0
   *  (older repaint sections were saved that way). */
  headDuration: z.number().min(0).optional(),
  /** The engine the browser had active at submit; must still be active. */
  expectedBackend: z.string().min(1),
  /** getGlobalParams() at submit, captured like any other request. */
  engineParams: z.record(z.string(), z.unknown()),
});
export type GenerateSection = z.infer<typeof generateSectionSchema>;

/** A project as stored (snake_case columns). */
export interface BuilderProject {
  id: string;
  user_id: string;
  title: string;
  style: string;
  bpm: number;
  key_scale: string;
  time_signature: string;
  vocal_language: string;
  /** Default section length, seconds. */
  section_length: number;
  /** Candidates rendered per section (at most 16). */
  variant_count: number;
  /** Shared generation parameters as a JSON string. */
  gen_params: string;
  created_at: string;
  updated_at: string;
  /** Moves on every accepted section command and every project edit. */
  revision: number;
}

export type BuilderSectionStatus = 'pending' | 'generating' | 'ready' | 'failed' | 'chosen';

/** A section in a project view: its row, with candidates resolved to songs. */
export interface BuilderSection {
  id: string;
  project_id: string;
  /** Order in the song; sections are listed by position, then creation. */
  position: number;
  label: string;
  lyrics: string;
  direction: 'first' | 'append' | 'prepend';
  section_length: number;
  candidate_song_ids: string[];
  chosen_song_id: string | null;
  /** The builder-section workflow job rendering its candidates. */
  job_id: string | null;
  /** `generating` while its job runs; then `ready` (some candidates) or
   *  `failed` (none); `chosen` once a candidate is picked. A section whose job
   *  ended elsewhere (cancelled, server restart) is settled on the next read. */
  status: BuilderSectionStatus;
  created_at: string;
  updated_at: string;
  /** Candidate songs in candidate order; ids whose song was deleted are dropped. */
  candidates: Song[];
  chosen: Song | null;
}

/** GET/PATCH /api/builder/projects/:id, POST /api/builder/projects, and every
 *  section command: the whole project after the change. */
export interface BuilderProjectView { project: BuilderProject; sections: BuilderSection[] }
/** POST …/sections/generate adds the job and section it created (or, for a
 *  repeated idempotency key with the same body, the ones it created before). */
export interface GenerateSectionResponse extends BuilderProjectView { jobId: string; sectionId: string }
/** GET /api/builder/projects: newest-updated first. */
export interface BuilderProjectList { projects: Array<BuilderProject & { section_count: number }> }

/** POST /api/builder/projects and PATCH /api/builder/projects/:id. Absent
 *  fields keep their value (create: their default). A project edit takes no
 *  expectedRevision: last write wins per field, and it moves the revision. */
export interface BuilderProjectFields {
  title?: string;
  style?: string;
  bpm?: number;
  keyScale?: string;
  timeSignature?: string;
  vocalLanguage?: string;
  sectionLength?: number;
  variantCount?: number;
  genParams?: Record<string, unknown>;
}

/** POST /api/builder/sections/:id/choose. */
export interface ChooseCandidate { songId: string; expectedRevision: number }
/** POST /api/builder/sections/:id/stop: keep the candidates that landed. */
export interface StopSection { expectedRevision: number }
/** PATCH /api/builder/sections/:id. */
export interface EditSection { label?: string; lyrics?: string; expectedRevision: number }
/** DELETE /api/builder/sections/:id?expectedRevision=N also stops its renders. */

/** A refused builder command. `currentRevision` is set on a stale revision (409). */
export interface BuilderError { error: string; currentRevision?: number; issues?: Array<{ path: string; message: string }> }
