// contracts/preferences.ts — wire schemas for Batch 4 slice 7a: named VST,
// AI-continue and YuE2 joint presets, plus STORM's operational tuning
// knobs. zod 4; clients import the types.
//
// Every kind here is installation-scoped (contracts/workflow.ts
// DocumentScope): none of the source browser keys carry a user id, so there
// is one shared owner per machine, same as the existing VST active chain and
// `ace-settings`. See routes/preferences.ts for why these routes skip auth
// entirely, matching routes/vst.ts.

import { z } from 'zod/v4';

// ── VST chain presets (named snapshots of the VST3 post-processing chain) ──

export const vstChainPresetEntrySchema = z.object({
  uid: z.string().min(1),
  name: z.string(),
  vendor: z.string(),
  path: z.string(),
  enabled: z.boolean(),
  statePath: z.string(),
}).passthrough(); // an entry's unknown extra fields (e.g. future plugin metadata) survive a round trip

export const vstChainPresetBodySchema = z.object({
  name: z.string().min(1).max(200),
  entries: z.array(vstChainPresetEntrySchema).max(500),
});
export type VstChainPresetBody = z.infer<typeof vstChainPresetBodySchema>;

// ── AI continue presets (STORM lyric/style continuation) ───────────────────
//
// Style and lyric presets share this body shape but are separate kinds, one
// per source browser key, so a list of one category never mixes the other.

export const aiContinuePresetBodySchema = z.object({
  label: z.string().min(1).max(200),
  value: z.string().min(1).max(4000),
});
export type AiContinuePresetBody = z.infer<typeof aiContinuePresetBodySchema>;

/** The continuation prompt template. A single installation-wide document;
 *  `hs-ai-continue-template` has no "clear" state (an empty string falls back
 *  to DEFAULT_TEMPLATE client-side), so the body always carries a string. */
export const aiContinueTemplateBodySchema = z.object({ template: z.string().max(20_000) });
export type AiContinueTemplateBody = z.infer<typeof aiContinueTemplateBodySchema>;

// ── YuE2 joint training presets ─────────────────────────────────────────────
//
// `version: 2` marks the current preset shape; its absence marks a legacy
// preset. That is a business-level version inside the body, not this
// document's schemaVersion — the legacy-upgrade behavior (adapterType,
// LoRA stop defaults, dropping a stale `lr`) is Yue2AitkTrainCard's own
// load-time logic (Slice J) and is preserved here by not reshaping `settings`.

export const yue2JointPresetBodySchema = z.object({
  name: z.string().min(1).max(200),
  version: z.literal(2).optional(),
  settings: z.record(z.string(), z.unknown()),
});
export type Yue2JointPresetBody = z.infer<typeof yue2JointPresetBodySchema>;

// ── STORM tuning (operational settings, one current-value document) ────────
//
// Unlike the presets above, this is not a named collection: STORM's 19
// `hs-storm-*` tuning keys (seed, solver/scheduler/guidance knobs, extra
// param maps) describe the single live session's current settings, the same
// "current value, no revision history" shape as `ace-settings`. One document
// per installation; every field optional and the whole body passthrough, so
// an older or newer client's extra fields survive, matching today's direct
// localStorage readers (they already tolerate missing fields).

export const stormTuningBodySchema = z.object({
  seed: z.number().optional(),
  seedLock: z.boolean().optional(),
  autoExpand: z.boolean().optional(),
  lss: z.number().optional(),
  xfade: z.number().optional(),
  maxbuf: z.number().optional(),
  stiffness: z.number().optional(),
  lbLambda: z.number().optional(),
  lbSnr: z.number().optional(),
  rkOrder: z.union([z.string(), z.number()]).optional(),
  cacheDepth: z.number().optional(),
  cacheRatio: z.number().optional(),
  cfgCutoff: z.number().optional(),
  inferMethod: z.string().optional(),
  scheduler: z.string().optional(),
  guidanceMode: z.string().optional(),
  solverExtra: z.record(z.string(), z.union([z.number(), z.string()])).optional(),
  schedulerExtra: z.record(z.string(), z.union([z.number(), z.string()])).optional(),
  guiderExtra: z.record(z.string(), z.union([z.number(), z.string()])).optional(),
  pageMode: z.string().optional(),
}).passthrough();
export type StormTuningBody = z.infer<typeof stormTuningBodySchema>;

// ── Requests ─────────────────────────────────────────────────────────────

/** POST /api/preferences/:family — create a new named preset. */
export const createPreferenceSchema = z.object({ body: z.unknown() });

/** PUT /api/preferences/:family/:id */
export const updatePreferenceSchema = z.object({ expectedRevision: z.number().int().min(1), body: z.unknown() });

/** One browser value to import, keyed by the exact source key and a hash of
 *  its serialized content (see ui/src/services/preferencesApi.ts). A named
 *  preset family also carries `name`, so the server can detect a same-name,
 *  different-content collision against an already-imported preset and
 *  refuse it (400 reason 'name-conflict') instead of silently overwriting
 *  or duplicating it; the caller then resubmits with an explicit `resolution`. */
export const importPreferenceItemSchema = z.object({
  storageKey: z.string().min(1).max(500),
  sourceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  name: z.string().min(1).max(200).optional(),
  body: z.unknown(),
  resolution: z.enum(['replace', 'keep-both']).optional(),
});
export type ImportPreferenceItem = z.infer<typeof importPreferenceItemSchema>;

export const importPreferenceBatchSchema = z.object({ items: z.array(importPreferenceItemSchema).max(500) });

/** Result of one item of an import batch. */
export interface ImportPreferenceResult {
  storageKey: string;
  outcome: 'imported' | 'unchanged' | 'replaced' | 'kept-both' | 'name-conflict';
  documentId?: string;
  /** The name actually stored, when 'kept-both' disambiguated it. */
  storedName?: string;
}
