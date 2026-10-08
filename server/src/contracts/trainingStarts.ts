// contracts/trainingStarts.ts — how a client starts each training family.
//
// Training starts in two steps, and this file names both for every family so
// a client never reads routes/training.ts to find them:
//   1. Resolve: POST /api/training/ops/recipes/resolve with the operation
//      envelope (contracts/trainingOperation.ts) and a recipe payload
//      (contracts/trainingRecipes.ts). Node checks the worker and dataset
//      revision and returns the recipe, including `execution`: the complete
//      start body.
//   2. Start: POST `execution`, unchanged, to the family's existing start
//      route. These routes predate the envelope. They read their fields by
//      hand, ignore unknown ones, and do their own execution checks.
// YuE2 joint is the exception: a new run starts through the preparation
// operation, which resolves, checks and starts in one accepted command. Its
// direct route still accepts a plain body (the Refine panel's continuation
// uses it) but checks only what the route itself checks.
//
// Nothing here changes how any route validates or dispatches. The shapes
// below describe what the routes return today (docs/dev/frontend-training.md).

import { TRAINING_RECIPE_VERSION, type TrainingRecipeFamily } from './trainingRecipes.js';
import { TRAINING_SNAPSHOT_VERSION, type TrainingDatasetRef, type TrainingWorkerRef } from './trainingOperation.js';
import type { Yue2PreparationPayload } from './trainingPreparation.js';

export interface TrainingStartRoute {
  /** How the start is reached. */
  via: 'recipe-then-start' | 'preparation';
  /** The start route, under /api/training (or a worker's proxy base, below). `:id` is the dataset id. */
  start: { method: 'POST'; path: string; successStatus: 200 | 202 };
  /** Where the run, once started, is listed or read back. */
  runs: { method: 'GET'; path: string };
}

export const TRAINING_STARTS: Record<TrainingRecipeFamily, TrainingStartRoute> = {
  'ace-lm': { via: 'recipe-then-start', start: { method: 'POST', path: '/datasets/:id/train-lm', successStatus: 202 },
    runs: { method: 'GET', path: '/datasets/:id/train-lm' } },
  'ace-dit': { via: 'recipe-then-start', start: { method: 'POST', path: '/datasets/:id/train-dit', successStatus: 202 },
    runs: { method: 'GET', path: '/datasets/:id/train-dit' } },
  'mm3-lm': { via: 'recipe-then-start', start: { method: 'POST', path: '/datasets/:id/mm3-train-lm', successStatus: 200 },
    runs: { method: 'GET', path: '/datasets/:id/mm3-runs' } },
  'yue2-nar': { via: 'recipe-then-start', start: { method: 'POST', path: '/datasets/:id/yue2-train', successStatus: 200 },
    runs: { method: 'GET', path: '/datasets/:id/yue2-runs' } },
  'yue2-ar': { via: 'recipe-then-start', start: { method: 'POST', path: '/datasets/:id/yue2-ar-train', successStatus: 200 },
    runs: { method: 'GET', path: '/datasets/:id/yue2-ar-runs' } },
  // Started by POST /api/training/ops/preparation, whose joint stage posts to
  // this route itself. Posting to it directly skips those checks.
  'yue2-joint': { via: 'preparation', start: { method: 'POST', path: '/datasets/:id/yue2-joint-train', successStatus: 200 },
    runs: { method: 'GET', path: '/datasets/:id/yue2-joint-runs' } },
};

/** Job routes shared by every family, under /api/training. */
export const TRAINING_JOB_ROUTES = {
  list: { method: 'GET', path: '/jobs' },          // ?datasetId= → { jobs: TrainingJobStatus[] }
  get: { method: 'GET', path: '/jobs/:jobId' },    // → TrainingJobStatus, 404
  stream: { method: 'GET', path: '/jobs/:jobId/stream' }, // SSE: data: { type, ... }, replays recent events
  cancel: { method: 'DELETE', path: '/jobs/:jobId' }, // → { ok: true }, 404; kills the trainer process
} as const;

/** The start route's path for a dataset, relative to the training base. */
export function trainingStartPath(family: TrainingRecipeFamily, datasetId: string): string {
  return TRAINING_STARTS[family].start.path.replace(':id', encodeURIComponent(datasetId));
}

/** The training base for direct start routes. A remote worker's start goes
 *  through the worker proxy; operations (/ops/...) never do and always go to
 *  this machine with the worker named in the envelope. */
export function trainingStartBase(worker: TrainingWorkerRef): string {
  return worker.kind === 'local' ? '/api/training' : `/api/workers/${encodeURIComponent(worker.name)}/api/training`;
}

/** Body for POST /api/training/ops/recipes/resolve. */
export function recipeResolveRequest(input: {
  family: Exclude<TrainingRecipeFamily, 'yue2-joint'>;
  idempotencyKey: string;
  worker: TrainingWorkerRef;
  dataset?: TrainingDatasetRef;
  overrides?: Record<string, unknown>;
  preset?: string;
}) {
  return {
    version: TRAINING_SNAPSHOT_VERSION,
    operation: { kind: `recipe:${input.family}`, idempotencyKey: input.idempotencyKey },
    worker: input.worker,
    ...(input.dataset ? { dataset: input.dataset } : {}),
    sources: [],
    payload: { family: input.family, recipeVersion: TRAINING_RECIPE_VERSION, overrides: input.overrides ?? {},
      ...(input.preset ? { preset: input.preset } : {}) },
  };
}

/** Body for POST /api/training/ops/preparation that starts joint training on
 *  already prepared inputs (the bundled Start-training button). Add earlier
 *  stages ('latents', 'codes', ...) to prepare first; a joint stage with any
 *  other stage needs 'latents'. */
export function jointStartRequest(input: {
  idempotencyKey: string;
  worker: TrainingWorkerRef;
  dataset: TrainingDatasetRef;
  /** From GET /api/training/ops/preparation/context/:datasetId. */
  source: { kind: string; id: string; revision: string };
  overrides: Record<string, unknown>;
  preset?: string;
  trigger?: string;
  lyricTiming?: boolean;
}) {
  const payload: Yue2PreparationPayload = {
    recipes: { joint: { version: TRAINING_RECIPE_VERSION, overrides: input.overrides, ...(input.preset ? { preset: input.preset } : {}) } },
    mode: 'train-after-preparation',
    stages: ['joint'],
    trigger: input.trigger ?? '',
    lyricTiming: input.lyricTiming ?? true,
  };
  return {
    version: TRAINING_SNAPSHOT_VERSION,
    operation: { kind: 'yue2-preparation', idempotencyKey: input.idempotencyKey },
    worker: input.worker,
    dataset: input.dataset,
    sources: [input.source],
    payload,
  };
}

/** What POST /api/training/ops/recipes/resolve returns. `execution` is the
 *  start body; `resolved` is the form it came from. */
export interface RecipeResolveResponse {
  recipeVersion: typeof TRAINING_RECIPE_VERSION;
  family: TrainingRecipeFamily;
  builtin: Record<string, unknown>;
  stored: Record<string, unknown> | null;
  resolved: Record<string, unknown>;
  execution: Record<string, unknown>;
  provenance: Record<string, unknown>;
  deferred: unknown;
  /** yue2-joint with method 'base-matched' only. */
  effective?: Record<string, unknown>;
  worker: TrainingWorkerRef;
  operation: { kind: string; idempotencyKey: string };
}

/** Success bodies of the start routes. Fields beyond these may be added. */
export interface TrainingStartResponses {
  'ace-lm': { jobId: string };
  'ace-dit': { jobId: string };
  'mm3-lm': { jobId: string; kind: string; runName: string; outDir: string; attnBackend: string };
  'yue2-nar': { jobId: string; kind: 'yue2-nar-train'; runName: string; outDir: string; clips: number; lmType: string;
    target: string; tensors: number; estimatedMs: number; license: unknown };
  'yue2-ar': { jobId: string; kind: string; runName: string; outDir: string; sources: unknown; clips: number; lmType: string;
    target: string; styleTemplate: string; steps: number; minted: unknown; ckptPickStep: unknown; warnings: string[]; license: unknown };
  'yue2-joint': { jobId: string; kind: string; trainingMethod: 'aitk'; recipeVersion: string; method: string; outDir: string;
    steps: number; saveEvery: number; [field: string]: unknown };
}

/** A start route's refusal. 400 for a bad or unusable body, 404 for an
 *  unknown dataset, 409 when a job already runs for the dataset (or a joint
 *  body carries an admission that was never issued or was already used; a
 *  body without one is accepted), 503 when the trainer binary is missing. */
export interface TrainingStartError { error: string }

/** GET /api/training/jobs/:jobId. */
export interface TrainingJobStatus {
  id: string;
  datasetId: string;
  kind: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  total: number;
  done: number;
  failed: number;
  currentSampleId: string | null;
  phase: string;
  /** ace-server jobs ahead of this one; 0 if unknown. */
  engineQueueDepth: number;
  error: string | null;
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
}
