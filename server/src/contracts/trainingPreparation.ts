import { z } from 'zod/v4';
import { TRAINING_RECIPE_VERSION } from './trainingRecipes.js';

export const yue2PreparationStageSchema = z.enum(['latents', 'codes', 'sheet', 'stems', 'align', 'nar', 'ar', 'joint']);
export type Yue2PreparationStage = z.infer<typeof yue2PreparationStageSchema>;

export const yue2PreparationPayloadSchema = z.object({
  recipes: z.partialRecord(z.enum(['nar', 'ar', 'joint']), z.object({
    version: z.literal(TRAINING_RECIPE_VERSION),
    overrides: z.record(z.string(), z.unknown()),
    preset: z.string().optional(),
  })),
  mode: z.enum(['prepare-only', 'train-after-preparation']),
  stages: z.array(yue2PreparationStageSchema).min(1),
  trigger: z.string().default(''),
  lyricTiming: z.boolean().default(true),
});
export type Yue2PreparationPayload = z.infer<typeof yue2PreparationPayloadSchema>;

export interface Yue2PreparationStageStatus {
  stage: Yue2PreparationStage;
  status: 'pending' | 'running' | 'done' | 'skipped' | 'failed' | 'cancelled' | 'interrupted';
  jobId: string | null;
  error: string | null;
}

export interface Yue2PreparationSummary {
  id: string;
  status: 'running' | 'pausing' | 'paused' | 'cancelling' | 'cancelled' | 'done' | 'failed' | 'interrupted';
  datasetId: string;
  worker: { kind: 'local' } | { kind: 'remote'; name: string };
  stages: Yue2PreparationStageStatus[];
  createdAt: number;
  updatedAt: number;
  error: string | null;
}

// ── Responses of /api/training/ops/preparation/* ──
/** GET /context/:datasetId: the revisions a preparation command must carry. */
export interface Yue2PreparationContext {
  dataset: { id: string; revision: string };
  source: { kind: 'dataset-sources'; id: string; revision: string };
}
/** POST /, GET /:id and POST /:id/{pause,resume,retry,cancel}. */
export interface Yue2PreparationResponse { pipeline: Yue2PreparationSummary }
/** GET /?datasetId=. */
export interface Yue2PreparationList { pipelines: Yue2PreparationSummary[] }
