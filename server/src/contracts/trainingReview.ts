import { z } from 'zod/v4';

export const reviewPickSchema = z.object({
  datasetId: z.string().min(1),
  runId: z.string().min(1),
  runRevision: z.number().int().nonnegative(),
  reviewRevision: z.string().length(64),
  step: z.number().int().nonnegative(),
  checkpointDir: z.string().min(1),
  blindLabel: z.string().optional(),
});
export const reviewFinishSchema = z.object({
  entries: z.array(reviewPickSchema.extend({ datasetRevision: z.string().min(1) })).min(1).max(32),
  knee: z.boolean().default(true),
});
export const reviewCleanupSchema = reviewPickSchema.extend({
  choice: z.object({ caches: z.boolean(), otherCheckpoints: z.boolean(), otherRuns: z.boolean(),
    resume: z.boolean(), otherPreviews: z.boolean() }),
});

export const auditionDraftSchema = z.object({
  datasetId: z.string().min(1),
  previewId: z.string().uuid(),
  slot: z.enum(['base', 'adapter']),
  cell: z.enum(['bare', 'adapter']),
});
export const reviewUseSchema = reviewPickSchema.extend({
  narFurther: z.boolean(),
  nar: z.object({ budget: z.number().int().min(10), lrScale: z.number().min(0.05).max(1),
    keepDelta: z.number().min(0), target: z.number().min(0).max(10).nullable(), knee: z.boolean() }),
});