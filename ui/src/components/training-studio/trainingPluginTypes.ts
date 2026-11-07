// trainingPluginTypes.ts — the React plug-in and presentation contracts of Training Studio.
//
// FROZEN for the Training unification batch (HOTSTEP-674); the server half is
// server/src/contracts/training.ts. Changed since: previewSource 'mm3-records'
// became 'mm3-ladder' when MM3 moved onto the shared checkpoint ladder
// (HOTSTEP-1043). Each training backend is one plug-in. The
// shared shell owns layout, phase selection and perform-all sequencing; a
// plug-in owns its own parameters and serialization. Server validation stays
// authoritative. This is deliberately not a generic form engine.
//
// Static facts (phases, stage IDs, recipes) live in trainingTargets.ts so they
// can be tested without React; the registry in trainingPlugins.tsx joins them
// to the rendering half defined here.

import type { ReactNode } from 'react';
import type { TrainingRecipeFamily } from '../../../../server/src/contracts/trainingRecipes';
import type { TrainingTargetId } from '../../../../server/src/contracts/training';
import type { DatasetAssets } from '../../services/trainingApi';
import type { TrainingPhase } from '../../stores/trainingStore';

export type { TrainingTargetId } from '../../../../server/src/contracts/training';
export type { TrainingPhase } from '../../stores/trainingStore';

/** A stable stage: `id` is a wire/test identifier and never changes with copy. */
export interface TrainingStageSpec {
  id: string;
  labelKey: string;
  label: string;
}

/** Which recipe a stage resolves. SA3 has no server recipe family: its recipe is the status `defaults`. */
export interface TrainingRecipeSpec {
  stage: string;
  family: TrainingRecipeFamily | 'sa3';
}

/** Where a backend's checkpoint previews come from. The ladder sources
 *  ('yue2-ladder', 'mm3-ladder') both draw CheckpointLadder.tsx. */
export type TrainingPreviewSource = 'none' | 'yue2-ladder' | 'mm3-ladder' | 'sa3-previews' | 'heartmula-previews';

export interface TrainingPhaseLabel {
  key: string;
  fallback: string;
}

/** The target-bound, translation-free half of a plug-in. */
export interface TrainingTargetSpec {
  id: TrainingTargetId;
  /** Stepper chips, in order. */
  phases: readonly TrainingPhase[];
  /** Every phase this target can stand on; a phase outside it falls back to 'train'. */
  reachablePhases: readonly TrainingPhase[];
  /** Per-phase label overrides (e.g. Prepare, Codes). */
  phaseLabels: Partial<Record<TrainingPhase, TrainingPhaseLabel>>;
  prepareStages: readonly TrainingStageSpec[];
  trainStages: readonly TrainingStageSpec[];
  recipeFields: readonly TrainingRecipeSpec[];
  previewSource: TrainingPreviewSource;
}

// ── Chips ────────────────────────────────────────────────────────────────

export interface TrainingChipSpec {
  id: string;
  on: boolean;
  label: string;
  title: string;
  suffix?: string;
}

export interface TrainingChipContext {
  t: (key: string, options?: Record<string, unknown>) => string;
  formatWhen: (iso: string) => string;
}

export interface TrainingReadiness {
  prepared: boolean;
  trained: boolean;
}

// ── Rendering ────────────────────────────────────────────────────────────

/** The page of a phase a plug-in is asked to draw. */
export type TrainingSection = 'prepare' | 'optimise' | 'train' | 'review';

export interface TrainingRenderContext {
  datasetId: string;
  slug: string;
  /** The dataset's custom trigger tag, '' when none. */
  trigger: string;
  section: TrainingSection;
}

export interface TrainingExecuteContext {
  datasetId: string;
  trigger: string;
}

export interface TrainingPlugin extends TrainingTargetSpec {
  /** Pipeline pills after Labeled/Built. Missing data renders OFF, never hidden. */
  chips(assets: DatasetAssets, ctx: TrainingChipContext): TrainingChipSpec[];
  readiness(assets: DatasetAssets | undefined): TrainingReadiness;
  /** The prepare page; undefined when the target shares the generic Preprocess panel or has none. */
  renderPrepare?(ctx: TrainingRenderContext): ReactNode;
  /** The Optimise page (YuE2 only). */
  renderOptimise?(ctx: TrainingRenderContext): ReactNode;
  /** The train page body for targets that own it. ACE's remains in TrainPanel until it migrates. */
  renderTrain?(ctx: TrainingRenderContext): ReactNode;
  /** The Review page for targets whose checkpoint previews are per dataset (HeartMuLa); others keep the shared review list. */
  renderReview?(ctx: TrainingRenderContext): ReactNode;
  /** Optional "perform all" entry, for shells that sequence stages. */
  performAll?(ctx: TrainingExecuteContext): Promise<void>;
}
