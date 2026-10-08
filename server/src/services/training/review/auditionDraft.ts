import type { AuditionPreview, AuditionSideResult } from '../types.js';
import { WorkflowDocuments } from '../../workflows/revisions.js';
import { getDb } from '../../../db/database.js';

export type AuditionRenderCell = 'bare' | 'adapter';
export interface TrainingCreateDraft {
  kind: 'training-audition-create';
  source: { datasetId: string; previewId: string; slot: 'base' | 'adapter'; cell: AuditionRenderCell };
  content: Record<string, string | number | boolean>;
  params: Record<string, unknown>;
  settings: { cacheLmCodes: false };
}

/** The complete mirrored-generation form, from a stored audition receipt. */
export function mirroredGenerationDraft(preview: AuditionPreview, side: AuditionSideResult, cell: AuditionRenderCell): TrainingCreateDraft {
  const caption = side.lmAdapter ? (preview.captionInput ?? preview.caption) : preview.caption;
  const ditAdapter = cell === 'adapter' ? (preview.renderDitAdapter || '') : '';
  return {
    kind: 'training-audition-create',
    source: { datasetId: preview.datasetId, previewId: preview.previewId, slot: side.slot, cell },
    content: {
      'hs-caption': caption || '', 'hs-lyrics': preview.lyrics || '', 'hs-instrumental': false,
      'hs-negative-prompt': '', 'hs-lora-trigger': '', 'hs-beat-intro': false,
      'hs-bpm': preview.bpm ?? 0, 'hs-keyScale': preview.keyscale ?? '',
      'hs-timeSignature': preview.timesignature ?? '', 'hs-duration': preview.durationSec || 180,
      'hs-vocalLanguage': 'en', 'hs-sourceLatentUrl': '',
    },
    params: {
      ditModel: preview.renderDitModel || preview.ditModel || '', lmModel: preview.lmModel || '',
      vaeModel: preview.vaeModel || '', embeddingModel: '',
      lmAdapter: side.lmAdapter || '', lmAdapterScale: side.lmAdapterScale ?? 1,
      adapter: ditAdapter, adapterScale: 1, adapterStack: [], advancedAdapters: false,
      adapterMode: 'runtime', adapterGroupScales: { self_attn: 1, cross_attn: 1, mlp: 1,
        cond_embed: 1, time_embed: 1, proj_in: 1 }, rebaseSource: '',
      seed: preview.seed, randomSeed: false, lmSeedFollowsDit: true, batchSize: 1,
      inferenceSteps: preview.renderSteps || 8, guidanceScale: 1, shift: 0,
      inferMethod: 'euler', scheduler: 'linear', guidanceMode: 'apg',
      apgMomentum: 0.75, apgNormThreshold: 2.5, cfgCutoffRatio: 1, lmCfgCutoffRatio: 1,
      cacheRatio: 0, customTimesteps: '', dcwEnabled: false, latentShift: 0,
      latentRescale: 1, denoiseStrength: 0, lssStrength: 0, pluginParams: {},
      skipLm: false, useCotCaption: true, lmTemperature: preview.lmTemperature ?? 0.85,
      lmTopP: preview.lmTopP ?? 0.9, lmCfgScale: preview.lmCfgScale ?? 2,
      lmRepPenalty: preview.lmRepPenalty ?? 1.1, lmRepMode: 'presence', lmRepWindow: 64,
      lmTopK: 0, lmNegativePrompt: 'NO USER INPUT', lmCodesStrength: 1, lmCodesMode: 'ratio',
      postProcessingEnabled: false, stableStepOn: false, timbreReference: false,
      timbreAudioPath: '', autoTrimEnabled: false, skipLrc: true, coverArtEnabled: false,
      qualityEvalEnabled: false, whisperLyricsEnabled: false,
    },
    settings: { cacheLmCodes: false },
  };
}

let documents: WorkflowDocuments | null = null;
const docs = () => documents ??= new WorkflowDocuments(getDb());
export function createTrainingCreateDraft(userId: string, operationKey: string, draft: TrainingCreateDraft, store = docs()) {
  const prior = store.list(userId, draft.kind).find(doc => doc.data.operationKey === operationKey);
  return prior ?? store.create(userId, draft.kind, { ...draft, operationKey });
}
export function getTrainingCreateDraft(userId: string, id: string, store = docs()) {
  const doc = store.get(id, userId);
  if (doc.kind !== 'training-audition-create') throw new Error('Not a training audition draft');
  return doc;
}
