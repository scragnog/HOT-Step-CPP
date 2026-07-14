// ABCompareModal.tsx — Parameter diff modal for A/B track comparison
//
// Shows a side-by-side table of generation parameter differences between
// two pinned tracks. Ported from HOT-Step 9000 with CPP-specific param labels.

import React from 'react';
import { X, ArrowLeftRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useABCompareSelector, closeModal } from '../../stores/abCompareStore';

// Human-friendly labels for generation param keys
const PARAM_LABELS: Record<string, string> = {
  ditModel: 'DiT Model',
  lmModel: 'LM Model',
  vaeModel: 'VAE Model',
  embeddingModel: 'Embedding Model',
  instrumental: 'Instrumental',
  vocalLanguage: 'Vocal Language',
  bpm: 'BPM',
  keyScale: 'Key',
  timeSignature: 'Time Signature',
  duration: 'Duration',
  inferenceSteps: 'Inference Steps',
  guidanceScale: 'Guidance Scale',
  cfgCutoffRatio: 'CFG Cutoff Ratio',
  lmCfgCutoffRatio: 'LM CFG Cutoff Ratio',
  cacheRatio: 'Step Cache',
  shift: 'Shift',
  inferMethod: 'Solver',
  scheduler: 'Scheduler',
  guidanceMode: 'Guidance Mode',
  batchSize: 'Batch Size',
  seed: 'Seed',
  taskType: 'Task Type',
  loraPath: 'Adapter',
  loraScale: 'Adapter Scale',
  adapterMode: 'Adapter Mode',
  triggerWord: 'Trigger Word',
  triggerPlacement: 'Trigger Placement',
  // Guidance sub-params
  apgMomentum: 'APG Momentum',
  apgNormThreshold: 'APG Norm Threshold',
  // DCW
  dcwEnabled: 'DCW Enabled',
  dcwMode: 'DCW Mode',
  dcwLowScaler: 'DCW Low Scaler',
  dcwHighScaler: 'DCW High Scaler',
  // Post-processing
  denoiseStrength: 'Denoiser Strength',
  denoiseSmoothing: 'Denoiser Smoothing',
  denoiseMix: 'Denoiser Mix',
  spectralLifterEnabled: 'Spectral Lifter',
  slDenoiseStrength: 'SL Denoise',
  slNoiseFloor: 'SL Noise Floor',
  slHfMix: 'SL HF Mix',
  slTransientBoost: 'SL Transient Boost',
  slShimmerReduction: 'SL Shimmer Reduction',
  masteringEnabled: 'Mastering',
  masteringReference: 'Mastering Reference',
  timbreReference: 'Timbre Reference',
  // Latent
  latentShift: 'Latent Shift',
  latentRescale: 'Latent Rescale',
  customTimesteps: 'Custom Timesteps',
  // Auto-trim
  autoTrimEnabled: 'Auto-Trim',
  durationBuffer: 'Duration Buffer',
  autoTrimFadeMs: 'Fade-Out (ms)',
  // Cover/repaint
  audioCoverStrength: 'Cover Strength',
  coverNoiseStrength: 'Cover Noise',
  repaintingStart: 'Repainting Start',
  repaintingEnd: 'Repainting End',
  // LM
  skipLm: 'Skip LM',
  lmTemperature: 'LM Temperature',
  lmCfgScale: 'LM CFG Scale',
  lmTopK: 'LM Top-K',
  lmTopP: 'LM Top-P',
  useCotCaption: 'CoT Caption',
  // PP-VAE
  ppVaeReencode: 'PP-VAE Re-encode',
  ppVaeBlend: 'PP-VAE Blend',
  // Vocal Naturalizer
  vocalNaturalizerEnabled: 'Vocal Naturalizer',
  naturalizeAmount: 'Naturalize Amount',
  // Misc
  postProcessingEnabled: 'Post-Processing',
  coverArtEnabled: 'Cover Art',
  qualityEvalEnabled: 'Quality Eval',
};

// Keys to skip in comparison (always different or internal)
const SKIP_KEYS = new Set([
  'prompt', 'lyrics', 'title', 'caption', 'songDescription', 'customMode',
  'randomSeed', 'seed', 'lmSeedFollowsDit', 'lmSeed', 'source', 'style', 'artist', 'subject', 'trackName',
  'sourceAudioUrl', 'sourceLatentUrl', 'lmNegativePrompt',
  'pluginParams',  // shown separately if needed
  'adapterGroupScales',
  'coverArtSubject', 'qualityEvalTarget',
]);

function formatValue(val: unknown): string {
  if (val === undefined || val === null) return '—';
  if (typeof val === 'boolean') return val ? 'Yes' : 'No';
  if (typeof val === 'number') return String(val);
  if (typeof val === 'string') return val || '—';
  if (Array.isArray(val)) return val.join(', ') || '—';
  return JSON.stringify(val);
}

interface DiffRow {
  key: string;
  label: string;
  valueA: string;
  valueB: string;
}

function computeDiffs(paramsA: Record<string, any> | undefined, paramsB: Record<string, any> | undefined): DiffRow[] {
  const allKeys = new Set([
    ...Object.keys(paramsA || {}),
    ...Object.keys(paramsB || {}),
  ]);
  const diffs: DiffRow[] = [];

  for (const key of allKeys) {
    if (SKIP_KEYS.has(key)) continue;
    const a = (paramsA || {} as any)[key];
    const b = (paramsB || {} as any)[key];
    const fa = formatValue(a);
    const fb = formatValue(b);
    if (fa !== fb) {
      diffs.push({
        key,
        label: PARAM_LABELS[key] || key,
        valueA: fa,
        valueB: fb,
      });
    }
  }
  return diffs.sort((a, b) => a.label.localeCompare(b.label));
}

export const ABCompareModal: React.FC = () => {
  const isOpen = useABCompareSelector(s => s.isModalOpen);
  const trackA = useABCompareSelector(s => s.trackA);
  const trackB = useABCompareSelector(s => s.trackB);
  const { } = useTranslation();

  if (!isOpen || !trackA || !trackB) return null;

  const diffs = computeDiffs(trackA.generationParams, trackB.generationParams);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={closeModal}>
      <div
        className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl border border-zinc-200 dark:border-white/10 w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/10">
          <div className="flex items-center gap-3">
            <ArrowLeftRight size={20} className="text-pink-500" />
            <h2 className="text-lg font-bold text-zinc-900 dark:text-white">A/B Parameter Comparison</h2>
          </div>
          <button onClick={closeModal} className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors">
            <X size={18} className="text-zinc-500" />
          </button>
        </div>

        {/* Track Names */}
        <div className="grid grid-cols-[1fr_1fr] gap-4 px-6 py-3 bg-zinc-50 dark:bg-black/20 border-b border-zinc-200 dark:border-white/10">
          <div className="text-center">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
              A
            </span>
            <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200 mt-1 truncate">{trackA.title || 'Untitled'}</p>
          </div>
          <div className="text-center">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-orange-500/10 text-orange-600 dark:text-orange-400 border border-orange-500/20">
              B
            </span>
            <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200 mt-1 truncate">{trackB.title || 'Untitled'}</p>
          </div>
        </div>

        {/* Diff Table */}
        <div className="flex-1 overflow-y-auto px-6 py-3">
          {diffs.length === 0 ? (
            <div className="text-center py-12 text-zinc-500">
              <ArrowLeftRight size={32} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm font-medium">No parameter differences found</p>
              <p className="text-xs mt-1">Both tracks used identical generation settings</p>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-zinc-500 border-b border-zinc-200 dark:border-white/5">
                  <th className="text-left py-2 font-semibold">Parameter</th>
                  <th className="text-center py-2 font-semibold text-blue-500">Track A</th>
                  <th className="text-center py-2 font-semibold text-orange-500">Track B</th>
                </tr>
              </thead>
              <tbody>
                {diffs.map((d) => (
                  <tr key={d.key} className="border-b border-zinc-100 dark:border-white/5 hover:bg-zinc-50 dark:hover:bg-white/[0.02]">
                    <td className="py-2 pr-4 text-xs font-medium text-zinc-700 dark:text-zinc-300">{d.label}</td>
                    <td className="py-2 text-center text-xs font-mono text-blue-600 dark:text-blue-400 max-w-[200px] truncate">{d.valueA}</td>
                    <td className="py-2 text-center text-xs font-mono text-orange-600 dark:text-orange-400 max-w-[200px] truncate">{d.valueB}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-black/20 flex items-center justify-between">
          <span className="text-xs text-zinc-500">{diffs.length} difference{diffs.length !== 1 ? 's' : ''} found</span>
          <button
            onClick={closeModal}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ABCompareModal;
