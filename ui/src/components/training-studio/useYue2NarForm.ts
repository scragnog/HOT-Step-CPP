// useYue2NarForm.ts — the YuE2 NAR form's state and derivations.
//
// DERIVED, not seeded: the server defaults sit underneath and the user's edits
// on top, so there is no effect copying one into the other and no window where
// the form holds stale numbers while the measured recipe still has one home.

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  estimateYue2PeakMb, type Yue2NarTarget, type Yue2OptimOptions, type Yue2PresetName,
  type Yue2Status,
} from '../../services/trainingApi';

export interface TrainForm extends Yue2OptimOptions {
  lmType: string;
  trigger: string;
  rank: number;
  alpha: number;
  target: Yue2NarTarget;
  lr: number;
  lrScheduler: 'cosine' | 'constant';
  steps: number;
  warmup: number;
  saveEvery: number;
  logEvery: number;
  gradAccum: number;
  maxGradNorm: number;
  weightDecay: number;
  captionDropout: number;
  abcDropout: number;
  tSampling: 'logit-normal' | 'uniform';
  seed: number;
  kvCache: number;
  clipBlock: number;
}

export const PRESET_ORDER: Yue2PresetName[] = ['balanced', 'thorough', 'fast'];

export interface Yue2Peak {
  text: string;
  tone: string;
}

export function useYue2NarForm(status: Yue2Status | null, trigger?: string) {
  const { t } = useTranslation();
  const [edits, setEdits] = useState<Partial<TrainForm>>({});

  const d = status?.defaults;
  const base: TrainForm | null = status && d ? {
    lmType: d.lmType,
    // The dataset's own trigger word first, then whatever the status route
    // reports, and only then blank. Without one the adapter has no handle at
    // generation time and the route refuses the run.
    trigger: trigger || status.trigger || '',
    optimizer: d.optimizer,
    prodigyD0: d.prodigyD0,
    muonLrScale: d.muonLrScale,
    muonNsSteps: d.muonNsSteps,
    rank: d.rank,
    alpha: d.alpha,
    target: d.target,
    lr: d.lr,
    lrScheduler: d.lrScheduler,
    steps: d.steps,
    warmup: d.warmup,
    saveEvery: d.saveEvery,
    logEvery: d.logEvery,
    gradAccum: d.gradAccum,
    maxGradNorm: d.maxGradNorm,
    weightDecay: d.weightDecay,
    captionDropout: d.captionDropout,
    abcDropout: d.abcDropout,
    tSampling: d.tSampling,
    seed: d.seed,
    kvCache: d.kvCache,
    clipBlock: d.clipBlock,
  } : null;
  const form: TrainForm | null = base ? { ...base, ...edits } : null;

  const set = <K extends keyof TrainForm>(k: K, v: TrainForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));
  const reset = <K extends keyof TrainForm>(k: K) =>
    setEdits(e => { const rest = { ...e }; delete rest[k]; return rest; });

  // Presets are DERIVED from the fields, never stored, so a hand edit shows as
  // Custom rather than misreporting a preset. Step count is the only lever
  // measured here, so it is the only thing they move.
  const presets = status?.presets;
  const activePreset: Yue2PresetName | 'custom' = (() => {
    if (!presets || !form) return 'custom';
    const hit = PRESET_ORDER.find(p => {
      const q = presets[p];
      return q && form.steps === q.steps && form.saveEvery === q.saveEvery;
    });
    return hit ?? 'custom';
  })();
  const applyPreset = (p: Yue2PresetName) => {
    const q = presets?.[p];
    if (!q) return;
    setEdits(e => ({ ...e, steps: q.steps, saveEvery: q.saveEvery }));
  };

  const chosenBase = status?.bases.find(b => b.id === form?.lmType);
  const peak: Yue2Peak | null = (() => {
    if (!form || !status) return null;
    const mb = estimateYue2PeakMb(chosenBase?.bytes ?? 0, form.rank, status.vramModel, form.optimizer);
    const gbStr = (mb / 1024).toFixed(1);
    const total = status.gpuTotalMb || 0;
    // 0 means the engine could not be read, NOT a card with no memory. Show the
    // estimate without a verdict rather than inventing a scary one.
    if (total <= 0) {
      return { text: t('trainingStudio.yue2.peakUnknown', 'about {{gb}} GB of VRAM', { gb: gbStr }),
               tone: 'text-zinc-500' };
    }
    const totalGb = (total / 1024).toFixed(1);
    if (mb + 1536 <= total) {
      return { text: t('trainingStudio.yue2.peakFits', 'about {{gb}} GB of your {{totalGb}} GB',
                       { gb: gbStr, totalGb }), tone: 'text-emerald-500' };
    }
    if (mb <= total) {
      return { text: t('trainingStudio.yue2.peakTight',
                       'about {{gb}} GB of your {{totalGb}} GB — fits with nothing left for the desktop',
                       { gb: gbStr, totalGb }), tone: 'text-amber-500' };
    }
    return { text: t('trainingStudio.yue2.peakOver',
                     'about {{gb}} GB, more than your {{totalGb}} GB — lower the rank',
                     { gb: gbStr, totalGb }), tone: 'text-rose-500' };
  })();

  return { base, form, set, reset, setEdits, presets, activePreset, applyPreset, peak };
}
