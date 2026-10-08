// Versioned recipe seed values shared by Node and the training forms.
// The existing trainer routes remain responsible for execution validation.
import { z } from 'zod/v4';

export const TRAINING_RECIPE_VERSION = 1;
export const trainingRecipeFamilySchema = z.enum(['ace-lm', 'ace-dit', 'mm3-lm', 'yue2-nar', 'yue2-ar', 'yue2-joint']);
export type TrainingRecipeFamily = z.infer<typeof trainingRecipeFamilySchema>;
export const trainingRecipePayloadSchema = z.object({
  family: trainingRecipeFamilySchema,
  recipeVersion: z.literal(TRAINING_RECIPE_VERSION),
  overrides: z.record(z.string(), z.unknown()),
  preset: z.string().optional(),
});

export const ACE_LM_FORM_DEFAULTS = {
  "lmSize": "4B",
  "lmModel": "",
  "adapterName": "",
  "targetLoss": 4,
  "epochs": 150,
  "adapterType": "lora",
  "optimizer": "adamw",
  "muonLrScale": 20,
  "rank": 16,
  "alpha": 32,
  "lokrDim": 128,
  "lokrAlpha": 128,
  "lokrFactor": 6,
  "learningRate": 0.0001,
  "gradAccum": 2,
  "gradClip": 1,
  "warmupRatio": 0.05,
  "weightDecay": 0.01,
  "maxLen": 0,
  "seed": 42,
  "order": "shuffle",
  "lossOnCot": true,
  "milestoneStep": 0,
  "milestoneKeep": 6,
  "stages": [
    "extract",
    "train",
    "export"
  ],
  "overwrite": false,
  "stopEngine": true,
  "resumeFromLatest": true,
  "calibrate": false,
  "calibrateRepoint": true,
  "weights": "bf16",
  "batch": 1,
  "bwd": "outprod",
  "captionDropout": 0.3,
  "rslora": false,
  "dora": false,
  "hira": false,
  "loha": false,
  "pissa": false,
  "hotPizza": false,
  "hra": false,
  "loraPlusRatio": 1,
  "artistTokenOn": true,
  "artistToken": "",
  "artistTokenK": 32,
  "artistTokenLr": 0.005,
  "prefixN": 8,
  "regEvery": 0,
  "regTopk": 64,
  "regSongs": 24,
  "regTeacher": "cached",
  "attnBackend": "flash",
  "attnHeadBlock": 0
};
export const ACE_DIT_FORM_DEFAULTS = {
  "adapterName": "",
  "quality": "balanced",
  "targetLoss": 0.3,
  "epochs": 500,
  "adapterType": "lora",
  "rank": 128,
  "alpha": 256,
  "lokrDim": 512,
  "lokrAlpha": 512,
  "lokrFactor": 6,
  "lokrDecomposeBoth": true,
  "targetMlp": true,
  "dora": false,
  "rslora": false,
  "hira": false,
  "loha": false,
  "pissa": false,
  "hra": false,
  "loraPlusRatio": 1,
  "layers": 0,
  "crop": 0,
  "cropMin": 375,
  "cropMax": 0,
  "learningRate": 0.0005,
  "gradAccum": 4,
  "gradClip": 1,
  "warmupRatio": 0.05,
  "weightDecay": 0.01,
  "lossWeighting": "flow_snr",
  "snrGamma": 5,
  "tBias": 0.5,
  "channelBalance": true,
  "timestepMu": -0.4,
  "timestepSigma": 1,
  "tMin": 0,
  "tMax": 1,
  "cfgRatio": 0.15,
  "genreRatio": 30,
  "seed": 42,
  "order": "shuffle",
  "milestoneStep": 0,
  "milestoneKeep": 6,
  "vramReserveMb": 2048,
  "mirror": "bf16-f32",
  "bwd": "mm",
  "attnBackend": "flash",
  "cropJitter": false,
  "optimizer": "prodigy",
  "muonLrScale": 20,
  "muonNsSteps": 5,
  "batch": 1,
  "ckptSegments": 1,
  "stages": [
    "train",
    "export"
  ],
  "overwrite": false,
  "stopEngine": true,
  "resumeFromLatest": true,
  "calibrate": false,
  "calibrateRepoint": true
};
export const ACE_DIT_LOKR_FORM_DEFAULTS = {
  ...ACE_DIT_FORM_DEFAULTS,
  adapterType: "lokr",
  learningRate: 0.002,
  weightDecay: 0.001,
  lossWeighting: "none",
};

export const YUE2_JOINT_FORM_DEFAULTS = {
  "trainingMethod": "aitk",
  "checkpoint": "",
  "dataset": "",
  "output": "",
  "method": "base-matched",
  "steps": 200,
  "saveEvery": 20,
  "gradAccum": 4,
  "narCropFrames": 1500,
  "seed": 42,
  "device": "",
  "base": "",
  "lyricTiming": false,
  "cursorWeight": 0,
  "optimizer": "adamw-lm",
  "cautious": false,
  "prodigyD0": 0.000001,
  "muonLrScale": 1,
  "muonNsSteps": 5,
  "rank": 64,
  "alpha": 256,
  "adapterType": "lokr",
  "lokrDim": 128,
  "lokrFactor": 4,
  "stopMode": "steps",
  "narExtraSteps": 0,
  "captionDropout": 0,
  "autoRefine": false,
  "preview": {
    "enabled": true,
    "everySteps": 0,
    "parallel": true,
    "takes": 2,
    "odeSteps": 12,
    "narCacheRatio": 0,
    "seconds": 300,
    "seed": 424242,
    "previewMaxFrames": 7500,
    "baseline": false,
    "control": false
  },
  "stopEngine": false
};

// Pure projections preserve the exact legacy request shapes. Inactive controls
// remain in the form but are omitted from the execution body.
export function aceLmExecutionBody(form: typeof ACE_LM_FORM_DEFAULTS, variantKey?: string) {
  return {
      lmSize: form.lmSize,
      ...(form.lmModel ? { lmModel: form.lmModel } : {}),
      ...(variantKey ? { variantKey: variantKey } : {}),
      adapterName: form.adapterName.trim(),
      targetLoss: form.targetLoss,
      epochs: form.epochs,
      rank: form.rank,
      alpha: form.alpha,
      learningRate: form.learningRate,
      gradAccum: form.gradAccum,
      gradClip: form.gradClip,
      warmupRatio: form.warmupRatio,
      weightDecay: form.weightDecay,
      maxLen: form.maxLen,
      seed: form.seed,
      lossOnCot: form.lossOnCot,
      order: form.order,
      milestoneStep: form.milestoneStep,
      milestoneKeep: form.milestoneKeep,
      stages: form.stages,
      overwrite: form.overwrite,
      stopEngine: form.stopEngine,
      initAdapter: form.resumeFromLatest ? 'latest' : '',
      calibrate: form.calibrate,
      calibrateRepoint: form.calibrateRepoint,
      ...(form.weights !== 'f32-window' ? { weights: form.weights } : {}),
      ...(form.batch !== 1 ? { batch: form.batch } : {}),
      bwd: form.bwd,
      adapterType: form.adapterType,
      optimizer: form.optimizer,
      ...(form.optimizer === 'muon' ? { muonLrScale: form.muonLrScale } : {}),
      ...(form.adapterType === 'lokr'
        ? { lokrDim: form.lokrDim, lokrAlpha: form.lokrAlpha, lokrFactor: form.lokrFactor }
        : {}),
      ...(form.captionDropout > 0 ? { captionDropout: form.captionDropout } : {}),
      ...(form.adapterType === 'lora' && form.dora ? { dora: true } : {}),
      ...(form.adapterType === 'lora' && form.hira ? { hira: true } : {}),
      ...(form.adapterType === 'lora' && form.loha ? { loha: true } : {}),
      ...(form.adapterType === 'lora' && form.rslora ? { rslora: true } : {}),
      ...(form.adapterType === 'lora' && form.pissa && !form.dora && !form.hira && !form.loha ? { pissa: true } : {}),
      ...(form.adapterType === 'lora' && form.hra && !form.dora && !form.hira && !form.loha && !form.pissa ? { hra: true } : {}),
      ...(form.adapterType === 'lora' && form.loraPlusRatio !== 1 ? { loraPlusRatio: form.loraPlusRatio } : {}),
      ...(form.adapterType === 'lora' && form.artistTokenOn
        ? { artistToken: form.artistToken || form.adapterName, artistTokenK: form.artistTokenK, artistTokenLr: form.artistTokenLr }
        : { artistToken: '' }),
      ...(form.adapterType === 'lora' && form.prefixN > 0 ? { prefixN: form.prefixN } : {}),
      ...(form.regEvery > 0
        ? {
            regEvery: form.regEvery, regTopk: form.regTopk, regSongs: form.regSongs,
            ...(form.regTeacher !== 'cached' ? { regTeacher: form.regTeacher } : {}),
          }
        : {}),
      ...(form.attnBackend !== 'exact' && !(form.artistTokenOn && form.prefixN > 0) ? { attnBackend: form.attnBackend } : {}),
    };
}

export function aceDitExecutionBody(ditForm: typeof ACE_DIT_FORM_DEFAULTS, variantKey?: string) {
  return {
      ...(variantKey ? { variantKey: variantKey } : {}),
      adapterName: ditForm.adapterName.trim(),
      adapterType: ditForm.adapterType,
      ...(ditForm.adapterType === 'lokr'
        ? {
            lokrDim: ditForm.lokrDim,
            lokrAlpha: ditForm.lokrAlpha,
            lokrFactor: ditForm.lokrFactor,
            lokrDecomposeBoth: ditForm.lokrDecomposeBoth,
          }
        : { rank: ditForm.rank, alpha: ditForm.alpha }),
      targetMlp: ditForm.targetMlp,
      ...(ditForm.adapterType === 'lora' && ditForm.dora ? { dora: true } : {}),
      ...(ditForm.adapterType === 'lora' && ditForm.hira ? { hira: true } : {}),
      ...(ditForm.adapterType === 'lora' && ditForm.loha ? { loha: true } : {}),
      ...(ditForm.adapterType === 'lora' && ditForm.rslora ? { rslora: true } : {}),
      ...(ditForm.adapterType === 'lora' && ditForm.pissa && !ditForm.dora && !ditForm.hira && !ditForm.loha ? { pissa: true } : {}),
      ...(ditForm.adapterType === 'lora' && ditForm.hra && !ditForm.dora && !ditForm.hira && !ditForm.loha && !ditForm.pissa ? { hra: true } : {}),
      ...(ditForm.adapterType === 'lora' && ditForm.loraPlusRatio !== 1 ? { loraPlusRatio: ditForm.loraPlusRatio } : {}),
      layers: ditForm.layers,
      crop: ditForm.crop,
      cropMin: ditForm.cropMin,
      cropMax: ditForm.cropMax,
      targetLoss: ditForm.targetLoss,
      epochs: ditForm.epochs,
      learningRate: ditForm.learningRate,
      gradAccum: ditForm.gradAccum,
      gradClip: ditForm.gradClip,
      warmupRatio: ditForm.warmupRatio,
      weightDecay: ditForm.weightDecay,
      lossWeighting: ditForm.lossWeighting,
      snrGamma: ditForm.snrGamma,
      tBias: ditForm.tBias,
      channelBalance: ditForm.channelBalance,
      timestepMu: ditForm.timestepMu,
      timestepSigma: ditForm.timestepSigma,
      tMin: ditForm.tMin,
      tMax: ditForm.tMax,
      cfgRatio: ditForm.cfgRatio,
      genreRatio: ditForm.genreRatio,
      seed: ditForm.seed,
      order: ditForm.order,
      initAdapter: ditForm.resumeFromLatest ? 'latest' : '',
      calibrate: ditForm.calibrate,
      calibrateRepoint: ditForm.calibrateRepoint,
      milestoneStep: ditForm.milestoneStep,
      milestoneKeep: ditForm.milestoneKeep,
      vramReserveMb: ditForm.vramReserveMb,
      mirror: ditForm.mirror,
      bwd: ditForm.bwd,
      attnBackend: ditForm.attnBackend,
      optimizer: ditForm.optimizer,
      ...(ditForm.optimizer === 'muon'
        ? { muonLrScale: ditForm.muonLrScale, muonNsSteps: ditForm.muonNsSteps }
        : {}),
      batch: ditForm.batch,
      ckptSegments: ditForm.ckptSegments,
      stages: ditForm.stages,
      overwrite: ditForm.overwrite,
      stopEngine: ditForm.stopEngine,
    };
}

export const ACE_DIT_QUALITY_PRESETS = {
  "fast": {
    "epochs": 150,
    "cropMax": 0,
    "milestoneStep": 0,
    "targetLoss": 0.3
  },
  "balanced": {
    "epochs": 500,
    "cropMax": 0,
    "milestoneStep": 0,
    "targetLoss": 0.3
  },
  "thorough": {
    "epochs": 900,
    "cropMax": 0,
    "milestoneStep": 0,
    "targetLoss": 0.2
  }
};

export const ACE_DIT_LOKR_TARGET_LOSS = {
  "fast": 0.3,
  "balanced": 0.3,
  "thorough": 0.2
};

export const ACE_DIT_LOKR_EPOCHS = {
  "fast": 150,
  "balanced": 500,
  "thorough": 900
};

export const YUE2_JOINT_LADDER_PREVIEW = { enabled: true, everySteps: 0, parallel: true, takes: 2, odeSteps: 12, narCacheRatio: 0, seconds: 300, seed: 424242,
  previewMaxFrames: 7500, baseline: false, control: false };

export const YUE2_JOINT_TUNED_KEYS = ['targetKl', 'targetLoss', 'targetKlMode', 'narExtraSteps', 'klWeight', 'captionDropout', 'plannerLrScale', 'narLrScale',
  'spikeFactor', 'spikeStop', 'spikeStopWindow', 'reconStop', 'reconStopWindow', 'reconTarget', 'lrSchedule', 'lrFloor', 'lrDecaySteps',
  'lrDecayShape', 'klOvershootMargin', 'lrCycleSteps', 'lrCycleMult', 'klCheckpointEvery', 'refineWarmup', 'rungAdaptiveLr', 'reconKeepDelta'] as const;

export const YUE2_JOINT_LORA_STOP = { targetKl: 1.4, plannerLrScale: 0.3, narLrScale: undefined };

export const YUE2_JOINT_LOKR_STOP = { targetKl: 1.0, plannerLrScale: 0.6, narLrScale: 1 };

export const YUE2_JOINT_LEGACY_VALUES = {
  method: 'tuned', optimizer: 'prodigy', cautious: true, lr: 2e-4, plannerLrScale: 0.6, narLrScale: 1,
  stopMode: 'kl', targetKl: 1.0, targetKlMode: 'trend', captionDropout: 0.5,
  // Decoder phase on by default (2026-10-06): the planner freezes at the KL
  // target and the decoder trains on to the reconstruction plateau.
  narExtraSteps: 250, reconStop: 0.005, reconStopWindow: 10, reconKeepDelta: 0.003,
  spikeFactor: 5, spikeStop: 3, spikeStopWindow: 20, adapterType: 'lokr', lokrDim: 256, lokrFactor: 4, alpha: 256,
  // Base-matched knobs back to the engine's own defaults.
  warmup: undefined, weightDecay: undefined, beta2: undefined, abcDropout: undefined, arLossWeight: undefined,
  textDropout: undefined, lyricDropout: undefined, bothDropout: undefined, arCropFrames: undefined,
};

export const YUE2_JOINT_BASE_MATCHED_VALUES = {
  method: 'base-matched', optimizer: 'adamw-lm', cautious: false, lr: undefined,
  ...Object.fromEntries(YUE2_JOINT_TUNED_KEYS.map(key => [key, undefined])),
};

export const YUE2_JOINT_PRESETS = [
  { key: 'legacy', label: 'Legacy', steps: 500, gradAccum: 1, narCropFrames: 1500, saveEvery: 25, lokrDim: 256, minutes: 22 },
  { key: 'fast', label: 'Fast', steps: 100, gradAccum: 4, narCropFrames: 1500, saveEvery: 10, lokrDim: 128, minutes: 28 },
  { key: 'balanced', label: 'Balanced', steps: 200, gradAccum: 4, narCropFrames: 1500, saveEvery: 20, lokrDim: 128, minutes: 57 },
  { key: 'thorough', label: 'Thorough', steps: 300, gradAccum: 8, narCropFrames: 1500, saveEvery: 30, lokrDim: 128, minutes: 170 },
] as const;

export const YUE2_JOINT_BASE_MATCHED_DEFAULTS = {
  lr: 1e-4, weightDecay: 0.1, beta2: 0.95, abcDropout: 0.5, narCropFrames: 1500, arLossWeight: 0.25,
  gradAccum: 4, textDropout: 0.1, lyricDropout: 0.1, bothDropout: 0.1, warmupFraction: 0.03,
  arCropFrames: 0,
} as const;
