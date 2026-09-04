// TrainDitForm.tsx — the DiT LoRA training controls
//
// Structural sibling of TrainLmForm: four visible controls (name, quality dial,
// target loss, epoch cap) and everything else behind the Advanced drawer, same
// pattern as PreprocessOptionsForm.
//
// The base model is a READ-ONLY chip, not a picker. The cached latents, context
// latents and encoder states in a preprocess variant are that DiT's own outputs,
// so training against a different base is meaningless — the server resolves
// --dit from preprocess_meta.json and ignores anything the client sends.

import React from 'react';
import { Loader2, Lock, Waves } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import type {
  DitAdapterType,
  TrainDitStage,
} from '../../services/trainingApi';

export type DitQuality = 'fast' | 'balanced' | 'thorough';

/** How the crop box is being *displayed*. Form-local only — the wire field stays
 *  `crop`, in raw latent frames, exactly as the engine and the API expect. */
type CropMode = 'auto' | 'seconds' | 'frames';

/** Latent frame rate of the preprocess cache (preprocess_meta.json latent_fps). */
const LATENT_FPS = 25;

/** The DiT patch size is 2, so an odd crop is a frame the trainer cannot use. */
const secondsToFrames = (seconds: number): number => {
  const f = Math.round(seconds * LATENT_FPS);
  return f - (f % 2);
};

/** Everything the panel needs to POST, with every default already resolved. */
export interface TrainDitFormState {
  adapterName: string;
  quality: DitQuality;
  targetLoss: number;
  epochs: number;
  adapterType: DitAdapterType;
  rank: number;
  alpha: number;
  /** LyCORIS LoKR factors (K2). Only meaningful when adapterType==='lokr' —
   *  always present so a type toggle never has to invent them. */
  lokrDim: number;
  lokrAlpha: number;
  lokrFactor: number;
  lokrDecomposeBoth: boolean;
  targetMlp: boolean;
  /** Parameterization (2026-09-04): DoRA / HiRA / LoHa are mutually exclusive
   *  reshapings of the LoRA update; rsLoRA and LoRA+ stack with any of them.
   *  All inert under 'lokr'. None ear-validated yet — measured gates only. */
  dora: boolean;
  rslora: boolean;
  hira: boolean;
  loha: boolean;
  pissa: boolean;            // init only; plain LoRA
  hra: boolean;              // orthogonal (reflections); excludes the others
  loraPlusRatio: number;     // 1 = off
  layers: number;            // 0 = auto (top-K depth)
  crop: number;              // 0 = auto-fit
  cropMin: number;
  cropMax: number;
  learningRate: number;
  gradAccum: number;
  gradClip: number;
  warmupRatio: number;
  weightDecay: number;
  lossWeighting: 'none' | 'flow_snr';
  snrGamma: number;
  tBias: number;
  channelBalance: boolean;
  timestepMu: number;
  timestepSigma: number;
  tMin: number;
  tMax: number;
  cfgRatio: number;
  genreRatio: number;
  seed: number;
  order: 'shuffle' | 'fixed';
  milestoneStep: number;
  milestoneKeep: number;
  vramReserveMb: number;
  /** Frozen-weight mirror precision — storage and compute are separate choices
   *  (see trainingApi.ts). 'bf16-f32' is bf16 storage with f32 arithmetic. */
  mirror: 'f32' | 'bf16' | 'bf16-f32';
  /** MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch).
   *  'mm' is the fast tensor-core backward and the server default. */
  bwd: 'outprod' | 'mm';
  /** Attention backend (2026-09-01, docs/plans/2026-09-01-flash-attn-backward.md).
   *  'exact' is today's byte-identical dit_attn_f32 graph, the default. 'flash'
   *  routes through the fused ggml_flash_attn_train/_back kernels, which is
   *  what makes full-song training crops affordable — experimental, and slower
   *  per step. */
  attnBackend: 'exact' | 'flash';
  cropJitter: boolean;
  optimizer: 'adamw' | 'muon' | 'prodigy';
  muonLrScale: number;
  muonNsSteps: number;
  /** Crops per micro-batch, from that many DIFFERENT songs (design §2.2). */
  batch: number;
  /** Gradient-checkpointing segments: 0 = off, 1 = auto, 2-32 = fixed. */
  ckptSegments: number;
  stages: TrainDitStage[];
  overwrite: boolean;
  stopEngine: boolean;
  /** Resume: continue from the newest non-calibrated run of this adapter name.
   *  Default ON (Rob, 2026-08-13). */
  resumeFromLatest: boolean;
  /** Post-training latent-Frechet calibration. Default OFF (Rob, 2026-08-13);
   *  it was ON from 2026-08-11. */
  calibrate: boolean;
  /** Let calibration repoint this artist's album preset(s). Default ON. */
  calibrateRepoint: boolean;
}

/** §2.6 defaults, with the DiT retune applied: 400 epochs, r128/a256, 30 %
 *  genre conditioning and the MLP projections trained. Kept in lockstep with
 *  the train-dit route's own numOpt fallbacks (server/src/routes/training.ts)
 *  and DitTrainArgs (engine/src/train/dit-train-run.h). */
export const TRAIN_DIT_DEFAULTS: TrainDitFormState = {
  adapterName: '',
  quality: 'balanced',
  // 0.1 / 500 (Rob, 2026-08-13) — targetLoss was 0.2 from 2026-07-30. Both are
  // deliberately beyond what a run typically reaches: the run always leaves the
  // BEST adapter behind, so an unreached target costs nothing and simply means
  // "use the whole horizon". The train-dit route's own fallback tracks this
  // number so a batch-pipeline run (which POSTs an empty option bag) stops at
  // the same loss a manual run does.
  targetLoss: 0.3,
  epochs: 500,
  adapterType: 'lora',
  rank: 128,
  alpha: 256,
  // Inert while adapterType==='lora'; kept at the K2 values so a toggle to
  // 'lokr' via TRAIN_DIT_LOKR_DEFAULTS below is the only place they change.
  lokrDim: 512,
  lokrAlpha: 512,
  lokrFactor: 6,
  lokrDecomposeBoth: true,
  targetMlp: true,
  dora: false,
  rslora: false,
  hira: false,
  loha: false,
  pissa: false,
  hra: false,
  loraPlusRatio: 1,
  layers: 0,
  crop: 0,
  cropMin: 375,
  // 0 = no pin: the engine lifts the cap to the dataset's longest track in
  // flash mode. A number pins it in either mode (flash included).
  cropMax: 0,
  learningRate: 0.0005,
  gradAccum: 4,
  gradClip: 1.0,
  warmupRatio: 0.05,
  weightDecay: 0.01,
  lossWeighting: 'flow_snr',
  snrGamma: 5.0,
  tBias: 0.5,
  channelBalance: true,
  timestepMu: -0.4,
  timestepSigma: 1.0,
  tMin: 0,
  tMax: 1,
  cfgRatio: 0.15,
  genreRatio: 30,
  seed: 42,
  order: 'shuffle',
  // Milestones OFF by default (Rob, 2026-07-30) — the best adapter is now
  // always what lands in the run dir, so mid-run snapshots are opt-in.
  milestoneStep: 0,
  milestoneKeep: 6,
  vramReserveMb: 2048,
  // 'bf16-f32' (Rob, 2026-09-02). Supersedes the 2026-09-01 plain-'bf16'
  // default, which was taken on VRAM alone: bf16 storage is lossless (the GGUF
  // is BF16), but bf16 COMPUTE also rounded the activations and the gradients
  // at every trainable-layer GEMM, and adapters trained that way rendered
  // coarse and "bitty" where the same recipe at 'f32' did not. 'bf16-f32'
  // keeps bf16's storage — so the flash-mode crop stays long (1556 vs f32's
  // 610 on mika_lifeincartoonmotion) — and casts each weight to F32 in the
  // graph, which measured bit-identical loss and gnorm to the f32 mirror over
  // three epochs. TRAIN_DIT_LOKR_DEFAULTS spreads this object, so LoKR
  // inherits it too.
  mirror: 'bf16-f32',
  // MUL_MAT activation-gradient formulation. 'mm' mirrors the server default
  // (2026-07-29): identical maths to out_prod but dtype-agnostic, so the BF16
  // mirror above rides BF16 tensor cores instead of being promoted to F32.
  bwd: 'mm',
  // 'flash' by default (2026-09-01, Rob-directed after the smoke test): fused
  // attention, full-song crops via the lifted crop cap. Unchecking the box
  // returns to the byte-identical exact graph AND restores the exact-mode
  // companions (f32 mirror, cropMax 1250) — see the checkbox handler.
  attnBackend: 'flash',
  cropJitter: false,
  // Prodigy is the DEFAULT (the 2026-07-30 Muon note below is history: Muon won
  // the epoch race then, Prodigy replaced it as the shipped default and the
  // route's absent-field fallback agrees). Kept for the record:
  // Muon was the default from 2026-07-30. The 10-epoch comparison that suggested
  // parity was too short a window: over a full run Muon reached ma5 0.6 in 161
  // epochs against AdamW's 227, and once the Newton-Schulz was bucketed that
  // became ~1.23x on wall-clock. Ear-validated before the flip.
  // muonMomentum/muonMinDim/muonBucket are left to the engine defaults.
  // Prodigy (Rob, 2026-09-03): automatic step size, so learningRate is ignored
  // under it. Muon (ear-validated 2026-07-30) and AdamW remain selectable.
  optimizer: 'prodigy',
  muonLrScale: 20,
  muonNsSteps: 5,
  // Micro-batching + checkpointing defaults (design §2.2 / C3/C5) — the engine's
  // own train-dit defaults. batch 1 = OFF (2026-07-29, measured): batching is
  // ~2.5x SLOWER at full depth on a 32 GB card and ~2.4x faster on shallow /
  // partial-depth runs, so it is opt-in for the latter. ckptSegments 1 = auto,
  // which resolves to a single (i.e. unsegmented) run whenever full depth fits.
  batch: 1,
  ckptSegments: 1,
  stages: ['train', 'export'],
  overwrite: false,
  stopEngine: true,
  // Resume ON, calibrate OFF (Rob, 2026-08-13) — the inverse of the 2026-08-11
  // seed, matching the LM form's 2026-08-12 flip. With target loss at 0.1 a
  // re-run is almost always "keep going", not "start over", and resume is soft
  // on the server: an adapter name with no previous run trains from scratch
  // rather than failing. TRAIN_DIT_LOKR_DEFAULTS spreads this object and
  // doesn't override either flag, so LoKR inherits both.
  resumeFromLatest: true,
  calibrate: false,
  calibrateRepoint: true,
};

/** K1/K2 (lokr-dit-training plan §0): LoKR is the UI's DEFAULT adapter type —
 *  Rob's validated preference (Uber-LoKR-4). Same base as TRAIN_DIT_DEFAULTS,
 *  patched with the preset's four adapter fields plus the loss-shape knobs the
 *  preset agrees with (lr/loss-weighting/target-loss/weight-decay), plus
 *  gradAccum, which the preset's lr is inseparable from. Everything else
 *  (epochs, targetMlp, cfgRatio, channelBalance, seed, milestoneStep…) stays at
 *  the shared default.
 *
 *  crop stays 0 (auto-fit) like the shared default. It shipped as a fixed 60 s
 *  window to match Side-Step's chunk_duration, but the auto-fit reduces CROP
 *  before it reduces DEPTH: pinning the window is what forces the depth ladder
 *  down instead, so a 32 GB card would train a partial-depth adapter to keep the
 *  60 s. Auto keeps all 32 layers and takes the longest window that genuinely
 *  fits. DIT_LOKR_CROP_FRAMES below is still where the Seconds/Frames control
 *  lands when the user leaves auto. */
export const DIT_LOKR_CROP_FRAMES = 1500;   // 60 s at 25 latent fps

export const TRAIN_DIT_LOKR_DEFAULTS: TrainDitFormState = {
  ...TRAIN_DIT_DEFAULTS,
  adapterType: 'lokr',
  lokrDim: 512,
  lokrAlpha: 512,
  lokrFactor: 6,
  lokrDecomposeBoth: true,
  // 2026-07-30 retune, from a five-run A/B on gunship_unicorn (14 songs, XL
  // thirds). GA 20 @ 1e-2 was Side-Step's effective batch of 20 reached by
  // accumulation; GA 4 @ 2e-3 is the same effective LR per sample by linear
  // scaling, and measured IDENTICAL epochs-to-target (227 vs 228) with strictly
  // better-behaved gradients: median grad-norm 0.062 vs 0.031 (the sqrt(5) the
  // smaller batch predicts) and NO warmup spike — the GA 20 run peaked at 13.5
  // on epoch 1, which is the same cliff three LoKR runs fell off on 2026-07-29.
  // The LR must move WITH the accumulation; 2e-3 without GA 4 is a different,
  // untested config. Mirrored by the train-dit route's isLokr fallbacks.
  learningRate: 0.002,
  gradAccum: 4,
  // 250, not 400: every 400-horizon run stopped with the cosine only halfway
  // down (LR still ~50% of peak at the stop), so the schedule never decayed
  // INTO the target. Shortening the horizon cut epochs-to-0.6 from 228 to 203
  // (-11%) and audio-seconds-to-target by 14% — the only knob of five tested
  // that reduced the work required rather than rearranging it. Don't shorten
  // much further: that run used 203 of its 250, and a horizon below the epochs
  // actually needed leaves the LR at its 10% floor with the tail unfinished.
  // 500 / 0.2 (Rob, 2026-07-30), overriding the 250/0.6 that the horizon study
  // landed on. That study's point stands — a horizon SHORTER than the epochs
  // actually needed leaves the LR pinned at its 10% floor — but it was written
  // when an unreached target meant shipping the LAST adapter. Now the run keeps
  // the best, so a long horizon plus an ambitious target is the safe direction:
  // you get the whole cosine decay and still keep the best point on it.
  epochs: 500,
  lossWeighting: 'none',
  targetLoss: 0.3,
  weightDecay: 0.001,
  crop: 0,
};

/** Which unit to show `crop` in. State, not a pure derivation: once the box holds
 *  a number, seconds and frames are the same value and only the user's last
 *  choice distinguishes them. This is the seed for that state. */
const deriveCropMode = (state: TrainDitFormState): CropMode => {
  if (state.crop === 0) return 'auto';
  // The LoKR window is authored in seconds, so show it in seconds.
  if (state.adapterType === 'lokr' && state.crop === DIT_LOKR_CROP_FRAMES) return 'seconds';
  return 'frames';
};

/** §5.2 quality dial. Each preset patches the SAME key set so flipping back and
 *  forth is reversible — the plan names targetLoss only under Thorough, which on
 *  its own would leave 0.3 stuck after switching away from it. */
// Balanced IS the default state, so its values have to BE the new defaults —
// leaving them behind would mean the dial silently undid the retune the moment
// anyone clicked the button that was already highlighted. Fast and Thorough are
// re-spaced around it to keep the dial monotone. milestoneStep stays 0 across
// all three (2026-07-30): the best adapter is kept regardless, so snapshots are
// an explicit choice, not something a quality click turns back on.
// Target-loss ladder re-spaced 2026-08-13 around the new 0.1 default (was
// 0.3/0.2/0.15): Balanced must equal TRAIN_DIT_DEFAULTS.targetLoss, and
// Thorough has to stay below it to keep the dial monotone.
const DIT_QUALITY_PRESETS: Record<DitQuality, Partial<TrainDitFormState>> = {
  // cropMax 0 = the engine's cap (800 since 2026-09-03); the old 750/1250 pins
  // went with the long-crop regression. Targets 0.3/0.3/0.2 (Rob, same day).
  fast:     { epochs: 150, cropMax: 0, milestoneStep: 0, targetLoss: 0.3 },
  balanced: { epochs: 500, cropMax: 0, milestoneStep: 0, targetLoss: 0.3 },
  thorough: { epochs: 900, cropMax: 0, milestoneStep: 0, targetLoss: 0.2 },
};

/** LoKR's targets were 0.6/0.6/0.5 — its validated auto-stop under the old
 *  "unreached target ships the LAST adapter" rule. With the best adapter now
 *  always kept, LoKR tracks the same ambitious targets as LoRA. Balanced MUST
 *  equal TRAIN_DIT_LOKR_DEFAULTS.targetLoss so the already-highlighted button
 *  is a no-op. */
const DIT_LOKR_TARGET_LOSS: Record<DitQuality, number> = {
  fast: 0.3, balanced: 0.3, thorough: 0.2,
};

/** Same reasoning for the epoch counts: the presets above are LoRA's 100/400/800,
 *  and LoKR's validated horizon is 250 (2026-07-30 A/B — see the epochs comment
 *  in TRAIN_DIT_LOKR_DEFAULTS). Balanced MUST be 250 here for the same reason it
 *  is 400 there: it is the default state, so clicking the already-highlighted
 *  button has to be a no-op rather than a silent undo of the retune. Fast and
 *  Thorough keep the dial monotone around it. */
const DIT_LOKR_EPOCHS: Record<DitQuality, number> = {
  fast: 150, balanced: 500, thorough: 900,
};

const ALL_STAGES: TrainDitStage[] = ['train', 'export'];
const QUALITIES: DitQuality[] = ['fast', 'balanced', 'thorough'];

const FIELD =
  'rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500 disabled:opacity-50 disabled:cursor-not-allowed';
const LABEL = 'text-xs font-semibold text-zinc-600 dark:text-zinc-400';
/** Checkbox-row label text — matches the row it sits in, so a hoverable
 *  caption doesn't change weight next to a plain one. */
const CHECK_LABEL = 'text-xs text-zinc-700 dark:text-zinc-300';

interface Props {
  value: TrainDitFormState;
  onChange: (patch: Partial<TrainDitFormState>) => void;
  /** The variant's own DiT, from trainDitStatus.ditModel. Read-only. */
  ditModel: string;
  /** Usable cached songs in the variant, from trainDitStatus.sampleCount. Only
   *  feeds the schedule preview; 0/absent simply hides it. */
  sampleCount?: number;
  /** Locks every control (a job is running, or a start is in flight). */
  disabled?: boolean;
  starting?: boolean;
  onStart: () => void;
}

export const TrainDitForm: React.FC<Props> = ({
  value, onChange, ditModel, sampleCount, disabled, starting, onStart,
}) => {
  const { t } = useTranslation();

  const lock = !!disabled;

  // Schedule preview — deliberately the SAME arithmetic the engine runs in
  // dit-train-run.h: stepsPerEpoch = ceil(n / (batch * gradAccum)), then a
  // warmup floored at 50 steps and capped at half the run. Surfaced because
  // effective batch and warmup length are exactly the two quantities a ported
  // learning rate silently depends on, and getting them wrong blew up three
  // LoKR runs (2026-07-29). Hidden until the variant reports a sample count.
  const schedule = React.useMemo(() => {
    const n = Math.max(0, Math.trunc(sampleCount ?? 0));
    if (n === 0) return null;
    const winSongs = Math.max(1, Math.trunc(value.batch) * Math.trunc(value.gradAccum));
    const steps = Math.max(1, Math.ceil(n / winSongs) * Math.max(1, Math.trunc(value.epochs)));
    let warmup = 0;
    if (value.warmupRatio > 0) {
      warmup = Math.min(Math.max(50, Math.trunc(steps * value.warmupRatio)), Math.trunc(steps / 2));
      if (warmup < 1) warmup = 0;   // only reachable at steps === 1
    }
    return { steps, effectiveBatch: Math.min(winSongs, n), warmup };
  }, [sampleCount, value.batch, value.gradAccum, value.epochs, value.warmupRatio]);

  // Display unit for `crop`. Seeded from the incoming state, then re-seeded
  // whenever something outside the component (a preset load, an adapter-type
  // swap) moves `crop` across the auto boundary — which is the only move the
  // current mode can no longer represent.
  const [cropMode, setCropMode] = React.useState<CropMode>(() => deriveCropMode(value));
  React.useEffect(() => {
    if ((cropMode === 'auto') !== (value.crop === 0)) {
      setCropMode(deriveCropMode(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value.crop, value.adapterType, cropMode]);

  // Mirrors the server's ADAPTER_NAME_RE (§4.5 step 6). The GET path sanitises
  // the same input instead of rejecting it, so without this the panel can render
  // a green "Adapter ready" card for a directory the POST will refuse.
  const trimmed = value.adapterName.trim();
  const adapterNameOk = /^[A-Za-z0-9._-]{1,64}$/.test(trimmed)
    && !/^\.+$/.test(trimmed)
    && !trimmed.startsWith('.');

  const num = (raw: string, fallback: number): number => {
    // '' -> Number('') === 0, which is finite. targetLoss 0 legitimately means
    // "no auto-stop" and crop/layers 0 mean "auto", so an emptied box must fall
    // back to the default rather than silently selecting one of those modes.
    if (raw.trim() === '') return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };

  const toggleStage = (stage: TrainDitStage, on: boolean) => {
    const next = ALL_STAGES.filter(s => (s === stage ? on : value.stages.includes(s)));
    onChange({ stages: next });
  };

  const pickQuality = (q: DitQuality) => onChange({
    quality: q,
    ...DIT_QUALITY_PRESETS[q],
    ...(value.adapterType === 'lokr' ? { targetLoss: DIT_LOKR_TARGET_LOSS[q], epochs: DIT_LOKR_EPOCHS[q] } : {}),
    // Presets carry exact-mode crop caps; re-pinning cropMax in flash mode
    // would silently kill the full-song lift.
    ...(value.attnBackend === 'flash' ? { cropMax: 0 } : {}),
  });

  // §4: switching adapter type swaps the WHOLE form state to that type's
  // default set (simplest option the plan sanctions over per-field "was it
  // touched" tracking) — except the adapter name, which is the one field
  // losing on a toggle would actually cost the user something.
  const pickType = (ty: DitAdapterType) => {
    if (ty === value.adapterType) return;
    const defaults = ty === 'lokr' ? TRAIN_DIT_LOKR_DEFAULTS : TRAIN_DIT_DEFAULTS;
    setCropMode(deriveCropMode(defaults));
    onChange({ ...defaults, adapterName: value.adapterName });
  };

  const pickCropMode = (mode: CropMode) => {
    setCropMode(mode);
    if (mode === 'auto') {
      onChange({ crop: 0 });
    } else if (value.crop === 0) {
      // seconds <-> frames is lossless (same field, same number), so only the
      // hop out of auto has to invent one. Land on the documented 60 s window.
      onChange({ crop: DIT_LOKR_CROP_FRAMES });
    }
  };

  const setCropValue = (raw: string) => {
    if (cropMode === 'seconds') {
      onChange({ crop: secondsToFrames(num(raw, DIT_LOKR_CROP_FRAMES / LATENT_FPS)) });
    } else {
      onChange({ crop: num(raw, 0) });
    }
  };

  /** Field label + hover explanation. `key` is the i18n suffix under
   *  trainingStudio.train.dit.*; the explanation is that key + "Info", so the
   *  two cannot drift apart. `meta` is the default/range line — numbers only,
   *  which is why it is not a translated string. */
  const P = (key: string, meta?: string, className = LABEL) => (
    <ParamLabel
      label={t(`trainingStudio.train.dit.${key}`)}
      info={t(`trainingStudio.train.dit.${key}Info`)}
      meta={meta}
      className={className}
    />
  );

  const cropShown = cropMode === 'seconds'
    ? Math.round((value.crop / LATENT_FPS) * 10) / 10
    : value.crop;

  // cropMax below cropMin is a 400 from the server (§4.5 step 7); flag it here.
  // 0 = unpinned (flash mode lifts the cap server-side), so only a real pin
  // has to clear cropMin.
  const cropRangeOk = value.cropMax === 0 || value.cropMax >= value.cropMin;
  const tWindowOk = value.tMin < value.tMax;
  const isLokr = value.adapterType === 'lokr';
  // Server range: -1 or [2,64] (§2.1). 0/1/negative-not-(-1) values are refused.
  const lokrFactorOk = value.lokrFactor === -1 || (value.lokrFactor >= 2 && value.lokrFactor <= 64);
  // The server accepts crop 0 (auto) or 128..8192 — the 2..126 band the stepper
  // walks through is a guaranteed 400. Same for lr (strictly > 0) and seed.
  const cropOk = value.crop === 0 || (value.crop >= 128 && value.crop <= 8192);
  const lrOk = value.learningRate > 0 && value.learningRate <= 1;
  const seedOk = value.seed >= 0 && value.seed <= 2147483647;

  return (
    <div className="flex flex-col gap-4">
      {/* ── Adapter type (K1: LoKR is the default) ──────────────────────── */}
      <div className="flex flex-col gap-1.5">
        {P('adapterType', 'Default LoKR')}
        <div className="flex items-center gap-1.5">
          {(['lokr', 'lora'] as DitAdapterType[]).map(ty => {
            const active = value.adapterType === ty;
            return (
              <button
                key={ty}
                type="button"
                disabled={lock}
                onClick={() => pickType(ty)}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  active
                    ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                    : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {ty === 'lokr' ? t('trainingStudio.train.dit.adapterTypeLokr') : t('trainingStudio.train.dit.adapterTypeLora')}
              </button>
            );
          })}
        </div>
        <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.adapterTypeHelp')}</span>
      </div>

      {/* ── Base model (read-only) ────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        {P('baseModel', 'Read-only · set by the preprocess variant')}
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-zinc-300 dark:border-white/10 bg-zinc-200/50 dark:bg-white/5">
          <Lock size={12} className="text-zinc-500 flex-shrink-0" />
          <code className="min-w-0 truncate font-mono text-[11px] text-zinc-700 dark:text-zinc-300" title={ditModel}>
            {ditModel || '—'}
          </code>
        </div>
        <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.baseModelHelp')}</span>
      </div>

      {/* ── Quality dial ──────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        {P('quality', 'Default Balanced')}
        <div className="flex items-center gap-1.5 flex-wrap">
          {QUALITIES.map(q => {
            const active = value.quality === q;
            const label = q === 'fast'
              ? t('trainingStudio.train.dit.qualityFast')
              : q === 'balanced'
                ? t('trainingStudio.train.dit.qualityBalanced')
                : t('trainingStudio.train.dit.qualityThorough');
            return (
              <button
                key={q}
                type="button"
                disabled={lock}
                onClick={() => pickQuality(q)}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  active
                    ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                    : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* ── Adapter name ────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          {P('adapterName')}
          <input
            type="text"
            value={value.adapterName}
            disabled={lock}
            spellCheck={false}
            onChange={(e) => onChange({ adapterName: e.target.value })}
            className={`${FIELD}${trimmed && !adapterNameOk ? ' border-red-500/50' : ''}`}
          />
          {trimmed && !adapterNameOk && (
            <span className="text-[11px] text-red-500 dark:text-red-400">
              {t('trainingStudio.train.adapterNameInvalid')}
            </span>
          )}
        </label>

        {/* ── Target loss ─────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          {P('targetLoss', 'Default 0.1 · 0 = no auto-stop')}
          <input
            type="number"
            min={0}
            max={20}
            step={0.05}
            value={value.targetLoss}
            disabled={lock}
            onChange={(e) => onChange({ targetLoss: num(e.target.value, 0.1) })}
            className={FIELD}
          />
        </label>
      </div>

      <span className="text-[11px] text-zinc-500 -mt-2">{t('trainingStudio.train.dit.targetLossHelp')}</span>

      {/* ── Max epochs ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="flex flex-col gap-1.5">
          {P('maxEpochs', 'Default 500 · 1–2000')}
          <input
            type="number"
            min={1}
            max={2000}
            step={1}
            value={value.epochs}
            disabled={lock}
            onChange={(e) => onChange({ epochs: num(e.target.value, 500) })}
            className={FIELD}
          />
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.maxEpochsHelp')}</span>
        </label>
      </div>

      {schedule && (
        <span className="text-[11px] text-zinc-500 -mt-2">
          {t('trainingStudio.train.dit.scheduleSummary', {
            steps: schedule.steps,
            effectiveBatch: schedule.effectiveBatch,
            warmup: schedule.warmup,
          })}
        </span>
      )}

      {/* ── Resume + calibration (2026-08-11) ─────────────────────────── */}
      <div className="flex flex-col gap-2 rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2.5">
        <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
          <input
            type="checkbox"
            checked={value.resumeFromLatest}
            disabled={lock}
            onChange={(e) => onChange({ resumeFromLatest: e.target.checked })}
            className="accent-amber-500"
          />
          {P('resumeFromLatest', 'Default on', CHECK_LABEL)}
        </label>
        <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
          <input
            type="checkbox"
            checked={value.calibrate}
            disabled={lock}
            onChange={(e) => onChange({ calibrate: e.target.checked })}
            className="accent-amber-500"
          />
          {P('calibrate', 'Default off', CHECK_LABEL)}
        </label>
        {value.calibrate && (
          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 pl-6">
            <input
              type="checkbox"
              checked={value.calibrateRepoint}
              disabled={lock}
              onChange={(e) => onChange({ calibrateRepoint: e.target.checked })}
              className="accent-amber-500"
            />
            {P('calibrateRepoint', 'Default on', CHECK_LABEL)}
          </label>
        )}
      </div>

      {/* ── Advanced ──────────────────────────────────────────────────── */}
      <details className="rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-zinc-600 dark:text-zinc-400 select-none">
          {t('trainingStudio.train.dit.advanced')}
        </summary>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {isLokr ? (
            <>
              <label className="flex flex-col gap-1.5">
                {P('lokrDim', 'Default 512 · 4–4096')}
                <input
                  type="number" min={4} max={4096} step={1}
                  value={value.lokrDim} disabled={lock}
                  onChange={(e) => onChange({ lokrDim: num(e.target.value, 512) })}
                  className={FIELD}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                {P('lokrAlpha', 'Default 512 (= dim, 1×) · 0–8192')}
                <input
                  type="number" min={0} max={8192} step={1}
                  value={value.lokrAlpha} disabled={lock}
                  onChange={(e) => onChange({ lokrAlpha: num(e.target.value, 512) })}
                  className={FIELD}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                {P('lokrFactor', 'Default 6 · -1 or 2–64')}
                <input
                  type="number" min={-1} max={64} step={1}
                  value={value.lokrFactor} disabled={lock}
                  onChange={(e) => onChange({ lokrFactor: num(e.target.value, 6) })}
                  className={`${FIELD}${lokrFactorOk ? '' : ' border-red-500/60'}`}
                />
                <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.lokrFactorHelp')}</span>
                {!lokrFactorOk && (
                  <span className="text-[11px] text-red-500">
                    {t('trainingStudio.train.dit.lokrFactorInvalid')}
                  </span>
                )}
              </label>

              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 sm:col-span-2">
                <input
                  type="checkbox"
                  checked={value.lokrDecomposeBoth}
                  disabled={lock}
                  onChange={(e) => onChange({ lokrDecomposeBoth: e.target.checked })}
                  className="accent-amber-500"
                />
                {P('lokrDecomposeBoth', 'Default on', CHECK_LABEL)}
              </label>
            </>
          ) : (
            <>
              <label className="flex flex-col gap-1.5">
                {P('rank', 'Default 128 · 1–256')}
                <input
                  type="number" min={1} max={256} step={1}
                  value={value.rank} disabled={lock}
                  onChange={(e) => onChange({ rank: num(e.target.value, 128) })}
                  className={FIELD}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                {P('alpha', 'Default 256 (2× at rank 128) · 1–1024')}
                <input
                  type="number" min={1} max={1024} step={1}
                  value={value.alpha} disabled={lock}
                  onChange={(e) => onChange({ alpha: num(e.target.value, 256) })}
                  className={FIELD}
                />
              </label>
            </>
          )}

          <label className="flex flex-col gap-1.5">
            {P('layers', 'Default 0 (auto) · 0–64')}
            <input
              type="number" min={0} max={64} step={1}
              value={value.layers} disabled={lock}
              onChange={(e) => onChange({ layers: num(e.target.value, 0) })}
              className={FIELD}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.layersAuto')}</span>
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.layersHelp')}</span>
          </label>

          <div className="flex flex-col gap-1.5">
            {P('crop', 'Default Auto · 128–8192 frames when pinned')}
            <div className="flex items-center gap-1.5">
              {/* Fixed width lives on a wrapper, not on the trigger: the
                  trigger already carries w-full, and two conflicting width
                  utilities resolve by stylesheet order, not class order. */}
              <div className="w-28 shrink-0">
                <StyledSelect
                  accent="amber"
                  value={cropMode}
                  disabled={lock}
                  onChange={(v) => pickCropMode(v)}
                  options={[
                    { value: 'auto' as const, label: t('trainingStudio.train.dit.cropModeAuto') },
                    { value: 'seconds' as const, label: t('trainingStudio.train.dit.cropModeSeconds') },
                    { value: 'frames' as const, label: t('trainingStudio.train.dit.cropModeFrames') },
                  ]}
                />
              </div>
              {cropMode !== 'auto' && (
                <input
                  type="number"
                  min={cropMode === 'seconds' ? 5 : 0}
                  max={cropMode === 'seconds' ? 320 : 8192}
                  step={cropMode === 'seconds' ? 1 : 2}
                  value={cropShown} disabled={lock}
                  onChange={(e) => setCropValue(e.target.value)}
                  className={`${FIELD} w-full min-w-0${cropOk ? '' : ' border-red-500/60'}`}
                />
              )}
            </div>
            <span className="text-[11px] text-zinc-500">
              {cropMode === 'auto'
                ? t('trainingStudio.train.dit.cropAuto')
                : cropMode === 'seconds'
                  ? t('trainingStudio.train.dit.cropSecondsHelp')
                  : t('trainingStudio.train.dit.cropFramesHelp')}
            </span>
            {!cropOk && (
              <span className="text-[11px] text-red-500">
                {t('trainingStudio.train.dit.cropHelp')}
              </span>
            )}
          </div>

          <label className="flex flex-col gap-1.5">
            {P('cropMin', 'Default 375 frames (≈15 s) · 128–8192')}
            <input
              type="number" min={128} max={8192} step={1}
              value={value.cropMin} disabled={lock}
              onChange={(e) => onChange({ cropMin: num(e.target.value, 375) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('cropMax', value.attnBackend === 'flash'
              ? '0 = auto — lifts to the dataset’s longest track · set a number to pin'
              : 'Default 1250 frames (≈50 s) · 128–8192')}
            <input
              type="number" min={0} max={8192} step={1}
              value={value.cropMax} disabled={lock}
              onChange={(e) => onChange({ cropMax: num(e.target.value, 0) })}
              className={`${FIELD}${cropRangeOk ? '' : ' border-red-500/50'}`}
            />
          </label>

          <span className="text-[11px] text-zinc-500 sm:col-span-2 -mt-1">
            {t('trainingStudio.train.dit.cropHelp')}
          </span>

          <label className="flex flex-col gap-1.5">
            {P('learningRate', value.optimizer === 'prodigy' ? 'Ignored under Prodigy (auto step size)' : isLokr ? 'LoKR default 0.01' : 'LoRA default 5e-4')}
            <input
              type="number" min={0.00001} max={1} step={0.00001}
              value={value.learningRate} disabled={lock}
              onChange={(e) => onChange({ learningRate: num(e.target.value, isLokr ? 0.01 : 0.0005) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('gradAccum', isLokr ? 'LoKR default 20 micro-batches · 1–64' : 'LoRA default 4 micro-batches · 1–64')}
            <input
              type="number" min={1} max={64} step={1}
              value={value.gradAccum} disabled={lock}
              onChange={(e) => onChange({ gradAccum: num(e.target.value, isLokr ? 20 : 4) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('gradClip', 'Default 1.0 · 0 = off')}
            <input
              type="number" min={0} max={100} step={0.1}
              value={value.gradClip} disabled={lock}
              onChange={(e) => onChange({ gradClip: num(e.target.value, 1) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('warmupRatio', 'Default 0.05 · 0–0.5')}
            <input
              type="number" min={0} max={0.5} step={0.01}
              value={value.warmupRatio} disabled={lock}
              onChange={(e) => onChange({ warmupRatio: num(e.target.value, 0.05) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('weightDecay', isLokr ? 'LoKR default 0.001 · 0–1' : 'LoRA default 0.01 · 0–1')}
            <input
              type="number" min={0} max={1} step={0.005}
              value={value.weightDecay} disabled={lock}
              onChange={(e) => onChange({ weightDecay: num(e.target.value, isLokr ? 0.001 : 0.01) })}
              className={FIELD}
            />
          </label>

          <div className="flex flex-col gap-1.5">
            {P('lossWeighting', isLokr ? 'LoKR default none' : 'LoRA default flow_snr')}
            <StyledSelect
              accent="amber"
              value={value.lossWeighting}
              disabled={lock}
              onChange={(v) => onChange({ lossWeighting: v })}
              options={[
                { value: 'flow_snr' as const, label: 'flow_snr' },
                { value: 'none' as const, label: 'none' },
              ]}
            />
          </div>

          <label className="flex flex-col gap-1.5">
            {P('snrGamma', 'Default 5.0 · 1–100 · flow_snr only')}
            <input
              type="number" min={1} max={100} step={0.5}
              value={value.snrGamma} disabled={lock}
              onChange={(e) => onChange({ snrGamma: num(e.target.value, 5) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('tBias', 'Default 0.5 · 0–4 · flow_snr only')}
            <input
              type="number" min={0} max={4} step={0.05}
              value={value.tBias} disabled={lock}
              onChange={(e) => onChange({ tBias: num(e.target.value, 0.5) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('timestepMu', 'Default -0.4 · -4 to 4')}
            <input
              type="number" min={-4} max={4} step={0.05}
              value={value.timestepMu} disabled={lock}
              onChange={(e) => onChange({ timestepMu: num(e.target.value, -0.4) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('timestepSigma', 'Default 1.0 · 0.01–4')}
            <input
              type="number" min={0.01} max={4} step={0.05}
              value={value.timestepSigma} disabled={lock}
              onChange={(e) => onChange({ timestepSigma: num(e.target.value, 1) })}
              className={FIELD}
            />
          </label>

          <div className="flex flex-col gap-1.5 sm:col-span-2">
            {P('tWindow', 'Default 0 → 1 (whole schedule)')}
            <div className="flex items-center gap-2">
              <input
                type="number" min={0} max={1} step={0.01}
                value={value.tMin} disabled={lock}
                onChange={(e) => onChange({ tMin: num(e.target.value, 0) })}
                className={`${FIELD} flex-1 min-w-0${tWindowOk ? '' : ' border-red-500/50'}`}
              />
              <span className="text-xs text-zinc-500">→</span>
              <input
                type="number" min={0} max={1} step={0.01}
                value={value.tMax} disabled={lock}
                onChange={(e) => onChange({ tMax: num(e.target.value, 1) })}
                className={`${FIELD} flex-1 min-w-0${tWindowOk ? '' : ' border-red-500/50'}`}
              />
            </div>
          </div>

          <label className="flex flex-col gap-1.5">
            {P('cfgRatio', 'Default 0.15 (15%) · 0–1')}
            <input
              type="number" min={0} max={1} step={0.05}
              value={value.cfgRatio} disabled={lock}
              onChange={(e) => onChange({ cfgRatio: num(e.target.value, 0.15) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('genreRatio', 'Default 30% · 0–100')}
            <input
              type="number" min={0} max={100} step={1}
              value={value.genreRatio} disabled={lock}
              onChange={(e) => onChange({ genreRatio: num(e.target.value, 30) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('seed', 'Default 42 · 0–2147483647')}
            <input
              type="number" min={0} max={2147483647} step={1}
              value={value.seed} disabled={lock}
              onChange={(e) => onChange({ seed: num(e.target.value, 42) })}
              className={FIELD}
            />
          </label>

          <div className="flex flex-col gap-1.5">
            {P('order', 'Default shuffle')}
            <StyledSelect
              accent="amber"
              value={value.order}
              disabled={lock}
              onChange={(v) => onChange({ order: v })}
              options={[
                { value: 'shuffle' as const, label: 'shuffle' },
                { value: 'fixed' as const, label: 'fixed' },
              ]}
            />
          </div>

          <label className="flex flex-col gap-1.5">
            {P('milestoneStep', 'Default 0 (off) · 0 = off')}
            <input
              type="number" min={0} max={5} step={0.05}
              value={value.milestoneStep} disabled={lock}
              onChange={(e) => onChange({ milestoneStep: num(e.target.value, 0) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('milestoneKeep', 'Default 6 · 0–64')}
            <input
              type="number" min={0} max={64} step={1}
              value={value.milestoneKeep} disabled={lock}
              onChange={(e) => onChange({ milestoneKeep: num(e.target.value, 6) })}
              className={FIELD}
            />
          </label>

          <span className="text-[11px] text-zinc-500 sm:col-span-2 -mt-1">
            {t('trainingStudio.train.dit.milestoneDiskHelp')}
          </span>

          <label className="flex flex-col gap-1.5">
            {P('vramReserve', 'Default 2048 MB · 0–16384')}
            <input
              type="number" min={0} max={16384} step={128}
              value={value.vramReserveMb} disabled={lock}
              onChange={(e) => onChange({ vramReserveMb: num(e.target.value, 2048) })}
              className={FIELD}
            />
          </label>

          <div className="flex flex-col gap-1.5">
            {P('mirror', 'Default BF16 + F32 compute · CUDA only')}
            <StyledSelect
              accent="amber"
              value={value.mirror}
              disabled={lock}
              onChange={(v) => onChange({ mirror: v })}
              options={[
                { value: 'bf16-f32' as const, label: t('trainingStudio.train.dit.mirrorBf16F32') },
                { value: 'f32' as const, label: t('trainingStudio.train.dit.mirrorF32') },
                { value: 'bf16' as const, label: t('trainingStudio.train.dit.mirrorBf16') },
              ]}
            />
          </div>

          <span className="text-[11px] text-zinc-500 sm:col-span-2 -mt-1">
            {t('trainingStudio.train.dit.mirrorHelp')}
          </span>

          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
              <input
                type="checkbox"
                checked={value.attnBackend === 'flash'}
                disabled={lock}
                // The toggle carries the crop cap with it: flash unpins it so
                // the engine can lift to full-song; exact restores the classic
                // 1250. The mirror is NOT coupled — 'bf16-f32' is the default
                // in both modes (Rob, 2026-09-02).
                onChange={(e) => onChange(e.target.checked
                  ? { attnBackend: 'flash', cropMax: 0 }
                  : { attnBackend: 'exact', cropMax: 0 })}
                className="accent-amber-500"
              />
              {P('attnBackend', 'Default on · fused attention', CHECK_LABEL)}
            </label>
            <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.train.dit.attnBackendHelp')}</span>
          </div>

          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
              <input
                type="checkbox"
                checked={value.cropJitter}
                disabled={lock}
                onChange={(e) => onChange({ cropJitter: e.target.checked })}
                className="accent-amber-500"
              />
              {P('cropJitter', 'Experimental · off', CHECK_LABEL)}
            </label>
            <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.train.dit.cropJitterHelp')}</span>
          </div>

          <div className="flex flex-col gap-1.5">
            {/* Label key is shared with the LM form, so it is not one of P()'s
                dit.* keys — but the explanation is DiT-specific (here mm is the
                default and there is no collision to warn about). */}
            <ParamLabel
              label={t('trainingStudio.train.bwd')}
              info={t('trainingStudio.train.dit.bwdInfo')}
              meta="Default mm (DiT)"
              className={LABEL}
            />
            <StyledSelect
              accent="amber"
              value={value.bwd}
              disabled={lock}
              onChange={(v) => onChange({ bwd: v })}
              options={[
                { value: 'mm' as const, label: t('trainingStudio.train.bwdMm') },
                { value: 'outprod' as const, label: t('trainingStudio.train.bwdOutprod') },
              ]}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.bwdHint')}</span>
          </div>

          <div className="flex flex-col gap-1.5">
            {P('optimizer', 'Default Prodigy (auto step size) · Muon and AdamW to compare')}
            <StyledSelect
              accent="amber"
              value={value.optimizer}
              disabled={lock}
              onChange={(v) => onChange({ optimizer: v })}
              options={[
                { value: 'prodigy' as const, label: t('trainingStudio.train.dit.optimizerProdigy') },
                { value: 'muon' as const, label: t('trainingStudio.train.dit.optimizerMuon') },
                { value: 'adamw' as const, label: t('trainingStudio.train.dit.optimizerAdamw') },
              ]}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.optimizerHint')}</span>
          </div>

          {/* Muon-only knobs. Hidden rather than disabled on the AdamW path —
              they are genuinely inert there, and a greyed-out row invites the
              question of what it would do. */}
          {value.optimizer === 'muon' && (
            <label className="flex flex-col gap-1.5">
              {P('muonLrScale', 'Default 20 · multiplies the LR for Muon params only')}
              <input
                type="number" min={0.001} max={1000} step={1}
                value={value.muonLrScale} disabled={lock}
                onChange={(e) => onChange({ muonLrScale: num(e.target.value, 20) })}
                className={FIELD}
              />
            </label>
          )}
          {value.optimizer === 'muon' && (
            <label className="flex flex-col gap-1.5">
              {P('muonNsSteps', 'Default 5 · Newton-Schulz iterations')}
              <input
                type="number" min={1} max={20} step={1}
                value={value.muonNsSteps} disabled={lock}
                onChange={(e) => onChange({ muonNsSteps: num(e.target.value, 5) })}
                className={FIELD}
              />
            </label>
          )}

          <label className="flex flex-col gap-1.5">
            {P('batch', 'Default 1 (off) · 1–16')}
            <input
              type="number" min={1} max={16} step={1}
              value={value.batch} disabled={lock}
              onChange={(e) => onChange({ batch: num(e.target.value, 1) })}
              className={FIELD}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.batchHelp')}</span>
          </label>

          <div className="flex flex-col gap-1.5">
            {P('ckptSegments', 'Default Auto · Off, or 2–32 segments')}
            <StyledSelect
              accent="amber"
              value={value.ckptSegments}
              disabled={lock}
              onChange={(v) => onChange({ ckptSegments: v })}
              options={[
                { value: 1, label: t('trainingStudio.train.dit.ckptAuto') },
                { value: 0, label: t('trainingStudio.train.dit.ckptOff') },
                { value: 2, label: '2' },
                { value: 4, label: '4' },
                { value: 8, label: '8' },
                { value: 16, label: '16' },
              ]}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.ckptSegmentsHelp')}</span>
          </div>
        </div>

        <div className="mt-3 flex flex-col gap-2">
          {P('stages')}
          <div className="flex items-center gap-4 flex-wrap">
            {ALL_STAGES.map(stage => (
              <label key={stage} className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input
                  type="checkbox"
                  checked={value.stages.includes(stage)}
                  disabled={lock}
                  onChange={(e) => toggleStage(stage, e.target.checked)}
                  className="accent-amber-500"
                />
                {stage}
              </label>
            ))}
          </div>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.targetMlp}
              disabled={lock}
              onChange={(e) => onChange({ targetMlp: e.target.checked })}
              className="accent-amber-500"
            />
            {P('targetMlp', 'Default on', CHECK_LABEL)}
          </label>
          <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.train.dit.targetMlpHelp')}</span>

          {/* ── Parameterization (2026-09-04) ── DoRA / HiRA / LoHa reshape the
              update and exclude each other; rsLoRA and LoRA+ stack with any.
              Each is gated by finite differences (T4) and a merge check, not
              by ear — docs/plans/adapter-parameterizations-roadmap.md. */}
          {!isLokr && (
            <div className="flex flex-col gap-2 rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2.5 mt-1">
              <span className="text-[11px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.paramGroup')}</span>
              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input type="checkbox" checked={value.dora} disabled={lock} className="accent-amber-500"
                  onChange={(e) => onChange({ dora: e.target.checked, hira: false, loha: false, hra: false })} />
                {P('dora', 'Default off', CHECK_LABEL)}
              </label>
              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input type="checkbox" checked={value.hira} disabled={lock} className="accent-amber-500"
                  onChange={(e) => onChange({ hira: e.target.checked, dora: false, loha: false, hra: false })} />
                {P('hira', 'Default off · merge mode only', CHECK_LABEL)}
              </label>
              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input type="checkbox" checked={value.loha} disabled={lock} className="accent-amber-500"
                  onChange={(e) => onChange({ loha: e.target.checked, dora: false, hira: false, hra: false })} />
                {P('loha', 'Default off · merge mode only', CHECK_LABEL)}
              </label>
              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input type="checkbox" checked={value.hra} disabled={lock} className="accent-amber-500"
                  onChange={(e) => onChange({ hra: e.target.checked, dora: false, hira: false, loha: false, pissa: false, rslora: false })} />
                {P('hra', 'Default off · rank = reflections, even', CHECK_LABEL)}
              </label>
              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input type="checkbox" checked={value.pissa} disabled={lock || value.dora || value.hira || value.loha} className="accent-amber-500"
                  onChange={(e) => onChange({ pissa: e.target.checked })} />
                {P('pissa', 'Default off · plain LoRA only', CHECK_LABEL)}
              </label>
              <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                <input type="checkbox" checked={value.rslora} disabled={lock} className="accent-amber-500"
                  onChange={(e) => onChange({ rslora: e.target.checked })} />
                {P('rslora', 'Default off', CHECK_LABEL)}
              </label>
              <label className="flex flex-col gap-1.5">
                {P('loraPlusRatio', 'Default 1 = off · paper 16 · AdamW/Prodigy only')}
                <input
                  type="number" min={1} max={64} step={1}
                  value={value.loraPlusRatio} disabled={lock}
                  onChange={(e) => onChange({ loraPlusRatio: Math.max(1, num(e.target.value, 1)) })}
                  className={FIELD}
                />
              </label>
              <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.dit.paramGroupHelp')}</span>
            </div>
          )}

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.channelBalance}
              disabled={lock}
              onChange={(e) => onChange({ channelBalance: e.target.checked })}
              className="accent-amber-500"
            />
            {P('channelBalance', 'Default on · needs channel_stats.json', CHECK_LABEL)}
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.stopEngine}
              disabled={lock}
              onChange={(e) => onChange({ stopEngine: e.target.checked })}
              className="accent-amber-500"
            />
            {P('stopEngine', 'Default on', CHECK_LABEL)}
          </label>
          <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.preprocess.stopEngineHelp')}</span>
        </div>
      </details>

      <button
        onClick={onStart}
        disabled={
          lock || !adapterNameOk || !cropRangeOk || !tWindowOk || !cropOk || !lrOk || !seedOk ||
          !lokrFactorOk || value.stages.length === 0
        }
        className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {starting ? <Loader2 size={15} className="animate-spin" /> : <Waves size={15} />}
        {t('trainingStudio.train.dit.start')}
      </button>
    </div>
  );
};

export default TrainDitForm;
