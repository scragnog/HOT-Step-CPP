// TrainLmForm.tsx — the LM LoRA training controls
//
// Four controls are always visible because they are the only ones a normal run
// needs: which base size, what the adapter is called, when to stop, and a
// backstop epoch cap. Everything else is Side-Step's own defaults and lives
// behind the Advanced drawer, same pattern as PreprocessOptionsForm.

import React from 'react';
import { Cpu, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import type {
  LmSize,
  TrainLmStage,
  TrainingCapabilities,
} from '../../services/trainingApi';

/** Everything the panel needs to POST, with every default already resolved. */
export interface TrainLmFormState {
  lmSize: LmSize;
  lmModel: string;          // '' = let the server pick the BF16 base for lmSize
  adapterName: string;
  targetLoss: number;
  epochs: number;
  adapterType: 'lora' | 'lokr';
  optimizer: 'adamw' | 'muon';
  muonLrScale: number;
  rank: number;
  alpha: number;
  lokrDim: number;
  lokrAlpha: number;
  lokrFactor: number;
  learningRate: number;
  gradAccum: number;
  gradClip: number;
  warmupRatio: number;
  weightDecay: number;
  maxLen: number;           // 0 = auto-fit from free VRAM
  seed: number;
  order: 'shuffle' | 'fixed';
  lossOnCot: boolean;
  milestoneStep: number;
  milestoneKeep: number;
  stages: TrainLmStage[];
  overwrite: boolean;
  stopEngine: boolean;
  /** Resume: continue from the newest non-calibrated run of this adapter name
   *  (server resolves the 'latest' sentinel; engine adopts identity).
   *  Default ON (Rob, 2026-08-12). */
  resumeFromLatest: boolean;
  /** Post-training calibration: eval old-vs-new x scales, pick under guards,
   *  bake the winner, write the score sidecar. Default OFF (Rob, 2026-08-12);
   *  it was ON from 2026-08-10. */
  calibrate: boolean;
  /** Let calibration repoint this artist's album preset(s). Default ON. */
  calibrateRepoint: boolean;
  /** Projection GEMM dtype (speed lever A). 'f32-window' is the default; note
   *  that on the LM 'bf16' is also the only route to the mul_mat backward, so
   *  this is a precision/speed trade, not a free dial. */
  weights: 'f32-window' | 'bf16';
  /** Micro-batch size (speed lever B). 1 is the shipped path. */
  batch: number | 'auto';
  /** MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch).
   *  'mm' is the fast tensor-core backward and the server default. */
  bwd: 'outprod' | 'mm';
}

export const TRAIN_LM_DEFAULTS: TrainLmFormState = {
  // 4B is the base Rob actually trains against (Rob, 2026-07-29) — 0.6B was
  // only ever the safe-for-any-GPU seed and meant every real run started with
  // a size click. The server contract's own default stays 0.6B; this is the
  // FORM seed, and stored per-stage defaults still override it.
  lmSize: '4B',
  lmModel: '',
  adapterName: '',
  // 0.1 (Rob, 2026-08-12) — was 0.2, and 2.0 before that. The server route's
  // own fallback tracks this number so a batch-pipeline run (which POSTs an
  // empty option bag) stops at the same loss a manual run does.
  // 1.5, not 0.1 (Rob, 2026-09-02): the render-scored stage ladder showed everything an LM adapter
  // can do is in by CE ~2.0, and the last leg to 0.1 only doubled the looping-plan rate.
  targetLoss: 1.5,
  epochs: 150,
  adapterType: 'lokr',
  optimizer: 'muon',
  muonLrScale: 20,
  rank: 16,
  alpha: 32,
  // dim 128, NOT the DiT's 512 (Rob, 2026-07-30). At 512 every attention site
  // is monolithic and every MLP site factorized; at 128 EVERY site factorizes
  // (the mono test is dim >= max(out_k,in_n)/2, and the smallest max on a 4B
  // Qwen3 is 512). That cuts a 4B adapter from ~420 MB to ~124 MB.
  //
  // lokrAlpha MUST TRACK lokrDim. scale = alpha_eff/dim, and LyCORIS forces
  // alpha_eff = dim only on MONOLITHIC sites — at dim 128 there are none, so
  // alpha 512 would leave every site running at scale 4.0, i.e. a silent 4x on
  // the adapter's effective learning rate. That is the same class of failure as
  // the Uber-LoKR alpha asymmetry. Keep them equal unless you mean it.
  lokrDim: 128,
  lokrAlpha: 128,
  // 6 stays. The 4B LM's hidden size is 2560 — the same as the DiT's — so
  // factor 6 yields the identical, proven splits: 2560 -> 5x512, 4096 -> 4x1024,
  // 1024 -> 4x256, 9728 -> 4x2432. Verified against a trained adapter's tensor
  // shapes, not assumed.
  lokrFactor: 6,
  learningRate: 0.0001,
  gradAccum: 2,
  gradClip: 1.0,
  warmupRatio: 0.05,
  weightDecay: 0.01,
  maxLen: 0,
  seed: 42,
  order: 'shuffle',
  lossOnCot: true,
  // Milestones OFF by default again (Rob, 2026-08-12) — they were ON at 0.1
  // from 2026-08-10 to feed the post-training calibration step with candidate
  // snapshots. Set a step > 0 to re-enable; milestone-keep caps the disk cost.
  milestoneStep: 0,
  milestoneKeep: 6,
  stages: ['extract', 'train', 'export'],
  overwrite: false,
  stopEngine: true,
  // Resume ON, calibrate OFF (Rob, 2026-08-12) — the inverse of the 2026-08-10
  // seed. Continuing the newest run is now the normal way to train: with target
  // loss at 0.1 a re-run is almost always "keep going", not "start over".
  // Resume is soft on the server: an adapter name with no previous run trains
  // from scratch rather than failing, which is what makes it safe as a default.
  resumeFromLatest: true,
  calibrate: false,
  calibrateRepoint: true,
  // batch still defaults to the shipped CLI behaviour (no --batch flag emitted).
  //
  // f32-window (Rob, 2026-07-30, final) — matching the DiT's F32 mirror.
  //
  // KNOWN AND ACCEPTED COST: on the LM this is NOT a free precision dial the way
  // the DiT's mirror is. `weights: 'bf16'` is also what reaches the mul_mat
  // backward, by rewriting ggml's out_prod nodes in place (lm-bf16.h Lever A,
  // ~1.7-1.8x on the GEMM mix), and NOTHING buys that back here — pairing
  // --bwd mm with f32-window is refused for the LM, because there the
  // transposed weight IS the F32 window, so the GEMM stays TF32 and you only
  // pay an extra cont. Expect LM training to be meaningfully slower than it was
  // under bf16; that is the trade being made, not a regression.
  weights: 'f32-window',
  batch: 1,
  // MUL_MAT activation-gradient formulation. 'outprod' here, unlike train-dit's
  // 'mm': `weights: 'bf16'` above ALREADY gives this base the mul_mat backward,
  // by rewriting ggml's out_prod nodes in place (lm-bf16.h). The two cannot be
  // combined — the rewrite asserts it found 7 out_prod nodes per segment and
  // aborts when --bwd mm leaves it none — and the server refuses the pair with
  // a 400. Choosing 'mm' here means also choosing weights 'f32-window'.
  bwd: 'outprod',
};

const ALL_STAGES: TrainLmStage[] = ['extract', 'train', 'export'];
const LM_SIZES: LmSize[] = ['0.6B', '1.7B', '4B'];
// Lever B (micro-batching) was closed by its own build gate and never written:
// ace-train refuses `--batch >1` at exit 2 and the server 400s before the job is
// queued. Offering 2/4/8/auto would be a dropdown whose every non-default value
// is a guaranteed failure — and, because the runner stops ace-server before it
// spawns ace-train, a failure that bounces the user's engine on the way. The
// control stays visible and disabled so the decision is documented where the
// question gets asked, rather than silently vanishing.
const BATCH_CHOICES: Array<number | 'auto'> = [1];

const FIELD =
  'rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500 disabled:opacity-50 disabled:cursor-not-allowed';
const LABEL = 'text-xs font-semibold text-zinc-600 dark:text-zinc-400';
/** Checkbox-row label text — same size as the row it sits in, not the field
 *  label above a box, so a hoverable checkbox caption doesn't change weight. */
const CHECK_LABEL = 'text-xs text-zinc-700 dark:text-zinc-300';

interface Props {
  capabilities: TrainingCapabilities | null;
  value: TrainLmFormState;
  onChange: (patch: Partial<TrainLmFormState>) => void;
  /** Locks every control (a job is running, or a start is in flight). */
  disabled?: boolean;
  starting?: boolean;
  onStart: () => void;
  /** The one `vram` metric event of the live/last run (4B plan §1.3). null
   *  before the engine has reported, in which case the static per-size hint
   *  stands on its own. */
  vram?: {
    mode: string; maxLen: number; estMb: number; freeMb: number;
    baseMb: number; ckptMb: number; segPeakMb: number;
  } | null;
}

export const TrainLmForm: React.FC<Props> = ({
  capabilities, value, onChange, disabled, starting, onStart, vram,
}) => {
  const { t } = useTranslation();

  const tl = capabilities?.trainLm;
  const sizes = tl?.sizes?.length ? tl.sizes : LM_SIZES;
  // Only bases whose name carries the selected size token — the same `-<size>-`
  // rule pickLmFor() uses. All three sizes are now trainable, but the base file
  // must still match the declared size: picking a 4B base at size 0.6B is
  // accepted by the route, the runner then STOPS ace-server, and only then does
  // ace-train infer the real size from the weights and exit 1.
  const sizeToken = new RegExp(`-${value.lmSize.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-`, 'i');
  const lmModels = (tl?.lmModels ?? []).filter(n => sizeToken.test(n));
  // '' means "no installed base matches this size" — the server would 400.
  const defaultForSize = tl?.defaultLmBySize?.[value.lmSize] ?? '';
  // A cold boot with ace-server unreachable leaves the CACHED model snapshot
  // empty, so defaultLmBySize is '' for every size even though the file is
  // installed. Saying "download one in the Model Manager" then is simply wrong,
  // and the POST would succeed (the route re-probes). Distinguish the two, as
  // PreprocessOptionsForm does for its own model list.
  const modelsUnknown = capabilities?.engine?.up === false && (tl?.lmModels?.length ?? 0) === 0;
  const noLmForSize = !value.lmModel && defaultForSize === '' && !modelsUnknown;
  const lock = !!disabled;

  // Mirrors the server's ADAPTER_NAME_RE. The GET path sanitises the same input
  // instead of rejecting it, so without this the panel can render a green
  // "Adapter ready" card for a directory the POST will refuse outright.
  const adapterNameOk = /^[A-Za-z0-9._-]{1,64}$/.test(value.adapterName.trim())
    && !/^\.+$/.test(value.adapterName.trim())
    && !value.adapterName.trim().startsWith('.');

  const num = (raw: string, fallback: number): number => {
    // '' -> Number('') === 0, which is finite. For every other field 0 is
    // rejected by the server, but targetLoss 0 legitimately means "no auto-stop",
    // so an emptied box would silently disable the panel's headline feature.
    if (raw.trim() === '') return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };

  const toggleStage = (stage: TrainLmStage, on: boolean) => {
    const next = ALL_STAGES.filter(s => (s === stage ? on : value.stages.includes(s)));
    onChange({ stages: next });
  };

  /** Field label + hover explanation. `key` is the i18n suffix under
   *  trainingStudio.train.*; the explanation is that key + "Info", so the two
   *  can never drift apart. `meta` is the default/range line — numbers only,
   *  which is why it is not a translated string. */
  const P = (key: string, meta?: string, className = LABEL) => (
    <ParamLabel
      label={t(`trainingStudio.train.${key}`)}
      info={t(`trainingStudio.train.${key}Info`)}
      meta={meta}
      className={className}
    />
  );

  return (
    <div className="flex flex-col gap-4">
      {/* ── Base size ─────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-1.5">
        {P('baseSize', 'Default 4B · 0.6B / 1.7B / 4B')}
        <div className="flex items-center gap-1.5">
          {sizes.map(size => {
            const active = value.lmSize === size;
            return (
              <button
                key={size}
                type="button"
                disabled={lock}
                onClick={() => onChange({ lmSize: size, lmModel: '' })}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  active
                    ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                    : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {size}
              </button>
            );
          })}
        </div>
        {value.lmSize === '4B' && (
          <span className="text-[11px] text-zinc-500">
            {t('trainingStudio.train.lmSizeHint4B')}
          </span>
        )}
        {/* §1.3: the measured line, once the engine has reported. Replaces
            nothing — the static hint above is what a user sees before a run. */}
        {vram && (
          <span className="text-[11px] text-zinc-500">
            {vram.mode === 'lowvram'
              ? t('trainingStudio.train.lmVramHintLow', {
                  est: vram.estMb, free: vram.freeMb, maxLen: vram.maxLen,
                  base: vram.baseMb, ckpt: vram.ckptMb, seg: vram.segPeakMb,
                })
              : t('trainingStudio.train.lmVramHintNaive', {
                  est: vram.estMb, free: vram.freeMb, maxLen: vram.maxLen,
                })}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* ── Adapter name ────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          {P('adapterName')}
          <div className="flex items-stretch">
            <input
              type="text"
              value={value.adapterName}
              disabled={lock}
              spellCheck={false}
              onChange={(e) => onChange({ adapterName: e.target.value })}
              className={`${FIELD} flex-1 min-w-0 rounded-r-none border-r-0${
                value.adapterName.trim() && !adapterNameOk ? ' border-red-500/50' : ''}`}
            />
            {/* Per-base layout: the size lives in the parent folder, not a
                name suffix — show where the adapter will land. */}
            <span className="flex items-center px-2.5 rounded-r-lg border border-l-0 border-zinc-300 dark:border-white/10 bg-zinc-200/60 dark:bg-white/5 text-[11px] font-mono font-semibold text-zinc-500 whitespace-nowrap">
              → lm-{value.lmSize.toLowerCase().replace('.', '')}/
            </span>
          </div>
          {value.adapterName.trim() && !adapterNameOk && (
            <span className="text-[11px] text-red-500 dark:text-red-400">
              {t('trainingStudio.train.adapterNameInvalid')}
            </span>
          )}
        </label>

        {/* ── Target loss ─────────────────────────────────────────────── */}
        <label className="flex flex-col gap-1.5">
          {P('targetLoss', 'Final target · default 1.5 (trains 2.0 → 1.5) · lower rungs loop · 0 = no auto-stop')}
          <input
            type="number"
            min={0}
            max={20}
            step={0.05}
            value={value.targetLoss}
            disabled={lock}
            onChange={(e) => onChange({ targetLoss: num(e.target.value, 1.5) })}
            className={FIELD}
          />
        </label>
      </div>

      <span className="text-[11px] text-zinc-500 -mt-2">{t('trainingStudio.train.targetLossHelp')}</span>

      {/* ── Max epochs ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="flex flex-col gap-1.5">
          {P('maxEpochs', 'Default 150 · 1–200')}
          <input
            type="number"
            min={1}
            max={200}
            step={1}
            value={value.epochs}
            disabled={lock}
            onChange={(e) => onChange({ epochs: num(e.target.value, 150) })}
            className={FIELD}
          />
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.maxEpochsHelp')}</span>
        </label>
      </div>

      {/* ── Resume + calibration (2026-08-10) ─────────────────────────── */}
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
          {t('trainingStudio.train.advanced')}
        </summary>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="flex flex-col gap-1.5 sm:col-span-2">
            {P('lmModel')}
            <StyledSelect
              accent="amber"
              value={value.lmModel}
              disabled={lock}
              onChange={(v) => onChange({ lmModel: v })}
              options={[
                { value: '', label: defaultForSize || '—' },
                ...lmModels.map(n => ({ value: n, label: n })),
              ]}
              searchPlaceholder="Filter bases…"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            {P('lm.optimizer', 'Default Muon · AdamW remains fully supported')}
            <StyledSelect
              accent="amber"
              value={value.optimizer}
              disabled={lock}
              onChange={(v) => onChange({ optimizer: v })}
              options={[
                { value: 'adamw' as const, label: t('trainingStudio.train.lm.optimizerAdamw') },
                { value: 'muon' as const, label: t('trainingStudio.train.lm.optimizerMuon') },
              ]}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.lm.optimizerHint')}</span>
          </div>

          {value.optimizer === 'muon' && (
            <label className="flex flex-col gap-1.5">
              {P('lm.muonLrScale', 'Default 20 · multiplies the LR for Muon params only')}
              <input
                type="number" min={0.001} max={1000} step={1}
                value={value.muonLrScale} disabled={lock}
                onChange={(e) => onChange({ muonLrScale: num(e.target.value, 20) })}
                className={FIELD}
              />
            </label>
          )}

          <div className="flex flex-col gap-1.5">
            {P('lm.adapterType', 'Default LoKr · LoRA remains fully supported')}
            <StyledSelect
              accent="amber"
              value={value.adapterType}
              disabled={lock}
              onChange={(v) => onChange({ adapterType: v })}
              options={[
                { value: 'lora' as const, label: t('trainingStudio.train.lm.adapterLora') },
                { value: 'lokr' as const, label: t('trainingStudio.train.lm.adapterLokr') },
              ]}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.lm.adapterTypeHint')}</span>
          </div>

          {value.adapterType === 'lokr' && (
            <label className="flex flex-col gap-1.5">
              {P('lm.lokrDim', 'Default 128 · 4–4096 · keep alpha equal to it')}
              <input
                type="number" min={4} max={4096} step={1}
                value={value.lokrDim} disabled={lock}
                onChange={(e) => onChange({ lokrDim: num(e.target.value, 128) })}
                className={FIELD}
              />
            </label>
          )}
          {value.adapterType === 'lokr' && (
            <label className="flex flex-col gap-1.5">
              {P('lm.lokrFactor', 'Default 6 · -1 or 2–64')}
              <input
                type="number" min={-1} max={64} step={1}
                value={value.lokrFactor} disabled={lock}
                onChange={(e) => onChange({ lokrFactor: num(e.target.value, 6) })}
                className={FIELD}
              />
            </label>
          )}

          {value.adapterType === 'lora' && (
          <label className="flex flex-col gap-1.5">
            {P('rank', 'Default 16 · 1–256')}
            <input
              type="number" min={1} max={256} step={1}
              value={value.rank} disabled={lock}
              onChange={(e) => onChange({ rank: num(e.target.value, 16) })}
              className={FIELD}
            />
          </label>
          )}

          {value.adapterType === 'lora' && (
          <label className="flex flex-col gap-1.5">
            {P('alpha', 'Default 32 · 1–1024')}
            <input
              type="number" min={1} max={1024} step={1}
              value={value.alpha} disabled={lock}
              onChange={(e) => onChange({ alpha: num(e.target.value, 32) })}
              className={FIELD}
            />
          </label>
          )}

          <label className="flex flex-col gap-1.5">
            {P('learningRate', 'Default 1e-4')}
            <input
              type="number" min={0} max={1} step={0.00001}
              value={value.learningRate} disabled={lock}
              onChange={(e) => onChange({ learningRate: num(e.target.value, 0.0001) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('gradAccum', 'Default 2 · 1–64')}
            <input
              type="number" min={1} max={64} step={1}
              value={value.gradAccum} disabled={lock}
              onChange={(e) => onChange({ gradAccum: num(e.target.value, 2) })}
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
            {P('weightDecay', 'Default 0.01 · 0–1')}
            <input
              type="number" min={0} max={1} step={0.005}
              value={value.weightDecay} disabled={lock}
              onChange={(e) => onChange({ weightDecay: num(e.target.value, 0.01) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            {P('maxLen', 'Default 0 (auto) · 0–16384 tokens')}
            <input
              type="number" min={0} max={16384} step={64}
              value={value.maxLen} disabled={lock}
              onChange={(e) => onChange({ maxLen: num(e.target.value, 0) })}
              className={FIELD}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.maxLenAuto')}</span>
          </label>

          {/* ── Speed levers (2026-07-28 plan §1.4) ───────────────────────
              Both ship at their default value, and the server omits the CLI
              flag entirely at that value, so leaving these alone reproduces
              today's run exactly. */}
          <div className="flex flex-col gap-1.5">
            {P('weights')}
            <StyledSelect
              accent="amber"
              value={value.weights}
              disabled={lock}
              onChange={(v) => onChange({ weights: v })}
              options={[
                { value: 'f32-window' as const, label: 'f32-window' },
                { value: 'bf16' as const, label: 'bf16' },
              ]}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.weightsHint')}</span>
          </div>

          <div className="flex flex-col gap-1.5">
            {P('bwd', 'Default outprod (LM)')}
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
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.bwdHintLm')}</span>
          </div>

          <div className="flex flex-col gap-1.5">
            {P('batch', 'Fixed at 1')}
            <StyledSelect
              accent="amber"
              value={String(value.batch)}
              disabled
              onChange={(v) => onChange({ batch: v === 'auto' ? 'auto' : Number(v) })}
              options={BATCH_CHOICES.map(b => ({ value: String(b), label: String(b) }))}
            />
            <span className="text-[11px] text-zinc-500">{t('trainingStudio.train.batchHint')}</span>
          </div>

          <label className="flex flex-col gap-1.5">
            {P('seed', 'Default 42')}
            <input
              type="number" min={0} step={1}
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
            {P('milestoneStep', 'Default 0 (off) · the best adapter is kept regardless')}
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
              checked={value.lossOnCot}
              disabled={lock}
              onChange={(e) => onChange({ lossOnCot: e.target.checked })}
              className="accent-amber-500"
            />
            {P('lossOnCot', 'Default on', CHECK_LABEL)}
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.overwrite}
              disabled={lock}
              onChange={(e) => onChange({ overwrite: e.target.checked })}
              className="accent-amber-500"
            />
            {P('reextract', 'Default off', CHECK_LABEL)}
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

      {noLmForSize && (
        <div className="px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          {t('trainingStudio.train.noLmForSize', { size: value.lmSize })}
        </div>
      )}

      {modelsUnknown && (
        <div className="px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          {t('trainingStudio.preprocess.engineDownForModels')}
        </div>
      )}

      <button
        onClick={onStart}
        disabled={lock || noLmForSize || !adapterNameOk || value.stages.length === 0}
        className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {starting ? <Loader2 size={15} className="animate-spin" /> : <Cpu size={15} />}
        {t('trainingStudio.train.start')}
      </button>
    </div>
  );
};

export default TrainLmForm;
