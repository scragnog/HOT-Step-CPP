// backends/yue2/index.ts — YuE2 backend
//
// The third registered generation backend (docs/plans/yue2/06-engine-port-
// plan.md §7, docs/plans/yue2/01-seam-design.md §4.2). Thin by design, same
// shape as backends/minimax/index.ts: the heavy lifting is C++/GGML inside
// the SAME ace-server process, so this module is a capability manifest + a
// client wrapper, with the generation path in ./generate.ts.
//
// LICENSE NOTE: YuE2 weights are CC BY-NC 4.0 (non-commercial). This backend
// must never be presented as unconditionally licensed for a commercial
// product — the picker and Model Manager carry a plain notice plus the
// upstream commercial-licence contact (gezhang@umich.edu). `licenseNotice`
// below is additive on BackendCapabilities so the UI has one place to read it
// from rather than hardcoding the string a second time.

import fs from 'fs';
import path from 'path';

import { engineReady } from '../../../engineState.js';
import { isEngineSuspended } from '../../aceEngineProcess.js';
import { getSetting, setSetting } from '../../../db/lireekDb.js';
import { listAllYue2Runs, type Yue2AdapterMeta } from '../../training/yue2Runs.js';
import { listAllYue2ArRuns } from '../../training/yue2ArRuns.js';
import { runYue2Generation } from './generate.js';
import {
  yue2Props, yue2PropsCached, yue2SelectModel, yue2Unload,
} from './client.js';
import type { Yue2AdapterRef, Yue2AdapterScales, Yue2Props, Yue2Selection } from './client.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
  GenerationArtifact,
  GenerationContext,
  GenerationOperation,
  GenerationOutcome,
  ResolvedRequest,
} from '../types.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

/** Mirrors capabilities().core.duration.max — the plan/semantic stages end on
 *  their own terminator; this is the honest v1 ceiling (§7 request fields:
 *  "duration ... max: 360, auto: true"). */
const YUE2_MAX_DURATION_SEC = 360;

/** CC BY-NC 4.0 — non-commercial only. See the header note above and the
 *  Model Manager entry for the same text. */
export const YUE2_LICENSE_NOTICE =
  'YuE2 weights are licensed CC BY-NC 4.0 (non-commercial use only). '
  + 'For a commercial license, contact the upstream authors at gezhang@umich.edu.';

const LM_TYPE_SETTING = 'yue2_lm_type';
const VAE_VARIANT_SETTING = 'yue2_vae_variant';
/** Legacy single-slot keys, from before the picker had an AR slot and a NAR
 *  slot. Read once and migrated into the per-slot keys below, then cleared —
 *  see migrateLegacyAdapterPick. */
const LM_ADAPTER_SETTING = 'yue2_lm_adapter';
const LM_ADAPTER_SCALE_SETTING = 'yue2_lm_adapter_scale';

/** Absolute path to the LoRA merged into each half, '' for none. Absolute
 *  because the engine opens the path as given (yue2-adapter.h) — see
 *  Yue2Selection. The halves are separate settings rather than one list
 *  because they are separate picks: AR and NAR share no weights, an adapter
 *  trained on one is refused on the other, and clearing one must not disturb
 *  the other. */
const SLOT_PATH_SETTING: Record<Yue2LmAdapterKind, string> = {
  ar: 'yue2_lm_adapter_ar',
  nar: 'yue2_lm_adapter_nar',
};
/** That slot's six dials, as one JSON blob rather than six more key names. */
const SLOT_SCALES_SETTING: Record<Yue2LmAdapterKind, string> = {
  ar: 'yue2_lm_adapter_ar_scales',
  nar: 'yue2_lm_adapter_nar_scales',
};

/** The order the stack is sent in, load-bearing rather than cosmetic: the
 *  engine builds its change key by walking the list it was given, so the same
 *  two adapters in the other order read as a different request and cost a full
 *  model reload. One order, defined once. */
export const YUE2_ADAPTER_SLOTS: readonly Yue2LmAdapterKind[] = ['ar', 'nar'];

const YUE2_ADAPTER_DEFAULT_SCALE = 1.0;

function yue2DefaultScales(): Yue2AdapterScales {
  return { global: YUE2_ADAPTER_DEFAULT_SCALE, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 };
}

/** A dial value the engine will accept: finite and not negative. Anything else
 *  (a blank field, a stray letter, a setting written by hand) falls back to the
 *  default rather than travelling on to become a silently wrong merge. 0 is
 *  allowed and means "do not adapt this part of the model", which the merge
 *  honours by skipping those tensors. */
function yue2Dial(raw: unknown, fallback: number): number {
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

function yue2ReadScales(key: string): Yue2AdapterScales {
  const base = yue2DefaultScales();
  const raw = getSetting(key, '');
  if (!raw) return base;
  try {
    const parsed = JSON.parse(raw) as Partial<Record<keyof Yue2AdapterScales, unknown>>;
    return {
      global: yue2Dial(parsed.global, base.global),
      attn: yue2Dial(parsed.attn, base.attn),
      mlp: yue2Dial(parsed.mlp, base.mlp),
      early: yue2Dial(parsed.early, base.early),
      mid: yue2Dial(parsed.mid, base.mid),
      late: yue2Dial(parsed.late, base.late),
    };
  } catch {
    return base;
  }
}

/** Which half an adapter file trains, from the training-run catalogue. null for
 *  a file the catalogue does not know (a hand-copied export, say) — the
 *  engine's own format gate is the real authority, and this is here only so the
 *  picker can refuse an obviously wrong slot before paying for a reload. */
function yue2AdapterKindOf(ref: string): Yue2LmAdapterKind | null {
  try {
    return yue2LmAdapterCatalogue().meta[path.resolve(ref)]?.kind ?? null;
  } catch {
    return null;
  }
}

let legacyAdapterMigrated = false;

/** Move a pre-two-slot pick into whichever slot it belongs to.
 *
 *  The old setting recorded a path and a strength and never recorded which half
 *  the file trains, so the half is re-derived from the run catalogue. A file the
 *  catalogue no longer knows falls back to NAR, the only thing the single-slot
 *  picker ever called itself.
 *
 *  Runs once per process and writes the result back, so the catalogue scan is
 *  not repeated on every read of the selection. */
function migrateLegacyAdapterPick(): void {
  if (legacyAdapterMigrated) return;
  legacyAdapterMigrated = true;
  const legacy = getSetting(LM_ADAPTER_SETTING, '');
  if (!legacy) return;
  const alreadySplit = YUE2_ADAPTER_SLOTS.some(k => getSetting(SLOT_PATH_SETTING[k], ''));
  if (!alreadySplit) {
    const kind = yue2AdapterKindOf(legacy) ?? 'nar';
    const scales = {
      ...yue2DefaultScales(),
      global: yue2Dial(getSetting(LM_ADAPTER_SCALE_SETTING, ''), YUE2_ADAPTER_DEFAULT_SCALE),
    };
    setSetting(SLOT_PATH_SETTING[kind], legacy);
    setSetting(SLOT_SCALES_SETTING[kind], JSON.stringify(scales));
    console.log(`[Backends] YuE2 adapter pick migrated into the ${kind.toUpperCase()} slot: `
      + path.basename(legacy));
  }
  setSetting(LM_ADAPTER_SETTING, '');
  setSetting(LM_ADAPTER_SCALE_SETTING, '');
}

export interface Yue2PersistedSlot {
  /** '' = base model, nothing merged into this half. */
  path: string;
  scales: Yue2AdapterScales;
}

export interface Yue2PersistedSelection {
  lm: string;
  vae_variant: string;
  adapters: Record<Yue2LmAdapterKind, Yue2PersistedSlot>;
}

/** The persisted selection. '' = auto (engine best-first / standard). */
export function yue2PersistedSelection(): Yue2PersistedSelection {
  migrateLegacyAdapterPick();
  return {
    lm: getSetting(LM_TYPE_SETTING, ''),
    vae_variant: getSetting(VAE_VARIANT_SETTING, ''),
    adapters: {
      ar: { path: getSetting(SLOT_PATH_SETTING.ar, ''), scales: yue2ReadScales(SLOT_SCALES_SETTING.ar) },
      nar: { path: getSetting(SLOT_PATH_SETTING.nar, ''), scales: yue2ReadScales(SLOT_SCALES_SETTING.nar) },
    },
  };
}

/** The engine's own key spelling for an adapter set (yue2_adapter_key):
 *  `<path>@<global>,a<attn>,m<mlp>,e<early>,i<mid>,l<late>`, every number to
 *  4dp, `; `-joined for a stack. Built here so the persisted pick can be
 *  compared against what GET /yue2/props reports without guessing at
 *  formatting.
 *
 *  This mirrors the C++ byte for byte and the two have to change together.
 *  Drift does not throw: it makes reconcileSelection believe the engine has
 *  wandered off on every poll, and re-select on every poll with it. */
function yue2AdapterKey(specs: ReadonlyArray<Yue2AdapterRef>, legacy = false): string {
  if (legacy) {
    // An engine built before the group dials renders `<path>@<scale>` and
    // nothing else. Comparing the six-dial spelling against that would read as
    // permanent drift, and reconcileSelection would re-select on every poll
    // forever. Speak the old spelling to an old engine.
    return specs.map(a => `${a.path}@${a.scales.global.toFixed(4)}`).join('; ');
  }
  return specs.map(a => `${a.path}@${a.scales.global.toFixed(4)},a${a.scales.attn.toFixed(4)}`
    + `,m${a.scales.mlp.toFixed(4)},e${a.scales.early.toFixed(4)}`
    + `,i${a.scales.mid.toFixed(4)},l${a.scales.late.toFixed(4)}`).join('; ');
}

/** Is this engine old enough to predate the group dials?
 *
 *  `synth_ready` and the dials landed in the same build, so its absence is the
 *  marker. On such an engine the extra scale_* fields are parsed and dropped:
 *  the adapter still merges at its master strength, the group dials simply do
 *  nothing until the binary is rebuilt. That is a real limitation and the UI
 *  cannot tell — but silently thrashing the selection every ten seconds on
 *  top of it would be worse. */
function yue2EngineIsPreDials(props: Yue2Props | null): boolean {
  return !!props && typeof props.synth_ready !== 'boolean';
}

/** The stack as the engine should receive it: every slot that holds a usable
 *  file, in YUE2_ADAPTER_SLOTS order. */
function yue2StackFrom(adapters: Record<Yue2LmAdapterKind, Yue2PersistedSlot>): Yue2AdapterRef[] {
  const out: Yue2AdapterRef[] = [];
  for (const kind of YUE2_ADAPTER_SLOTS) {
    const slot = adapters[kind];
    if (slot.path) out.push({ path: slot.path, scales: slot.scales });
  }
  return out;
}

/** An adapter reference the engine can actually open, or null.
 *
 *  Deliberately NOT confined to the training root: a hand-trained or copied
 *  adapter anywhere on disk is a legitimate pick, and the reference comes from
 *  the local user either way. What is refused is a reference that would fail
 *  silently later — a relative path (the engine would resolve it against
 *  ace-server's working directory, which is nobody's intent), a missing file,
 *  or something that is not a .safetensors. A bad path must fail the pick, not
 *  the next generation. */
function resolveYue2Adapter(ref: string): string | null {
  const trimmed = ref.trim();
  if (!trimmed) return null;
  if (!path.isAbsolute(trimmed)) return null;
  if (!/\.safetensors$/i.test(trimmed)) return null;
  try {
    if (!fs.statSync(trimmed).isFile()) return null;
  } catch {
    return null;
  }
  return path.normalize(trimmed);
}

const YUE2_OPERATIONS: readonly GenerationOperation[] = ['text2music'];

function firstText(submission: Readonly<Record<string, unknown>>, keys: string[]): string {
  for (const key of keys) {
    const value = submission[key];
    if (typeof value === 'string' && value) return value;
  }
  return '';
}

/** Pure descriptive snapshot. The live mapper remains mapYue2Params(). */
function resolveRequest(submission: Readonly<Record<string, unknown>>): ResolvedRequest {
  const instrumental = typeof submission.instrumental === 'boolean'
    ? submission.instrumental : undefined;
  const options: Record<string, unknown> = {};
  for (const key of Object.keys(submission)) {
    if (key.startsWith('yue2')) options[key] = submission[key];
  }
  const common = {
    caption: firstText(submission, ['prompt', 'songDescription', 'caption', 'style']),
    lyrics: instrumental === true ? '' : typeof submission.lyrics === 'string' ? submission.lyrics : '',
    ...(instrumental === undefined ? {} : { instrumental }),
    ...(typeof submission.seed === 'number' && Number.isFinite(submission.seed) ? { seed: submission.seed } : {}),
    ...(typeof submission.randomSeed === 'boolean' ? { randomSeed: submission.randomSeed } : {}),
    ...(typeof submission.title === 'string' ? { title: submission.title } : {}),
  };
  const picked = yue2PersistedSelection();
  // BackendModelSelection is string-valued, so the dials are stringified rather
  // than dropped: the snapshot records what the request was resolved against,
  // and how hard each merged adapter pushed is part of that. A slot holding
  // nothing contributes nothing.
  const pickedModels: Record<string, string> = { lm: picked.lm, vae_variant: picked.vae_variant };
  for (const kind of YUE2_ADAPTER_SLOTS) {
    const slot = picked.adapters[kind];
    if (!slot.path) continue;
    pickedModels[`lm_adapter_${kind}`] = slot.path;
    pickedModels[`lm_adapter_${kind}_scale`] = String(slot.scales.global);
    pickedModels[`lm_adapter_${kind}_scale_attn`] = String(slot.scales.attn);
    pickedModels[`lm_adapter_${kind}_scale_mlp`] = String(slot.scales.mlp);
    pickedModels[`lm_adapter_${kind}_scale_early`] = String(slot.scales.early);
    pickedModels[`lm_adapter_${kind}_scale_mid`] = String(slot.scales.mid);
    pickedModels[`lm_adapter_${kind}_scale_late`] = String(slot.scales.late);
  }
  return {
    operation: 'text2music',
    common,
    models: pickedModels,
    options,
    policy: { retry: { maxAttempts: 2, reseedOnRetry: true } },
  };
}

function outcomeFromJob(job: GenerationJob): GenerationOutcome {
  const result = job.result;
  const artifacts: GenerationArtifact[] = (result?.audioUrls ?? []).map((url, trackIndex) => ({
    kind: 'audio', trackIndex, url,
  }));
  return {
    endReason: job.status === 'succeeded' ? 'completed' : job.status === 'cancelled' ? 'cancelled' : 'failed',
    stages: result?.timing ?? [],
    artifacts,
    songIds: result?.songIds ?? [],
    result,
    error: job.error,
  };
}

/** Is the engine in a state where POST /yue2/synth would get as far as loading
 *  weights?
 *
 *  The engine answers this directly with `synth_ready`, and that is the signal
 *  to prefer. But engines built before that key existed simply omit it, and an
 *  omitted key is undefined, and undefined is not true — so reading it alone
 *  reports this backend permanently down on an older binary. That is not a
 *  cosmetic lie: the UI refuses to cache a down manifest (issue #153), so
 *  every cluster in the top bar falls back to ACE's controls for the whole
 *  session, which is exactly how this surfaced.
 *
 *  So: trust the key when the engine sends one, and otherwise derive the same
 *  answer from the file probe it has always sent — the LM found and parsed,
 *  plus at least one VAE variant the same. Same definition, one build older. */
function yue2SynthReady(props: Yue2Props | null): boolean {
  if (!props) return false;
  if (typeof props.synth_ready === 'boolean') return props.synth_ready;
  const vaeOk = props.files?.vae_standard?.found === true || props.files?.vae_legacy?.found === true;
  return props.available === true && vaeOk;
}

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (!engineReady) return 'down';
  return yue2SynthReady(yue2PropsCached()) ? 'ready' : 'down';
}

async function capabilities(): Promise<BackendCapabilities> {
  const { props, stale } = await yue2Props();
  // Self-healing restore, same as MM3's: the UI polls this, so a crash-respawn
  // that reset the in-memory selection is repaired without anyone reopening
  // the picker. Fire-and-forget — capabilities must stay fast and must never
  // fail over a residency concern.
  void reconcileSelection(props, stale);
  const synthReady = yue2SynthReady(props ?? null);
  const up = engineReady && !isEngineSuspended() && synthReady;

  // modelsMissing: honest, narrower than `!up` — true only when the engine
  // is reachable but the weight files themselves weren't found. Fail open
  // (false) on a stale/never-fetched manifest, same contract as MM3's own.
  const modelsMissing = !stale && props != null && !synthReady &&
    (props.files?.lm?.found === false
      || (props.files?.vae_standard?.found === false && props.files?.vae_legacy?.found === false));

  return {
    backend: 'yue2',
    up,
    core: {
      // auto: the AR plan/semantic stages end on their own terminator; a
      // requested length has no wire slot at all (see generate.ts's duration
      // note), so `editable: false` hides the UI control the same way MM3's
      // does.
      duration: { max: YUE2_MAX_DURATION_SEC, auto: true, editable: false },
      bpm: false,
      keyscale: false,
      negativePrompt: false,
      batch: { max: 1 },
      seed: true,
      captionFormat: 'freeform',
      timeSignature: false,
      languageMeans: 'lyrics',
      modelsMissing,
      modelsMissingHint: 'yue2-lm + yue2-vae GGUFs (~8 GB)',
      propsStale: stale,
    },
    features: {
      models: true,
      lm: false,
      plugins: false,
      samplerPlugins: false,
      adapters: false,
      // AR and NAR LoRAs trained by the Training Studio, merged at load. Not
      // ACE's DiT stack (no masking, no runtime mode, no per-request delta) —
      // hence lmAdapters and not `adapters`. There ARE group scales now, one
      // set per half.
      lmAdapters: true,
      // ...and held as ENGINE STATE: the delta is baked into the resident
      // weights, so the pick is a POST that evicts the model rather than a
      // field on the next generation request.
      lmAdapterSelectable: true,
      postProcess: true,
      stableStep: true,
      whisper: true,
      // YuE2's DiT (the NAR/flow stack) has no lyric cross-attention or
      // decode-alignment head yet — no route to per-line timestamps in v1.
      lyricTimestamps: false,
      cover: false,
      repaint: false,
      lego: false,
      extract: false,
      streaming: false,
      training: false,
      midi: false,
      stems: false,
      understand: false,
      conceptSteering: false,
      captionDatasetSource: false,
      timbreReference: false,
    },
    license: YUE2_LICENSE_NOTICE,
    extensions: [
      {
        key: 'yue2Cot',
        type: 'select',
        label: 'Chain of Thought',
        hint: 'How much of the song structure the planner writes out before generating audio. '
            + '"full" is the reference default; "off" skips the ABC plan entirely and is the '
            + 'fastest, least-structured mode (its CFG is on by default, unlike the other two).',
        default: 'full',
        options: [
          { value: 'full', label: 'Full (structure + melody)' },
          { value: 'melody', label: 'Melody only' },
          { value: 'off', label: 'Off (fastest, least structure)' },
        ],
      },
      {
        key: 'yue2CfgScale',
        type: 'text',
        label: 'CFG Scale',
        hint: 'Blank = the checkpoint\'s own default for the selected Chain of Thought mode '
            + '(1.0 for melody/full, 1.01 for off). Set a number to override it.',
        default: '',
      },
      {
        key: 'yue2OdeSteps',
        type: 'slider',
        label: 'ODE Steps',
        hint: 'Midpoint-solver steps per NAR chunk (the checkpoint default is 32; each step is '
            + 'two network evaluations). Lower is faster and less refined.',
        default: 32,
        min: 8,
        max: 64,
        step: 1,
      },
      {
        key: 'yue2VaeVariant',
        type: 'select',
        label: 'VAE Variant',
        hint: 'Which of the two shipped decoder checkpoints renders the final audio.',
        default: 'standard',
        options: [
          { value: 'standard', label: 'Standard' },
          { value: 'legacy', label: 'Legacy' },
        ],
      },
    ],
  };
}

/** Which half of the model an adapter trains. Not interchangeable: the engine
 *  gates the legal tensor-name family on the file's own `format`, so an AR
 *  adapter picked where a NAR one is wanted is REFUSED at load
 *  (yue2-adapter.h). A flat list of both halves is a pick the user cannot get
 *  right by reading it, which is why the kind travels with every entry. */
export type Yue2LmAdapterKind = 'ar' | 'nar';

/** A catalogue entry: the shared lmAdapterMeta shape plus the three things
 *  only this backend has to say. They are additive rather than pushed into
 *  BackendModels because no other backend has two halves or a style template
 *  to report; they reach the UI as JSON on the same object. */
export type Yue2LmAdapterEntry = NonNullable<BackendModels['lmAdapterMeta']>[string] & {
  kind: Yue2LmAdapterKind;
  /** The prompt shape this checkpoint was trained under — what generate.ts
   *  composes with. */
  styleTemplate?: string;
  captionDropout?: number;
};

/** What the catalogue needs from a run. Structural on purpose: the AR and NAR
 *  summaries are separate types from separate scanners (they share a directory
 *  layout and nothing else), and both satisfy this without either knowing the
 *  picker exists. */
interface Yue2CataloguedRun {
  runName: string;
  updatedAt: number;
  configuredSteps: number;
  datasetName?: string;
  trigger?: string;
  rank?: number;
  checkpoints: ReadonlyArray<{
    step: number; path: string; bytes: number; final: boolean;
    loss?: number; meta?: Yue2AdapterMeta;
  }>;
}

/** Every trained adapter on disk, both halves, newest run first, each
 *  checkpoint labelled with what its own safetensors header records.
 *
 *  The run directories are enumerated by training/yue2Runs.ts and
 *  training/yue2ArRuns.ts — the same scanners the Training Studio reads —
 *  rather than a third walk of the same tree that could disagree with them
 *  about what counts as a checkpoint. Paths are ABSOLUTE because that is what
 *  the engine needs (see Yue2Selection). */
function yue2LmAdapterCatalogue(): {
  paths: string[];
  meta: Record<string, Yue2LmAdapterEntry>;
} {
  const paths: string[] = [];
  const meta: Record<string, Yue2LmAdapterEntry> = {};
  const runs: Array<{ kind: Yue2LmAdapterKind; run: Yue2CataloguedRun }> = [
    ...listAllYue2ArRuns().map(run => ({ kind: 'ar' as const, run })),
    ...listAllYue2Runs().map(run => ({ kind: 'nar' as const, run })),
  ].sort((a, b) => b.run.updatedAt - a.run.updatedAt);

  for (const { kind, run } of runs) {
    // Newest checkpoint first within a run: the final export is what anyone
    // wants by default, and the snapshot ladder is the "it was better at 4000
    // steps" escape hatch below it.
    for (const ckpt of [...run.checkpoints].reverse()) {
      const trigger = ckpt.meta?.trigger || run.trigger || '';
      const rank = ckpt.meta?.rank ?? run.rank;
      const steps = ckpt.meta?.steps ?? (ckpt.final ? run.configuredSteps : ckpt.step);
      // Absolute, always: the adapter root is absolute by default but an
      // ACESTEPCPP_ADAPTERS override need not be, and a relative path would be
      // read by the ENGINE against ace-server's working directory.
      const abs = path.resolve(ckpt.path);
      paths.push(abs);
      meta[abs] = {
        // The half leads the label as well as riding on `kind`: a UI that has
        // not yet split the list still shows two adapters from one dataset as
        // the different things they are.
        label: [
          kind.toUpperCase(),
          run.runName,
          ckpt.final ? 'final' : `step ${ckpt.step}`,
          trigger ? `"${trigger}"` : '',
        ].filter(Boolean).join(' · '),
        kind,
        runName: run.runName,
        trigger: trigger || undefined,
        rank,
        steps: Number.isFinite(steps) && steps > 0 && steps < Number.MAX_SAFE_INTEGER ? steps : undefined,
        bytes: ckpt.bytes,
        dataset: run.datasetName,
        final: ckpt.final,
        loss: ckpt.loss,
        styleTemplate: ckpt.meta?.styleTemplate,
        captionDropout: ckpt.meta?.captionDropout,
      };
    }
  }
  return { paths, meta };
}

async function models(): Promise<BackendModels> {
  const { props } = await yue2Props();
  const v = props?.variants;

  const meta: NonNullable<BackendModels['meta']> = {};
  if (v?.lm?.available?.length) {
    meta.lm = {};
    for (const f of v.lm.available) meta.lm[f.type] = { label: f.filename, bytes: f.bytes };
  }

  const adapters = yue2LmAdapterCatalogue();
  const persisted = yue2PersistedSelection();

  return {
    buckets: {
      lm: (v?.lm?.available ?? []).map(f => f.type),
      // Fixed two-choice enum, not a disk scan — matches the extension's own
      // yue2VaeVariant select. modelsMissing/props.files is what tells the
      // user whether a given variant is actually installed.
      vae: ['standard', 'legacy'],
    },
    adapters: [],
    lmAdapters: adapters.paths,
    lmAdapterMeta: adapters.meta,
    defaults: {
      lm: v?.lm?.selected ?? '',
      vae: props?.files?.vae_standard?.found ? 'standard' : (props?.files?.vae_legacy?.found ? 'legacy' : ''),
      ...slotDefaults(persisted, props?.adapter),
      // Aggregate, kept for anything that just wants to know whether the
      // resident model is adapted at all.
      lmAdapterMerged: props?.adapter?.merged ?? '',
      lmAdapterInForce: props?.adapter?.in_force === true,
    },
    meta,
  };
}

/** Per-slot picker state: the pick, its six dials, and whether that pick is
 *  actually merged into the resident weights right now.
 *
 *  The PICK is the persisted value, never props.adapter.merged: `merged` is
 *  empty for the whole window between a selection and the next warm, and a
 *  picker that blanked itself there would read as "the choice was lost". What
 *  is in force is a separate flag, per slot, because with two slots "adapted"
 *  stopped being one answer — one half can be merged while the other is still
 *  waiting for the next load. */
function slotDefaults(
  persisted: Yue2PersistedSelection,
  adapter: { entries?: Array<{ path?: string; family?: string }>; in_force?: boolean } | undefined,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const entries = adapter?.entries ?? [];
  for (const kind of YUE2_ADAPTER_SLOTS) {
    const slot = persisted.adapters[kind];
    const Slot = kind === 'ar' ? 'Ar' : 'Nar';
    // An engine too old to send `entries` says nothing per half; fall back to
    // the aggregate rather than claiming a half is unmerged when it may not be.
    const inForce = entries.length > 0
      ? entries.some(e => e.family === kind && (!e.path || !slot.path || path.resolve(e.path) === path.resolve(slot.path)))
      : (adapter?.in_force === true && !!slot.path);
    out[`lmAdapter${Slot}`] = slot.path;
    out[`lmAdapter${Slot}Scale`] = slot.scales.global;
    out[`lmAdapter${Slot}ScaleAttn`] = slot.scales.attn;
    out[`lmAdapter${Slot}ScaleMlp`] = slot.scales.mlp;
    out[`lmAdapter${Slot}ScaleEarly`] = slot.scales.early;
    out[`lmAdapter${Slot}ScaleMid`] = slot.scales.mid;
    out[`lmAdapter${Slot}ScaleLate`] = slot.scales.late;
    out[`lmAdapter${Slot}InForce`] = !!slot.path && inForce;
  }
  return out;
}

/** Read one slot's buckets out of a selection body.
 *
 *  Three states per field, and the difference between the last two is the whole
 *  point: a key that is ABSENT leaves that slot alone, an EXPLICIT '' clears it,
 *  and a path picks it. A picker that posts only its own slot must not disturb
 *  the other one, and a caller that serialises its whole option struct must not
 *  clear an adapter it never meant to touch. */
function readSlotSelection(
  selection: Record<string, string>,
  kind: Yue2LmAdapterKind,
  persisted: Yue2PersistedSlot,
): { given: boolean; ref: string; scales: Yue2AdapterScales } {
  const Slot = kind === 'ar' ? 'Ar' : 'Nar';
  const pathKey = `lmAdapter${Slot}`;
  const given = typeof selection[pathKey] === 'string';
  const dial = (suffix: string, fallback: number) =>
    selection[`lmAdapter${Slot}Scale${suffix}`] === undefined
      ? fallback
      : yue2Dial(selection[`lmAdapter${Slot}Scale${suffix}`], fallback);
  return {
    given,
    ref: given ? selection[pathKey].trim() : persisted.path,
    scales: {
      global: selection[`lmAdapter${Slot}Scale`] === undefined
        ? persisted.scales.global
        : yue2Dial(selection[`lmAdapter${Slot}Scale`], persisted.scales.global),
      attn: dial('Attn', persisted.scales.attn),
      mlp: dial('Mlp', persisted.scales.mlp),
      early: dial('Early', persisted.scales.early),
      mid: dial('Mid', persisted.scales.mid),
      late: dial('Late', persisted.scales.late),
    },
  };
}

async function selectModel(selection: Record<string, string>) {
  const persisted = yue2PersistedSelection();
  const sel: Yue2Selection = {
    lm: selection.lm ?? persisted.lm,
    vae_variant: (selection.vae as Yue2Selection['vae_variant']) ?? (persisted.vae_variant as Yue2Selection['vae_variant']),
  };

  // A pre-two-slot caller sending the old flat `lmAdapter` still works: route
  // it to whichever half the catalogue says the file trains. An unknown file
  // goes to NAR, which is what the single-slot picker always meant.
  const routed: Record<string, string> = { ...selection };
  if (typeof selection.lmAdapter === 'string') {
    const ref = selection.lmAdapter.trim();
    const kind = ref ? (yue2AdapterKindOf(ref) ?? 'nar') : 'nar';
    const Slot = kind === 'ar' ? 'Ar' : 'Nar';
    if (routed[`lmAdapter${Slot}`] === undefined) routed[`lmAdapter${Slot}`] = ref;
    if (selection.lmAdapterScale !== undefined && routed[`lmAdapter${Slot}Scale`] === undefined) {
      routed[`lmAdapter${Slot}Scale`] = selection.lmAdapterScale;
    }
    // An explicit clear on the legacy key clears BOTH halves: the old picker
    // had one "None", and it meant no adapter at all.
    if (!ref) {
      for (const k of YUE2_ADAPTER_SLOTS) {
        const S = k === 'ar' ? 'Ar' : 'Nar';
        if (routed[`lmAdapter${S}`] === undefined) routed[`lmAdapter${S}`] = '';
      }
    }
  }

  const next: Record<Yue2LmAdapterKind, Yue2PersistedSlot> = {
    ar: { path: '', scales: persisted.adapters.ar.scales },
    nar: { path: '', scales: persisted.adapters.nar.scales },
  };
  let anyGiven = false;

  for (const kind of YUE2_ADAPTER_SLOTS) {
    const want = readSlotSelection(routed, kind, persisted.adapters[kind]);
    anyGiven = anyGiven || want.given;
    next[kind].scales = want.scales;
    if (!want.ref) continue;

    const resolved = resolveYue2Adapter(want.ref);
    if (!resolved) {
      if (want.given) {
        // Fail the pick loudly (the route answers 400). Merging nothing while
        // the UI shows an adapter is exactly the silent failure this whole path
        // exists to avoid.
        throw new Error(`YuE2 ${kind.toUpperCase()} adapter not usable: ${want.ref} `
          + '(needs an absolute path to an existing .safetensors)');
      }
      // A persisted pick whose file has since moved or been deleted: clear it
      // rather than wedge every later selection on a 400.
      console.warn(`[Backends] YuE2 ${kind.toUpperCase()} adapter gone, clearing persisted pick: ${want.ref}`);
      continue;
    }
    // Wrong-half guard. The engine refuses this at load anyway, on the file's
    // own __metadata__.format — but by then the model has already been torn
    // down, so the refusal costs a reload and reads as "generation broke".
    // Catch it while it is still only a failed click.
    const declared = yue2AdapterKindOf(resolved);
    if (want.given && declared && declared !== kind) {
      throw new Error(`That adapter trains the ${declared.toUpperCase()} half, `
        + `so it cannot go in the ${kind.toUpperCase()} slot (the engine would refuse it at load).`);
    }
    next[kind].path = resolved;
  }

  const wantStack = yue2StackFrom(next);
  const haveStack = yue2StackFrom(persisted.adapters);
  // Say nothing about adapters when the caller said nothing and we hold
  // nothing: that is the omitted case the engine's omitted-vs-null distinction
  // exists for, and it keeps a VAE-only POST from disturbing a pick made
  // out-of-band.
  if (anyGiven || haveStack.length > 0) {
    sel.lm_adapter = wantStack;
  }

  // `changed` isn't part of the engine's own response (yue2_handle_select_model
  // returns {selected, vae_variant, lm_type_want, lm_file, lm_found} — no
  // `changed`/`lm` field, unlike mm3SelectModel's shape); EngineBackend's
  // interface requires it, so it's derived here from the persisted values.
  const changed = (sel.lm ?? '') !== persisted.lm
    || (sel.vae_variant ?? '') !== persisted.vae_variant
    || yue2AdapterKey(wantStack) !== yue2AdapterKey(haveStack);

  const result = await yue2SelectModel(sel);
  setSetting(LM_TYPE_SETTING, sel.lm ?? '');
  setSetting(VAE_VARIANT_SETTING, sel.vae_variant ?? '');
  // Persist only after the engine accepted it — a refused pick that was
  // written back would be replayed on every boot from then on.
  for (const kind of YUE2_ADAPTER_SLOTS) {
    setSetting(SLOT_PATH_SETTING[kind], next[kind].path);
    setSetting(SLOT_SCALES_SETTING[kind], JSON.stringify(next[kind].scales));
  }
  if (changed) {
    const named = wantStack.length
      ? wantStack.map(a => path.basename(a.path)).join(' + ')
      : '(none)';
    console.log(`[Backends] YuE2 models: lm_type=${result.lm_type_want || '(auto)'} vae_variant=${result.vae_variant}`
      + ` lm_found=${result.lm_found} lm_adapter=${named}`);
  }
  return { ...result, changed };
}

// ── Persisted selection replay ──────────────────────────────────────────────

/** Push the persisted selection back into the engine if it has drifted.
 *
 *  Same contract as MM3's reconcileSelection(), for the same reason: the
 *  engine's own `requested` fields reset to '' when ace-server restarts, which
 *  makes them an exact drift signal rather than a guess. Without this a
 *  restart silently drops the adapter pick and the next generation renders the
 *  base model while the picker still shows the adapter — the MM3 trap
 *  (project-mm3-selection-engine-only), one family over.
 *
 *  Idempotent, so it is safe on both a cold boot and a crash-respawn. */
async function reconcileSelection(props: Yue2Props | null, stale: boolean): Promise<void> {
  // A stale manifest means the props probe timed out, which nearly always
  // means a generation holds the engine mutex. Re-selecting then would block
  // and could evict weights out from under the running job.
  if (stale || !props) return;

  const want = yue2PersistedSelection();
  // Only slots whose file is still on disk. A pick that has been deleted since
  // must not be replayed: the engine would refuse the load, and the refusal
  // would land on the next generation rather than here.
  const stack: Yue2AdapterRef[] = [];
  for (const kind of YUE2_ADAPTER_SLOTS) {
    const slot = want.adapters[kind];
    if (!slot.path) continue;
    const resolved = resolveYue2Adapter(slot.path);
    if (resolved) stack.push({ path: resolved, scales: slot.scales });
  }
  if (!want.lm && !want.vae_variant && stack.length === 0) return;  // never chosen — the engine default is right

  // Only ask for an LM type the engine can actually see; a file deleted since
  // the choice was made must fall back, not wedge the poll on a 400.
  const lmSeen = !want.lm
    || (props.variants?.lm?.available?.some(f => f.type === want.lm) ?? false);
  const targetLm = lmSeen ? want.lm : '';

  const lmDrifted = targetLm !== (props.variants?.lm?.requested ?? '');
  const legacyKey = yue2EngineIsPreDials(props);
  const adapterDrifted = yue2AdapterKey(stack, legacyKey) !== (props.adapter?.requested ?? '');
  if (!lmDrifted && !adapterDrifted) return;

  const sel: Yue2Selection = { lm: targetLm };
  if (want.vae_variant === 'standard' || want.vae_variant === 'legacy') {
    sel.vae_variant = want.vae_variant;
  }
  // Only speak about the adapter when there is something to say: replaying the
  // stack we hold, or clearing one the engine has but we do not.
  if (stack.length > 0) {
    sel.lm_adapter = stack;
  } else if (props.adapter?.requested) {
    sel.lm_adapter = null;
  }

  try {
    await yue2SelectModel(sel);
    const named = stack.length ? stack.map(a => path.basename(a.path)).join(' + ') : '(none)';
    console.log('[Backends] YuE2 restored persisted models:'
      + ` lm_type=${targetLm || '(auto)'} lm_adapter=${named}`);
  } catch (err: any) {
    // Advisory: a failed restore leaves the engine on its defaults, which
    // still generate. Logging beats throwing out of a capability poll.
    console.warn('[Backends] YuE2 selection restore failed:', err?.message || err);
  }
}

/** Called once when the engine reports ready, and again after every engine
 *  restart (server/src/index.ts), so the persisted choice is in force before
 *  the first generation rather than after the UI happens to poll. */
export async function restoreYue2Selection(): Promise<void> {
  const { props, stale } = await yue2Props();
  await reconcileSelection(props, stale);
}

export const yue2Backend: EngineBackend = {
  id: 'yue2',
  displayName: 'YuE2',
  resourcePool: 'gpu',
  lifecycle: {
    // No process of its own — a model family inside ace-server, same as MM3.
    async start() { /* no separate process — ace-server owns the lifecycle */ },
    async stop() { await yue2Unload(); },
    status,
  },
  capabilities,
  models,
  selectModel,
  operations: YUE2_OPERATIONS,
  resolveRequest,
  async generate(job: GenerationJob, ctx: GenerationContext): Promise<GenerationOutcome> {
    await runYue2Generation(job, {
      attempt: ctx.attempt,
      pollUntilDone: ctx.pollUntilDone,
      signal: ctx.signal,
    });
    return outcomeFromJob(job);
  },
  // v1 choice per docs/plans/yue2/06-engine-port-plan.md §7 Residency: YuE2
  // has no engine-side cross-family arbitration code (unlike MM3's
  // mm3_arbitrate_vram), so the shared Node eviction runner handles it —
  // same value and same reasoning as ACE's.
  arbitratesResidencyInEngine: false,
  /** Model-residency arbitration: frees YuE2's ~8 GB when the active backend
   *  switches away, so it never sits in VRAM next to another family. No
   *  keepCaches concept to preserve (no AR-cache analog in v1). */
  async releaseVram() {
    const r = await yue2Unload();
    if (r?.unloaded) {
      console.log(`[Backends] YuE2 unloaded (${(r.freed_mb ?? 0).toFixed(0)} MB freed)`);
    }
  },
};
