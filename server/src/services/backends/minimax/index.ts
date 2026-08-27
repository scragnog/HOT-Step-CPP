// backends/minimax/index.ts — MiniMax-Music3 backend
//
// The second registered generation backend (plan §4.3). Thin by design: the
// heavy lifting is C++/GGML inside the SAME ace-server process as the ACE
// pipeline, so this module is a capability manifest + a client wrapper, with
// the generation path in ./generate.ts.
//
// LICENSE NOTE (plan §1.5): the MiniMax-Music3 Community License requires a
// commercial product to prominently display the string "MiniMax-Music3" in its
// UI. `displayName` below IS that display — the backend toggle renders it
// verbatim. Do not shorten, prettify, or localize it.

import { engineReady } from '../../../engineState.js';
import { isEngineSuspended } from '../../aceEngineProcess.js';
import { getSetting, setSetting } from '../../../db/lireekDb.js';
import { listMm3Planks } from './plank.js';
import { listMm3LmAdapters } from './lmAdapter.js';
import { mm3Props, mm3PropsCached, mm3SelectModel, mm3Unload } from './client.js';
import type { Mm3Props, Mm3RoleVariants } from './client.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
} from '../types.js';

/** Duration ceiling we expose. The checkpoint's own cap is 9,000 frames
 *  (= 360 s @ 25 fps) and the engine clamps to it; 300 s is the honest
 *  user-facing limit for v1 (the model card's "5 minutes"). */
const MM3_MAX_DURATION_SEC = 300;

// ── Persisted quant selection ───────────────────────────────────────────────
//
// The engine holds the chosen quant in memory only, so a restart (or a
// crash-respawn) silently reverts to best-first — f16 — and the user's pick is
// lost. ACE doesn't have this problem because its model names ride on every
// generate call; MM3's selection is engine STATE, so somebody has to remember
// it and put it back.
//
// Stored in the same generic settings table backends/registry.ts uses for the
// active backend id: no new mechanism, no schema change. Server-side rather
// than in localStorage deliberately — the engine must be restored even when no
// browser has been opened.
// One setting per role since the 5-way model split. The legacy
// 'mm3_synth_quant' key seeds the dit/depth roles on first boot after the
// upgrade (cond/voc stay auto: they are near-native-only), then per-role
// settings take over.
const LM_SETTING           = 'mm3_lm_quant';
const LEGACY_SYNTH_SETTING = 'mm3_synth_quant';
const MM3_ROLES = ['depth', 'cond', 'dit', 'voc'] as const;
type Mm3Role = (typeof MM3_ROLES)[number];
const ROLE_SETTING: Record<Mm3Role, string> = {
  depth: 'mm3_depth_quant',
  cond:  'mm3_cond_quant',
  dit:   'mm3_dit_quant',
  voc:   'mm3_voc_quant',
};

/** The persisted per-role selection, with one-time migration from the legacy
 *  single-synth setting. */
function persistedSelection(): { lm: string } & Record<Mm3Role, string> {
  const legacy = getSetting(LEGACY_SYNTH_SETTING, '');
  const roleOf = (role: Mm3Role): string => {
    const v = getSetting(ROLE_SETTING[role], '');
    if (v) return v;
    // Legacy seed: the old bundle token maps to the two roles that ship a
    // matching quant ladder. cond/voc stay '' (auto → their near-native file).
    return legacy && (role === 'dit' || role === 'depth') ? legacy : '';
  };
  return {
    lm:    getSetting(LM_SETTING, ''),
    depth: roleOf('depth'),
    cond:  roleOf('cond'),
    dit:   roleOf('dit'),
    voc:   roleOf('voc'),
  };
}

/** Push the persisted selection back into the engine if it has drifted.
 *
 *  `variants.<role>.requested` is the engine's own record of what it was last
 *  asked for, and it resets to '' when the process restarts — which makes it an
 *  exact drift signal rather than a guess. Idempotent and self-healing: safe to
 *  call on a timer, and it repairs a crash-respawn as well as a cold boot. */
async function reconcileSelection(props: Mm3Props | null, stale: boolean): Promise<void> {
  // A stale manifest means the props probe timed out, which nearly always means
  // a generation is holding the engine mutex. Re-selecting then would block and
  // then evict weights out from under the very next job — never reconcile on a
  // guess.
  if (stale || !props?.variants) return;

  const want = persistedSelection();
  if (!want.lm && MM3_ROLES.every(r => !want[r])) return;  // never chosen — engine default is right

  // Only ask for a quant the engine can actually see; a file deleted since the
  // choice was made must fall back, not wedge every capability poll on a 400.
  const variantsOf = (key: 'lm' | Mm3Role) => props.variants?.[key];
  const has = (key: 'lm' | Mm3Role, q: string) =>
    !q || (variantsOf(key)?.available?.some(f => f.quant === q) ?? false);

  const target = {
    lm:    has('lm', want.lm) ? want.lm : '',
    depth: has('depth', want.depth) ? want.depth : '',
    cond:  has('cond', want.cond) ? want.cond : '',
    dit:   has('dit', want.dit) ? want.dit : '',
    voc:   has('voc', want.voc) ? want.voc : '',
  };

  // Engines predating the split have no per-role variants; skip rather than
  // send a per-role body they would misread.
  if (!props.variants.dit) return;

  const drifted = (['lm', ...MM3_ROLES] as const)
    .some(k => target[k] !== (variantsOf(k)?.requested ?? ''));
  if (!drifted) return;

  try {
    const r = await mm3SelectModel(target);
    if (r.changed) {
      console.log(`[Backends] MiniMax-Music3 restored persisted models: ${r.lm} + ${r.dit ?? r.synth}`);
    }
  } catch (err: any) {
    // Advisory: a failed restore leaves the engine on its default, which still
    // generates. Logging beats throwing out of a capability poll.
    console.warn('[Backends] MiniMax-Music3 selection restore failed:', err?.message || err);
  }
}

/** Called once when the engine reports ready (server/src/index.ts), so the
 *  persisted choice is in force before the first generation rather than after
 *  the UI happens to poll. */
export async function restoreMm3Selection(): Promise<void> {
  const { props, stale } = await mm3Props();
  await reconcileSelection(props, stale);
}

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (!engineReady) return 'down';
  // Cached-only: status() is called from route handlers and must never block
  // behind an in-flight MM3 generation (GET /mm3/props holds the MM3 mutex).
  const props = mm3PropsCached();
  return props?.synth_ready ? 'ready' : 'down';
}

/** The saved planks as select options, newest first. Built here rather than
 *  held as a static schema so a plank saved on the last render shows up on the
 *  next capabilities poll without a restart. Never throws — a missing or
 *  unreadable plank directory just means "None" is the only choice. */
function mm3PlankOptions(): { value: string; label: string }[] {
  const none = { value: '', label: 'None — generate a fresh AR plan' };
  return [none, ...listMm3Planks().map(p => {
    const when = typeof p.created === 'string' ? p.created.slice(0, 16).replace('T', ' ') : '';
    const cap = p.caption ? p.caption.slice(0, 40) : p.file.replace('.mm3plank', '').slice(0, 8);
    const frames = Number.isFinite(p.frames) ? `${p.frames}f` : '';
    return { value: p.file, label: [when, cap, frames].filter(Boolean).join(' · ') };
  })];
}

async function capabilities(): Promise<BackendCapabilities> {
  const { props, stale } = await mm3Props();
  // Self-healing restore: the UI polls this, so an engine crash-respawn (which
  // resets the in-memory selection) is repaired without anyone touching the
  // dropdown. Fire-and-forget — capabilities must stay fast and never fail
  // because of a residency concern.
  void reconcileSelection(props, stale);
  // stale === true means the props probe timed out — nearly always "an MM3
  // generation is running and holds the engine mutex", which is the opposite
  // of down. Keep the last-known-good answer rather than flapping to false.
  const synthReady = props?.synth_ready === true;
  const up = engineReady && !isEngineSuspended() && synthReady;

  // modelsMissing: honest, narrower than `!up` — true only when the engine IS
  // reachable but the weight files themselves weren't found (as opposed to
  // the engine being down/suspended, or files found-but-corrupt). This is
  // what the UI needs to decide "point the user at the Model Manager" vs
  // "the engine hasn't started yet". `found` is per-file (see Mm3PropsFile);
  // fail open (false) on a stale/never-fetched manifest — a probe timeout
  // must never be misread as "go nag the user to download 24 GB again".
  const modelsMissing = !stale && props != null && !synthReady &&
    (props.files?.lm?.found === false || props.files?.synth?.found === false ||
     props.files?.depth?.found === false || props.files?.cond?.found === false ||
     props.files?.dit?.found === false || props.files?.voc?.found === false);

  return {
    backend: 'minimax-m3',
    up,
    core: {
      // auto: the planner LM emits EOS when the song is over and the render
      // stops there, so a duration is a ceiling rather than a target — and the
      // LM never sees it either way (it is not in the assembled prompt).
      // Asking for nothing means "end it where you think it ends".
      duration: { max: MM3_MAX_DURATION_SEC, auto: true },
      // MM3 takes no structured musical metadata — tempo/key live inside the
      // Structured Caption prose, not as fields.
      bpm: false,
      keyscale: false,
      // No negative/uncond prompt on the wire: the flow stage's uncond pass is
      // zeros-conditioning, fixed by the checkpoint.
      negativePrompt: false,
      // v1 generates one take per job (the engine has no batch axis exposed).
      batch: { max: 1 },
      seed: true,
      // Non-standard extras, honestly reported (BackendCoreCapabilities is an
      // open type). The UI ignores what it doesn't know.
      promptTokenLimit: props?.prompt_token_limit ?? 5000,
      maxAudioFrames: props?.max_audio_frames_limit ?? 9000,
      sampleRate: 44100,
      propsStale: stale,
      modelsMissing,
    },
    // Everything except `models` is false. This is the honest v1 manifest:
    // none of these subsystems exist for MM3 — they are not "coming soon"
    // flags, they gate UI regions. `models` IS true: MM3 ships a quant ladder
    // (mm3-{lm,synth}-<quant>.gguf) and the picker is live.
    features: {
      models: true,
      lm: false,
      plugins: false,
      // The one subsystem on this list that stopped being false. MM3's flow DiT
      // now runs the SAME Lua solvers/schedulers/guidance as ACE, via the
      // convention bridge in engine minimax/mm3-plugins.h — no plugin was
      // modified and no plugin API was widened.
      //
      // `plugins` stays false on purpose: it picks ACE's Generation dropdown,
      // and MM3 needs the generic one for its steps/cfg extension knobs. The
      // plugin controls render alongside those, gated on THIS flag.
      //
      // Still opt-in per request (params.mm3SamplerPlugins): the picks are
      // shared global UI state, so defaulting them on would move every render
      // off the parity-proven native flow loop without anyone asking.
      samplerPlugins: true,
      adapters: false,
      // Runtime LM LoRAs (engine mm3-lm-adapter.h): a picker + strength dials
      // for the ARTIST adapters, not ACE's DiT adapter stack. Deliberately a
      // flag of its own rather than `adapters: true` — that one gates ACE's
      // whole stack UI (merge/runtime modes, per-section masking, trigger
      // embedding, group scales for a DiT), none of which exists here. The
      // catalogue is GET /api/mm3/lm-adapters; the request fields are
      // params.mm3LmAdapter + mm3LmAdapterScale* (backends/minimax/generate.ts).
      lmAdapters: true,
      // Model-agnostic post stages: the VST chain reads the rate from the WAV,
      // mastering reads and writes it, and SA3 is natively 44.1 kHz — none of
      // the three assume ACE's 48 kHz output. PP-VAE and Spectral Lifter stay
      // excluded (ACE-VAE coupled) and ride the `plugins` flag above.
      postProcess: true,
      stableStep: true,
      // Whisper needs nothing but a rendered file — whisper-cli resamples
      // internally, so it is as valid here as on ACE.
      whisper: true,
      // LRC timestamps. ACE reads these from its DiT's lyric cross-attention;
      // MM3's DiT has none and never sees lyrics, so they come instead from
      // the LM's own decode attention over the lyric span (engine
      // minimax/mm3-align.h). Line-level, ~1 s — the granularity ACE ships.
      lyricTimestamps: true,
      cover: false,
      repaint: false,
      lego: false,
      extract: false,
      // STORM-style continuous streaming (an endless chain of independent
      // generations) — genuinely not supported. NOT the same thing as the
      // `mm3Stream` extension below, which plays ONE render's windows as they
      // finish; that is gated by the extension toggle, not by this flag.
      streaming: false,
      // Native MM3 LM LoRA training shipped 2026-08-20 (ace-train mm3-codes +
      // mm3-lm-train, wired as Training Studio job kinds). Nothing reads this
      // flag today, but a manifest that says `false` about a feature the
      // backend has is a lie waiting to mislead someone.
      training: true,
      midi: false,
      stems: false,
      understand: false,
      conceptSteering: false,
    },
    // cfg_flow / steps were held back as "checkpoint-fixed sampling contract".
    // That was the right call for a bring-up, but the flow stage is ~56% of
    // wall time on a full-length track, and `steps` is a direct linear dial on
    // it — the single most useful speed/quality control MM3 has. Rendered
    // generically by the existing PluginControls schema renderer.
    extensions: [
      {
        key: 'mm3Steps',
        type: 'slider',
        label: 'Flow Steps',
        hint: 'Euler steps per window. The checkpoint default is 30; lower is '
            + 'proportionally faster (the flow stage dominates long renders). '
            + 'Below 30, Low-Step Compensation reshapes the schedule to keep the '
            + 'stereo image and low end intact — without it, low step counts go '
            + 'thin and phasey rather than merely soft. Under 8 steps is '
            + 'experimental territory.',
        default: 30,
        min: 2,
        max: 60,
        step: 1,
      },
      {
        key: 'mm3CfgFlow',
        type: 'slider',
        label: 'Flow Guidance (CFG)',
        hint: 'Checkpoint default 1.7. Higher follows the caption harder; '
            + 'lower is looser. 1.0 disables guidance.',
        default: 1.7,
        min: 1.0,
        max: 5.0,
        step: 0.1,
      },
      {
        key: 'mm3LmRepPenalty',
        type: 'slider',
        label: 'LM Repetition Penalty',
        hint: 'Breaks verbatim code loops in the planner — the "same riff forever" '
            + 'failure, which sharpened adapters make likelier. 1.0 = off (the '
            + 'reference recipe); 1.05-1.15 is the useful range. Uses the DRY mode '
            + 'by default: only codes that would EXTEND a verbatim recent cycle are '
            + 'punished, so ordinary musical restatement is untouched.',
        default: 1.0,
        min: 1.0,
        max: 1.5,
        step: 0.01,
      },
      {
        key: 'mm3LmRepMode',
        type: 'select',
        label: 'Repetition Mode',
        hint: 'dry: annihilates verbatim loops only (recommended). frequency: '
            + 'penalises codes by how often they recurred. presence: flat penalty '
            + 'on anything recent — bluntest, also flattens legitimate restatement.',
        default: 'dry',
        options: [
          { value: 'dry', label: 'DRY (verbatim loops only)' },
          { value: 'frequency', label: 'Frequency' },
          { value: 'presence', label: 'Presence' },
        ],
      },
      {
        key: 'mm3LmRepWindow',
        type: 'slider',
        label: 'Repetition Window',
        hint: 'How far back the penalty looks, in 25fps semantic frames. '
            + '320 = 12.8 s. (The ACE LM used 64 at 5 Hz — same duration.)',
        default: 320,
        min: 25,
        max: 2000,
        step: 25,
      },
      {
        key: 'mm3LmTemperature',
        type: 'slider',
        label: 'LM Temperature',
        hint: 'Sampling temperature on the planner. The reference recipe has '
            + 'none (1.0). Below 1 is safer/more repetitive; above 1 is wilder. '
            + 'Small moves — 0.9 or 1.1 — are already audible.',
        default: 1.0,
        min: 0.5,
        max: 1.6,
        step: 0.05,
      },
      {
        key: 'mm3LmTopK',
        type: 'slider',
        label: 'LM Top-K',
        hint: '0 = the checkpoint\'s own 50. Lower narrows the planner to safer '
            + 'choices; higher lets rarer codes through.',
        default: 0,
        min: 0,
        max: 500,
        step: 5,
      },
      {
        key: 'mm3LmTopP',
        type: 'slider',
        label: 'LM Top-P',
        hint: 'Nucleus sampling over the top-K survivors. 0 = off (reference). '
            + '0.9-0.95 trims the improbable tail adaptively.',
        default: 0,
        min: 0,
        max: 1,
        step: 0.01,
      },
      {
        // Backend-NEUTRAL key: BackendGenerationDropdown reads it to decide
        // whether to render the shared solver/scheduler/guidance pickers, and
        // that component must stay free of MM3-specific names. Any future
        // backend claiming features.samplerPlugins declares the same key.
        key: 'samplerPluginsEnabled',
        type: 'toggle',
        label: 'Sampler Plugins',
        hint: 'Drive the flow sampler with the shared Lua solver/scheduler/guidance '
            + 'plugins instead of MiniMax-Music3\'s own Euler loop. Off is the tested '
            + 'default; on is experimental — the plugins were written and tuned against '
            + 'the ACE-Step DiT, and stochastic solvers in particular can disturb the '
            + 'window seams this backend stitches across.',
        default: false,
      },
      {
        key: 'mm3FlowShift',
        type: 'slider',
        label: 'Schedule Shift',
        hint: 'Timestep warp passed to a scheduler plugin. 1.0 matches MiniMax-Music3\'s '
            + 'own schedule; higher spends more steps near the noisy end, where '
            + 'global structure and stereo coherence are decided. Left at 1.0 this '
            + 'defers to Low-Step Compensation; move it and your value wins outright.',
        default: 1.0,
        min: 0.5,
        max: 20.0,
        step: 0.05,
      },
      {
        // Opt-OUT rather than opt-in. MM3's schedule is uniform/shift=1 and the
        // checkpoint declares steps=30, so every sub-30 render was already
        // paying this cost silently — the toggle only decides whether it is
        // absorbed or handed to the user. Default ON is the honest default.
        key: 'mm3AutoLowStep',
        type: 'toggle',
        label: 'Low-Step Compensation',
        hint: 'Below 30 steps, run the flow on the linear schedule with '
            + 'flow_shift = 29/(steps-1), so the noisy end of the trajectory keeps '
            + 'the resolution it gets at 30 steps. Without it, low step counts lose '
            + 'stereo coherence and low end rather than just detail (measured at 10 '
            + 'steps: L/R correlation -0.07 vs +0.77, and -8 dB at 60 Hz). Solver and '
            + 'guidance stay native. No effect at 30 steps or above.',
        default: true,
      },
      {
        // ── MM3 Plank ──
        // Read the hint carefully before believing this is a speed feature: it
        // is not. The AR loop still runs every per-frame forward pass on a
        // replay. What it buys is an IDENTICAL semantic bed, so that comparing
        // two solver/scheduler/guidance settings compares those settings rather
        // than two different AR samplings.
        key: 'mm3SaveArCodes',
        type: 'toggle',
        label: 'Save AR Plan (Plank)',
        hint: 'Save this render\'s AR planner output so it can be replayed later. '
            + 'Costs nothing to enable — the codes already exist in memory.',
        default: false,
      },
      {
        // ── AR cache ──
        // The speed feature the plank is NOT. Default ON: the engine only
        // fills the slot when a render happens, and the whole point is that
        // iterating on flow settings should not re-plan.
        //
        // NOTE the default must be mirrored in generate.ts as `!== false` —
        // the UI sends nothing for an untouched control (see the comment
        // there), so an absent value has to resolve the same way.
        key: 'mm3ReuseAr',
        type: 'toggle',
        label: 'Reuse Planner Output',
        hint: 'Skip the AR planner when nothing upstream of the flow stage changed — '
            + 'roughly halves the render when you are only tweaking steps, CFG, solver '
            + 'or scheduler. Any change to caption, lyrics, duration, seed, LM adapter '
            + 'or LM/depth model re-plans automatically. Holds one block of engine RAM '
            + '(~3 MB per second of audio, so roughly 600 MB for a 200 s song). Needs a '
            + 'fixed seed to be able to hit.',
        default: true,
      },
      {
        // ── Streaming player ──
        // Lives here rather than as a core UI control on purpose: it is a
        // backend-specific capability (windowed rendering is what makes it
        // possible at all), and the extensions channel already carries it into
        // getGlobalParams() with no store field, no route change and no
        // ACE-side risk. The ACE `streamMode` block in CreatePanel.tsx is a
        // different, shelved feature and stays shelved.
        key: 'mm3Stream',
        type: 'toggle',
        label: 'Play While Rendering',
        hint: 'Start listening a few seconds in, while the rest of the song is still '
            + 'being computed. MiniMax-Music3 renders in overlapping windows, so each '
            + 'window is final long before the last one exists. The complete '
            + 'file is still written and saved exactly as usual — this only adds a live '
            + 'preview. Off by default: it holds the rendered audio in engine memory '
            + 'until the browser drains it.',
        default: false,
      },
      {
        // ── Ensemble takes ──
        // The autoregressive planner reads the whole 8B LM plus seven 0.6B
        // depth passes for EVERY audio frame, and that read costs the same
        // whether it serves one song or four — so decoding several takes in
        // lockstep shares it. Measured on an RTX 5090 at q8_0, the AR stage
        // does 4 takes in the time it used to do 1.4.
        //
        // A slider rather than a free number: the ceiling is a CUDA kernel
        // limit (ggml amortises a quantised weight read across at most 8
        // matrix-vector columns, and a CFG pair costs two of them), not a
        // preference, so offering 12 would just be clamped silently.
        key: 'mm3Takes',
        type: 'slider',
        label: 'Variations Per Render',
        min: 1,
        max: 4,
        step: 1,
        hint: 'Render several DIFFERENT songs from one prompt in a single pass, each '
            + 'from its own seed. The planner is shared, so the extra takes are close '
            + 'to free — three variations cost about the time of one and a half. Each '
            + 'take is saved as its own track. The flow stage is not shared, so this '
            + 'is a throughput win, not an instant one.',
        default: 1,
      },
      {
        key: 'mm3ArSeed',
        type: 'text',
        label: 'Planner Seed',
        hint: 'Seed for the AR planner only. Blank ties it to the main seed, which is '
            + 'MiniMax-Music3’s own behaviour. Set a number to pin the plan while the '
            + 'main seed still rerolls the flow noise — the one way to change the seed '
            + 'and still reuse the planner.',
        default: '',
      },
      {
        key: 'mm3PlankPath',
        type: 'select',
        label: 'Replay AR Plan',
        hint: 'Reuse a saved AR plan instead of generating a new one, fixing the '
            + 'semantic content so flow-stage settings can be compared fairly. '
            + 'NOT a speedup: the AR stage still runs its full compute, it just '
            + 'emits the saved codes. Duration and lyrics come from the plank.',
        default: '',
        options: mm3PlankOptions(),
      },
    ],
  };
}

async function models(): Promise<BackendModels> {
  // Since the 5-way split MM3 ships one GGUF per component, each at several
  // quant levels — the roles are the buckets and the quant tokens the options.
  // cond/voc exist only near-native, so their buckets are usually one entry;
  // the UI's dropdowns degrade to a fixed label there. A pre-split engine
  // (variants.dit absent) still gets the legacy lm+synth pair.
  const { props } = await mm3Props();
  const v = props?.variants;

  const meta: NonNullable<BackendModels['meta']> = {};
  const bucketOf = (variants: Mm3RoleVariants | undefined, key: string): string[] => {
    if (!variants?.available?.length) return [];
    meta[key] = {};
    for (const f of variants.available) {
      meta[key][f.quant] = { label: f.filename, bytes: f.bytes };
    }
    return variants.available.map(f => f.quant);
  };

  if (!v?.dit) {
    // Legacy engine: two-bucket shape, untouched.
    return {
      buckets: { lm: bucketOf(v?.lm, 'lm'), synth: bucketOf(v?.synth, 'synth') },
      adapters: [],
      lmAdapters: [],
      defaults: { lm: v?.lm?.selected ?? '', synth: v?.synth?.selected ?? '' },
      meta,
    };
  }

  return {
    buckets: {
      lm:    bucketOf(v.lm, 'lm'),
      dit:   bucketOf(v.dit, 'dit'),
      depth: bucketOf(v.depth, 'depth'),
      cond:  bucketOf(v.cond, 'cond'),
      voc:   bucketOf(v.voc, 'voc'),
    },
    // No DiT-adapter subsystem for MM3 yet (features.adapters is false), so
    // that cluster stays an empty placeholder. LM LoRAs DO exist — the bare
    // file list here keeps the generic catalogue honest; the picker reads
    // GET /api/mm3/lm-adapters for the sidecar detail (trigger, rank,
    // recommended scales) this shape has no room for.
    adapters: [],
    lmAdapters: listMm3LmAdapters().map(a => a.file),
    // The quant actually in force, not what was requested: if a selected file
    // is deleted the engine falls back and the UI must show the truth.
    defaults: {
      lm:    v.lm?.selected ?? '',
      dit:   v.dit?.selected ?? '',
      depth: v.depth?.selected ?? '',
      cond:  v.cond?.selected ?? '',
      voc:   v.voc?.selected ?? '',
    },
    meta,
  };
}

/** Switch which quant of each role runs. The engine drops only the residency
 *  the change feeds (a DiT swap keeps the 17 GB LM warm), so the next
 *  generation pays a partial warm — that is the honest cost of the switch. */
async function selectModel(selection: Record<string, string>) {
  // A MISSING role means "leave it alone", never "reset to auto". The engine
  // treats an absent/empty role as auto (best-first, which is f16), so a
  // partial body like {depth: "q8_0"} would silently revert a deliberately
  // picked q8_0 LM to f16 — and this function would then PERSIST the revert.
  // Measured as a mystery 2× LM slowdown on 2026-08-21 before the cause was
  // found. The UI already sends every role; this guards API callers and
  // future panels. An EXPLICIT "" still means auto.
  const persisted = persistedSelection();
  const sel = {
    lm:    selection.lm ?? persisted.lm,
    depth: selection.depth ?? persisted.depth,
    cond:  selection.cond ?? persisted.cond,
    dit:   selection.dit ?? selection.synth ?? persisted.dit,
    voc:   selection.voc ?? persisted.voc,
  };
  const result = await mm3SelectModel(sel);
  // Persist only after the engine accepted it, so a rejected quant can never
  // be written back and then replayed on every subsequent boot.
  setSetting(LM_SETTING, sel.lm);
  for (const role of MM3_ROLES) {
    setSetting(ROLE_SETTING[role], sel[role]);
  }
  if (result.changed) {
    console.log(
      `[Backends] MiniMax-Music3 models: lm=${result.lm} dit=${result.dit ?? result.synth}` +
      ` depth=${result.depth ?? ''}` + (result.unloaded ? ' (evicted affected weights)' : ''));
  }
  return { ...result };
}

export const minimaxBackend: EngineBackend = {
  id: 'minimax-m3',
  displayName: 'MiniMax-Music3',   // license-mandated exact string — see header
  resourcePool: 'gpu',
  lifecycle: {
    // MM3 has no process of its own: it is a model family inside ace-server.
    // "Start" is therefore whatever ace-server's own lifecycle already did —
    // weights load lazily on the first job (the job's own VRAM arbitration
    // evicts the ACE side if needed).
    async start() { /* no separate process — ace-server owns the lifecycle */ },
    // "Clean shutdown (frees VRAM/resources)" for a residency-only backend is
    // exactly the unload.
    async stop() { await mm3Unload(); },
    status,
  },
  capabilities,
  models,
  selectModel,
  /** Model-residency arbitration (plan §4.4): switching away from MM3 frees
   *  its ~13 GB rather than leaving it parked next to the ACE pipeline.
   *
   *  Keeps the AR plan cache. This is VRAM arbitration the user did not ask
   *  for — flipping to ACE to check something and coming back must not cost a
   *  cached plan. Correctness is held by the AR key, which pins LM + depth
   *  model identity; a real model change drops the slot on its own path. */
  async releaseVram() {
    const r = await mm3Unload(undefined, /*keepArCache=*/true);
    if (r?.unloaded) {
      console.log(`[Backends] MiniMax-Music3 unloaded (${(r.freed_mb ?? 0).toFixed(0)} MB freed)`
        + `${r.ar_cache_kept ? ' — AR plan cache kept' : ''}`);
    }
  },
};
