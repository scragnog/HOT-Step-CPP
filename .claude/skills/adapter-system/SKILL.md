---
name: adapter-system
description: Explains how HOT-Step's LoRA/LoKr adapter system loads, merges, caches, stacks, and regionally masks adapters at runtime, including hard-won failure modes. Use when working on adapter loading/merging, multi-adapter stacking, per-section adapter masking, adapter cache keys, cross-base adapter conversion, debugging adapters that sound wrong or silent, or when UI adapter knobs (Adapter VRAM, Alignment Timing, Sum/Blend, trigger words, [Section]{k=v} lyric directives) appear dead or ignored.
---

# Adapter (LoRA/LoKR) System

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

The deepest institutional-knowledge subsystem in HOT-Step CPP. Every rule below was
paid for with a shipped bug or a failed research direction. Deeper detail (merge/runtime
internals, cache-key construction, cross-arch research history) lives in
[reference.md](reference.md) in this folder. Line references verified 2026-10-23; grep for
the named symbol when one has drifted.

## Glossary (read first — no prior context assumed)

YuE2 ComfyUI import preserves per-module alpha, ordinary distilled slider keys and
NAR flow heads. It captures `lora_alpha` or `alpha` from adjacent `adapter_config.json`
with tensor alpha taking precedence, followed by file metadata and then the sidecar.
Native `yue2.{vae2llm,llm2vae}.{diff,diff_b}` are scaled deltas;
`weight` and `bias` are full replacements independent of nonzero strength. A flow-head
import sets `decoder_companion=off` so it does not stack on the automatic companion.
Validate imports with the published adapter, including an output comparison.

Album-preset maintenance: Settings > General > Adapters > **Update adapter references**
calls `POST /api/lireek/presets/update-adapter-references`. The service in
`server/src/services/lireek/adapterReferences.ts` checks all five adapter columns in
SQLite `album_presets`, clears missing weights in a transaction, and reports each
preset and dropped path. It preserves live references and other columns. This is an
explicit action, never a startup migration or a remapping operation.

- **DiT** — Diffusion Transformer, the C++ model that denoises audio latents (32 layers on the XL base). Lives in `engine/src/dit.h` / `dit-graph.h`.
- **LoRA** adds a low-rank delta `B@A` to a base weight. **LoKr** uses LyCORIS Kronecker-product factors. **DoRA** adds a per-row magnitude rescale, named `dora_scale` in LyCORIS or `lora_magnitude_vector` in PEFT. Merge supports both; runtime rescaling requires `adapter_runtime_dora:true`, default false. Keep existing strength semantics: the delta carries alpha/rank, and user strength times group scale blends the rescale factor, so strength 0 does not fully disable an existing-format DoRA adapter. ACE preflight rejects configs whose `use_dora` contradicts the magnitude payload, even with runtime rescaling off.
- **LoHa** (`hada_*` tensors) and **HiRA** (`peft_type: HIRA`, delta = W ⊙ BA) are merge-only: their deltas are not low-rank, so the runtime loaders refuse them. **rsLoRA** (`use_rslora: true`, alpha/√r scaling) is honoured in merge and runtime.
- **Delta (Δ)** — the full-size weight difference an adapter contributes: `W_adapted = W_base + scale·Δ`.
- **Merge mode** — deltas are baked into base weights on load, before GPU upload (`adapter-merge.h`). **Runtime mode** — deltas live as separate GPU tensors, applied each sampling step as `y = W@x + Δ@x` (`adapter-runtime.h` + `dit-graph.h`). **Low-Rank mode** (`adapter_mode: "runtime_lowrank"`) keeps the raw factors on the GPU (LoRA A/B with the scale folded into B; LoKr w1/w2), and the graph applies `y += B@(A@x)` per step, with no full-size deltas and no precompute. The UI and server default is `runtime` (`server/src/services/generation/defaults.ts:18`); the engine's own default when the field is absent is `merge` (`hot-step-params.h:165`).
- **aceReq** — the JSON generation request (`AceRequest`) the Node server sends to the C++ engine's `/synth` endpoint.
- **Sideband** — extra HOT-Step-only request fields the engine parses separately from `AceRequest` (struct `ServerFields`, `engine/tools/hot-step-server.cpp:697`) and stores in the global `g_hotstep_params` (`engine/src/hot-step-params.h:288`).
- **ModelKey** — the cache key under which a loaded DiT is stored in `model-store.{h,cpp}`. Two requests with the same key reuse the same loaded (adapter-merged) model.
- **Basin re-base** — before merging, nudge the loaded base T toward the adapter's original training base S: `base ← base + β·(S − base)` (see Institutional knowledge).

## When to use this skill

- Adding/changing anything in `engine/src/adapter-*.h`, the adapter parts of `dit.h`, `dit-graph.h`, `hot-step-sampler.h`, or `hot-step-server.cpp`.
- Working on multi-adapter stacking, Sum/Blend, per-section `[Header]{k=v}` lyric directives, adapter VRAM quantization, or trigger words.
- Debugging: adapter inaudible, wrong strength, wrong flavor after toggling settings, CUDA errors during sampling with section adapters, slow adapter loads.
- Anything involving converting adapters between DiT bases or architectures.

## Golden rules (each one reverses a shipped bug or dead research month)

1. **Every constant GPU graph input must stay valid on both graph paths.** On the scheduler fallback, GGML clobbers input-tensor buffers after every compute, so constants, especially the per-frame LoRA section masks, must be re-uploaded every evaluation, in all compute paths, and again after any mid-sampling graph rebuild (CFG-cutoff). Upload-once produced the "nil adapter" bug: masks read ~0, output is pure base model. On the direct static graph (since `73e057b6`), the masks are flagged input and output in `mark_resident_inputs()` so they stay resident, and are re-uploaded only when timestep gains are active. All mask uploads go through `upload_lora_masks(t_val)` (`engine/src/hot-step-sampler.h:438`), called at :829, :894, :954 (the `!dit_graph.direct || lora_gains_active` sites) and :1171, :1331 (the two CFG-cutoff rebuilds). A NEW graph input must join `force_inputs_to_backend()`, `mark_resident_inputs()` and every one of those sites. (Commits `a6db135`, `2052bc3`.)
2. **Cache-key discipline (commit `168dcb5` — the law).** Every input that changes the merged weights or loaded deltas MUST be part of `ModelKey`, and a failed adapter load must NEVER be cached as a success. Key construction: `engine/src/pipeline-synth.cpp:176-241`; hash/eq: `engine/src/model-store.cpp:43-115`; failure path: `engine/src/dit.h:868-879`. If you add any parameter that affects merged/loaded weights, add it to the key AND the hash AND the equality — all three.
3. **Basin re-base applies ONCE PER STACK, first adapter only** (merge: `engine/src/dit.h:778-801`; runtime: `adapter_runtime_rebase` in `adapter-runtime.h`, invoked after the first adapter stages, :1193-1198). Applying it per adapter resets the running base on every merge (at β=1 only the last adapter survives) — in runtime mode it would duplicate the base correction N×. Works in BOTH modes since `7aac3c2` (runtime folds β·(S−T) into the staged delta sum — matmul distributivity makes it output-identical to merge). In Low-Rank mode the first adapter stages zero-filled full-size correction deltas for the nudge to land in (:1181). NOT supported on the per-section masking path (masked per-frame deltas can't carry an always-on base correction; engine warns + skips, and the cache key mirrors that skip in `pipeline-synth.cpp`).
4. **Do NOT re-attempt regional self-attention isolation.** Implemented in commit `0f3bf6d`, reverted in `ee041e1`: penalising cross-section self-attn logits broke musical continuity (multi-second silence gaps, degenerate later sections). Self-attention carries BOTH adapter identity AND musical coherence — you cannot suppress one without the other. The `adapter_section_isolation` param is still plumbed end-to-end but **dormant** (engine ignores it). See reference.md §Reverted for the structurally safer untried alternative.
5. **Weight similarity does NOT imply adapter transferability.** LoKR cross-base conversion to non-turbo XL bases fails even though the bases are ~99% weight-identical — the root cause is basin-sensitivity, not weight drift (Institutional knowledge below).
6. **Never use real Q4_K for runtime delta storage.** Its per-superblock optimizer stalls the load for minutes across ~360 tensors. The `"q4_k"` string is now aliased to Q4_0 (`adapter_runtime_storage_type`, `engine/src/adapter-runtime.h:159-168`). Q4_0/Q8_0 require `ne0 % 32 == 0`; anything else falls back to BF16.
7. **Never assume runtime deltas are BF16.** They may be Q4_0/Q8_0. `mul_mat` dequantizes transparently, but any code that reads delta bytes directly assuming BF16 crashes (a delta-L2 diagnostic did exactly this — removed in `46603bf`).
8. **LM echo sideband gotcha: never whitelist synth-request fields.** Sideband/ServerFields-only params (`adapter_runtime_quant`, `adapter_section_align_at`, `rebase_*`, …) do NOT survive the engine `/lm` round trip. `server/src/services/backends/ace/generate.ts` rebuilds synth requests as `{...aceReq, ...LM-generated-fields-only}` (cache hit :100-111, cache miss :199-212, plan-guard resample :252-261). A field whitelist here once made the Adapter VRAM and Alignment Timing knobs silently dead on the default path (fixed in `8ea519b`/`168dcb5`).
9. **Edit the right files.** The compiled server is `engine/tools/hot-step-server.cpp` — `engine/tools/ace-server.cpp` is upstream reference code, not compiled. The live sampler is `engine/src/hot-step-sampler.h` — `dit-sampler.h` is dead upstream code. `pipeline-synth-ops.cpp:9` includes `hot-step-sampler.h`; losing that include on an upstream sync is **silent** (compiles, all solvers/schedulers/guidance go dead). After any sync run `engine/verify-hooks.ps1`.
10. **Build & git rules.** After ANY engine edit: `.\dev-rebuild.bat` from repo root, immediately — NEVER `engine/build.cmd` directly (you cannot reliably tell whether the app is running; Node auto-respawns ace-server → infinite respawn + file-lock loop), never `cmake --clean-first` (20+ min CUDA recompile). TypeScript changes: `npx tsc --noEmit` in `server/` (in `ui/`: `npx tsc --noEmit -p tsconfig.app.json`), not `npm run build`. Git: stage explicit paths only — never `git add -A` or `add -f` (re-adds gitignored <data root>/models/checkpoints/local docs); commit locally often; push only with explicit user approval.

## How an adapter reaches the GPU (request flow)

1. **UI → Node**: `server/src/services/generation/translateParams.ts:190-299` maps `params.loraStack` → `req.adapters` `[{name, scale, gain_curve?, gain_domain?}]` (supersedes the single `req.adapter`), plus `adapter_mode`, `adapter_runtime_quant`, `adapter_runtime_dora`, `adapter_merge_lowvram`, per-group scales (:227-231), basin re-base fields (:273-276), and trigger words for every loaded adapter, the planner LM adapter included (:278-299; read from each file's safetensors `__metadata__` by `server/src/services/adapters/stMetadata.ts`). Per-section directives in lyrics are parsed here (see below).
2. **Node → engine `/synth`**: the engine parses the JSON twice — once as `AceRequest` (`engine/src/request.{h,cpp}`), once as the `ServerFields` sideband (`parse_server_fields`, `hot-step-server.cpp:774`).
3. **Worker resolves and stages**: adapter names → paths via the registry, with a path fallback for unregistered adapters: a file, or a folder holding `adapter_model.safetensors` or the Training Studio's `lokr_weights.safetensors` (`hot-step-server.cpp:1373-1404`). Fills `g_hotstep_params.adapters` and all sideband fields **BEFORE `ace_synth_load`** — the merge reads them during load (`hot-step-server.cpp:1450-1453` comment).
4. **Cache lookup**: `pipeline-synth.cpp:176-241` builds the DiT `ModelKey`; `model-store` returns the cached DiT or loads via `dit_ggml_load`.
5. **Load** (`dit.h`): QKV and gate/up fusion is skipped whenever an adapter is present in any of the three modes (`dit.h:577-590` — runtime deltas need individual projections, and an HQ merge promotes them to F32). Merge mode → the whole stack's DoRA config is preflighted first (:734-746), merge is refused on ConvRot models (:750-754), then sequential `adapter_merge` per stacked adapter (deltas accumulate: `W ← W + s1·Δ1 + s2·Δ2 + …`, :778-806). Runtime and Low-Rank modes → after GPU alloc, load runtime deltas or factors (`dit.h:821+`).
6. **Sampling** (`hot-step-sampler.h`): deltas applied per step; section masks (if active) built and uploaded as Golden rule 1 describes.

## Per-section (regional) adapter masking — shipped, P1+P2

Different adapters active in different song sections, driven by lyric directives like
`[Chorus]{albumB2=1; albumN=0}` (keys = adapter filename stem, or positional `#2`/`2`, 1-based).

- **Gate**: activates only with directives in lyrics AND ≥2 stacked adapters (`translateParams.ts:237`). Forces `adapter_mode=runtime` (server :247; engine double-checks and also turns `runtime_lowrank` into `runtime`, `hot-step-server.cpp:1487-1497`). When the gate is unmet, directives are still **stripped** from lyrics (`stripAdapterDirectives`, :255) so they never reach the LM/encoder as garbage tokens. The UI stack controls sit behind Advanced mode (`advancedAdapters` in `ui/src/stores/globalParamsStore.ts`).
- **Parser** (`server/src/services/generation/adapterSections.ts`): Sum/Blend applied per section (blend budget default 0.75); directive-less sections use stack default scales; `{…}` with no `key=val` pair is treated as lyric text and left alone; all-typo keys warn and fall back to defaults; weights clamp ≥ 0.
- **Engine load** (`dit.h:828-862`): each adapter goes into its OWN `DiTLoRA` in `m->loras[]`, **UNIT-scaled (1.0)** — the per-frame mask carries the effective scale; loading with the stack scale would double-scale (`dit.h:853-857`). This costs N× VRAM vs the summed path, hence Q4_0 delta quantization (~¼ size).
- **Graph** (`dit-graph.h`): one `[1,S,1]` F32 mask tensor per adapter, shared across layers (`dit-graph.h:783-820`). Frame-indexed projections get `Σᵢ (Δᵢ@x) ⊙ maskᵢ`; token/global projections (cross-attn k/v, cond_embed) get the adapter's scalar **mean** section weight instead — a frame mask is the wrong axis there (`dit-graph.h:527`, `:903`). Known limitation: text conditioning is therefore a constant blend across the song.
- **Sampler** (`hot-step-sampler.h`): P1 builds an initial frame→section map proportional to section character counts (:385-402), with a ~0.5 s triangular crossfade between sections (`xf = max(3, S/300)`, `ce57675`). P2: at `adapter_section_align_at` fraction of steps (default 0.55, UI "Alignment Timing"), estimate x0, run alignment extraction on a **private scheduler** (`dit-alignment-graph.h:322`), map frames → dominant lyric token → section via the header-anchored token map (`pipeline-synth-ops.cpp:1376-1458`), median-smooth, rebuild masks (`run_p2_alignment`, :533-583). Falls back to the P1 map on failure.

## Timestep-dependent adapter gating (interval experts / MoE) — shipped 2026-07-20

Per-adapter gain curves g(t) over flow-matching t (1=noise → 0=clean): each stacked
adapter's per-frame mask is multiplied by its interpolated g(t) at EVERY model
evaluation (`upload_lora_masks(t_val)` in `hot-step-sampler.h` — the single helper
that replaced all five raw mask-upload sites). Different adapters can own different
slices of the trajectory ("structure" expert early / "timbre" expert late — TD-LoRA
/ TimeStep Master scalar mixing). Key facts:

- **Rides the per-section machinery**: a curve-carrying stack with no lyric
  directives gets a synthetic single whole-song section from `translateParams.ts`
  (:263-270, + forced runtime mode); allowed at stack size 1 (gates in `dit.h` and
  `hot-step-server.cpp` accept `>=2 adapters OR gains active`). Composes with real
  per-section directives (mask × gain).
- **Curves are NOT in the ModelKey** — deliberate: they change per-step mask
  uploads only, never loaded weights, so mixing retunes with zero model reload
  (the `|sect` marker still applies via the synthetic section). Do not "fix" this
  by adding them to the key.
- **Direct-graph caveat**: masks are pinned resident on direct graphs and normally
  skip per-step re-upload; with gains active `upload_lora_masks` runs on EVERY
  evaluation regardless (`!dit_graph.direct || lora_gains_active`). Host masks stay
  UNSCALED — scaling happens into a scratch buffer at upload so P2 rebuilds and
  repeated evals never compound gains.
- **Wire format**: `adapters[].gain_curve` = uniform samples of g(x) over [0,1]
  (33 when the server compiles a UI window, `windowToGainCurve` in `translateParams.ts`)
  + `gain_domain`: `"steps"` (x = remaining-steps fraction,
  engine maps each eval's t to its nearest schedule index) or `"t"` (x = raw
  flow-matching t). UI windows are ALWAYS `"steps"`; trained-expert / router
  curves must be `"t"` (must match the training axis). This split exists because
  shifted schedules are wildly nonuniform in t — shift-3 @ 20 steps spends 17
  steps above t=0.5, so a t-domain "0–50%" window starved the late adapter to 3
  tail steps ("all I hear is the first adapter"). Window edges are smoothstep
  ramps CENTERED on the bound so adjacent windows crossfade summing to 1. Every
  `[DiT] Step` log line appends ` gains=[..]` when gating is active — check
  there first when a mix sounds one-sided.
- **Training side**: Side-Step `--timestep-window-min/max` (rejection-resampled
  logit-normal; discrete mode filters the 8-step schedule) trains matching
  interval experts. T-LoRA: the high-noise expert overfits fastest — lower rank.
- Inherits per-section constraints: runtime-only, N× VRAM
  (Q4_0 knob), no basin re-base. `runtime_lowrank` is CLOBBERED to plain runtime
  (server + engine) — windows currently require the full-delta path; masked
  low-rank apply is untried future work.
  DoRA rescaling requires the opt-in `adapter_runtime_dora` flag.
- **VRAM trap (shipped bug, fixed same day)**: windows force runtime mode
  server-side, but `getGlobalParams()` gated `adapterRuntimeQuant` on the
  *selected* mode — a Merge-mode user with windows got full BF16 deltas
  (~8 GB/adapter on XL, 32 GB with 2 + reload churn) and their Merge-VRAM "Low"
  setting meant nothing. Quant now flows (and the Adapter VRAM knob shows)
  whenever windows are active. Any future knob that matters on a *forced*
  path must gate on the EFFECTIVE mode, not the selected one.

### Per-section hard constraints (violating any reproduces a shipped bug)

1. Mask uploads per Golden rule 1, including after graph rebuilds.
2. Mid-sampling alignment needs its **own** `backend_sched_new` (`dit-alignment-graph.h:322`); sharing `dit->sched` corrupts CUDA state → `CUDA error: invalid argument` (`4e48176`).
3. Graph node budget AND scheduler hash-set must scale with adapter count: `dit-graph.h:727` (`graph_cap = 8192 + loras*4096 + ...`) and `dit.h:425-443` (`sched_nodes = 8192 + adapters*4096`, sized from the intended stack because `m->loras` isn't populated yet; Low-Rank and concept steering add their own terms). Miss either → crash with N section adapters (`e09d6a3`).
4. In `HOTSTEP_SECTION_NOMASK` debug mode, mask tensors must not be created at all — unconsumed graph inputs get no buffer, so uploading to them crashes (`34dce60`).

## Low-Rank mode, ConvRot and the TRT path

- **Low-Rank** (`runtime_lowrank`, `4da6e562`): ~20–40× less adapter VRAM than full deltas and no ~17 s precompute. Conv1d-tiled tensors (`proj_in`) keep the full-delta path. With **Runtime DoRA magnitudes** on, a DoRA stack falls back to full-size effective deltas (`adapter_runtime_dora_stack`), because the magnitude correction is full rank. Its cache-key marker is `|mode:runtime_lr`, and the `|q:` quant marker covers both runtime modes.
- **ConvRot** (int8 rotated weights, GGUF key `acestep.convrot_map`): merge mode is refused with `[Adapter] FATAL: adapter merge mode is not supported on ConvRot models — use adapter_mode=runtime` (`dit.h:750-754`); runtime deltas run as separate matmuls on the unrotated activations.
- **TensorRT DiT** (an ONNX DiT): adapters go through `adapter-trt.h` refit instead (the `engine-performance` skill). That path refits the primary adapter only; stacks, section masks and the runtime modes are GGML-only.

## Planner (LM) adapters

`engine/src/lm-adapter.h` loads adapters for the 5 Hz planner LM: PEFT LoRA (also LoKr and DoRA), applied at graph-build time through `qwen3_linear_lora()` and never merged, so the base LM can be any quant. The request carries `lm_adapter` / `lm_adapter_scale` in `AceRequest` itself, not the sideband, so they survive the LM echo by construction (`translateParams.ts:172-177`). The LM `ModelKey` includes the adapter path and the scale's bit pattern (`model-store.cpp:47-54`). The loader refuses, rather than guesses at, a LoHa or HiRA file (not low-rank, and this LM has no merge mode), a PiSSA delta adapter (MM3-only), a DiT LoKr (`carries cross_attn sites`), a marker that disagrees with `adapter_config.json`, and a `use_dora` flag that disagrees with the tensors. With an LM adapter the server defaults `lm_rep_penalty` to 1.05, and the plan guard in `services/backends/ace/generate.ts` resamples a degenerate plan up to twice with a shifted seed and a firmer penalty, then renders anyway with a warning. The UI list comes from `GET /api/adapters/lm`.

## Key files

| Path | Role |
|---|---|
| `engine/src/adapter-merge.h` | Merge mode: safetensors/PEFT/LyCORIS parsing, LoKr detection (`adapter_detect_lokr` :514), LoHa (`adapter_merge_loha` :1339), GPU merge graph (`adapter_merge_on_backend` :599, contains the DoRA weight-decompose for both GPU and host paths), delta compute (`adapter_compute_delta` :556), basin re-base (`adapter_rebase_fetch` :79), PEFT DoRA magnitude detection (`lora_is_magnitude` :184), DoRA preflight (`adapter_dora_preflight` :894), rsLoRA (`adapter_read_rslora` :243), per-module `alpha_pattern` parser (`adapter_read_alpha_pattern` :321 — REQUIRED for PEFT rank_pattern adapters, else per-module strength is alpha_global/rank_actual ≈ wildly wrong), entry `adapter_merge` (:1914) |
| `engine/src/adapter-runtime.h` | Runtime mode: `DiTLoRA` storage, slot mapping, stack delta staging, Low-Rank factor staging, quantized finalization and `adapter_load_runtime_stack`; opt-in `adapter_runtime_dora_stack` builds effective DoRA deltas through the merger. |
| `engine/src/adapter-cancel.h` | `g_adapter_cancel` atomic + `adapter_cancel_requested()` — cooperative cancel of the ~17 s delta precompute; `g_adapter_progress`, the live delta index the `/job` poll reports as a watchdog heartbeat; separate header so the server avoids ggml deps |
| `engine/src/adapter-trt.h` | TensorRT IRefitter merge variant (`#ifdef HOT_STEP_TRT`) |
| `engine/src/lokr-common.h`, `engine/src/lokr-apply.h` | LoKr shape rules and the factor-by-factor forward delta, shared by the trainers (`train/dit-adapter-lokr.h`, `train/lm-graph.h`). A trainer/loader disagreement loads fine and computes something else; the inference loaders parse LoKr themselves |
| `engine/src/lm-adapter.h` | Planner-LM runtime adapters (see above) |
| `engine/src/dit.h` | Load orchestration: scheduler budget (:425), fusion skip (:577), DoRA stack preflight (:734), ConvRot refusal (:750), once-per-stack re-base (:778), per-section per-adapter loads (:828), fail-don't-cache (:868). Fork hook: includes `adapter-merge.h`/`adapter-runtime.h` (:11/:14) |
| `engine/src/dit-graph.h` | Forward graph: `dit_ggml_linear_lora` (:118), `DiTLoRASectionCtx` (:148), per-adapter masks (:783), NOMASK debug flag (:155-158), `graph_cap` (:727) |
| `engine/src/hot-step-sampler.h` | Live sampling loop: mask building/rebuilding, `upload_lora_masks`, P2 alignment, direct vs scheduler-fallback graph |
| `engine/src/dit-alignment-graph.h` | `dit_alignment_extract` with its private scheduler (P2) |
| `engine/src/hot-step-params.h` | `AdapterSpec` (:49), `g_hotstep_params` sideband global (:288) + `hotstep_adapter_stack_sig` (:306); included by `model-store.h` (fork hook) |
| `engine/src/model-store.{h,cpp}` | `ModelKey` adapter fields (model-store.h:83-96), hash/eq (model-store.cpp:43-115) |
| `engine/src/pipeline-synth.cpp` | Builds the DiT cache key (:176-241) |
| `engine/src/pipeline-synth-ops.cpp` | Header-anchored token→section map for P2 (:1376-1458); carries the `hot-step-sampler.h` fork hook (:9) |
| `engine/tools/hot-step-server.cpp` | The REAL ace-server. `ServerFields` (:697), name→path resolution + `g_hotstep_params` fill (:1360-1500), cancel wiring (:1567-1570), job phase `ADAPTER_PRECOMPUTE` (:316) |
| `server/src/services/generation/adapterSections.ts` | `[Section]{k=v}` directive parser + `stripAdapterDirectives` |
| `server/src/services/generation/translateParams.ts` | UI params → aceReq adapter fields (:190-299), `windowToGainCurve` |
| `server/src/services/generation/triggerWords.ts`, `server/src/services/adapters/stMetadata.ts` | Trigger-word composition, and reading the trigger and its trained position from a safetensors `__metadata__` header |
| `server/src/services/backends/ace/generate.ts` | LM-echo rebuild `{...aceReq, ...lmFields}` (:100-111, :199-212, :252-261), plan guard |
| `server/src/routes/adapters.ts` | Filesystem only: `GET /api/adapters/browse`, `POST /api/adapters/scan` (`.safetensors` files plus adapter folders, descending into `dit-*` per-base folders and their run folders, each with its trigger), `GET /api/adapters/lm` (planner adapters) |
| `ui/src/stores/globalParamsStore.ts`, `ui/src/components/global-bar/AdaptersDropdown.tsx` | Adapter stack, Sum/Blend, loading mode, Adapter VRAM, Runtime DoRA magnitudes, Alignment Timing UI |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| Adapters silently inaudible in per-section mode; mask logs show ~0 | Mask/input upload missing for steps 2..N on the scheduler fallback, after a graph rebuild, or a new input not marked resident on the direct graph — Golden rule 1 |
| Base-model output despite adapter selected, only after an earlier failed load | Load failure cached as success under the adapter-bearing key — check `dit.h:868` path (fixed `168dcb5`) |
| Wrong adapter flavor after toggling merge↔runtime↔Low-Rank or changing quant/scale/group scales | Missing cache-key component — audit `pipeline-synth.cpp:176-241` + `model-store.cpp` hash AND eq |
| `CUDA error: invalid argument` at the alignment step | Alignment shared the main scheduler — must be private (`4e48176`) |
| Crash/overflow when loading N section adapters | Node budget / sched hash-set not scaled with adapter count (`dit-graph.h:727`, `dit.h:425`) |
| Load stalls minutes at "quantize" | Real Q4_K requested — use Q4_0 (Golden rule 6) |
| Crash reading delta tensor bytes in diagnostics | Assumed BF16; deltas may be Q4_0/Q8_0 (Golden rule 7) |
| Silence gaps between sections, later sections degenerate | Someone re-enabled self-attn isolation — REVERTED, Golden rule 4 |
| Adapter VRAM / Alignment Timing knobs "do nothing" | A ServerFields param dropped in an LM round-trip whitelist — Golden rule 8 |
| Merge stack with re-base outputs only the LAST adapter | Re-base applied per adapter instead of once per stack (`dit.h:778`) |
| Section adapters roughly twice as strong as expected | Section deltas loaded with stack scale instead of unit scale (`dit.h:853`) |
| DoRA adapter sounds wrong in runtime mode | Runtime DoRA magnitudes defaults off for compatibility. Enable it in the ACE Adapters section, or send `adapter_runtime_dora:true` to ACE/MM3. Low-Rank may materialize full-size deltas. Merge and LM behavior are unchanged. |
| DoRA adapter refuses to load in any mode | `adapter_dora_preflight` found `use_dora` in `adapter_config.json` disagreeing with the magnitude tensors; replace the incomplete export |
| `[Adapter-RT] LoHa adapters have no runtime path ...` or `[Adapter] HiRA adapter cannot use the runtime low-rank path ...` | LoHa and HiRA are merge-only. Switch the loading mode to Merge |
| `[Adapter] FATAL: adapter merge mode is not supported on ConvRot models` | ConvRot int8 base: use Runtime or Low-Rank |
| PEFT adapter with per-module ranks (`rank_pattern` in adapter_config.json) sounds weak/distorted | `alpha_pattern` must be parsed per module (`adapter_read_alpha_pattern`) — the global `lora_alpha` over the actual per-tensor rank gives e.g. 128/512 instead of the trained 1024/512. Also requires `adapter_config.json` to sit NEXT TO the .safetensors (alpha fallback is rank ⇒ wrong strength without it) |
| All solvers/schedulers/guidance dead after an upstream sync (compiles fine) | `pipeline-synth-ops.cpp` lost the `hot-step-sampler.h` include — run `engine/verify-hooks.ps1` |

## Institutional knowledge (from the departing lead engineer)

- **LoKR cross-base conversion — VALIDATED failure, VALIDATED fix.** Converting adapters to non-turbo XL bases FAILS even though the bases are ~99% weight-identical. Root cause is **basin-sensitivity** (the adapter delta lands in a different loss basin), NOT weight drift. Fix: the **β·(S−T) basin nudge** (`adapter_rebase_fetch`, `adapter-merge.h:79`, applied from `dit.h:778-801` for merge; `adapter_runtime_rebase` in `adapter-runtime.h` for runtime) — **user-validated by ear in merge mode, 2026-07-14** ("works fantastically"); the runtime-mode port (`7aac3c2`, same day) is mathematically output-identical but awaits its own by-ear A/B. Never assume weight similarity implies adapter transferability.
- **Multi-adapter stacking (issue #72) — VALIDATED.** Implemented as a sideband stack (`g_hotstep_params.adapters`) with read-from-pending merge accumulation (merge mode sources each tensor's current post-prior-merge value) and delta-sum in runtime mode (`adapter_stage_delta` — per-step cost and VRAM flat regardless of stack depth on the non-section path). Low-Rank mode keeps each adapter's factors separate instead. CRITICAL: basin re-base once per stack, not per adapter (Golden rule 3).
- **Per-section masking — VALIDATED (shipped P1+P2).** GGML clobbers input buffers on the scheduler path, so masks/weights must be re-uploaded every step there, never cached on device without the resident flags (Golden rule 1). Phase 1 = proportional token map; Phase 2 = alignment-based timing; the token→section map is now header-anchored (falls back to char-proportional when header count ≠ section count — the log line says which).
- **Regional self-attn isolation — VALIDATED failure.** `0f3bf6d` → reverted `ee041e1`. Do not re-attempt without a new design for cross-section coherence (Golden rule 4; safer untried idea in reference.md).
- **Cache-key gaps + load-failure caching — VALIDATED fix (`168dcb5`).** Golden rule 2. The specific historical gaps were the time_embed/proj_in group scales and the runtime-mode marker; the failure-caching bug installed adapter-less models under adapter-bearing keys.
- **Cross-ARCH conversion (XL↔2B) — UNSOLVED; per-layer linear surgery is ruled out.** Even with excellent per-layer alignment (CKA 0.90-0.99), independent per-layer linear transforms compound across 24 layers to ~5% end-to-end fidelity. Do not retry crop/pad, ridge, CCA, or Procrustes weight surgery. Distillation is the unblocked direction (details + open items in reference.md — several artifacts there are cited from local docs and unverified).

## Debug tooling

- `HOTSTEP_BCAST_TEST=1` then run `engine\build\Release\ace-synth.exe` — numeric self-test of mask broadcast ops on the real backend, no models needed (`engine/tools/ace-synth.cpp:361`).
- `HOTSTEP_SECTION_NOMASK=1` — apply section deltas unmasked (`dit-graph.h:155-158`). Adapters audible ⇒ mask values are the problem; still silent ⇒ wiring/upload is the problem.
- `HOTSTEP_KRON_TEST=1` then run `ace-synth.exe`: a LoKr Kronecker-apply self-test from factors alone on the real backend, no models needed (`engine/tools/ace-synth.cpp:364`, `8aa7c089`); the Low-Rank LoKr graph was validated against it.
- Logs: newest `<data root>/logs\<session>\ace_engine.log` or `<data root>/logs\<session>\generations\gen_*.log`. Useful greps: `[Adapter-RT] Loaded`, `[Adapter-RT] P2:`, `header-anchored`, `Basin re-base` (`[Adapter-RT] Basin re-base` for runtime, `[Adapter] Basin re-base` for a LoKr merge; a LoRA merge does not log the nudge), `Per-section masking`, `Timestep gains active`, `[DiT] Graph exec:`.
- Audible validation is **by ear only** — ask the user to listen; do not use a browser agent for UI/audio verification. Do not delete experiment output files on your own prediction of their worth.

## Deeper reading

- [reference.md](reference.md) (this folder) — merge/runtime internals, exact cache-key recipe, P2 alignment pipeline, cross-base/cross-arch research history, status ledger.
- [docs/user/adapters.md](../../../docs/user/adapters.md) — the user-facing loading modes, stacking, windows, re-base and per-section directives.
- `docs/plans/per-section-adapter-masking.md`, `docs/plans/cross-arch-adapter-conversion.md`, `docs/plans/multi-adapter-runtime-switching-handoff.md`, `docs/plans/2026-04-18-adapter-group-scales-investigation.md` — **local-only, gitignored; may be absent on a fresh clone.** Their load-bearing content is distilled into this skill and reference.md.
- `docs/dev/engine.md` — engine-wide context (committed); `engine/docs/ARCHITECTURE.md` documents the request fields.
