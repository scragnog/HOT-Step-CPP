---
name: adapter-system
description: Explains how HOT-Step's LoRA/LoKr adapter system loads, merges, caches, stacks, and regionally masks adapters at runtime, including hard-won failure modes. Use when working on adapter loading/merging, multi-adapter stacking, per-section adapter masking, adapter cache keys, cross-base adapter conversion, debugging adapters that sound wrong or silent, or when UI adapter knobs (Adapter VRAM, Alignment Timing, Sum/Blend, trigger words, [Section]{k=v} lyric directives) appear dead or ignored.
---

# Adapter (LoRA/LoKR) System

The deepest institutional-knowledge subsystem in HOT-Step CPP. Every rule below was
paid for with a shipped bug or a failed research direction. Deeper detail (merge/runtime
internals, cache-key construction, cross-arch research history) lives in
[reference.md](reference.md) in this folder.

## Glossary (read first — no prior context assumed)

- **DiT** — Diffusion Transformer, the C++ model that denoises audio latents (32 layers on the XL base). Lives in `engine/src/dit.h` / `dit-graph.h`.
- **LoRA** — low-rank adapter: two small matrices A, B whose product `B@A` is a delta added to a base weight. **LoKr** — LyCORIS Kronecker-product variant (`.lokr_w1/.lokr_w2` tensors). **DoRA** — adds a per-row multiplicative rescale; two on-disk namings, both supported in MERGE MODE ONLY: LyCORIS `dora_scale` (LoKr path) and PEFT `lora_magnitude_vector` (LoRA path, since 2026-07-19). With DoRA the delta carries only alpha/rank; user strength × group scale blends the decompose factor (`s = u·(m/‖W+Δ‖) + (1−u)`), so strength 0 does NOT fully disable a DoRA adapter.
- **Delta (Δ)** — the full-size weight difference an adapter contributes: `W_adapted = W_base + scale·Δ`.
- **Merge mode** — deltas are baked into base weights on load, before GPU upload (`adapter-merge.h`). **Runtime mode** — deltas live as separate GPU tensors, applied each sampling step as `y = W@x + Δ@x` (`adapter-runtime.h` + `dit-graph.h`).
- **aceReq** — the JSON generation request (`AceRequest`) the Node server sends to the C++ engine's `/synth` endpoint.
- **Sideband** — extra HOT-Step-only request fields the engine parses separately from `AceRequest` (struct `ServerFields`, `engine/tools/hot-step-server.cpp:569`) and stores in the global `g_hotstep_params` (`engine/src/hot-step-params.h:205`).
- **ModelKey** — the cache key under which a loaded DiT is stored in `model-store.{h,cpp}`. Two requests with the same key reuse the same loaded (adapter-merged) model.
- **Basin re-base** — before merging, nudge the loaded base T toward the adapter's original training base S: `base ← base + β·(S − base)` (see Institutional knowledge).

## When to use this skill

- Adding/changing anything in `engine/src/adapter-*.h`, the adapter parts of `dit.h`, `dit-graph.h`, `hot-step-sampler.h`, or `hot-step-server.cpp`.
- Working on multi-adapter stacking, Sum/Blend, per-section `[Header]{k=v}` lyric directives, adapter VRAM quantization, or trigger words.
- Debugging: adapter inaudible, wrong strength, wrong flavor after toggling settings, CUDA errors during sampling with section adapters, slow adapter loads.
- Anything involving converting adapters between DiT bases or architectures.

## Golden rules (each one reverses a shipped bug or dead research month)

1. **GGML clobbers input-tensor buffers after every compute.** Any constant GPU input — especially the per-frame LoRA section masks — must be re-uploaded **every sampling step**, in all compute paths, and again after any mid-sampling graph rebuild (CFG-cutoff). Upload-once produces the "nil adapter" bug: masks read ~0, output is pure base model. Existing re-upload sites: `engine/src/hot-step-sampler.h:605, :659, :716, :940, :1097`. Any NEW graph input must join every one of those sites. (Commits `a6db135`, `2052bc3`.)
2. **Cache-key discipline (commit `168dcb5` — the law).** Every input that changes the merged weights or loaded deltas MUST be part of `ModelKey`, and a failed adapter load must NEVER be cached as a success. Key construction: `engine/src/pipeline-synth.cpp:168-199`; hash/eq: `engine/src/model-store.cpp:48-94`; failure path: `engine/src/dit.h:648-658`. If you add any parameter that affects merged/loaded weights, add it to the key AND the hash AND the equality — all three.
3. **Basin re-base applies ONCE PER STACK, first adapter only** (merge: `engine/src/dit.h:565-586`; runtime: `adapter_runtime_rebase` in `adapter-runtime.h`, invoked after the first adapter stages). Applying it per adapter resets the running base on every merge (at β=1 only the last adapter survives) — in runtime mode it would duplicate the base correction N×. Works in BOTH modes since `7aac3c2` (runtime folds β·(S−T) into the staged delta sum — matmul distributivity makes it output-identical to merge). NOT supported on the per-section masking path (masked per-frame deltas can't carry an always-on base correction; engine warns + skips, and the cache key mirrors that skip in `pipeline-synth.cpp`).
4. **Do NOT re-attempt regional self-attention isolation.** Implemented in commit `0f3bf6d`, reverted in `ee041e1`: penalising cross-section self-attn logits broke musical continuity (multi-second silence gaps, degenerate later sections). Self-attention carries BOTH adapter identity AND musical coherence — you cannot suppress one without the other. The `adapter_section_isolation` param is still plumbed end-to-end but **dormant** (engine ignores it). See reference.md §Reverted for the structurally safer untried alternative.
5. **Weight similarity does NOT imply adapter transferability.** LoKR cross-base conversion to non-turbo XL bases fails even though the bases are ~99% weight-identical — the root cause is basin-sensitivity, not weight drift (Institutional knowledge below).
6. **Never use real Q4_K for runtime delta storage.** Its per-superblock optimizer stalls the load for minutes across ~360 tensors. The `"q4_k"` string is now aliased to Q4_0 (`engine/src/adapter-runtime.h:136-141`). Q4_0/Q8_0 require `ne0 % 32 == 0`; anything else falls back to BF16.
7. **Never assume runtime deltas are BF16.** They may be Q4_0/Q8_0. `mul_mat` dequantizes transparently, but any code that reads delta bytes directly assuming BF16 crashes (a delta-L2 diagnostic did exactly this — removed in `46603bf`).
8. **LM echo sideband gotcha: never whitelist synth-request fields.** Sideband/ServerFields-only params (`adapter_runtime_quant`, `adapter_section_align_at`, `rebase_*`, …) do NOT survive the engine `/lm` round trip. `server/src/routes/generate.ts` rebuilds synth requests as `{...aceReq, ...LM-generated-fields-only}` (lines 238 and 320). A field whitelist here once made the Adapter VRAM and Alignment Timing knobs silently dead on the default path (fixed in `8ea519b`/`168dcb5`).
9. **Edit the right files.** The compiled server is `engine/tools/hot-step-server.cpp` — `engine/tools/ace-server.cpp` is upstream reference code, not compiled. The live sampler is `engine/src/hot-step-sampler.h` — `dit-sampler.h` is dead upstream code. `pipeline-synth-ops.cpp:9` includes `hot-step-sampler.h`; losing that include on an upstream sync is **silent** (compiles, all solvers/schedulers/guidance go dead). After any sync run `engine/verify-hooks.ps1`.
10. **Build & git rules.** After ANY engine edit: `.\dev-rebuild.bat` from repo root, immediately — NEVER `engine/build.cmd` directly (you cannot reliably tell whether the app is running; Node auto-respawns ace-server → infinite respawn + file-lock loop), never `cmake --clean-first` (20+ min CUDA recompile). TypeScript changes: `npx tsc --noEmit`, not `npm run build`. Git: stage explicit paths only — never `git add -A` or `add -f` (re-adds gitignored models/checkpoints/local docs); commit locally often; push only with explicit user approval.

## How an adapter reaches the GPU (request flow)

1. **UI → Node**: `server/src/services/generation/translateParams.ts:83-143` maps `params.loraStack` → `req.adapters` `[{name, scale}]` (supersedes the single `req.adapter`), plus `adapter_mode`, `adapter_runtime_quant`, per-group scales, basin re-base fields (through :128), and trigger words for all stacked adapters (:131-143). Per-section directives in lyrics are parsed here (see below).
2. **Node → engine `/synth`**: the engine parses the JSON twice — once as `AceRequest` (`engine/src/request.{h,cpp}`), once as the `ServerFields` sideband (`hot-step-server.cpp:569+`).
3. **Worker resolves and stages**: adapter names → paths via the registry, with an absolute-path fallback for unregistered adapters (`hot-step-server.cpp:1079`). Fills `g_hotstep_params.adapters` and all sideband fields **BEFORE `ace_synth_load`** — the merge reads them during load (`hot-step-server.cpp:1129` comment).
4. **Cache lookup**: `pipeline-synth.cpp:168-199` builds the DiT `ModelKey`; `model-store` returns the cached DiT or loads via `dit_ggml_load`.
5. **Load** (`dit.h`): merge mode → sequential `adapter_merge` per stacked adapter (deltas accumulate: `W ← W + s1·Δ1 + s2·Δ2 + …`); runtime mode → skip QKV/gate_up fusion (`dit.h:420` — runtime deltas need individual projections), then after GPU alloc load runtime deltas (`dit.h:608+`).
6. **Sampling** (`hot-step-sampler.h`): deltas applied per step; section masks (if active) built and re-uploaded every step.

## Per-section (regional) adapter masking — shipped, P1+P2

Different adapters active in different song sections, driven by lyric directives like
`[Chorus]{greenday_idiot=1; blink_selftitled=0}` (keys = adapter filename stem, or positional `#2`/`2`, 1-based).

- **Gate**: activates only with directives in lyrics AND ≥2 stacked adapters (`translateParams.ts:104`). Forces `adapter_mode=runtime` (server :114; engine double-checks, `hot-step-server.cpp:1157-1159`). When the gate is unmet, directives are still **stripped** from lyrics (`stripAdapterDirectives`) so they never reach the LM/encoder as garbage tokens. The UI stack controls sit behind Advanced mode (`advancedAdapters` in `ui/src/stores/globalParamsStore.ts`).
- **Parser** (`server/src/services/generation/adapterSections.ts`): Sum/Blend applied per section (blend budget default 0.75); directive-less sections use stack default scales; `{…}` with no `key=val` pair is treated as lyric text and left alone; all-typo keys warn and fall back to defaults; weights clamp ≥ 0.
- **Engine load** (`dit.h:618-640`): each adapter goes into its OWN `DiTLoRA` in `m->loras[]`, **UNIT-scaled (1.0)** — the per-frame mask carries the effective scale; loading with the stack scale would double-scale (`dit.h:628-631`). This costs N× VRAM vs the summed path, hence Q4_0 delta quantization (~¼ size).
- **Graph** (`dit-graph.h`): one `[1,S,1]` F32 mask tensor per adapter, shared across layers. Frame-indexed projections get `Σᵢ (Δᵢ@x) ⊙ maskᵢ`; token/global projections (cross-attn k/v, cond_embed) get the adapter's scalar **mean** section weight instead — a frame mask is the wrong axis there (`dit-graph.h:430-433`). Known limitation: text conditioning is therefore a constant blend across the song.
- **Sampler** (`hot-step-sampler.h`): P1 builds an initial frame→section map proportional to section character counts (:306-323), with a ~0.5 s triangular crossfade between sections (`ce57675`). P2: at `adapter_section_align_at` fraction of steps (default 0.55, UI "Alignment Timing"), estimate x0, run alignment extraction on a **private scheduler**, map frames → dominant lyric token → section via the header-anchored token map (`pipeline-synth-ops.cpp:1419-1495`), median-smooth, rebuild masks (:338-385). Falls back to the P1 map on failure.

## Timestep-dependent adapter gating (interval experts / MoE) — shipped 2026-07-20

Per-adapter gain curves g(t) over flow-matching t (1=noise → 0=clean): each stacked
adapter's per-frame mask is multiplied by its interpolated g(t) at EVERY model
evaluation (`upload_lora_masks(t_val)` in `hot-step-sampler.h` — the single helper
that replaced all five raw mask-upload sites). Different adapters can own different
slices of the trajectory ("structure" expert early / "timbre" expert late — TD-LoRA
/ TimeStep Master scalar mixing). Key facts:

- **Rides the per-section machinery**: a curve-carrying stack with no lyric
  directives gets a synthetic single whole-song section from `translateParams.ts`
  (+ forced runtime mode); allowed at stack size 1 (gates in `dit.h` and
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
- **Wire format**: `adapters[].gain_curve` = uniform samples of g(t) over [0,1]
  (33 from the UI). UI "Active phase X–Y%" is % of denoising, flipped to t
  (`stepStart/stepEnd` in t-domain on stack entries); window edges are smoothstep
  ramps CENTERED on the bound so adjacent windows crossfade summing to 1.
- **Training side**: Side-Step `--timestep-window-min/max` (rejection-resampled
  logit-normal; discrete mode filters the 8-step schedule) trains matching
  interval experts. T-LoRA: the high-noise expert overfits fastest — lower rank.
- Inherits per-section constraints: runtime-only (no DoRA rescale), N× VRAM
  (Q4_0 knob), no basin re-base.

### Per-section hard constraints (violating any reproduces a shipped bug)

1. Re-upload masks every step + after graph rebuilds — Golden rule 1.
2. Mid-sampling alignment needs its **own** `backend_sched_new`; sharing `dit->sched` corrupts CUDA state → `CUDA error: invalid argument` (`4e48176`).
3. Graph node budget AND scheduler hash-set must scale with adapter count: `dit-graph.h:590` (`graph_cap = 8192 + loras*4096`) and `dit.h:328-331` (`sched_nodes = 8192 + adapters*4096`, sized from the intended stack because `m->loras` isn't populated yet). Miss either → crash with N section adapters (`e09d6a3`).
4. In `HOTSTEP_SECTION_NOMASK` debug mode, mask tensors must not be created at all — unconsumed graph inputs get no buffer, so uploading to them crashes (`34dce60`).

## Key files

| Path | Role |
|---|---|
| `engine/src/adapter-merge.h` | Merge mode: safetensors/PEFT/LyCORIS parsing, LoKr detection (:440), GPU merge graph (`adapter_merge_on_backend` :523, contains the DoRA weight-decompose for both GPU and host paths), delta compute (`adapter_compute_delta` :480), basin re-base (`adapter_rebase_fetch` :78), PEFT DoRA magnitude detection (`lora_is_magnitude` :183), per-module `alpha_pattern` parser (`adapter_read_alpha_pattern` :238 — REQUIRED for PEFT rank_pattern adapters, else per-module strength is alpha_global/rank_actual ≈ wildly wrong), entry `adapter_merge` (:1453) |
| `engine/src/adapter-runtime.h` | Runtime mode: `DiTLoRA*` structs (:39-70), slot map `dit_lora_slot` (:105), stack delta-sum `adapter_stage_delta` (:173), quantized finalize (:804), `adapter_load_runtime_stack` (:1011), DoRA merge-only warnings (PEFT :299, LoKr :510) |
| `engine/src/adapter-cancel.h` | `g_adapter_cancel` atomic + `adapter_cancel_requested()` — cooperative cancel of the ~17 s delta precompute; separate header so the server avoids ggml deps |
| `engine/src/adapter-trt.h` | TensorRT IRefitter merge variant (`#ifdef HOT_STEP_TRT`) |
| `engine/src/dit.h` | Load orchestration: merge-vs-runtime, fusion skip (:420), once-per-stack re-base (:565), per-section per-adapter loads (:618), fail-don't-cache (:648). Fork hook: includes `adapter-merge.h`/`adapter-runtime.h` (:11/:13) |
| `engine/src/dit-graph.h` | Forward graph: `dit_ggml_linear_lora` (:45), `DiTLoRASectionCtx` (:80), per-adapter masks, NOMASK debug flag (:90) |
| `engine/src/hot-step-sampler.h` | Live sampling loop: mask building/rebuilding, per-step re-uploads, P2 alignment |
| `engine/src/hot-step-params.h` | `g_hotstep_params` sideband global (:205) + `hotstep_adapter_stack_sig` (:211); included by `model-store.h` (fork hook) |
| `engine/src/model-store.{h,cpp}` | `ModelKey` adapter fields (model-store.h:90-102), hash/eq (model-store.cpp:48-94) |
| `engine/src/pipeline-synth.cpp` | Builds the DiT cache key (:168-199) |
| `engine/src/pipeline-synth-ops.cpp` | Header-anchored token→section map for P2 (:1419-1495); carries the `hot-step-sampler.h` fork hook (:9) |
| `engine/tools/hot-step-server.cpp` | The REAL ace-server. `ServerFields` (:569), name→path resolution + `g_hotstep_params` fill (:1060-1162), cancel wiring (~:1214), job phase `ADAPTER_PRECOMPUTE` (:263) |
| `server/src/services/generation/adapterSections.ts` | `[Section]{k=v}` directive parser + `stripAdapterDirectives` |
| `server/src/services/generation/translateParams.ts` | UI params → aceReq adapter fields (:83-143) |
| `server/src/routes/generate.ts` | LM-echo rebuild `{...aceReq, ...lmFields}` (:238, :320) |
| `server/src/routes/adapters.ts` | Filesystem only: `GET /api/adapters/browse`, `POST /api/adapters/scan` (`.safetensors` listing) |
| `ui/src/stores/globalParamsStore.ts`, `ui/src/components/global-bar/AdaptersDropdown.tsx` | Adapter stack, Sum/Blend, Adapter VRAM, Alignment Timing UI |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| Adapters silently inaudible in per-section mode; mask logs show ~0 | Mask/input re-upload missing for steps 2..N or after a graph rebuild — Golden rule 1 |
| Base-model output despite adapter selected, only after an earlier failed load | Load failure cached as success under the adapter-bearing key — check `dit.h:648` path (fixed `168dcb5`) |
| Wrong adapter flavor after toggling merge↔runtime or changing quant/scale/group scales | Missing cache-key component — audit `pipeline-synth.cpp:168-199` + `model-store.cpp` hash AND eq |
| `CUDA error: invalid argument` at the alignment step | Alignment shared the main scheduler — must be private (`4e48176`) |
| Crash/overflow when loading N section adapters | Node budget / sched hash-set not scaled with adapter count (`dit-graph.h:590`, `dit.h:328`) |
| Load stalls minutes at "quantize" | Real Q4_K requested — use Q4_0 (Golden rule 6) |
| Crash reading delta tensor bytes in diagnostics | Assumed BF16; deltas may be Q4_0/Q8_0 (Golden rule 7) |
| Silence gaps between sections, later sections degenerate | Someone re-enabled self-attn isolation — REVERTED, Golden rule 4 |
| Adapter VRAM / Alignment Timing knobs "do nothing" | A ServerFields param dropped in an LM round-trip whitelist — Golden rule 8 |
| Merge stack with re-base outputs only the LAST adapter | Re-base applied per adapter instead of once per stack (`dit.h:565`) |
| Section adapters roughly twice as strong as expected | Section deltas loaded with stack scale instead of unit scale (`dit.h:628`) |
| DoRA adapter sounds wrong in runtime mode | DoRA is merge-only; runtime can't express the multiplicative rescale — engine warns and applies plain LoRA (`adapter-runtime.h:299` PEFT, `:510` LoKr) |
| PEFT adapter with per-module ranks (`rank_pattern` in adapter_config.json) sounds weak/distorted | `alpha_pattern` must be parsed per module (`adapter_read_alpha_pattern`) — the global `lora_alpha` over the actual per-tensor rank gives e.g. 128/512 instead of the trained 1024/512. Also requires `adapter_config.json` to sit NEXT TO the .safetensors (alpha fallback is rank ⇒ wrong strength without it) |
| All solvers/schedulers/guidance dead after an upstream sync (compiles fine) | `pipeline-synth-ops.cpp` lost the `hot-step-sampler.h` include — run `engine/verify-hooks.ps1` |

## Institutional knowledge (from the departing lead engineer)

- **LoKR cross-base conversion — VALIDATED failure, VALIDATED fix.** Converting adapters to non-turbo XL bases FAILS even though the bases are ~99% weight-identical. Root cause is **basin-sensitivity** (the adapter delta lands in a different loss basin), NOT weight drift. Fix: the **β·(S−T) basin nudge** (`adapter-merge.h:78`, applied in `dit.h:578` for merge; `adapter_runtime_rebase` in `adapter-runtime.h` for runtime) — **user-validated by ear in merge mode, 2026-07-14** ("works fantastically"); the runtime-mode port (`7aac3c2`, same day) is mathematically output-identical but awaits its own by-ear A/B. Never assume weight similarity implies adapter transferability.
- **Multi-adapter stacking (issue #72) — VALIDATED.** Implemented as a sideband stack (`g_hotstep_params.adapters`) with read-from-pending merge accumulation (merge mode sources each tensor's current post-prior-merge value) and delta-sum in runtime mode (`adapter_stage_delta` — per-step cost and VRAM flat regardless of stack depth on the non-section path). CRITICAL: basin re-base once per stack, not per adapter (Golden rule 3).
- **Per-section masking — VALIDATED (shipped P1+P2).** GGML clobbers input buffers, so masks/weights must be re-uploaded every step, never cached on device (Golden rule 1). Phase 1 = proportional token map; Phase 2 = alignment-based timing; the token→section map is now header-anchored (falls back to char-proportional when header count ≠ section count — the log line says which).
- **Regional self-attn isolation — VALIDATED failure.** `0f3bf6d` → reverted `ee041e1`. Do not re-attempt without a new design for cross-section coherence (Golden rule 4; safer untried idea in reference.md).
- **Cache-key gaps + load-failure caching — VALIDATED fix (`168dcb5`).** Golden rule 2. The specific historical gaps were the time_embed/proj_in group scales and the runtime-mode marker; the failure-caching bug installed adapter-less models under adapter-bearing keys.
- **Cross-ARCH conversion (XL↔2B) — UNSOLVED; per-layer linear surgery is ruled out.** Even with excellent per-layer alignment (CKA 0.90-0.99), independent per-layer linear transforms compound across 24 layers to ~5% end-to-end fidelity. Do not retry crop/pad, ridge, CCA, or Procrustes weight surgery. Distillation is the unblocked direction (details + open items in reference.md — several artifacts there are cited from local docs and unverified).

## Debug tooling

- `HOTSTEP_BCAST_TEST=1` then run `engine\build\Release\ace-synth.exe` — numeric self-test of mask broadcast ops on the real backend, no models needed (`engine/tools/ace-synth.cpp:160`).
- `HOTSTEP_SECTION_NOMASK=1` — apply section deltas unmasked (`dit-graph.h:90`). Adapters audible ⇒ mask values are the problem; still silent ⇒ wiring/upload is the problem.
- Logs: newest `logs\<session>\ace_engine.log` or `logs\<session>\generations\gen_*.log`. Useful greps: `[Adapter-RT] Loaded`, `[Adapter-RT] P2:`, `header-anchored`, `[Adapter] Basin re-base`, `Per-section masking`.
- Audible validation is **by ear only** — ask the user to listen; do not use a browser agent for UI/audio verification. Do not delete experiment output files on your own prediction of their worth.

## Deeper reading

- [reference.md](reference.md) (this folder) — merge/runtime internals, exact cache-key recipe, P2 alignment pipeline, cross-base/cross-arch research history, status ledger.
- `docs/plans/per-section-adapter-masking.md`, `docs/plans/cross-arch-adapter-conversion.md`, `docs/plans/multi-adapter-runtime-switching-handoff.md`, `docs/plans/2026-04-18-adapter-group-scales-investigation.md` — **local-only, gitignored; may be absent on a fresh clone.** Their load-bearing content is distilled into this skill and reference.md.
- `engine/docs/ARCHITECTURE.md` — engine-wide context (committed).
