# Adapter system — deep reference

Companion to [SKILL.md](SKILL.md). Line references verified against the repo on
2026-10-23 (first written at commit `168dcb5`, 2026-07-02). If line numbers have drifted,
grep for the named symbol.

## 1. Merge mode internals (`engine/src/adapter-merge.h`)

- Entry: `adapter_merge(wctx, ws, path, scale, backend, promote_f32, rebase_source, rebase_beta)` (:1914). Accepts a PEFT directory (`adapter_model.safetensors`, or `adapter_model.bin.safetensors`, + `adapter_config.json`), the Training Studio's LoKr folder (`lokr_weights.safetensors`, alpha per module, no config), or a flat LyCORIS file. The whole file is preflighted for DoRA consistency first (`adapter_dora_preflight`, :894). Then it dispatches: LoKr via `.lokr_w1/.lokr_w2` tensor names (`adapter_detect_lokr`, :514; reverse name map `lokr_build_reverse_map`, :482), LoHa (`adapter_detect_loha`, :1325), else LoRA (`adapter_merge_lora`, :1026, which also handles HiRA when `peft_type` is `HIRA`).
- Alpha resolution order: per-tensor baked alpha (ComfyUI style) → `adapter_config.json` `alpha_pattern` per module (`adapter_read_alpha_pattern`, :321) → `lora_alpha`/`alpha` → fallback `alpha = rank` (scaling 1.0) with a loud warning when the config is missing. `use_rslora: true` switches the scaling to alpha/√r (`adapter_read_rslora`, :243).
- Unified GPU merge graph per tensor (`adapter_merge_on_backend`, :599): upload base in native quant from mmap → `ggml_cast` dequant to F32 on the backend → add the BF16-rounded delta (mirrors PEFT/LyCORIS numerics) → optional DoRA per-row rescale → encode back to native quant when the backend supports it (`adapter_backend_can_encode`, :458), else host-side `ggml_quantize_chunk`. NVFP4/MXFP4 lack a GPU cast → a split path promotes to BF16 instead of requantizing (host FP4 requant is ~60 s per model). HiRA needs the unified path and is refused on the split path.
- `promote_f32 = true` (the default "merge"/merge_hq behavior): merged weights stay F32 to avoid catastrophic BF16 cancellation. It **skips non-layer tensors whose group scale is 0** (`adapter_hq_should_skip`, :876 — time_embed/cond_embed default to 0 in HQ mode). FP4 base models auto-disable promotion (`dit.h:756-765` — saves ~13 GB VRAM), and so does the opt-in low-VRAM merge (`adapter_merge_lowvram`, UI "Merge VRAM: Low ¼", `dit.h:766-771`).
- **Multi-adapter accumulation:** each merge sources the base from the tensor's *current pending copy by name*, so a stack merges as `W ← W + s1·Δ1 + s2·Δ2 + …` sequentially (see comment at `dit.h:773-776`). Pre-permuted Conv1d (proj_in 3D→2D) is handled.
- Conv1d shape expansion: adapter delta `[out, in_ch]` vs base `[in_ch*P, out]` → delta tiled with `ggml_repeat` across the patch dimension (both merge and runtime paths).
- **Basin re-base** (`adapter_rebase_fetch`, :79; applied inside the merge before the delta add): `base ← base + β·(S − base)`, where S is the adapter's home-base safetensors (absolute path in `g_hotstep_params.rebase_source`). Applied to the FIRST stack adapter only (`dit.h:778-801`). Shape-preserving, so useless for cross-architecture transfer. Fatal failure signature if misused: at β=1, per-adapter application replaces the base each merge — only the last adapter survives. Runtime mode has its own implementation (§2).
- DoRA (`dora_scale` / `lora_magnitude_vector`) merges in every case. Runtime mode applies it only with `adapter_runtime_dora` on (§2); otherwise the runtime loader warns that rescaling is not applied (`adapter-runtime.h:306-309` for PEFT, `:521-523` for LoKr), and the UI warns too.
- ConvRot (int8 rotated) bases refuse merge mode outright (`dit.h:750-754`): merged deltas would be unrotated.

## 2. Runtime mode internals (`engine/src/adapter-runtime.h`)

- Deltas are precomputed on GPU per projection (`adapter_compute_delta`, `adapter-merge.h:556`, shared with merge mode), staged host-side as F32, then finalized in one buffer alloc with parallel quantize + upload (`adapter_runtime_finalize`, :831; log `[Adapter-RT] Loaded N deltas (... MB, N quantized to ...) into VRAM`).
- Storage type (`adapter_runtime_storage_type`, :159): `"q4_0"` (and legacy `"q4_k"`, aliased after `47fedfe`) → Q4_0; `"q8_0"` → Q8_0; both require `ne0 % 32 == 0`, otherwise BF16 fallback. Q4_0 ≈ ¼ the VRAM (~2.2 GB per XL adapter unquantized).
- Slot model (`dit_lora_slot`, :105): 32 layers × 11 projections (self-attn q/k/v/o, cross-attn q/k/v/o, mlp gate/up/down) + proj_in, cond_emb, and 6 time_embed tensors. Tensors without a slot are merge-only and skipped with an INFO log.
- **Stacking = delta sum:** `adapter_stage_delta` (:173) sums a later adapter's F32 delta into the already-staged tensor for the same projection. Per-step compute cost and VRAM are flat regardless of stack depth on this (non-section) path. Entry for stacks: `adapter_load_runtime_stack` (:1121), which preflights every file for DoRA first.
- **Low-Rank** (`adapter_runtime_lowrank_active()`, :206): factors are staged instead (`adapter_stage_factor`, :214, tensors named `lrf_*` so the re-base never touches them) and applied by the graph as factor units; `dit_lora_unit_nodes` (:243) sizes the graph budget. A re-based Low-Rank stack stages zero-filled full-size correction deltas on the first adapter (`adapter_stage_zero_correction`, :230) for the nudge to fold into.
- **Runtime re-base** (`adapter_runtime_rebase`, :929): folds β·(S−T) into the first adapter's staged deltas, once per stack (`7aac3c2`); log `[Adapter-RT] Basin re-base: nudged N/M deltas ...`.
- **Opt-in DoRA** (`adapter_runtime_dora_stack`, :1039): when any stack file carries DoRA and `adapter_runtime_dora` is on, the loader builds merge-equivalent effective deltas through the merger (log `[Adapter-RT] DoRA opt-in: merge-equivalent effective deltas`), in Low-Rank mode too. Cache-key marker `|dora:merge-v1`.
- LoHa and HiRA have no runtime form and are refused (`[Adapter-RT] LoHa adapters have no runtime path ...`, `[Adapter] HiRA adapter cannot use the runtime low-rank path ...`).
- Vulkan: the precision-rounding cast uses F16 instead of BF16 (no bf16→f32 shader on that backend; log `[Adapter-RT] Vulkan backend: using F16 precision rounding`).
- Cancel and progress: the precompute loops check `adapter_cancel_requested()` between every delta (<100 ms cancel latency) and publish the current delta index in `g_adapter_progress` (`adapter-cancel.h`), which `/job` reports as `adapter_progress` and the Node watchdog counts as progress. The worker points `g_adapter_cancel` at the job's cancel flag around `ace_synth_load` with a scope guard (`hot-step-server.cpp:1567-1570`; `/warm` does the same at :2916). Job phase `ADAPTER_PRECOMPUTE` (= 7, `hot-step-server.cpp:316`) is reported during the ~17 s cold-start precompute.
- Graph application: `dit_ggml_linear_lora` (`dit-graph.h:118`) adds `delta@x` (and any factor units) beside `W@x`. `mul_mat` dequantizes quantized deltas transparently; nothing else may touch delta bytes assuming BF16 (`46603bf`).

## 3. Cache-key recipe (exact)

`ModelKey` DiT extras (`model-store.h:83-96`), built in `pipeline-synth.cpp:176-241`:

- `adapter_path` (primary adapter), `adapter_scale`
- all six `adapter_group_scales` (self_attn, cross_attn, mlp, cond_embed, time_embed, proj_in — the last two were the `168dcb5` gap)
- `rebase_source` / `rebase_beta` — merge mode AND runtime mode (since `7aac3c2`); explicitly cleared on the runtime per-section path (sections + 2+ stack), where the engine skips the nudge
- `adapter_stack` string = `hotstep_adapter_stack_sig(...)` (`hot-step-params.h:306` — each path + `@%08x|` float-bit-pattern of its scale; gain curves deliberately left out), then:
  - `+ "|sect"` when `adapter_sections` is non-empty (per-section loads N separate deltas — distinct DiT; the section weights themselves live in per-frame masks, not the model, so they are NOT keyed)
  - `+ "|dora:merge-v1"` when either runtime mode has `adapter_runtime_dora` on
  - `+ "|q:<quant>"` when either runtime mode has `adapter_runtime_quant != "bf16"`
  - `+ "|mlv"` when an adapter is loaded, the mode is not `runtime`, and `adapter_merge_lowvram` is set (low-VRAM merge keeps native-quant merged weights)
  - `+ "|mode:runtime"` when runtime mode with any adapter — a merge↔runtime toggle must never reuse the cached DiT
  - `+ "|mode:runtime_lr"` when Low-Rank mode with any adapter

Hash: `model-store.cpp:43-85`. Equality: `model-store.cpp:87-115`. **Adding a key field requires touching all three sites.** The same hash and equality key planner LMs on their adapter path and scale bits, and StableStep's SA3 DiT on `adapter_stack`.

Load-failure rule: a runtime adapter load failure FAILS `dit_ggml_load` (`dit.h:868-879`), freeing the already-GPU-resident wctx (`dit_ggml_free`) because the store's failure path only does `delete m`. Per-section requires ALL stack adapters to load (`dit.h:847-858`) — a partial stack would silently generate with adapters missing and be cached that way. A merge that changes no tensor fails too (`[Adapter] FATAL: no tensors merged (model mismatch)`).

## 4. Per-section masking — P1/P2 pipeline detail

- `AceRequest.adapter_sections` = `[{weights: number[], size: number}]`; `weights` indexed to stack order, `size` = section character count (frame-allocation hint). Built by `parseAdapterSections` (`server/src/services/generation/adapterSections.ts:113`).
- **P1** (`hot-step-sampler.h:385-402`): partition the S latent frames proportionally by section `size`; build per-adapter `[S]` float masks with a triangular crossfade of `xf = max(3, S/300)` (~0.5 s; shortened from a longer fade in `ce57675` to reduce transition garble). `rebuild_section_masks` (:353) writes `lora_mask_host` and uploads.
- **P2** (`run_p2_alignment`, :533; invoked at :1030 and :1353 when the step reaches `p2_align_step`, :529): fires at `adapter_section_align_at` × num_steps (default 0.55; exposed in the UI as "Alignment Timing"; useful tuning range 0.3-0.4 per the lead). Estimates `x0 = xt − t·vt`, runs `dit_alignment_extract` with flash attention off on a **private scheduler** (`dit-alignment-graph.h:322`, never `dit->sched` — `4e48176`), obtains cross-attn `[enc_S, S]`, maps each frame to its dominant token, token → section via `g_hotstep_params.adapter_section_token_map`, median-smooths, then `rebuild_section_masks(..., "alignment")`. Any failure keeps the P1 proportional map and logs why.
- **Token→section map** (`pipeline-synth-ops.cpp:1376-1458`, built only when sections and alignment are both on): scans lyric token text for `[...]` headers. Anchored when header count H == n_sections or H+1 == n_sections (server may emit a preamble section before the first header); otherwise falls back to char-proportional. Log line: `[Adapter-RT] P2: token→section map (... header-anchored|char-proportional ...)`.
- Masks must be forced to the backend in the scheduler (`force_inputs_to_backend`) and marked resident for the direct graph (`mark_resident_inputs`); in NOMASK mode they must not be created at all (`34dce60`).
- Known limitation: cross-attn k/v and cond_emb use the scalar mean section weight (`dit-graph.h:527`, `:903`), so text conditioning is a constant blend across the whole song — open issue.

## 5. Reverted: regional self-attention isolation

- `0f3bf6d` added a cross-section self-attn logit penalty (penalty = isolation × 8) to fight first-adapter bias. `ee041e1` reverted it: multi-second silence gaps between sections, degenerate later sections. Self-attention carries both adapter identity and musical coherence.
- `adapter_section_isolation` remains plumbed but dormant: `hot-step-params.h:207`, `ServerFields` field and parse `hot-step-server.cpp:715/:836`, sideband fill :1475, `translateParams.ts:249` — the engine reads it into `g_hotstep_params` and then ignores it.
- Structurally safer untried idea on record: **section-aware CROSS-attention** — a token-axis mask on ca k/v + cond_emb using the existing token→section map. Cross-attn does not carry musical continuity, so it should not reproduce the silence-gap failure. Uncertain payoff: first-adapter bias may already be near the ceiling for weight-masking on a coherence-trained model. Try free knobs first: Alignment Timing 0.3-0.4, and raising the intended adapter's per-section directive weight to 1.2-1.4 in the lyric directive itself (e.g. `[Chorus]{myadapter=1.3}` — there is no separate "dominant section weight" param).

## 6. Cross-base / cross-arch conversion research history

Source: `docs/plans/cross-arch-adapter-conversion.md` (LOCAL-ONLY, gitignored — may be absent). Distilled here; items marked (unverified) were cited from that doc without opening the artifacts.

### Cross-base (same architecture, different base weights)

- **Ground truth (departing lead):** LoKR cross-base conversion to non-turbo XL bases FAILS despite ~99% weight-identity. Root cause: **basin-sensitivity** — the adapter delta lands in a different loss basin — NOT weight drift. Candidate fix: β·(S−T) basin nudge, designed but **NOT validated** at the time.
- The nudge IS implemented and shipped (SKILL.md Golden rule 3; `W_merged = T + β·(S−T) + scale·ΔW`). **User-validated by ear in merge mode (2026-07-14, "works fantastically")**, resolving the earlier handoff conflict. Runtime-mode port (`7aac3c2`) folds β·(S−T) into the staged delta sum (`adapter_runtime_rebase`) — output-identical math, pending its own by-ear A/B. 2B→2B re-base: expected to work, untested.
- Re-base is shape-preserving → does nothing cross-architecture.

### Cross-arch (XL 32 layers/2560 hidden ↔ 2B 24 layers/2048 hidden) — UNSOLVED

- Six approaches failed. **Decisive finding:** even with excellent per-layer alignment (CCA on real inputs: CKA 0.90-0.99, fit residual 0.120), the baked model is garbled — independent per-layer linear transforms compound across 24 layers (~0.88²⁴ ≈ 5% end-to-end). **Any per-layer linear weight surgery (crop/pad, ridge alignment, CCA, Procrustes) is ruled out — do not retry.**
- Shared spaces that matter for future work: 64-channel output latent (both feed the same VAE), text-encoder 2048-dim, timestep embed 256, KV heads 8 × head_dim 128 (k/v output matches at 1024), `condition_embedder` / `time_embed.linear_1` as a "Rosetta Stone".
- **Distillation is unblocked:** the v1 non-convergence was an optimization bug (LR 2e-3 + single-sample SGD), not bf16 precision (measured bf16 noise = 4.5% of teacher delta). fp32 + per-sample-normalized loss + AdamW warmup/cosine reached rel error 0.09 by step 150 on an 8-tuple overfit. **Real, on-manifold conditioning inputs are mandatory** — synthetic Gaussian inputs are useless.
- Status (in progress, unverified artifacts): capture 384 real tuples (`capture_dit_inputs_v2.py`), rank-128 training (`distill_v3_train.py`), validate on held-out prompts. Scripts in `docs/plans/cross-arch-scripts/` (local-only, unverified). Real-input capture uses the Python sister project `D:\Ace-Step-Latest\hot-step-9000` (`ACESTEP_INIT_LLM=false`, `PYTHONIOENCODING=utf-8`) (unverified). Distilled students export as normal PEFT adapters — the engine already loads them.
- Ranked next directions: (A) learned paired converter from (XL, 2B) ground-truth adapter pairs; (B) distillation as a real training job; (C) conversion-as-init + light finetune. Unpursued: cross-model output-space guidance (run 2B + XL_adapted + XL_base each step, add `(XL_adapted − XL_base)` — zero training but 3 forward passes/step with both models resident).

## 7. Status ledger

**DONE / shipped:** merge, runtime and Low-Rank modes (LoRA, LoKr; DoRA in merge and, opt-in, in both runtime modes; LoHa and HiRA merge-only; rsLoRA and per-module `alpha_pattern`); DoRA config preflight; multi-adapter stacking (merge accumulation, runtime delta-sum, Sum/Blend UI, all trigger words injected); basin re-base (once-per-stack, merge + runtime summed paths + Low-Rank via zero corrections; not per-section); per-section masking P1 (proportional) + P2 (alignment, header-anchored token map); timestep-dependent gating; runtime delta quantization Q4_0/Q8_0 with parallel finalize; low-VRAM merge; runtime adapters on ConvRot bases; cooperative cancel and the `adapter_progress` heartbeat; cache-key hardening + no-failure-caching (`168dcb5`); LM-echo rebuild fix (`8ea519b`); parameter profiles that save the whole stack (`ui/src/utils/training/paramProfiles.ts`).

**REVERTED:** regional self-attn isolation (`0f3bf6d` → `ee041e1`); `adapter_section_isolation` param dormant.

**PLANNED / open:** first-adapter bias mitigation via section-aware cross-attention (untried); masked Low-Rank apply, so sections and step windows stop forcing full deltas; 2B→2B re-base confirmation; cross-arch distillation v3 training run; per-adapter group scales / per-adapter re-base (both still apply to the whole stack, `hot-step-params.h` `AdapterSpec`); multi-adapter album presets (the album-preset tables hold one adapter path per column).

**Commit archaeology (all hashes verified in git log):**

| Commit | What |
|---|---|
| `168dcb5` | Cache-key gaps (time_embed/proj_in group scales, runtime-mode marker), load-failure caching, sampler fixes |
| `8ea519b` | LM-echo adapter param plumbing (rebuild from `{...aceReq}`, no whitelist) |
| `ee041e1` / `0f3bf6d` | Revert / original self-attn isolation |
| `46603bf` | Removed BF16-assuming delta diagnostic (crashed on quantized deltas) |
| `a6db135` | Re-upload per-section masks every step (the "nil adapter" bug) |
| `2052bc3` | Re-upload per-section masks after CFG-cutoff graph rebuild |
| `4e48176` | Private scheduler for mid-sampling alignment (P2 CUDA crash) |
| `e09d6a3` | Scale scheduler hash-set with adapter count |
| `ce57675` | ~0.5 s per-section crossfade (transition garble) |
| `47fedfe` | Fast Q4_0 + parallel quantization for runtime delta VRAM saver |
| `34dce60` | NOMASK debug crash on unused mask input tensors |
| `7aac3c2` | Basin re-base in runtime mode |
| `73e057b6` | Upstream StaticGraph direct execution ported to the sampler (resident constant inputs) |
| `4da6e562` | `runtime_lowrank` mode: factor apply, no materialized deltas |
| `54254ade` | rsLoRA, LoRA+, HiRA and LoHa parameterizations |
| `1ef5cb1e` | Opt-in ACE runtime DoRA rescaling |
