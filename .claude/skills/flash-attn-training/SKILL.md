---
name: flash-attn-training
description: How HOT-Step's custom flash-attention training ops (GGML_OP_FLASH_ATTN_TRAIN/_BACK) work, what the AS1.5 DiT trainer campaign proved and disproved, and the exact contract for porting flash mode to the other trainers (AS1.5 LM, MM3 LM, MM3 DiT). Use when adding --attn flash to any ace-train subcommand, touching engine/ggml/src/ggml-cuda/fattn-train.*, changing a trainer's VRAM model, debugging "flash is slower/uses more VRAM than expected", or interpreting any flash-vs-exact measurement.
---

# Flash-attention training (the fused backward) — adoption playbook

Written 2026-09-02 from the AS1.5 DiT campaign (commits 28ca16d3 → 10c37556).
Everything here was measured on an RTX 5090 (32 GB, sm_120) unless it says otherwise.
The deep docs are **gitignored, local-only** (`docs/plans/2026-09-01-flash-attn-backward.md`,
`fattn-train-spec.md`, `fattn-train-tf32-design.md`); this skill is the committed distillation.

**Context for a reader with zero prior exposure:** ggml's autodiff had no attention backward,
so every trainer built attention as `mul_mat → soft_max_ext → mul_mat` and retained the
`[S,S,Nh]` softmax per layer for the backward — the O(S²) term that capped DiT training crops
at ~50 s of audio on 32 GB. We wrote our own fused forward+backward ops (CPU reference + CUDA
TF32 kernels), carried as a patch on the vendored ggml submodule. Attention memory is now
linear in S; the DiT auto-fit picks full-song crops. Rob ear-validated the first flash-trained
adapter as "fantastic".

## 1. What exists

| Piece | Where | Notes |
|---|---|---|
| Ops `GGML_OP_FLASH_ATTN_TRAIN` / `_BACK` | `engine/ggml/include/ggml.h`, `src/ggml.c` (constructors, view getters, **autodiff case**), `src/ggml-cpu/ops.cpp` (f32 reference), `src/ggml-backend-meta.cpp` | Appended at the END of the op enum. Forward output is ONE packed tensor: O `[D,Nh,S,B]` then LSE `[Nh,S,B]`; `ggml_flash_attn_train_get_o()` views O out. Backward packs dQ\|dK\|dV. |
| CUDA kernels | `engine/ggml/src/ggml-cuda/fattn-train.cu/.cuh` (NEW files — never touch the inference `fattn-*.cu/.cuh`) | Scalar f32 v1 kernels kept as strict mode + pre-sm_80 fallback; TF32 mma (m16n8k8) kernels are the default. Bitwise-deterministic in every mode: no fp atomics, fixed schedules. |
| Precision knob | `ggml_flash_attn_train_set_prec/get_prec` (op_params slot 3) | `GGML_PREC_DEFAULT` (= 0 = zero-init!) → TF32 on sm_80+; `GGML_PREC_F32` → v1 scalar. Autodiff copies the forward's prec onto the backward node. |
| Patch | `engine/patches/flash-attn-train.patch` (+ `alloc-free-blocks.patch`) | CI applies `engine/patches/*.patch` onto a clean submodule in every build job (release.yml ×3, cache-warm, rocm). `verify-hooks.ps1` Hook 12/13 grep the markers. |
| DiT trainer surface | `ace-train train-dit --attn exact\|flash\|flash-f32` (default `exact` in the CLI; the Training Studio form defaults to `flash`) | `dit_attn_flash()` in `engine/src/train/dit-train-graph.h` beside the untouched `dit_attn_f32()`. Both self- and cross-attention route through it. |
| Parity harness | `engine/tools/fattn-train-test.cpp`, target `fattn-train-test` | `--backend cpu\|cuda`, `--prec f32\|tf32`, `--extra`, `--large`, `--bench`, `--bench-tr` (the trainer's three real geometries). |
| Profilers | `--profile-step N` (coarse buckets); `DIT_PROFILE_NODES=1` per-node with site attribution (`engine/src/train/dit-node-profile.h`) | Node profiler is env-gated, zero cost when off. |
| Server/UI | `attnBackend: 'exact'\|'flash'\|'flash-f32'` through `types.ts` → `routes/training.ts` → `aceTrain.ts`; Training Studio checkbox | `cropMax 0` = "no pin" end to end (see trap 6). |

## 2. The adoption contract (non-negotiable, proven necessary)

Every trainer that gains flash mode must keep all of these. Each one exists because its
absence bit us.

1. **Per-trainer mode flag, default `exact`, and exact means byte-identical.** With the flag
   off the emitted graph must be the pre-flash graph to the byte — the DiT proves it with T3
   (`0.00e+00` on 17 named taps) and SC1–SC3 (`0.000e+00` grad delta) against a
   reverted-tree baseline. Gate the mode at the attention call sites only; restructure nothing else.
2. **A `supports_op` probe at trainer init, hard error on false.** `ggml_backend_supports_op`
   returning false is NOT an error in this engine: `backend_sched_new` registers the CPU backend
   alongside CUDA, so the scheduler silently splits attention onto the CPU — correct, unusably
   slow, low VRAM, tripwire silent, i.e. indistinguishable from a pass on every number the run
   reports. Build a scratch no_alloc node pair at the run's REAL shapes (both attention sites,
   effective Nkv) and abort with a named error. See DiT `dit-train-run.h` "spec 9.8 probe".
3. **A parity/selftest rung, exact vs flash, on CPU f32.** Gate on the CPU backend where both
   arms are f32; CUDA exact-vs-flash deltas (~3e-3) size the *reference's* cuBLAS TF32 rounding,
   not the fused op. Also gate the CUDA supports_op result so a silently-CPU flash arm can't pass.
4. **A measured drift class, documented like `--bwd mm`.** DiT: over 200 same-seed epochs flash
   drifted *less* than `--bwd mm`. Not identity — never claim identity.
5. **That trainer's VRAM model taught the flash branch** — otherwise the auto-fit keeps pricing
   the retained softmax and the flag buys nothing. See §5.
6. **Record the RESOLVED precision** (`attn_prec`) in the run log, not just the requested mode.
   Reason: op_params zero-init == `GGML_PREC_DEFAULT`, so every `--attn flash` run on Ampere+
   was ALREADY TF32 before the knob existed and said nothing about it.

## 3. Per-trainer porting checklist

Adoption is call-site wiring, not kernel work. The ops take any additive F16 mask
(`[S_kv,S]` or `[S_kv,S,1,B]` broadcast), GQA (Nkv < Nh at B=1), S_kv ≠ S, and
non-contiguous q/k/v views (only `nb[0]==4` is required — do not `ggml_cont` them, that
gives back the VRAM win).

| Trainer | Files | Specifics |
|---|---|---|
| **AS1.5 LM** (R2) | `engine/src/train/lm-graph.h`, `lm-train-run.h`, `lm-vram.h`, `lm-selftest.h` | Causal = one triangular −INF mask. The kernel skips all-−INF tiles, so causal gets ~half its compute skipped free. Qwen GQA at B=1 is the tested path. Add the LM VRAM model's flash branch (same over-predict rule). |
| **MM3 LM** (R3) | `mm3-lm-train-run.h`, `mm3-lm-load.h`, `lm-kvprefix.h` | The "sequence term was quadratic all along" retained softmax is exactly what goes. `--prefix-frames` (no-grad frozen K/V) composes but the fused backward computes dK/dV for the prefix columns and discards them — harmless, wasted; measure before building a no-dK/dV variant. Re-derive crop/prefix budgets afterwards. |
| **MM3 DiT** (R4) | `mm3-dit-train-*.h` | Bidirectional like the AS DiT; smallest win (shorter sequences). |

For each: (a) sibling `xxx_attn_flash()` returning exactly the shape the manual chain returned;
(b) flag + log fields; (c) probe; (d) selftest rung; (e) VRAM branch; (f) drift A/B;
(g) `--bench-tr`-style measurement at that trainer's REAL geometries (see §4).

## 4. Measurement discipline (where every wrong conclusion came from)

- **Pair arms at equal graph shape.** "Exact vs flash at crop 1250" once compared exact
  auto-fit crop 820 against flash's 1250; pinning both to 1250 forced exact into 2 checkpoint
  segments. Paired properly (same S, same segments, back to back): **flash is ~8.5% SLOWER than
  exact per token at equal shape on the DiT.** Flash's win is the CROP it affords, not per-token
  speed. Any claim otherwise needs a paired, interleaved measurement.
- **Interleave and repeat.** This box drifts ~7% between invocations; run-to-run contention is
  ±10%. Only within-invocation paired ratios are readable. Use 3 runs per arm.
- **Bench the real geometries.** `--bench` (window mask only) flattered fused. The trainer has
  three: windowed self (fused 0.89× cuBLAS), full self with NO mask (1.18×), cross at
  S_kv = enc_S (1.52×). `--bench-tr` covers all three. Half the DiT layers are full attention
  (`layer_type = i % 2`) and get no tile skip.
- **Attribute before fixing.** `DIT_PROFILE_NODES=1` found the whole flash deficit is the
  cross-attention BACKWARD (67.8 vs 34.8 ms/step); self-attention is a wash, cross forward is
  2× faster. Root cause: both TF32 backward kernels split warps by output d-range and recompute
  the shared S/dP tiles (dK/dV 1.5×, dQ 1.67× the needed mma). A dQ role split measured −2.7%
  end to end → **reverted under a 3% bar**. dK/dV split is blocked by ~128 B of static shared
  memory at the 3-blocks/SM occupancy cliff. Recorded in the plan doc; not worked around.
- **A 3% end-to-end bar for kernel churn.** Isolated-kernel wins of 15–25% can be 1% of a step.
- **Loss-to-target speed ≠ quality.** The overnight sweep's fastest-to-0.5 config (LoRA r128,
  pinned short crop) is a step-cost win that inverts at long crops; the 0.5 proxy's leader
  changed three times between ma5 0.8 and 0.5. Ear tests decide; nothing trained in flash mode
  after the first adapter has been heard.

## 5. VRAM model rules

- **Estimate must over-predict, never under** (target +5–15%); the NVML tripwire and the
  high-water probe are the backstop, never the plan.
- The exact-mode arena polynomial hides an **enc_S** dependence in its linear coefficient. The
  flash branch (`dit_vram_arena_bytes_flash`) takes enc_S explicitly; cross-attention scales
  with `enc_S×S` and at crop 1250 exceeds self-attention's S² — "enc_S is small" was refuted.
- **Read the arena log line as the TOTAL.** A "4319 est vs 7824 measured" line that omitted the
  LoKR-apply term sent a whole refit chasing a non-existent under-prediction; the total was
  over-predicting 73%. The line now prints both terms — keep it that way in every trainer.
- Fits are per-adapter-graph: `DIT_FLASH_LOKR_RETENTION` (0.62) was fitted before the LoKR
  apply reorder and now over-predicts +16.5% (safe direction, ~one crop step unspent). **Owed
  refit**; the batch>1 term is B=1-fitted and over-conservative.
- The flash lift raises `crop_max` to the dataset's longest track ONLY when the user passed
  no `--crop-max`; `a.crop_max_user` is the pin flag. See trap 6.

## 6. Trap list

1. **ggml.h edits invalidate ~141 CUDA objects — ~1 h rebuild.** Batch header changes.
   New `.cu` files need a cmake re-configure (the ggml-cuda CMake globs `*.cu`).
2. **DLL locks.** A running `ace-server` holds `ggml-base.dll`/`ggml-cuda.dll`; any ggml change
   needs the app down (`/api/shutdown` or `dev-rebuild.bat`). `ace-train.exe` is NOT held, so
   trainer-only edits build with the app up. Never kill ace-server (Node respawns it).
3. **Packed-output alignment gap.** Autodiff builds the packed gradient as
   `ggml_scale(packed, 0)` + `ggml_acc(dO)`; garbage in the O→LSE alignment gap becomes NaN.
   Both CUDA and CPU forwards zero the gap explicitly. Zero-width at every tested geometry, so
   tests never see it — keep the memset.
4. **In-place SCALE hazard.** `ggml_scale` is in `ggml_op_can_inplace`; it is safe only because
   the packed tensor always has a view child. The backward asserts `dst->data != fwd->data`.
5. **GQA at B>1 cannot be parity-tested against the manual chain** (ggml MUL_MAT backward
   asserts on broadcast src0) — that is why `dit_expand_heads` exists. Flash mode skips the
   expansion (native GQA), which also disarms the CUDA `REPEAT_BACK` cap on `Nkv·max(S,enc_S)·B`.
   Measured: **batch 1 still wins** on throughput and loss.
6. **The server always emitted `--crop-max`**, which the engine treats as a user pin → the flash
   lift never fired from the UI. `cropMax 0` now means "omit the flag". Quality presets must not
   re-pin it in flash mode. Any new trainer flag with an engine-side "user set it" sentinel has
   this exact failure mode — check the arg emitter.
7. **The parity tool must seed the loss gradient with 1.0** (`ggml_set_loss` only allocates)
   and assert a non-zero reference gradient, or both arms compare 0 vs 0 and pass vacuously.
8. **`dit_sa_mask` never produces a fully-masked key column** (pad columns stay open for padded
   query rows) — use `dit_ca_mask` for the exactly-zero-gradient assertion.
9. **Fully-masked query rows**: the fused op defines O=0, LSE=0; `soft_max_ext` produces NaN.
   Exclude them from reference diffs, check them directly.
10. **TF32 A-operand lane map ≠ accumulator map.** mma.cuh's `tile<16,8,float>` is the C/D map;
    using it as the tf32 A operand gives deterministic garbage. Derive with a probe kernel.
11. **Patch files are LF; a scratch tree extracted under `core.autocrlf=true` is CRLF** and every
    hunk fails. Replay with `git -c core.autocrlf=false -c core.eol=lf archive`. Export patches
    hunk-filtered: several patches share `ggml.c` and `ggml-cuda.cu`.
12. **rocm-build.yml did not apply patches** until 1b7e50d5 — every workflow that builds the
    engine needs the apply loop now that the trainer references patch-provided symbols.
13. **`MAX_FREE_BLOCKS`** (ggml-alloc) was 256; LoKR dim 256 (19k-node graph) overflowed it.
    Now 1024 via `alloc-free-blocks.patch`. Inference-shared → smoke generation after touching.
14. **Workflows die with the VSCode/Claude process.** Long unattended runs need the window open;
    machine sleep is "never" on this box (checked).
15. **Disk.** Probe runs write adapters; a campaign filled D: to 2.4 GB free and artifacts were
    deleted for space. Clean scratch dirs between grid cells.

## 7. Numbers worth remembering (5090)

| Measurement | Value |
|---|---|
| Fused TF32 vs cuBLAS per site, fwd+bwd, window mask | 0.94× / 0.64× / 0.49× at S=625/1250/3000 |
| Same at the trainer's real geometries | windowed 0.89×, full-self 1.18×, cross(S_kv 1877) 1.52× |
| Attention VRAM per site at S=3000 | 487 MB fused vs 4.9 GB manual |
| Parity worst rel err | f32 3.5e-6 (bar 1e-4); tf32 4.7e-4 (bar 5e-3, floor 1e-5) |
| Flash vs exact drift, 200 same-seed epochs | smaller than `--bwd mm` |
| Done-gate auto-fit, production LoKR, unpinned | nwa 1498 (enc_S 1877), fightstar 1616 (enc_S 640); LoRA r16 ~3400 |
| LoKR apply reorder | −10% step, LoKR:LoRA 1.35→1.21; the two copies are unavoidable, ~7% of step |
| 12 GB emulated card, flash+bf16+LoRA r16 | full 32-layer depth, crop 410, 4 segments |

## 8. Open items (as of 2026-09-02)

- R2/R3/R4 ports (this skill is their brief).
- Cross-attention backward kernel: dK/dV role split blocked by smem; a dQ split exists in
  the plan doc (reverted, −2.7%).
- `DIT_FLASH_LOKR_RETENTION` refit after the apply reorder; batch>1 VRAM term.
- Exact-mode arena polynomial under-predicts 13–18% (masked by LoKR over-count; fix gated to flash).
- Ear validation of anything trained since the first flash adapter, and of the LoKR reorder.
- Low-VRAM training profiles for users (B1) — deferred by Rob until the 32 GB path is nailed.
