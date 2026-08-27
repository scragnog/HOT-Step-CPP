# Engine Performance & TensorRT — Reference

Deep detail supporting [SKILL.md](SKILL.md). Line references verified 2026-07-02 against the working tree; they will drift — re-verify with Grep before editing code from them.

## 1. Native TRT DiT internals (`engine/src/dit-trt.h`, ~767 lines)

### Build — `dit_trt_build` (:198)
- ONNX → engine with `kSTRONGLY_TYPED` (mandatory in this TRT version), `kTF32`, and `kREFIT_IDENTICAL` + `kSTRIP_PLAN` (:258) — produces a weight-stripped plan (~50 MB); weights are refit from the ONNX at every load.
- 4 GB workspace. Dynamic shape profile: `input_latents [B,T,192]` T min/opt/max 64/2048/8192, `enc_hidden [B,S,2048]` S 64/512/2048, batch min 1 / opt 1–4 / max ≤16.
- Prints `[DiT-TRT] This will take 5-30 minutes (first run only).` (:207) and a **heartbeat thread** logs `Engine build in progress... (Ns elapsed)` every 30s (:324) so the Node stall-detector does not kill the job.
- `kSTRONGLY_TYPED` forbids per-layer precision overrides — precision problems (e.g. norms that must be fp32) can only be fixed by re-exporting the ONNX (see comments near :243).

### Load — `dit_trt_load` (:377)
1. Deserialize the `.engine` plan.
2. `nvonnxparser::createParserRefitter(...).refitFromFile(onnx)` (:425) — restores stripped weights from the ONNX.
3. Cache all BF16/FP16 weights in host RAM (`base_weights`) so adapters can be reverted without reloading.
4. Read `<onnx>.refit_manifest.json` sidecar (:464) — emitted by `tools/onnx-export/export_dit.py`, lists dynamo-transposed weights (stored `[in,out]`) so adapter deltas get transposed correctly.
5. Create a **dedicated CUDA stream** (:562-564) — fixes the `[TRT-WARN] Using default stream in enqueueV3()` perf warning.

### I/O dtype detection (~:539)
Three modes: **bf16** (standard dynamo export; host staging fp32↔bf16), **fp16**, **fp32** (FP8 QDQ modelopt export; direct upload). Adapters are only supported on BF16-I/O engines — FP8/fp32 engines skip adapters with a warning (`pipeline-synth-ops.cpp` ~:1320).

### Forward — `dit_trt_forward` (:674)
`setInputShape` + `setTensorAddress` + `enqueueV3(stream)`. First call dumps 8 output values as a NaN early-warning: `[DiT-TRT] DIAG first output:` (:743).

### Lifecycle
Static context in `pipeline-synth-ops.cpp` keyed on ONNX path (`s_trt_onnx_path`, :1230-1234) — model switch triggers reload/rebuild. Eviction (:1388-1395): `EVICT_STRICT` (default) frees the engine from VRAM after each generation; `EVICT_NEVER` ("Keep DiT & VAE loaded" / server `coResident`) keeps it resident.

## 2. TRT sampler (`engine/src/hot-step-sampler-trt.h`, ~850 lines)

Mirrors the GGML sampler (`hot-step-sampler.h`); only the forward pass differs. Full parity: Lua solver/guidance/scheduler plugins, custom timesteps, DCW, repaint, cover-mode context switch, SDE, batched CFG (2N batch) or 2-pass CFG, native APG fast path.

- Data path: host FP32 → staging bf16/fp16 → GPU **per step** (~:345-426). `enc_hidden` is re-uploaded every step here too.
- **Step-velocity cache** (`cache_ratio`): :484-515 — skips middle-step forwards reusing last velocity; first/last steps protected; logs the cached-step plan.
- **CFG cutoff mid-loop**: :571-611 — at the cutoff step, GPU buffers downsize 2N→N and guidance stops.
- First-step latency (incl. warmup) logged: `(first step: N ms)` (:700).
- Streaming batched step: `dit_trt_step` (:739-826) — one TRT forward for N ring-buffer slots with per-row timesteps; Euler-only per-slot integration (~:841). Pre-allocated `DitTrtStreamBuffers` live in `dit-trt.h` (~:120-187).

## 3. Adapter refit (`engine/src/adapter-trt.h`, ~675 lines)

- Entry `adapter_trt_apply` (:604): opens safetensors, detects LoKr vs LoRA, dispatches to `adapter_trt_apply_lokr` (:294) / `adapter_trt_apply_lora` (:166), then `dit_trt_refit_adapter` (`dit-trt.h:577`) sets named weights and calls `refitCudaEngine()` under a mutex.
- **The batching fix:** LoKr Kronecker deltas are computed on GPU via the GGML backend in **micro-batches of 32 modules per graph** (:458-460). The old per-module approach caused ~359 separate cudaMalloc/free cycles ≈ ~14s of pure CUDA driver overhead; now ~1.8s. Logs: `[Adapter-TRT] LoKr: N valid entries, computing in micro-batches...`, `Batched GPU merge: N ms (...)` (:588), `Applied in N ms`.
- Transposed weights: refit manifest tells the merge which deltas to transpose (dynamo `[in,out]` storage).
- Name mapping: adapter `decoder.X` ↔ TRT `dit.X`; LoKr lycoris prefixes reverse-mapped from `base_weights` keys. Failure = `[Adapter-TRT] No weights matched TRT engine — adapter has no effect` (:662).
- Revert = re-refit cached base weights (`dit-trt.h` ~:630). Adapter switch = revert-then-apply; wall clock logged if >50 ms.

## 4. TRT LM (`engine/src/lm-trt.h`, ~738 lines)

- Qwen3 4B, single full-vocab engine. `LM_TRT_VOCAB 217204` (:42) — **padded**; the planned trim to 151,936 never landed. Phase-2 audio-code logit slicing at offset 151645 done in C++ (~:44, ~:640).
- **KV cache: double-buffered** device memory per layer per set (`[1,8,max_seq,128]` bf16 × 36 layers × 2 buffers × n_kv_sets), swap A/B each step (~:85-91). The planned single-buffer append was not implemented. VRAM scales as n_kv_sets(=2×max_batch) × 36 layers × 4 buffers × max_seq — grows fast.
- Single wide optimization profile, decode-optimal (opt seq=1); a second prefill profile was planned, not implemented (~:226-279).
- Micro-optimizations present: pre-cached tensor-name strings, attention mask pre-filled once on device, static tensor addresses bound once, stack-allocated ≤16-token id staging, CFG batch defers stream sync to the last element.
- Dispatch: `pipeline-lm.cpp:78` (`s_use_trt`, declared at :66, routes vocab/forward/KV ops). GGML LM load is skipped entirely when TRT is active.
- **Teardown (critical):** `lm_trt_free` does `cudaStreamSynchronize` then `cudaDeviceSynchronize` **before** freeing (:702-709). Removing/reordering corrupts shared CUDA state → GGML crashes afterwards.
- LM probe order at load (`pipeline-lm.cpp:1110-1193`): (1) `trtllm-engine-*` dir → TRT-LLM (compile-time disabled, so effectively skipped); (2) `lm_full.onnx` → raw TRT LM, `lm_full.engine` built if missing; (3) GGUF/GGML.
- Speculative decoding (GGML path): `run_phase2_speculative` (`pipeline-lm.cpp:793`) — 0.6B draft (no CFG) proposes, 4B target verifies with CFG; enabled via `--draft-model`.

## 5. TRT-LLM Executor (`engine/src/lm-trtllm.h`, ~544 lines) — DEAD on native Windows

`engine/CMakeLists.txt:342-371`: separate from `HOT_STEP_TRT`; **STATUS: DISABLED (2026-06-02). "Native Windows TRT-LLM is not viable"** — TRT version mismatch between Docker-built engines (10.14) and the Windows SDK (10.16); ONNX-rebuilt engines lack TRT-LLM tensor bindings (`kv_cache_block_offsets` etc.). Code kept behind `#ifdef HOT_STEP_TRTLLM`; re-enable with `-DHOT_STEP_TRTLLM_ENABLE=ON` (intended for WSL2/cross-platform later). The built `trtllm-engine-RTX5090/` in `models/onnx/lm-4B/` is inert.

## 6. ORT + TRT-EP wrappers (VAE, PP-VAE, text/cond encoders)

Files: `engine/src/vae-ort.h`, `vae-enc-ort.h`, `text-enc-ort.h`, `cond-enc-ort.h`. Gated on `HOT_STEP_ORT_PATHS` (option, `ORT_PATHS_ENABLED` internally) together with `sa3-refine.h` — these are the last ONNX Runtime consumers in the engine. The option was called `HOT_STEP_SUPERSEP` until SuperSep was rewritten on GGML; SuperSep no longer touches ORT and always builds.

- Legacy **V1 TRT EP via C API** (`vae-ort.h:78`), fp16, engine cache written next to the ONNX, CUDA EP appended as fallback.
- **Blackwell (sm_120) gotcha:** ORT-TRT VAE can segfault on RTX 50xx (Myelin fusion bug). Known workaround `builder_optimization_level=1` **cannot be expressed** through the V1 EP options (`vae-ort.h:95` NOTE) — fixing requires the V2 EP API.
- The large `TensorrtExecutionProvider_TRTKernel_*.engine` files scattered under `models/onnx/*` are ORT EP caches — safe to delete, will rebuild.
- Mixed mode is supported: GGUF DiT + ONNX text encoder (`pipeline-synth.cpp` ~:294). When the DiT itself is ONNX and `text_encoder.onnx` + `cond_encoder.onnx` sit beside it, `ctx->is_onnx_pipeline` is set (~:284-325). PP-VAE auto-discovers `models/onnx/pp-vae/pp-vae_{encoder,decoder}.onnx` (`hot-step-server.cpp:2947` new layout, flat-file fallback).

## 7. Streaming pipeline (`engine/src/stream-pipeline.h`)

DEMON-style ring buffer: N in-flight slots at staggered denoising stages; one **batched** TRT forward per tick with per-row timesteps. Config: depth (default 8) and `preview_interval` — **hard-set to 0** at `pipeline-synth-ops.cpp:2251` because decoding partially-denoised latents through the VAE yields scrambled audio; real streaming previews need temporal chunking (PLANNED). Requires ONNX DiT (`:2098`); prefers a sibling `dit-stream/` FP32 model over a possibly-FP8 main DiT; builds a separate `*_stream.engine` with max_batch=8. Node parses `[Stream] tick` and `[STREAM_PREVIEW]` markers (`server/src/routes/generate.ts:787-798`).

## 8. On-disk model layout (as found 2026-07-02 on Rob's machine — machine-specific)

```
models/onnx/
  dit-fp8/     dit_fp8.onnx (+~8GB .onnx_data), text_encoder.onnx(.data), cond_encoder.onnx(.data),
               null_condition_emb.bin, embed_tokens.bin, ORT-TRT EP kernel caches
               NOTE: no dit_fp8.engine — first native-TRT use = full 5-30 min build
  dit-stream/  dit.onnx, dit_stream.engine, dumped decoder norm/scale_shift_table weights
  lm-4B/       lm_full.onnx(.data), lm_full.engine, lm_full.onnx.refit_manifest.json,
               trtllm-engine-RTX5090/ (inert, see §5)
  vae/         vae_decoder.onnx, scragvae_{encoder,decoder}.onnx
  pp-vae/      pp-vae_{encoder,decoder}.onnx(.data) + ORT-TRT caches
```

ONNX export tooling: `tools/onnx-export/` — `export_dit.py`, `export_fp8_dit.py`, `export_lm.py`, `export_vae.py`, `export_vae_encoder.py`, `export_pp_vae.py`, `export_text_enc.py`, `export_cond_enc.py`, `gen_weight_map.py`, `test_trt_vae.py`. These run in the hot-step-9000 Python venv (a **separate repo**), patterns adapted from the DEMON project (`D:\Ace-Step-Latest\Demon`) — do not expect them to run with this repo's Node/C++ toolchain.

## 9. Complete DONE / PLANNED / REVERTED ledger

**DONE (verified in code):**
- Native TRT DiT build/load/refit + full sampler parity (all Lua plugins work on TRT).
- LoRA + LoKr refit with batched GPU merge (~14s → ~1.8s).
- Dedicated CUDA streams (DiT `dit-trt.h:562`; LM has its own stream too).
- C++ wall-clock timing markers + flush (ground truth vs Node table skew).
- Raw TRT LM end-to-end (works; slower than GGUF at last measurement).
- ORT+TRT-EP VAE / PP-VAE / text-enc / cond-enc; ONNX auto-discovery (`--onnx-dir`, sibling-file detection).
- Streaming ring buffer (previews disabled); per-arch engine caching; build heartbeat.
- Server timing table incl. TRT-specific stages (`generate.ts:1257`).
- CFG cutoff, LM CFG cutoff, step cache; co-resident `EVICT_NEVER`.

**PLANNED / NOT implemented (do not claim these exist):**
- DiT cross-attention KV caching (est. ~2–4s of the plan).
- LM split-KV single-buffer append (double-buffer still ships).
- LM vocab trim 217204 → 151936 (`LM_TRT_VOCAB` unchanged at `lm-trt.h:42`).
- Second (prefill-optimal) TRT optimization profile for the LM.
- Explicit `dit_trt_warmup()` — first-step latency is only measured/logged instead.
- GGML CUDA graph capture (no `GGML_CUDA_GRAPH` reference anywhere in engine source).
- Cache-DiT layer caching; SageAttention.
- Real temporal-chunked streaming previews.
- Adapter support on FP8/fp32-I/O engines.

**REVERTED / DEAD / superseded:**
- TRT-LLM Executor on native Windows (CMake-disabled 2026-06-02; code kept).
- "Skip constant re-uploads" in the GGML sampler — a local plan doc marks it implemented, but `hot-step-sampler.h:593` re-uploads every step: "Confirmed: skipping these produces blank output" (scheduler aliases input buffers as scratch). Code wins.
- "Approach B (ORT+TRT EP) for the DiT" from the 2026-05-30 design — shipped DiT uses Approach A (native TRT) because LoRA refit requires `IRefitter`. ORT+TRT-EP kept only for refit-free models. The design's `--onnx-dir` + `use_ort_vae` did land as designed.

## 10. Open questions / not verified

- Current end-to-end TRT vs GGUF totals **after** the adapter fix + dedicated streams — the 77.6s/50.5s table predates both; no fresher benchmark in the repo.
- Whether native-TRT generation from `dit-fp8/` is actively used (no `.engine` on disk).
- The 2026-05-30 design's 30–50% DiT speedup claim (measured: ~25%).
- Exact vendored TRT SDK version in `engine/deps/tensorrt` — libs are `*_10` (TRT 10.x) while some code comments say "TRT 11" for `kSTRONGLY_TYPED` being mandatory.
