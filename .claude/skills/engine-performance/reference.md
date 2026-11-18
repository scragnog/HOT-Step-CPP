# Engine Performance & TensorRT — Reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail supporting [SKILL.md](SKILL.md). Line references verified 2026-10-23 against the working tree; they will drift — re-verify with Grep before editing code from them.

## 1. Native TRT DiT internals (`engine/src/dit-trt.h`, ~767 lines)

### Build — `dit_trt_build` (:198)
- ONNX → engine with `kSTRONGLY_TYPED` (mandatory in this TRT version), `kTF32`, and `kREFIT_IDENTICAL` + `kSTRIP_PLAN` (:257-258) — produces a weight-stripped plan (~50 MB); weights are refit from the ONNX at every load.
- 4 GB workspace. Dynamic shape profile: `input_latents [B,T,192]` T min/opt/max 64/2048/8192, `enc_hidden [B,S,2048]` S 64/512/2048, batch min 1 / opt 1–4 / max ≤16.
- Prints `[DiT-TRT] This will take 5-30 minutes (first run only).` (:207) and a **heartbeat thread** logs `Engine build in progress... (Ns elapsed)` every 30s (:324) so the Node stall-detector does not kill the job.
- `kSTRONGLY_TYPED` forbids per-layer precision overrides — precision problems (e.g. norms that must be fp32) can only be fixed by re-exporting the ONNX (see comments near :243).

### Load — `dit_trt_load` (:377)
1. Deserialize the `.engine` plan.
2. `nvonnxparser::createParserRefitter(...).refitFromFile(onnx)` (:425) — restores stripped weights from the ONNX.
3. Cache all BF16/FP16 weights in host RAM (`base_weights`) so adapters can be reverted without reloading.
4. Read `<onnx>.refit_manifest.json` sidecar (:464) — emitted by `tools/onnx-export/export_dit.py`, lists dynamo-transposed weights (stored `[in,out]`) so adapter deltas get transposed correctly. Without it: `[DiT-TRT] No refit manifest found (adapters will assume torch orientation)`.
5. Create a **dedicated CUDA stream** (:562-564) — fixes the `[TRT-WARN] Using default stream in enqueueV3()` perf warning.

Before any of this, `pipeline-synth-ops.cpp` calls `hot_step_trt_runtime_probe()` (`trt-runtime-probe.h`) and fails the request cleanly when `nvinfer_10.dll` cannot be loaded; the DLLs are delay-loaded on Windows (`engine/CMakeLists.txt:357-372`).

### I/O dtype detection (:540-551)
Three modes: **bf16** (standard dynamo export; host staging fp32↔bf16), **fp16**, **fp32** (FP8 QDQ modelopt export; direct upload). Adapters are only supported on BF16-I/O engines — FP8/fp32 engines skip adapters with `[Adapter-TRT] WARNING: LoRA adapters not yet supported with this engine I/O dtype, skipping` (`pipeline-synth-ops.cpp:1281-1286`).

### Forward — `dit_trt_forward` (:674)
`setInputShape` + `setTensorAddress` + `enqueueV3(stream)`. First call dumps 8 output values as a NaN early-warning: `[DiT-TRT] DIAG first output:` (:743).

### Lifecycle
Static context in `pipeline-synth-ops.cpp` keyed on ONNX path (`s_trt_onnx_path`, :1186-1190, set at :1276) — model switch triggers reload/rebuild. Eviction (:1347-1356): `EVICT_STRICT` (default) frees the engine from VRAM after each generation (`[DiT-TRT] Engine unloaded from VRAM`); `EVICT_NEVER` ("Keep DiT & VAE loaded" / server `coResident`) keeps it resident.
<!-- TODO(verify): the reload check compares s_trt_onnx_path (the resolved .onnx file) with ctx->dit_key.path, which is the directory for registry-listed ONNX DiTs, so a directory selection may reload the engine on every generation even under EVICT_NEVER. -->

An ONNX DiT carries no FSQ tokenizer/detokenizer and no condition encoder. `pipeline-synth.cpp:113-169` goes two folders up from the ONNX folder (`models/onnx/<dir>` → `models`) and takes the first of a fixed list of safetensors XL DiT folders that holds `model.safetensors`; FSQ and the condition encoder both run on GGML from it, and `null_condition_emb` (the CFG uncond embedding, outside the exported decoder) is read from its weights by `st_read_null_cond`. Without one: `[Synth-Load] WARNING: no safetensors DiT found for FSQ — covers/passthrough may fail` and `[Synth-Load] ONNX DiT: no safetensors XL DiT folder found for the condition encoder`, after which the request fails at text encoding.

## 2. TRT sampler (`engine/src/hot-step-sampler-trt.h`, ~851 lines)

Mirrors the GGML sampler (`hot-step-sampler.h`); only the forward pass differs. Full parity: Lua solver/guidance/scheduler plugins, custom timesteps, DCW, repaint, cover-mode context switch, SDE, batched CFG (2N batch) or 2-pass CFG, native APG fast path.

- Data path: host FP32 → staging bf16/fp16 → GPU **per step** (~:320-401). `enc_hidden` is re-uploaded every step here too.
- **Step-velocity cache** (`cache_ratio`): :476-506 — skips middle-step forwards reusing last velocity; first/last steps protected; logs the cached-step plan.
- **CFG cutoff mid-loop**: from :563 — at the cutoff step, GPU buffers downsize 2N→N and guidance stops.
- First-step latency (incl. warmup) logged: `[DiT-TRT] Step 1/N ... (first step: N ms)` (:694-702).
- The batched streaming step `dit_trt_step` (:740) and its Euler per-slot integration (~:838), with the pre-allocated `DitTrtStreamBuffers` in `dit-trt.h` (:117-187), are left over from the removed streaming pipeline and have no callers.

## 3. Adapter refit (`engine/src/adapter-trt.h`, ~754 lines)

- Entry `adapter_trt_apply` (:683): opens safetensors, detects LoKr vs LoRA, dispatches to `adapter_trt_apply_lokr` (:373) / `adapter_trt_apply_lora` (:208), then `dit_trt_refit_adapter` (`dit-trt.h:577`) sets named weights and calls `refitCudaEngine()` under a mutex. Group scales apply per weight (`adapter_group_scale_for`); DoRA magnitudes go through `trt_merge_delta_dora` (:120).
- The call site (`pipeline-synth-ops.cpp:1280-1331`) refits the request's primary adapter (`ctx->dit_key.adapter_path`) at its scale. Stacks, per-section masks and the runtime adapter modes are GGML-only.
- **The batching fix:** LoKr Kronecker deltas are computed on GPU via the GGML backend in **micro-batches of 32 modules per graph** (`CHUNK_SIZE`, :542). The old per-module approach caused ~359 separate cudaMalloc/free cycles ≈ ~14s of pure CUDA driver overhead; now ~1.8s. Logs: `[Adapter-TRT] LoKr: N valid entries, computing in micro-batches...`, `Batched GPU merge: N ms (...)` (:667), `Applied in N ms`.
- Transposed weights: refit manifest tells the merge which deltas to transpose (dynamo `[in,out]` storage).
- Name mapping: adapter `decoder.X` ↔ TRT `dit.X`; LoKr lycoris prefixes reverse-mapped from `base_weights` keys (`lokr_build_trt_reverse_map`, :162). Failure = `[Adapter-TRT] No weights matched TRT engine — adapter has no effect` (:741).
- Revert = re-refit cached base weights (`adapter_trt_revert`, :750). Adapter switch = revert-then-apply; wall clock logged if >50 ms (`[Adapter-TRT] Wall clock total: N ms`).

## 4. TRT LM (`engine/src/lm-trt.h`, ~738 lines)

- Qwen3 4B, single full-vocab engine. `LM_TRT_VOCAB 217204` (:42) — **padded**; the planned trim to 151,936 never landed. Phase-2 audio-code logit slicing at offset 151645 (`LM_TRT_PARTIAL_OFFSET`, :45) done in C++.
- **KV cache: double-buffered** device memory per layer per set (`[1,8,max_seq,128]` bf16 × 36 layers × 2 buffers × n_kv_sets), swap A/B each step (~:82-91). The planned single-buffer append was not implemented. VRAM scales as n_kv_sets(=2×max_batch) × 36 layers × 4 buffers × max_seq — grows fast.
- Single wide optimization profile, decode-optimal (opt seq=1); a second prefill profile was planned, not implemented.
- Micro-optimizations present: pre-cached tensor-name strings, attention mask pre-filled once on device, static tensor addresses bound once, stack-allocated ≤16-token id staging, CFG batch defers stream sync to the last element.
- Dispatch: `s_use_trt` (`pipeline-lm.cpp:84`) routes vocab/forward/KV ops (:96-160). GGML LM load is skipped entirely when TRT is active.
- **Teardown (critical):** `lm_trt_free` does `cudaStreamSynchronize` then `cudaDeviceSynchronize` **before** freeing (:702-709). Removing/reordering corrupts shared CUDA state → GGML crashes afterwards.
- LM probe order at load (`pipeline-lm.cpp:1347-1432`): (1) `trtllm-engine-*` dir → TRT-LLM (only compiled with `HOT_STEP_TRTLLM`, so effectively skipped); (2) `lm_full.onnx` → raw TRT LM, `lm_full.engine` built if missing; (3) GGUF/GGML.
- Speculative decoding (GGML path): `run_phase2_speculative` (`pipeline-lm.cpp:1025`) — 0.6B draft (no CFG) proposes, 4B target verifies with CFG; enabled via the ace-server flag `--draft-lm`, which the Node server passes only when `ACESTEPCPP_DRAFT_LM` is set.

## 5. TRT-LLM Executor (`engine/src/lm-trtllm.h`, ~544 lines) — DEAD on native Windows

`engine/CMakeLists.txt:403-439`: separate from `HOT_STEP_TRT`; **STATUS: DISABLED (2026-06-02). "Native Windows TRT-LLM is not viable"** — TRT version mismatch between Docker-built engines (10.14) and the Windows SDK (10.16); ONNX-rebuilt engines lack TRT-LLM tensor bindings (`kv_cache_block_offsets` etc.). Code kept behind `#ifdef HOT_STEP_TRTLLM`; re-enable with `-DHOT_STEP_TRTLLM_ENABLE=ON` (intended for WSL2/cross-platform later). A `trtllm-engine-*` folder beside an ONNX LM is ignored in a normal build. The build script lives in `tools/trtllm-build/`.

## 6. Removed with ONNX Runtime (2026-10-01, `a44753f1`)

The engine no longer links ONNX Runtime; CMake has no option, detection or DLL copy for it, and the last ORT consumers are gone:

- `vae-ort.h`, `vae-enc-ort.h`, `text-enc-ort.h`, `cond-enc-ort.h` (the ORT + TensorRT execution-provider wrappers) and the StableStep refiner's ONNX backend. VAE decode, VAE encode, PP-VAE, the text encoder and the condition encoder all run on GGML; for an ONNX DiT the condition encoder comes from the safetensors fallback folder (§1).
- ONNX VAEs and ONNX text encoders are no longer registered (`model-registry.h:329-335, 463-469`); `/synth` with an `.onnx` `vae_model` and `backend=onnx` on `/pp-vae-reencode`, `/sa3-refine` or `/sa3-synth` answer 400.
- `stream-pipeline.h`, the DEMON-style ring-buffer streaming pipeline, decoded only through ORT and went with it; `/synth` with `stream_mode: true` answers 400. Node no longer parses `[Stream] tick` or `[STREAM_PREVIEW]` markers.
- `--onnx-dir` followed on 2026-10-04 (`810a78eb`). The two `tools/onnx-export` scripts that take their own `--onnx-dir` argument are unrelated.

What remains of ONNX is TensorRT's input format: `nvonnxparser` builds and refits the DiT and LM engines. Leftover `TensorrtExecutionProvider_TRTKernel_*.engine` files under `<data root>/models/` are caches from the removed ORT provider; nothing reads them now.

## 7. Expected on-disk model layout

The registry scans `<data root>/models/` and its `onnx/` subfolder (`hot-step-server.cpp:3266-3270`). A subfolder holding a `.onnx` file counts as an ONNX model when it has no `model.safetensors`; it is classified by the ONNX filename prefix (`lm_*`, `dit_*`, `vae_*`, `text_enc*`), then by the folder name, and only DiT and LM folders are kept (`model-registry.h:390-485`):

```
<data root>/models/onnx/
  <dit folder>/   dit*.onnx (+ external .onnx_data), dit*.engine (built on first use),
                  <onnx>.refit_manifest.json
  <lm folder>/    lm_full.onnx (+ data), lm_full.engine (built on first use),
                  optionally an inert trtllm-engine-*/ folder
  sa3/            StableStep tokenizer files (owned by that backend)
<data root>/models/<safetensors XL DiT folder>/model.safetensors   FSQ + condition encoder + null_condition_emb for an ONNX DiT
```

As of 2026-10-23 the development machine's data root has no ONNX DiT or LM folders; `models/onnx/` holds only `sa3/`. A working DiT export (2026-11-18) sits outside the repo on the 4090 test PC. The ONNX DiT folder also needs `config.json` and `silence_latent.pt` (read by `store_dit_meta`). `ace-synth` does not scan `models/onnx/`; drive it through `ace-server`.

ONNX export tooling: `tools/onnx-export/` — `export_dit.py`, `export_fp8_dit.py`, `export_lm.py`, `export_vae.py`, `export_vae_encoder.py`, `export_pp_vae.py`, `export_text_enc.py`, `export_cond_enc.py`, `gen_weight_map.py`, `test_trt_vae.py`, plus the SA3 exporters. The VAE, PP-VAE and encoder exports fed the removed ORT paths and have no consumer in the engine now. These run in the hot-step-9000 Python venv (a **separate repo**), patterns adapted from the DEMON project (`D:\Ace-Step-Latest\Demon`) — do not expect them to run with this repo's Node/C++ toolchain. MM3's exporter is `tools/mm3-trt-export/`.

## 8. Complete DONE / PLANNED / REVERTED ledger

**DONE (verified in code):**
- Native TRT DiT build/load/refit + full sampler parity (all Lua plugins work on TRT).
- LoRA + LoKr refit with batched GPU merge (~14s → ~1.8s).
- Dedicated CUDA streams (DiT `dit-trt.h:562`; LM has its own stream too).
- C++ wall-clock timing markers + flush (ground truth vs Node table skew).
- Raw TRT LM end-to-end (works; slower than GGUF at last measurement).
- Delay-loaded TensorRT DLLs with `hot_step_trt_runtime_probe()` before every TRT entry point; the DLLs come through the Model Manager.
- MM3 TensorRT DiT (`minimax/mm3-dit-trt.h`; `mm3-backend` skill).
- Per-arch engine caching; build heartbeat.
- Server timing table incl. TRT-specific stages (`services/backends/ace/generate.ts:1359-1370`).
- CFG cutoff, LM CFG cutoff, step cache; co-resident `EVICT_NEVER`.
- ggml CUDA graph capture and replay: on by default (`engine/CMakeLists.txt:139-143`, `-DGGML_CUDA_GRAPHS=ON` in `build.cmd` since `02766043`), off at runtime with `GGML_CUDA_DISABLE_GRAPHS=1`.
- GGML sampler direct static-graph execution (`73e057b6`, 2026-07-14): no scheduler, constant inputs resident, per-step re-uploads only on the scheduler fallback.

**PLANNED / NOT implemented (do not claim these exist):**
- DiT cross-attention KV caching (est. ~2–4s of the plan).
- LM split-KV single-buffer append (double-buffer still ships).
- LM vocab trim 217204 → 151936 (`LM_TRT_VOCAB` unchanged at `lm-trt.h:42`).
- Second (prefill-optimal) TRT optimization profile for the LM.
- Explicit `dit_trt_warmup()` — first-step latency is only measured/logged instead.
- Cache-DiT layer caching; SageAttention.
- Real temporal-chunked streaming previews (and any streaming at all since the ORT pipeline went).
- Adapter support on FP8/fp32-I/O engines.
- Adapter stacks, per-section masks and runtime adapter modes on the TRT DiT (one primary adapter is refit).

**REVERTED / DEAD / superseded:**
- TRT-LLM Executor on native Windows (CMake-disabled 2026-06-02; code kept).
- Everything that ran on ONNX Runtime (§6), removed 2026-10-01.
- "Approach B (ORT+TRT EP) for the DiT" from the 2026-05-30 design — shipped DiT uses Approach A (native TRT) because LoRA refit requires `IRefitter`. ORT+TRT-EP was kept only for refit-free models, then removed with ONNX Runtime. The design's ONNX directory flag and `use_ort_vae` landed as designed, then went with the ONNX VAE decoder.
- "Skip constant re-uploads" in the GGML sampler: a local plan doc marked it implemented while the scheduler path still re-uploaded every step ("confirmed: skipping these produces blank output"). The direct graph later made it true for the direct path only (DONE above).

## 9. Open questions / not verified

- Current end-to-end TRT vs GGUF totals **after** the adapter fix + dedicated streams — the 77.6s/50.5s table predates both; no fresher benchmark in the repo.
- Whether the ACE-Step TRT LM path works end to end after the ORT removal (the DiT path was verified 2026-11-18; see SKILL.md).
- The 2026-05-30 design's 30–50% DiT speedup claim (measured: ~25%).
- The vendored SDK headers are TensorRT 10.16.1 (`engine/deps/tensorrt/include/NvInferVersion.h`) while code comments in `dit-trt.h` say "TRT 11" for `kSTRONGLY_TYPED` being mandatory.
