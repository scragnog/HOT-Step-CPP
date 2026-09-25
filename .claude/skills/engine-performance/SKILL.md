---
name: engine-performance
description: Explains where HOT-Step generation time goes (LM/DiT/VAE), how the TensorRT paths activate, how to benchmark from logs, and which knobs trade quality for speed. Use when profiling slow generations, working on dit-trt.h/lm-trt.h/adapter-trt.h/hot-step-sampler-trt.h, debugging TRT/ONNX issues, or deciding what performance work is DONE vs PLANNED.
---

# Engine Performance & TensorRT

The C++ engine ([engine/](../../../engine/)) generates music in three phases: **LM** (a Qwen3 language model produces audio-code tokens from caption+lyrics), **DiT** (a Diffusion Transformer denoises latents over N steps — this is the main compute loop), and **VAE** (decodes latents to 48 kHz stereo audio). Two inference backends coexist: **GGML** (quantized GGUF files, the default) and **TensorRT** ("TRT" — NVIDIA's compiled-engine runtime, fed by ONNX model exports). This skill maps the performance landscape, the TRT integration, and how to measure things.

Detailed line-references, the full DONE/PLANNED ledger, and the on-disk model layout live in [reference.md](reference.md).

## When to use this skill

- A generation is slow and you need to find out which phase is eating the time.
- You are editing any of `engine/src/dit-trt.h`, `lm-trt.h`, `adapter-trt.h`, `hot-step-sampler-trt.h`, `lm-trtllm.h`, `stream-pipeline.h`, or the ORT wrappers (`vae-ort.h`, `text-enc-ort.h`, `cond-enc-ort.h`, `vae-enc-ort.h`).
- TRT/ONNX generation produces NaNs, garbage audio, silent adapters, or "hangs" on first run.
- Someone asks "does the engine support X optimization?" — check the DONE/PLANNED ledger before promising anything.
- You want to speed up generation without touching C++ (quality↔speed knobs).

## Golden rules

1. **Rebuild C++ via `dev-rebuild.bat` at repo root, NEVER `engine/build.cmd` directly.** WHY: you cannot reliably tell whether the app is running; the Node server auto-respawns `ace-server.exe` on crash, and killing it mid-build causes an infinite respawn + file-lock loop.
2. **Never `cmake --build . --clean-first`** unless the GGML/CUDA layer itself changed. WHY: CUDA kernel recompilation takes 20+ minutes. For stale `.obj` issues delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
3. **Never commit, ship, or copy `.engine` files between machines.** WHY: TRT engines are locked to the GPU architecture (sm_86/89/120 are not interchangeable) and TRT version. They are built on first use and cached next to the ONNX file — that is the intended distribution model.
4. **Never delete a `.engine` file's sibling `.onnx`.** WHY: engines are built with `kSTRIP_PLAN` (weights stripped, ~50 MB plan); weights are re-refit **from the ONNX** on every load (`engine/src/dit-trt.h:375-377`). Engine without ONNX = unusable.
5. **Never remove or reorder the device syncs in `lm_trt_free`** (`engine/src/lm-trt.h:702-709`: `cudaStreamSynchronize` + `cudaDeviceSynchronize` *before* freeing). WHY: TRT teardown without them corrupts shared CUDA state and crashes GGML afterwards — this was learned from real crashes.
6. **Trust engine-side wall-clock log lines over the Node timing table when they disagree.** WHY: the Node table timestamps *log arrival*; stdout pipe buffering skews it. A phantom "16.56s adapter apply" was chased for this exact reason. Engine ground truth: `[Adapter-TRT] Applied in N ms`, `[DiT-Generate] TRT Total: ...`.
7. **Don't "optimize away" the per-step constant re-uploads in the GGML sampler.** WHY: `engine/src/hot-step-sampler.h:593` — "Confirmed: skipping these produces blank output." The GGML scheduler aliases input buffers as scratch, so encoder/pos/mask constants must be re-uploaded every step. The TRT sampler re-uploads `enc_hidden` per step for the same class of reason. A local design doc marks this optimization "implemented"; the code comment wins.
8. **Do not expect a "use TRT" flag — backend selection is path-driven.** Selecting an ONNX model path activates TRT; selecting a GGUF activates GGML (see procedure 2). Debugging "why is TRT not used" starts with the model path, not a setting.

## Where generation time goes

Measured breakdown from the local TRT optimization plan (RTX 5090, 4B LM + XL DiT, 50 steps, same song/adapter — **predates** the adapter-batching fix and dedicated CUDA streams, so treat as directional):

| Component | TRT | GGUF | Winner |
|---|---|---|---|
| LM Phase | 29.4s | 18.0s | GGUF |
| Adapter apply | 16.6s → later fixed to ~1.8s | 2.0s | ~tie now |
| DiT Denoising | **15.3s** | 20.5s | TRT (~−25%) |
| DiT Model Load | 7.5s | ~0s | GGUF |
| VAE Decode | **0.8s** | 1.3s | TRT (~−38%) |
| Text Encoding | ~6.4s | ~5s | ~tie |

Rules of thumb: DiT denoising dominates (turbo models = 8 steps, base/SFT = 50 steps, cost is linear in steps). Per-step GGML DiT cost is ~85–90% GPU transformer layers, ~5–8% sync/re-upload overhead, ~2–3% CPU-side guidance, <1% solver. TRT wins the DiT and VAE; GGUF wins the LM and load times. No fresher end-to-end A/B exists in the repo (unverified whether TRT now wins overall).

## How the TRT paths activate (path-driven, no flag)

Two distinct TRT integrations coexist:

1. **Native TRT (raw NvInfer API)** — DiT and LM. Chosen over ONNX Runtime because ORT's TRT execution provider does not expose `IRefitter`, which runtime LoRA/LoKr adapter switching requires (`engine/src/dit-trt.h` header comment).
2. **ORT + TensorRT execution provider** — VAE, PP-VAE, text/cond encoders (no adapters needed there). Legacy V1 EP via C API, fp16, engine cache written next to the ONNX (`engine/src/vae-ort.h:78`).

Triggers:

- **DiT:** selected DiT model path ends in (or its directory contains) `.onnx` → `dit_ends_with_onnx` gate at `engine/src/pipeline-synth-ops.cpp:1225` → `dit-trt.h`. Engine file = same name with `.engine`; **built on first use if missing (5–30 min), else loaded from cache**.
- **LM:** if `<model_path>/lm_full.onnx` exists → raw TRT LM (`lm-trt.h`), engine `lm_full.engine` built if missing (`engine/src/pipeline-lm.cpp:1159`). A TRT-LLM engine dir (`trtllm-engine-*`) is checked first but TRT-LLM is compile-time **disabled** (see ledger).
- **VAE decode:** request flag `use_ort_vae` (`engine/src/request.h:169`) AND server started with `--onnx-dir` containing `vae/vae_decoder.onnx` (`engine/tools/hot-step-server.cpp:2568-2587`). UI plumbing: `useOrtVae` → `server/src/services/generation/translateParams.ts:186`.
- **Text/cond encoders + PP-VAE:** auto-discovered ONNX files alongside the DiT / in `models/onnx/pp-vae/` — no user action.
- **Streaming** (`stream_mode` request): requires an ONNX DiT (`pipeline-synth-ops.cpp:2098`); ring-buffer batched generation, previews currently hard-disabled (`:2251` sets `preview_interval = 0` — partially-denoised latents through the VAE = scrambled audio).

Build requirement: the vendored TRT SDK at `engine/deps/tensorrt/` (present: `include/NvInfer.h`, libs `*_10` = TRT 10.x) defines `HOT_STEP_TRT` at compile time (`engine/CMakeLists.txt:302`). Without it, all TRT code compiles out and the build is GGML-only. ONNX export tooling: `tools/onnx-export/` (`export_dit.py`, `export_lm.py`, `export_vae.py`, etc.) — runs in a separate Python venv (hot-step-9000 repo), not in this repo's toolchain.

## Procedure: benchmark / profile a generation

There is **no dedicated bench tool** — instrumentation is log-based. Logs land in `logs\YYYY-MM-DD_HH-MM-SS\` per session (name-sorted = time-sorted).

1. Run a generation normally (via the UI, `dev.bat` or `LAUNCH.bat`).
2. Read the **server-side timing table** — best single overview. It is in `logs\<newest>\generations\gen_<uuid>_<task>.log` as a `[Timing] ── Pipeline Breakdown ──` block (rendered by `server/src/routes/generate.ts:1257`) with per-stage seconds/percent bars: LM Phase, model loads, FSQ Detokenize, Adapter Refit/Merge, DiT Denoising, VAE Decode, Text Encoding, plus gap rows (HTTP→Engine, DiT Model Load, etc.).
3. Cross-check against **engine-side ground truth** in `logs\<newest>\ace_engine.log` (see Golden rule 6). Key markers:
   - `[DiT-Generate] TRT Total: X ms (Y ms/sample)` (`pipeline-synth-ops.cpp:1384`) or the GGML equivalent `[DiT-Generate] Total: ...`
   - `[DiT-TRT] Step k/N ... (first step: N ms)` — first-step latency including warmup (`hot-step-sampler-trt.h:700`)
   - `[Adapter-TRT] Applied in N ms` / `Batched GPU merge: N ms`
   - `[DiT-TRT] Load + refit complete (N ms)`; `[LM-TRT] Load complete ...`
   - `[VAE-Decode ...] Decode: N ms (ORT|GGML)`
4. Quick pull of the newest session (PowerShell):
   ```powershell
   $s = Get-ChildItem D:\Ace-Step-Latest\hot-step-cpp\logs | Sort-Object Name -Descending | Select-Object -First 1
   Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern 'TRT Total|Applied in|Decode:|Load complete|first step'
   Select-String -Path "$($s.FullName)\generations\*.log" -Pattern '\[Timing\]'
   ```
5. **A/B GGML vs TRT:** switch the DiT model between a GGUF file and the ONNX directory in the UI's Model Manager, keep seed/params identical, compare the two timing tables. Expect the first TRT run to pay a one-time engine build (5–30 min) if no `.engine` cache exists.
6. For audio-quality comparisons: do not judge outputs yourself — preserve the generated files and ask the user (Rob) to verify by ear.

## Quality↔speed knobs (all implemented unless noted)

| Knob | Where | Trade-off |
|---|---|---|
| **CFG Cutoff** (`cfg_cutoff_ratio`, default 1.0) | UI Performance accordion (`GenerationDropdown.tsx`); engine `hot-step-params.h:170` | Classifier-free guidance (the 2× "conditional + unconditional" forward pass) runs only for the first ratio×steps, then conditional-only — halves per-step cost after the cutoff. ~0.5 ≈ 20% speedup; may reduce prompt adherence. TRT path also frees/halves GPU buffers at cutoff. |
| **LM CFG Cutoff** (`lm_cfg_cutoff_ratio`) | `request.h`, `pipeline-lm.cpp` | Same idea for LM token generation; ~0.7 ≈ 15% LM speedup. |
| **Step Cache** (`cache_ratio`, default 0.0, UI max 0.7) | `hot-step-params.h:176`; GGML `hot-step-sampler.h`; TRT `hot-step-sampler-trt.h:484` | Skips middle-step forward passes, reusing last velocity; first/last steps protected. Try 0.3–0.5. Stacks with CFG cutoff. |
| **Steps / model tier** | `num_steps`; turbo (8) vs base/SFT (50) | Linear cost. Biggest single lever. |
| **Quantization** | GGUF variants in `models\` (Q4_K_M → Q8_0 → BF16, plus NVFP4/MXFP4) | VRAM vs dequant overhead vs quality. |
| **ORT VAE** (`use_ort_vae` + server `--onnx-dir`) | see triggers above | TRT VAE decode ~0.8s vs 1.3s GGML; quality nominally identical. |
| **Co-resident models** (`coResident` → engine `EVICT_NEVER`) | `generate.ts:302/584`; `pipeline-synth-ops.cpp:1390` | Keeps DiT (including multi-GB TRT engine) + VAE in VRAM between jobs — saves ~7.5s reload per back-to-back run, costs GBs of VRAM. Default `EVICT_STRICT` frees after every generation. |
| **Batched CFG** (`use_batch_cfg`) | request | One 2N-batch forward vs two N passes; faster but doubles activation VRAM. |
| **Flash attention** (`use_fa`) | request | GGML paths. |
| **VAE tiling** (`vae_chunk`/`vae_overlap`, defaults 256/64) | synth params | Bigger tiles = fewer passes and fewer seams, more VRAM. |
| **LM draft model** (`--draft-model`) | `pipeline-lm.cpp:793` speculative decode | 0.6B draft proposes, 4B verifies. GGML-side LM speedup. |
| **Streaming** (`stream_mode`/`stream_depth`) | request; TRT-only | Ring-buffer batched denoising; previews disabled pending temporal chunking. |

## DONE vs PLANNED (summary — full ledger in reference.md)

- **DONE:** native TRT DiT (build/load/refit + full sampler parity with all Lua solver/guidance/scheduler plugins); LoRA+LoKr adapter refit with batched GPU merge (~14s → ~1.8s fix); dedicated CUDA streams; raw TRT LM end-to-end; ORT+TRT-EP VAE/PP-VAE/encoders; ONNX auto-discovery; streaming ring buffer (previews off); engine build heartbeat; CFG cutoff, LM CFG cutoff, step cache; co-resident `EVICT_NEVER`.
- **PLANNED / NOT implemented:** DiT cross-attention KV caching; LM single-buffer KV append (double-buffer still ships); LM vocab trim (still padded `LM_TRT_VOCAB 217204`, `lm-trt.h:42`); prefill optimization profile; explicit TRT warmup call; GGML CUDA graph capture (no `GGML_CUDA_GRAPH` anywhere in engine source); SageAttention; real streaming previews; adapter support on FP8/fp32-I/O engines.
- **DEAD:** TRT-LLM Executor on native Windows — CMake-disabled 2026-06-02 (`engine/CMakeLists.txt:344`, "Native Windows TRT-LLM is not viable"; TRT version mismatch between Docker-built engines and Windows SDK). Code kept behind `-DHOT_STEP_TRTLLM_ENABLE=ON` for a future WSL2 path.

## Key files

| Path | Role |
|---|---|
| `engine/src/dit-trt.h` | Native TRT DiT: engine build (`dit_trt_build:198`), load+refit (`dit_trt_load:377`), adapter refit (`:577`), forward (`:674`) |
| `engine/src/hot-step-sampler-trt.h` | TRT denoising loop — mirrors the GGML sampler (`hot-step-sampler.h`), full plugin parity; batched streaming step (`dit_trt_step:739`) |
| `engine/src/adapter-trt.h` | LoRA/LoKr → TRT refit; batched GPU LoKr merge (entry `adapter_trt_apply:604`) |
| `engine/src/lm-trt.h` | Raw TRT Qwen3-4B LM, double-buffered KV cache, teardown syncs at `:702-709` |
| `engine/src/lm-trtllm.h` | TRT-LLM Executor — compile-time disabled, do not assume it runs |
| `engine/src/stream-pipeline.h` | Ring-buffer streaming generation (TRT-only) |
| `engine/src/vae-ort.h`, `text-enc-ort.h`, `cond-enc-ort.h`, `vae-enc-ort.h` | ORT + TRT-EP wrappers for refit-free models |
| `engine/src/pipeline-synth-ops.cpp` | Backend dispatch: TRT-vs-GGML gate `:1225`, eviction policy `:1390`, ORT VAE gate `:1589`, streaming `:2095` |
| `engine/src/pipeline-lm.cpp` | LM backend dispatch (TRT-LLM → raw TRT → GGML probe order `:1110-1193`), speculative decoding `:790` |
| `engine/src/hot-step-params.h` | `cfg_cutoff_ratio:170`, `cache_ratio:176` |
| `engine/tools/hot-step-server.cpp` | `--onnx-dir` parsing + VAE/PP-VAE ONNX discovery (`:2568`, `:2947`) |
| `engine/CMakeLists.txt` | `HOT_STEP_TRT` define `:302`; TRT-LLM disable block `:342-371` |
| `server/src/routes/generate.ts` | Timing table `:1257`, stream markers `:787-798`, coResident `:302` |
| `server/src/services/generation/translateParams.ts` | UI param → engine request mapping (`useOrtVae:186`) |
| `tools/onnx-export/` | Python ONNX exporters (separate venv, not this repo's toolchain) |
| `docs/plans/2026-05-31-TRT-OPTIMIZATION-PLAN.md` | Local-only optimization plan + measurements (gitignored — may be absent) |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| First generation with an ONNX model "hangs" 5–30 min; log shows `[DiT-TRT] This will take 5-30 minutes` + `Engine build in progress... (Ns elapsed)` every 30s | Normal first-run engine build; result cached as `.engine` next to the ONNX. Deleting `.engine` retriggers it. The heartbeat exists so the Node stall-detector doesn't kill the job. |
| TRT loads but garbage/NaN audio | Check `[DiT-TRT] DIAG first output:` (`dit-trt.h:743`) for NaNs — ONNX export precision issue (norms must be fp32; `kSTRONGLY_TYPED` forbids per-layer precision overrides, so a re-export is required). |
| Adapter has no audible effect on TRT | Log shows either `[Adapter-TRT] No weights matched TRT engine` (name-mapping failure) or the FP8/fp32-engine "adapters not supported with this I/O dtype" warning (`pipeline-synth-ops.cpp` near `:1320`). |
| Adapter deltas subtly wrong on TRT only | Missing `<onnx>.refit_manifest.json` sidecar — dynamo-transposed weights get deltas in the wrong orientation. Log: `No refit manifest found`. |
| GGML CUDA crash right after LM-TRT teardown | Device syncs before TRT free were skipped — restore `lm-trt.h:702-709` ordering (Golden rule 5). |
| Node timing table shows huge Adapter/gap times that engine wall-clock contradicts | stdout pipe-buffering skew — trust engine `Applied in` / `Wall clock` lines. |
| `[TRT-WARN] Using default stream in enqueueV3()` | A call path passed a null stream; current code creates dedicated streams (`dit-trt.h:562`) — find the regressing caller. |
| VAE segfault on RTX 50xx (sm_120) via ORT-TRT | Known Blackwell Myelin fusion bug. The known workaround (`builder_optimization_level=1`) cannot be expressed via the legacy V1 EP options used in `vae-ort.h:78-98` — fixing requires migrating to the V2 EP API. |
| `Stream ERROR: streaming requires TRT (ONNX model)` | `stream_mode` requested with a GGUF DiT selected — switch to an ONNX DiT. |
| Wrong/stale engine after switching DiT models | Static TRT context keys on the ONNX path (`s_trt_onnx_path`, `pipeline-synth-ops.cpp:1230-1234`) and rebuilds on change — start staleness debugging there. |
| ONNX DiT + cover mode fails on FSQ | FSQ weights aren't in ONNX dirs; loader falls back to a hardcoded safetensors-dir list in `pipeline-synth.cpp`; log `WARNING: no safetensors DiT found for FSQ`. |
| Back-to-back generations each pay ~7.5s DiT load | Default `EVICT_STRICT` frees the TRT engine after every generation — enable co-resident ("Keep DiT & VAE loaded"). |

## Institutional knowledge

- **VALIDATED — the "16.56s adapter apply" was two bugs in one:** part log-timestamp skew (Node table vs reality), part real CUDA driver overhead: per-module LoKr Kronecker merges caused ~359 separate cudaMalloc/free cycles ≈ 14s. Fix: micro-batches of 32 modules per GGML graph → ~1.8s (`adapter-trt.h`, log `[Adapter-TRT] Applied in 1777 ms`).
- **VALIDATED — skipping GGML per-step constant re-uploads breaks output** (blank audio). The scheduler aliases input buffers as scratch. A local plan doc claims this optimization landed; it was effectively reverted. Code comment at `hot-step-sampler.h:593` is authoritative.
- **VALIDATED — native TRT (Approach A) beat ORT+TRT-EP (Approach B) for the DiT** solely because adapter refit needs `IRefitter`, which ORT does not expose. ORT+TRT-EP remains correct for refit-free models (VAE, encoders).
- **VALIDATED — TRT-LLM on native Windows is a dead end** (engine-version mismatch Docker 10.14 vs Windows SDK 10.16; ONNX-rebuilt engines lack TRT-LLM tensor bindings like `kv_cache_block_offsets`). A built `trtllm-engine-RTX5090/` still sits in `models/onnx/lm-4B/` — it is inert.
- **VALIDATED — LM-TRT teardown ordering** (Golden rule 5) prevents cross-runtime CUDA corruption.
- **UNVERIFIED — current TRT-vs-GGUF end-to-end totals.** The 77.6s-vs-50.5s table predates the adapter fix and dedicated streams; no fresher benchmark exists in the repo. Re-measure before making "TRT is slower overall" claims.
- **UNVERIFIED — whether the FP8 DiT native-TRT path is in active use.** `models/onnx/dit-fp8/` has no `dit_fp8.engine` on disk (checked 2026-07-02) — first native-TRT use would trigger a full rebuild; the dir may mainly serve as the home for the ONNX text/cond encoders.
- **HYPOTHESIS — the 2026-05-30 design's 30–50% DiT speedup claim.** Measured reality was ~25%. Remaining headroom (cross-attn KV cache, CUDA graphs) is unimplemented and unproven here.

## Deeper reading

- [reference.md](reference.md) — full line-referenced internals: TRT build/load/refit flow, LM-TRT KV design, adapter name-mapping, streaming, model dir layout, complete DONE/PLANNED ledger.
- `engine/docs/ARCHITECTURE.md` — committed engine internals, CLI, request JSON.
- `docs/dev/plugins-authoring.md` — committed Lua plugin authoring (solvers/schedulers/guidance run identically on GGML and TRT samplers).
- `docs/plans/2026-05-31-TRT-OPTIMIZATION-PLAN.md`, `docs/plans/2026-04-18-performance-optimizations.md`, `docs/plans/dit_optimization_analysis.md` — **local-only, gitignored; may be absent on other machines.** Where they disagree with code, code wins (see the re-upload discrepancy above).
