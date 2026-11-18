---
name: engine-performance
description: Explains where HOT-Step generation time goes (LM/DiT/VAE), how the TensorRT paths activate, how to benchmark from logs, and which knobs trade quality for speed. Use when profiling slow generations, working on dit-trt.h/lm-trt.h/adapter-trt.h/hot-step-sampler-trt.h, debugging TRT/ONNX issues, or deciding what performance work is DONE vs PLANNED.
---

# Engine Performance & TensorRT

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

The C++ engine ([engine/](../../../engine/)) generates music in three phases: **LM** (a Qwen3 language model produces audio-code tokens from caption+lyrics), **DiT** (a Diffusion Transformer denoises latents over N steps — this is the main compute loop), and **VAE** (decodes latents to 48 kHz stereo audio). Two inference backends coexist: **GGML** (quantized GGUF files, the default) and **TensorRT** ("TRT" — NVIDIA's compiled-engine runtime, fed by ONNX model exports). ONNX Runtime is no longer part of the engine: it was removed on 2026-10-01 (`a44753f1`), along with every path that ran on it. This skill maps the performance landscape of the ACE-Step pipeline, the TRT integration, and how to measure things. The MiniMax-Music3 TensorRT DiT has its own skill (`mm3-backend`).

Detailed line-references, the full DONE/PLANNED ledger, and the expected on-disk model layout live in [reference.md](reference.md).

## When to use this skill

- A generation is slow and you need to find out which phase is eating the time.
- You are editing any of `engine/src/dit-trt.h`, `lm-trt.h`, `adapter-trt.h`, `hot-step-sampler-trt.h`, `lm-trtllm.h` or `trt-runtime-probe.h`, or the GGML sampler's per-step upload logic in `hot-step-sampler.h`.
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
7. **Keep every constant GGML graph input either resident or re-uploaded.** WHY: on the scheduler-fallback path the GGML scheduler aliases input buffers as scratch, so encoder/pos/mask constants must be re-uploaded on every evaluation; `hot-step-sampler.h:812-815` records "confirmed: skipping these produces blank output". The direct static-graph path (ported from upstream 2026-07-14, `73e057b6`) skips those uploads only because `mark_resident_inputs()` flags each constant input *and* output, which keeps it out of gallocr's reuse pool. A new constant input needs both: the resident flag, and a re-upload in every `if (!dit_graph.direct)` block and after the CFG-cutoff graph rebuilds. `[DiT] Graph exec: direct (static, no scheduler)` or `scheduler fallback` in the engine log says which path ran. The TRT sampler re-uploads `enc_hidden` per step for the same class of reason.
8. **Do not expect a "use TRT" flag — backend selection is path-driven.** Selecting an ONNX model path activates TRT; selecting a GGUF activates GGML (see the triggers below). Debugging "why is TRT not used" starts with the model path, not a setting.

## Where generation time goes

Measured breakdown from the local TRT optimization plan (RTX 5090, 4B LM + XL DiT, 50 steps, same song/adapter — **predates** the adapter-batching fix, dedicated CUDA streams and the GGML direct graph, so treat as directional):

| Component | TRT | GGUF | Winner |
|---|---|---|---|
| LM Phase | 29.4s | 18.0s | GGUF |
| Adapter apply | 16.6s → later fixed to ~1.8s | 2.0s | ~tie now |
| DiT Denoising | **15.3s** | 20.5s | TRT (~−25%) |
| DiT Model Load | 7.5s | ~0s | GGUF |
| VAE Decode | **0.8s** | 1.3s | TRT (~−38%) |
| Text Encoding | ~6.4s | ~5s | ~tie |

The VAE row measured the ONNX Runtime decoder, which no longer exists: VAE decode is GGML-only now. Rules of thumb: DiT denoising dominates (turbo models = 8 steps, base/SFT = 50 steps, cost is linear in steps). Per-step GGML DiT cost was measured at ~85–90% GPU transformer layers, ~5–8% sync/re-upload overhead, ~2–3% CPU-side guidance, <1% solver; the re-upload share predates the direct graph. TRT wins the DiT; GGUF wins the LM and load times. No fresher end-to-end A/B exists in the repo (unverified whether TRT now wins overall).

## How the TRT paths activate (path-driven, no flag)

Native TensorRT (raw NvInfer API) is the only non-GGML runtime. It covers the ACE-Step DiT and LM here, and the MiniMax-Music3 DiT (`engine/src/minimax/mm3-dit-trt.h`, chosen by `"dit_backend": "tensorrt"` on `/mm3/synth`). Native TRT was chosen over ONNX Runtime's TRT execution provider because that provider does not expose `IRefitter`, which runtime LoRA/LoKr adapter switching requires (`engine/src/dit-trt.h` header comment).

Triggers:

- **DiT:** selected DiT model path ends in (or its directory contains) `.onnx` → `dit_ends_with_onnx` (`engine/src/dit.h:240`) gate at `engine/src/pipeline-synth-ops.cpp:1181` → `dit-trt.h`. The registry lists an ONNX DiT by its directory (`models/<dir>/` or `models/onnx/<dir>/` holding a `dit*.onnx`). Engine file = same name with `.engine`; **built on first use if missing (5–30 min), else loaded from cache**. An ONNX DiT still needs a safetensors XL DiT folder for its FSQ and condition encoder, which run on GGML (`pipeline-synth.cpp:113-169`).
- **LM:** if `<model_path>/lm_full.onnx` exists → raw TRT LM (`lm-trt.h`), engine `lm_full.engine` built if missing (`engine/src/pipeline-lm.cpp:1386-1432`). A TRT-LLM engine dir (`trtllm-engine-*`) is checked first only in a build with TRT-LLM, which is compile-time **disabled** (see ledger).
- **Runtime DLLs:** on Windows `nvinfer_10.dll` and `nvonnxparser_10.dll` are delay-loaded (`engine/CMakeLists.txt:357-372`), so every binary starts without them, and each TRT entry point first calls `hot_step_trt_runtime_probe()` (`trt-runtime-probe.h`). A missing runtime fails the request with `[DiT-Generate] FATAL: TensorRT runtime (nvinfer_10.dll) is not installed; get it from the Model Manager` instead of crashing. The Node server puts the TensorRT DLL folder on the engine's PATH (`server/src/services/aceEngineProcess.ts:227-242`): `TENSORRT_LIBS`, else the data folder's runtime folder in a release, else `engine/deps/tensorrt_libs/` (`config.aceServer.trtLibs`, `server/src/config.ts`).
- **VAE decode:** GGML only. ONNX VAEs and ONNX text encoders are not registered (`[Registry] skipping ... (ONNX VAE: no runtime for it)`, `model-registry.h:329-335` for files, `:463-469` for folders); the registry takes the first non-ONNX VAE (`registry_find_non_onnx`, `model-registry.h:66`), and `/synth` with an `.onnx` `vae_model` answers 400 (`synth_removed_feature()`, `engine/tools/hot-step-server.cpp:1949-1958`). There is no `useOrtVae` in `translateParams.ts` any more; it drops an `.onnx` `vaeModel` from old profiles.
- **Text/cond encoders + PP-VAE:** GGML. `backend=onnx` on `/pp-vae-reencode`, `/sa3-refine` or `/sa3-synth` answers 400.
- **Streaming** (`stream_mode` request): removed with the ONNX Runtime pipeline; `/synth` with `stream_mode: true` answers 400 (`hot-step-server.cpp:1950-1952`). `dit_trt_step` and `DitTrtStreamBuffers` remain in the headers with no callers.

Build requirement: the vendored TRT SDK at `engine/deps/tensorrt/` (Windows: `include/NvInfer.h` and `lib/nvinfer_10.lib`; the headers are TensorRT 10.16.1) defines `HOT_STEP_TRT` at compile time (`engine/CMakeLists.txt:353`); Linux uses a system-installed TensorRT. Without it, all TRT code compiles out and the build is GGML-only. ONNX export tooling: `tools/onnx-export/` (`export_dit.py`, `export_lm.py`, `export_fp8_dit.py`, etc.) for ACE-Step — runs in a separate Python venv (hot-step-9000 repo), not in this repo's toolchain. MM3's dev-only exporter is `tools/mm3-trt-export/` (see its README).

## Procedure: benchmark / profile a generation

There is **no dedicated bench tool** — instrumentation is log-based. Logs land in `<data root>/logs\YYYY-MM-DD_HH-MM-SS\` per session (name-sorted = time-sorted).

1. Run a generation normally, through the UI at `http://localhost:3000` (app started detached with `dev.bat`).
2. Read the **server-side timing table** — best single overview. It is in `<data root>/logs\<newest>\generations\gen_<uuid>_<task>.log` as a `[Timing] ── Pipeline Breakdown ──` block (ACE-Step only, rendered by `server/src/services/backends/ace/generate.ts:1359-1370`) with per-stage seconds/percent bars: LM Phase, model loads, FSQ Detokenize, Adapter Refit/Merge, DiT Denoising, VAE Decode, Text Encoding, plus gap rows (HTTP→Engine, DiT Model Load, etc.).
3. Cross-check against **engine-side ground truth** in `<data root>/logs\<newest>\ace_engine.log` (see Golden rule 6). Key markers:
   - `[DiT-Generate] TRT Total: X ms (Y ms/sample)` (`pipeline-synth-ops.cpp:1346`) or the GGML equivalent `[DiT-Generate] Total: ...` (`:1480`)
   - `[DiT-TRT] Step k/N ... (first step: N ms)` — first-step latency including warmup (`hot-step-sampler-trt.h:701`)
   - `[DiT] Graph exec: direct (static, no scheduler)` or `scheduler fallback`, which says which GGML sampler path ran (`hot-step-sampler.h:252`)
   - `[Adapter-TRT] Applied in N ms` / `Batched GPU merge: N ms`
   - `[DiT-TRT] Load + refit complete (N ms)`; `[LM-TRT] Load complete ...`
   - `[VAE-Decode BatchN] Decode: N ms`
4. Quick pull of the newest session (PowerShell 7, repo root):
   ```powershell
   Push-Location server; $logs = (npx tsx scripts/data-paths.ts | Out-String | ConvertFrom-Json).children.logs; Pop-Location
   $s = Get-ChildItem $logs -Directory | Where-Object Name -match '^\d{4}-\d{2}-\d{2}_' | Sort-Object Name -Descending | Select-Object -First 1
   Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern 'TRT Total|Applied in|Decode:|Load complete|first step|Graph exec'
   Select-String -Path "$($s.FullName)\generations\*.log" -Pattern '\[Timing\]'
   ```
5. **A/B GGML vs TRT:** switch the DiT model between a GGUF file and the ONNX directory in the UI's Model Manager, keep seed/params identical, compare the two timing tables. Expect the first TRT run to pay a one-time engine build (5–30 min) if no `.engine` cache exists.
6. For audio-quality comparisons: do not judge outputs yourself — preserve the generated files and ask the user to verify by ear.

## Quality↔speed knobs (all implemented unless noted)

| Knob | Where | Trade-off |
|---|---|---|
| **CFG Cutoff** (`cfg_cutoff_ratio`, default 1.0) | UI Performance accordion (`GenerationDropdown.tsx`); engine `hot-step-params.h:240` | Classifier-free guidance (the 2× "conditional + unconditional" forward pass) runs only for the first ratio×steps, then conditional-only — halves per-step cost after the cutoff. ~0.5 ≈ 20% speedup; may reduce prompt adherence. TRT path also frees/halves GPU buffers at cutoff. |
| **LM CFG Cutoff** (`lm_cfg_cutoff_ratio`) | `request.h:63`, `pipeline-lm.cpp` | Same idea for LM token generation; ~0.7 ≈ 15% LM speedup. |
| **Step Cache** (`cache_ratio`, default 0.0, UI max 0.7) | `hot-step-params.h:246`; GGML `hot-step-sampler.h`; TRT `hot-step-sampler-trt.h:476` | Skips middle-step forward passes, reusing last velocity; first/last steps protected. Try 0.3–0.5. Stacks with CFG cutoff. |
| **Steps / model tier** | `num_steps`; turbo (8) vs base/SFT (50) | Linear cost. Biggest single lever. |
| **Quantization** | GGUF variants in `<data root>/models\` (Q4_K_M → Q8_0 → BF16, plus NVFP4/MXFP4) | VRAM vs dequant overhead vs quality. |
| **Co-resident models** (`coResident` → per-request `keep_loaded`, engine `EVICT_NEVER`) | `services/backends/ace/generate.ts:169/391`; `pipeline-synth-ops.cpp:1350-1356` | Keeps DiT (including multi-GB TRT engine) + VAE in VRAM between jobs — saves ~7.5s reload per back-to-back run, costs GBs of VRAM. Once a request flips the store to `EVICT_NEVER` it stays that way until the engine restarts (`hot-step-server.cpp`). Default `EVICT_STRICT` frees after every generation; `--keep-loaded` (Settings → Environment → "Keep models in VRAM") starts in `EVICT_NEVER`. <!-- TODO(verify): on the TRT path the static context compares the resolved .onnx file path (s_trt_onnx_path) with ctx->dit_key.path, which is the directory for registry-listed ONNX DiTs, so the engine may reload on every generation even when co-resident. --> |
| **Batched CFG** | per-request `batch_cfg` (1 batch, 0 split, -1 default); `--no-batch-cfg` on the command line | One 2N-batch forward vs two N passes; batched is the default and faster, split halves DiT activation memory. |
| **Flash attention** | on by default; `--no-fa` command-line flag; `-DHOT_STEP_DISABLE_FA=ON` build option for pre-Turing GPUs | GGML paths. |
| **VAE tiling** (`vae_chunk`/`vae_overlap`, defaults 1024/64) | `--vae-chunk`/`--vae-overlap` (from `ACESTEPCPP_VAE_CHUNK`/`ACESTEPCPP_VAE_OVERLAP`); per-request `vae_chunk` override | Bigger tiles = fewer passes and fewer seams, more VRAM. |
| **LM draft model** (`--draft-lm`) | `run_phase2_speculative`, `pipeline-lm.cpp:1025` | 0.6B draft proposes, 4B verifies. GGML-side LM speculative decode. Off by default: the server passes it only when `ACESTEPCPP_DRAFT_LM` is set, because GGML per-call overhead made the draft nearly as expensive as the target (`hot-step-server.cpp:3272-3275`). |
| **CUDA graphs** (ggml) | on by default (`engine/CMakeLists.txt:139-143`, `build.cmd`); `GGML_CUDA_DISABLE_GRAPHS=1` turns it off at runtime | Replays the captured kernel launch sequence after warmup instead of dispatching each kernel. The `CUDA graph warmup` / `CUDA Graph id` log noise comes from it. |

## DONE vs PLANNED (summary — full ledger in reference.md)

- **DONE:** native TRT DiT (build/load/refit + full sampler parity with all Lua solver/guidance/scheduler plugins); LoRA+LoKr adapter refit with batched GPU merge (~14s → ~1.8s fix); dedicated CUDA streams; raw TRT LM end-to-end; delay-loaded TRT runtime with a probe before every entry point; MM3 TRT DiT (`mm3-backend` skill); engine build heartbeat; CFG cutoff, LM CFG cutoff, step cache; co-resident `EVICT_NEVER`; ggml CUDA graph capture (on by default); GGML sampler direct static-graph execution with resident constants (`73e057b6`).
- **PLANNED / NOT implemented:** DiT cross-attention KV caching; LM single-buffer KV append (double-buffer still ships); LM vocab trim (still padded `LM_TRT_VOCAB 217204`, `lm-trt.h:42`); prefill optimization profile; explicit TRT warmup call; SageAttention; real streaming previews; adapter support on FP8/fp32-I/O engines; adapter stacks, section masks and runtime adapter modes on the TRT DiT (it refits only the primary adapter).
- **REMOVED:** everything that ran on ONNX Runtime (2026-10-01, `a44753f1`): the ORT+TRT-EP VAE decoder, PP-VAE and text/condition encoders, ONNX VAE and text-encoder registration, and the streaming ring buffer (`stream-pipeline.h`) with its `stream_mode` request field; the `--onnx-dir` engine flag followed on 2026-10-04 (`810a78eb`).
- **DEAD:** TRT-LLM Executor on native Windows — CMake-disabled 2026-06-02 (`engine/CMakeLists.txt:403-439`, "Native Windows TRT-LLM is not viable"; TRT version mismatch between Docker-built engines and Windows SDK). Code kept behind `-DHOT_STEP_TRTLLM_ENABLE=ON` for a future WSL2 path.

## Key files

| Path | Role |
|---|---|
| `engine/src/dit-trt.h` | Native TRT DiT: engine build (`dit_trt_build:198`), load+refit (`dit_trt_load:377`), adapter refit (`dit_trt_refit_adapter:577`), forward (`dit_trt_forward:674`) |
| `engine/src/hot-step-sampler-trt.h` | TRT denoising loop — mirrors the GGML sampler (`hot-step-sampler.h`), full plugin parity; the old streaming step `dit_trt_step` (`:740`) has no callers |
| `engine/src/hot-step-sampler.h` | GGML sampling loop: direct vs scheduler-fallback graph (`hotstep_graph_alloc`, `mark_resident_inputs`), per-step uploads in `evaluate_velocity` |
| `engine/src/adapter-trt.h` | LoRA/LoKr → TRT refit; batched GPU LoKr merge (entry `adapter_trt_apply:683`, revert `adapter_trt_revert:750`) |
| `engine/src/lm-trt.h` | Raw TRT Qwen3-4B LM, double-buffered KV cache, teardown syncs at `:702-709` |
| `engine/src/lm-trtllm.h` | TRT-LLM Executor — compile-time disabled, do not assume it runs |
| `engine/src/trt-runtime-probe.h` | Checks the delay-loaded TensorRT DLLs are loadable before any TRT call |
| `engine/src/pipeline-synth-ops.cpp` | Backend dispatch: TRT-vs-GGML gate `:1181`, adapter refit `:1280-1331`, eviction policy `:1347-1356`, GGML `dit_ggml_generate` call `:1465` |
| `engine/src/pipeline-synth.cpp` | ONNX DiT sidecar handling: FSQ, condition-encoder and `null_condition_emb` fallback to a safetensors XL DiT folder (`:96-180`) |
| `engine/src/pipeline-lm.cpp` | LM backend dispatch (TRT-LLM → raw TRT → GGML probe order `:1347-1432`), speculative decoding `:1025` |
| `engine/src/hot-step-params.h` | `cfg_cutoff_ratio:240`, `cache_ratio:246` |
| `engine/tools/hot-step-server.cpp` | CLI parsing; registry scan (including the `onnx/` subdirectory, `:3266-3270`); 400s for removed ONNX Runtime features |
| `engine/CMakeLists.txt` | TRT detection `:290-334`, `HOT_STEP_TRT` define `:353`, DLL delay-load `:357-372`, TRT-LLM disable block `:403-439` |
| `server/src/services/backends/ace/generate.ts` | Timing table `:1359-1370`, TRT build stage lines `:778-779`, coResident `:169/391` |
| `server/src/services/generation/translateParams.ts` | UI param → engine request mapping (drops `.onnx` VAE picks from old profiles) |
| `tools/onnx-export/` | Python ONNX exporters (separate venv, not this repo's toolchain) |
| `docs/plans/2026-05-31-TRT-OPTIMIZATION-PLAN.md` | Local-only optimization plan + measurements (gitignored — may be absent) |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| First generation with an ONNX model "hangs" 5–30 min; log shows `[DiT-TRT] This will take 5-30 minutes` + `Engine build in progress... (Ns elapsed)` every 30s | Normal first-run engine build; result cached as `.engine` next to the ONNX. Deleting `.engine` retriggers it. The heartbeat exists so the Node stall-detector doesn't kill the job. |
| `[DiT-Generate] FATAL: TensorRT runtime (nvinfer_10.dll) is not installed; get it from the Model Manager` | The TRT DLLs are not in the release archive; download them through the Model Manager (or set `TENSORRT_LIBS`). |
| TRT loads but garbage/NaN audio | Check `[DiT-TRT] DIAG first output:` (`dit-trt.h:743`) for NaNs — ONNX export precision issue (norms must be fp32; `kSTRONGLY_TYPED` forbids per-layer precision overrides, so a re-export is required). |
| Adapter has no audible effect on TRT | Log shows either `[Adapter-TRT] No weights matched TRT engine` (name-mapping failure) or the FP8/fp32-engine `[Adapter-TRT] WARNING: LoRA adapters not yet supported with this engine I/O dtype, skipping` (`pipeline-synth-ops.cpp:1281-1286`). |
| Adapter deltas subtly wrong on TRT only | Missing `<onnx>.refit_manifest.json` sidecar — dynamo-transposed weights get deltas in the wrong orientation. Log: `No refit manifest found`. |
| GGML CUDA crash right after LM-TRT teardown | Device syncs before TRT free were skipped — restore `lm-trt.h:702-709` ordering (Golden rule 5). |
| Blank or noise output from the GGML sampler after adding a graph input | The new input is neither marked resident nor re-uploaded on the scheduler fallback (Golden rule 7). |
| Node timing table shows huge Adapter/gap times that engine wall-clock contradicts | stdout pipe-buffering skew — trust engine `Applied in` / `Wall clock` lines. |
| `[TRT-WARN] Using default stream in enqueueV3()` | A call path passed a null stream; current code creates dedicated streams (`dit-trt.h:562`) — find the regressing caller. |
| `/synth` answers 400 `stream_mode was removed ...` or `ONNX VAE decoding was removed with ONNX Runtime ...` | An old client or profile asked for a removed ONNX Runtime feature; drop `stream_mode`, pick a GGUF or safetensors VAE. |
| Wrong/stale engine after switching DiT models | Static TRT context keys on the ONNX path (`s_trt_onnx_path`, `pipeline-synth-ops.cpp:1186-1190`, set at `:1276`) and rebuilds on change — start staleness debugging there. |
| ONNX DiT + cover mode fails on FSQ, or any ONNX DiT request fails at text encoding | FSQ weights and the condition encoder aren't in ONNX dirs; the loader falls back to a hardcoded list of safetensors XL DiT folders in `pipeline-synth.cpp`. Logs: `WARNING: no safetensors DiT found for FSQ`, `ONNX DiT: no safetensors XL DiT folder found for the condition encoder`. Install one of those folders. |
| Back-to-back generations each pay ~7.5s DiT load | Default `EVICT_STRICT` frees the TRT engine after every generation — enable co-resident ("Keep DiT & VAE loaded"). |

## Institutional knowledge

- **VALIDATED — the "16.56s adapter apply" was two bugs in one:** part log-timestamp skew (Node table vs reality), part real CUDA driver overhead: per-module LoKr Kronecker merges caused ~359 separate cudaMalloc/free cycles ≈ 14s. Fix: micro-batches of 32 modules per GGML graph → ~1.8s (`adapter-trt.h`, `CHUNK_SIZE = 32`, log `[Adapter-TRT] Applied in 1777 ms`).
- **VALIDATED — on the scheduler path, skipping GGML per-step constant re-uploads breaks output** (blank audio): the scheduler aliases input buffers as scratch. The direct static graph (`73e057b6`, 2026-07-14) is what made skipping safe, by keeping those inputs resident; it did not make the re-uploads unnecessary on the fallback path (`evaluate_velocity`, `hot-step-sampler.h:801-830`).
- **VALIDATED — native TRT (Approach A) beat ORT+TRT-EP (Approach B) for the DiT** solely because adapter refit needs `IRefitter`, which ORT does not expose. ORT+TRT-EP was kept for refit-free models (VAE, encoders) until ONNX Runtime left the engine entirely.
- **VALIDATED — TRT-LLM on native Windows is a dead end** (engine-version mismatch Docker 10.14 vs Windows SDK 10.16; ONNX-rebuilt engines lack TRT-LLM tensor bindings like `kv_cache_block_offsets`). A `trtllm-engine-*` folder next to an ONNX LM is ignored while TRT-LLM is compiled out.
- **VALIDATED — LM-TRT teardown ordering** (Golden rule 5) prevents cross-runtime CUDA corruption.
- **MEASURED (2026-11-18, RTX 4090, HOTSTEP-1270) — ACE-Step TRT DiT works end to end**, including ADG and restricted CFG intervals. Export: `export_dit.py` bf16_mixed of the sft-turbo ta-0.7 XL merge (8.34 GB, 10 min); first engine build 20 min. 30 s / 32 steps / CFG: DiT sampling 1.05-1.22 s TRT vs 1.83 s GGML; load + refit 7-9 s per job vs GGML DiT load 3.5-4.1 s, so per-job wall was 12 s TRT vs 9-10 s GGML under default eviction (keep-loaded not measured). One forward vs fp32 PyTorch: GGML 0.9% rel. error, TRT 5.4% (the bf16 recipe itself is 5.1%), so 32-step renders drift (waveform corr about 0.7 to GGML). The TRT LM was not tested.
- **HYPOTHESIS — the 2026-05-30 design's 30–50% DiT speedup claim.** Measured reality was ~25%. Remaining headroom (cross-attn KV cache) is unimplemented and unproven here.

## Deeper reading

- [reference.md](reference.md) — full line-referenced internals: TRT build/load/refit flow, LM-TRT KV design, adapter name-mapping, what the ONNX Runtime removal took out, expected model dir layout, complete DONE/PLANNED ledger.
- `docs/dev/engine.md` — committed engine guide, including its TensorRT section.
- `engine/docs/ARCHITECTURE.md` — committed engine request JSON, CLI and HTTP reference.
- `docs/dev/plugins-authoring.md` — committed Lua plugin authoring (solvers/schedulers/guidance run identically on GGML and TRT samplers).
- `docs/plans/2026-05-31-TRT-OPTIMIZATION-PLAN.md`, `docs/plans/2026-04-18-performance-optimizations.md`, `docs/plans/dit_optimization_analysis.md` — **local-only, gitignored; may be absent on other machines.** Where they disagree with code, code wins.
