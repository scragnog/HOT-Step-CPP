# Engine guide

`engine/` is the C++17/GGML inference and training engine behind HOT-Step. It
started as a fork of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)
(ACE-Step 1.5 on GGML) and now also carries two more model families, MiniMax-Music3
(MM3) and YuE2, plus a training toolchain, a Lua sampler-plugin host and optional
TensorRT paths.

This page is the map. For the request JSON and every CLI flag, see
[engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md). For how the three
tiers fit together, see [architecture.md](architecture.md). For build scripts and
toolchains, see [building.md](building.md).

## Layout

| Path | Contents |
|---|---|
| `engine/src/` | Engine modules, header-only for the most part. `acestep-core` compiles only `request.cpp`, `model-store.cpp`, `pipeline-lm.cpp`, `pipeline-synth.cpp`, `pipeline-synth-ops.cpp` and `pipeline-understand.cpp` |
| `engine/src/minimax/` | MiniMax-Music3 backend (`mm3-*.h`) |
| `engine/src/yue2/` | YuE2 backend (`yue2-*.h`) and the SheetSage2 lead-sheet transcriber (`sheetsage-*.h`) |
| `engine/src/train/` | Trainers and dataset preprocessors used by `ace-train` |
| `engine/src/moss/` | MOSS-Music captioner used by `ace-caption` |
| `engine/src/solvers/`, `schedulers/`, `guidance/` | Older C++ sampler registries. The live sampler resolves solvers, schedulers and guidance through the Lua plugin registry instead; `guidance/apg-core.h` is still used for the native `apg()` bridge |
| `engine/tools/` | One `.cpp` per binary, plus Python converters (`convert-*.py`) and fixture scripts |
| `engine/plugins/` | Bundled Lua plugins: `solvers/`, `schedulers/`, `guidance/` |
| `engine/patches/` | Patches applied to the `engine/ggml` submodule (see [The ggml patch stack](#the-ggml-patch-stack)) |
| `engine/ggml/` | ggml submodule |
| `engine/vendor/` | yyjson, Lua 5.4, cpp-httplib, pocketfft, VST3 SDK |
| `engine/deps/` | Optional vendored SDK: `tensorrt/` |

## Binaries

All targets are defined in `engine/CMakeLists.txt`.

| Binary | Source | What it does |
|---|---|---|
| `ace-server` | `tools/hot-step-server.cpp` | The HTTP server the app runs. ACE-Step `/lm`, `/synth`, `/understand` plus every HOT-Step endpoint and the `/mm3/*` and `/yue2/*` families |
| `ace-lm` | `tools/ace-lm.cpp` | ACE-Step 5 Hz LM from the command line: request JSON in, `request0.json`..`requestN-1.json` out |
| `ace-synth` | `tools/ace-synth.cpp` | ACE-Step text encoder, DiT and VAE from the command line: request JSONs in, MP3 or WAV out |
| `ace-understand` | `tools/ace-understand.cpp` | Reverse pipeline: audio in, caption, lyrics and metadata out |
| `ace-caption` | `tools/ace-caption.cpp` | MOSS-Music captioning: audio in, prose caption, MM3-format caption or lyrics out |
| `ace-midi` | `tools/ace-midi.cpp` | MuScriptor audio-to-MIDI transcription (`--transcribe`), plus parity modes against a Python oracle |
| `ace-train` | `tools/ace-train.cpp` | Training toolchain, one subcommand per job (see [Training subcommands](#training-subcommands)) |
| `neural-codec` | `tools/neural-codec.cpp` | ACE-Step VAE encode and decode, with optional Q8/Q4 latent compression |
| `mp3-codec` | `tools/mp3-codec.cpp` | Standalone MP3 encoder and decoder |
| `quantize` | `tools/quantize.cpp` | GGUF requantizer for ACE-Step, MM3 and YuE2 files, with optional `--imatrix` |
| `mastering` | `tools/mastering.cpp` | Reference-based mastering (the matchering algorithm) |
| `vst-host` | `tools/vst-host.cpp` | VST3 host: `--scan`, `--gui`, `--process`, `--process-chain`. Each plugin gets a 10 s warm-up pre-roll of the track's opening, fed time-reversed so plugin buffers hold the song's own start when the real pass begins |
| `yue2-probe` | `tools/yue2-probe.cpp` | YuE2 bring-up and parity CLI (`--info`, `--load`, `--tokenize`, `--ar-parity` and more) |
| `sa3-ggml-test`, `fattn-train-test`, `moss-ggml-test`, `bs-roformer-test`, `mdx23c-test` | `tools/*-test.cpp` | Parity tests against reference outputs |

`tools/ace-server.cpp` is upstream's server, kept as a reference for syncs. It is
not compiled. The shipped `ace-server` is built from `hot-step-server.cpp`.

The Node server starts `ace-server` with `--models`, `--host` and `--port`, and adds
`--adapters`, `--keep-loaded` and `--noise-profile` when they are configured
(`server/src/services/aceEngineProcess.ts`). `--onnx-dir` is still accepted for older
launchers and ignored. The app's default port is
8085; the binary's own default is 8080.

## How ace-server is organised

One process, one port, one GPU worker thread. `POST /lm`, `/synth`, `/understand`,
`/vae`, `/codes-decode`, `/warm`, `/mm3/synth` and `/yue2/synth` create a job, push
it onto a FIFO queue and return `{"id":"..."}` straight away. Clients poll
`GET /job?id=` and fetch the result with `GET /job?id=&result=1`. ACE-Step, MM3 and
YuE2 jobs share that queue, so they never run on the GPU at the same time. The job
table keeps 32 entries and evicts the oldest finished ones first.

Models are discovered by scanning `--models` at startup. ACE-Step files are
classified by their GGUF `general.architecture` into LM, text encoder, DiT and VAE
buckets (`model-registry.h`). MM3 looks in `<models>/mm3/` and then `<models>/`;
YuE2 looks in `<models>/yue2/` and then `<models>/`. The server still boots when only
MM3 or YuE2 weights are present (`hot-step-families.h`); the ACE-Step endpoints then
answer 501.

GPU residency is owned by `model-store.h`. The default policy keeps at most one
module in VRAM at a time. `--keep-loaded`, or `?keep_loaded=1` on `/lm`, `/synth` or
`/warm`, switches to keeping everything resident. `POST /models/restore-policy` undoes
the per-request switch; it refuses when `--keep-loaded` came from the command line.

## Pipelines

### ACE-Step 1.5: LM, DiT, VAE

```
/lm         caption (+ lyrics, metadata)
              -> Qwen3 5 Hz LM (pipeline-lm.cpp, qwen3-lm.h)
                 phase 1: CoT metadata and lyrics, FSM-constrained (metadata-fsm.h)
                 phase 2: audio codes (5 Hz FSQ tokens)
              -> enriched request JSON

/synth      request JSON (+ source / reference audio or latents)
              -> text encoder (qwen3-enc.h)
              -> condition encoder (cond-enc.h; for an ONNX DiT, loaded from the
                 safetensors XL DiT folder the graph was exported from)
              -> FSQ detokenizer: codes -> 25 Hz source latents (fsq-detok.h)
              -> DiT flow matching (dit.h, dit-graph.h) driven by hot-step-sampler.h
                 or, for an ONNX DiT, hot-step-sampler-trt.h + dit-trt.h
              -> VAE decode (vae.h)
              -> optional LRC alignment (lrc-alignment.h), denoiser, PP-VAE
              -> MP3 or WAV, 48 kHz stereo

/understand audio -> VAE encode -> FSQ tokenize (fsq-tok.h) -> LM understand prompt
              -> caption, lyrics, metadata (pipeline-understand.cpp)
```

`pipeline-synth-ops.cpp` holds the per-request steps (resolve params, build schedule,
build context, run DiT, decode). Task routing (text2music, cover, repaint, lego,
extract, complete) is described in the
[generation modes reference](../../engine/docs/ARCHITECTURE.md#generation-modes).

### MiniMax-Music3: planner LM, then DiT renderer

```
caption + lyrics -> prompt assembly and caption cleaning (mm3-request.h)
  -> tokenizer (mm3-tokenizer.h)
  -> AR planning loop (mm3-ar-loop.h): global LM (mm3-lm-graph.h) emits one semantic
     code per 25 fps frame; the RVQ depth decoder (mm3-depth-graph.h) adds 7 acoustic
     codebooks per frame -> frame hiddens
  -> per 200-frame window, 100-frame hop (mm3-pipeline.h):
       condition encoder (mm3-cond-graph.h)
       flow DiT, Euler (mm3-dit-graph.h), or the TensorRT DiT (mm3-dit-trt.h)
       vocoder (mm3-vocoder-graph.h), 44.1 kHz
  -> overlap stitch -> WAV
```

`mm3-model.h` owns loading and residency; the weights come as five role files
(`mm3-{lm,depth,cond,dit,voc}-<quant>.gguf`) and older `mm3-synth-*` bundles still
load. `mm3-plugins.h` lets the same Lua solvers, schedulers and guidance run on the
MM3 flow stage. `mm3-ar-cache.h` and `mm3-hiddens-file.h` cache the planner output
so a re-render can skip stage 1. Deeper notes, including the trap list, are in
`.claude/skills/mm3-backend/SKILL.md`.

### YuE2: plan, compose, render

```
style + lyrics (+ optional ABC) -> tokenizer and prompt assembly (yue2-tokenizer.h)
  -> plan stage: ABC lead sheet, only when cot is "melody" or "full"
  -> semantic stage: the AR "composer" (yue2-lm-graph.h) samples codec ids
  -> NAR stage: mixture-of-transformers flow matching with a midpoint ODE solver
     (yue2-nar-graph.h), optionally chunked (nar_chunk_frames)
  -> Oobleck VAE decode, "standard" or "legacy" variant (yue2-vae-graph.h)
  -> stitch and clamp -> 16-bit WAV (yue2-pipeline.h, yue2-job.h)
```

YuE2 loads its weights itself, outside `model-store.h`. Under the default policy a
YuE2 job first evicts every unreferenced store module (`store_evict_all`), as MM3 does,
because the store keeps 0-byte entries resident after use and their real VRAM would
make the YuE2 load run out of memory (#160, #195; the ONNX Runtime SA3 refiner was the
case that hit it).

The AR and NAR halves live in one `yue2-lm-<type>.gguf` and are resident together;
the VAE is a separate `yue2-vae-{standard,legacy}-<type>.gguf` (`yue2-model.h`). The
NAR stage can use Lua solvers and schedulers through `infer_method` and
`scheduler`. A song's composer (semantic stage) stops at `plan_cutoff` (default 1.5)
times its plan's length plus 20 s (`yue2_plan_cap_frames`, the bars x beats / tempo
estimate `classifyYue2Score` makes) or at the stage cap, whichever comes first, and counts as
a runaway there (`limit_hit`). `semantic_retries` is then a per-song budget of redraws. They
run side by side in their own cache, up to `YUE2_RETRY_DRAWS` (4) at once shared between
the runaway songs, with seeds `seed + n x 1000003` for try n. A song is settled by the
first of its draws to end cleanly (`Yue2SemanticPass::first_wins`); its other draws are
abandoned, and the winner's rows are copied into the song's own set of the first pass's
cache, so songs that ended there are never replayed. A draw count that does not fit in
VRAM halves and tries again. Within any pass, a stream that has ended leaves the decode:
its slot is swapped behind the live ones through one spare slot (`yue2_swap_slots`) and
`yue2_ar_decode_batch(..., n_active)` decodes the first n sets only. The cache is
allocated without the spare when VRAM is short, and ended streams then stay in the
decode as before. Both decode loops sample with `yue2_sample_row`, which runs the reference
chain over the top-k survivors only and draws the same token for a seed as
`yue2_distribution` + `yue2_sample_draw`; `YUE2_SAMPLER_CHECK=1` runs both on every step
and logs any disagreement. A failed YuE2 job logs its reason
as `[YuE2-Job] <id>: FAILED: <error>`. A `plan_only` request takes up to
`YUE2_MAX_PLAN_BATCH` (8) `songs` entries against 4 for a render, reported by `/yue2/props`
as `max_plan_batch`; the server's auto-replan uses it to draw every batch member's plan
attempts in one pass. During the semantic stage, `GET /job` carries `songs_done`, a bitmask
of the batch's songs that have ended and are waiting on the rest (0 in every other stage and
family). `POST /job?id=N&drop_songs=MASK` takes songs out of a running batch
(`Job::songs_dropped`, read through `Yue2Request::songs_dropped`): the plan and semantic
loops end a dropped song at once (end reason `dropped`), the recompose retry ignores it, and
it is erased from `songs` before `yue2_seal_chunks`, so it costs no NAR or VAE time. Tracks
report `song` as the batch index (`Yue2SongState::src_song`), so a gap is visible rather
than shifting later tracks; the server picks each member's tracks by that index. Dropping
every song fails the job as cancelled. The server's queue X routes through
`dropYue2BatchMember` (backends/yue2/generate.ts) and cancels the engine job only when the
last live member goes.
Other modules in `yue2/` serve training and analysis: MERT and the
tokenizer head (`yue2-mert.h`, `yue2-tok-head.h`), MMS_FA forced alignment
(`yue2-mmsfa.h`, `yue2-ctc-align.h`, also behind `POST /yue2/align`), and
SheetSage2 (`sheetsage-*.h`).

YuE2 Cover Studio uses SheetSage2 for inference transcription through
`ace-train yue2-sheet --melody-only`, not through a `/yue2/` engine route.
The Node cover boundary creates a private one-source manifest and runs the
existing queued training job with progress and cancellation. It accepts only
a non-empty ABC result with no `abc_error`, then returns that score for user
review. An ABC supplied by the user skips this transcriber and its optional
GGUF. YuE2 generation captures the approved score as `yue2Abc` and source
identity as `yue2Cover`; the engine composes from the score rather than
conditioning its audio renderer on the recording. Without `--melody-only`,
`yue2-sheet` retains its full-score training default.

The post-render cover drift check compares forced-aligned lyric words with score
sections. `server/src/services/backends/yue2/scoreClock.ts` walks each Vocal bar
under its active `M:` and `Q:` fields; a later meter or tempo change therefore
changes the expected section time. The offline `server/scripts/yue2-c4-sidecar.ts`
uses that same clock to propose word-to-note matches from cached `cursor_words`
and ABC. It marks shared notes and timing gaps uncertain. The cache has no
original variable beat timestamps, so these matches need manual review before
use as training data. Drift results include a metric version in their saved
cache key; older results are recomputed when requested.

## HOT-Step hook files

Upstream files are kept as close to acestep.cpp as possible so syncs stay a copy.
Three upstream files carry HOT-Step `#include` hooks, and losing them is the main
risk of a sync:

| Upstream file | Hook | If a sync overwrites it |
|---|---|---|
| `pipeline-synth-ops.cpp` | `#include "hot-step-sampler.h"` in place of `dit-sampler.h` | Compiles, but every solver, scheduler, guidance mode, DCW and custom schedule goes dead. The link then fails on `hotstep_sampler_linked_`, a sentinel `hot-step-server.cpp` references on purpose |
| `model-store.h` | `hot-step-params.h` | Compile error |
| `dit.h` | `adapter-merge.h` and `adapter-runtime.h` | Compile error |

The HOT-Step files these hooks bring in:

- `hot-step-params.h` defines `g_hotstep_params`, a sideband struct. Upstream's
  pipeline has no way to carry solver names, adapter stacks, guidance settings or
  plugin params, so `hot-step-server.cpp` fills this global before each synth and
  the sampler and adapter loaders read it. It is safe because there is one GPU
  worker thread.
- `hot-step-sampler.h` replaces upstream's sampling loop with one that resolves
  solvers, schedulers and guidance from the Lua registry, and adds APG, DCW,
  custom timesteps, step caching, CFG cutoff, repaint injection and LRC alignment.
  `hot-step-sampler-trt.h` is the same loop over a TensorRT forward pass.
- `hot-step-families.h` lets the server boot when only MM3 or YuE2 weights exist.
- `hot-step-build-flags.h` carries `HOT_STEP_DISABLE_FA` for the Volta build.

`hot-step-server.cpp` wires the model families in with one include and one
registration call each: `minimax/mm3-server.h` plus `minimax/mm3-job.h`
(`mm3_register_routes`, `mm3_register_job_routes`) and `yue2/yue2-server.h`
(`yue2_register_routes`).

`engine/verify-hooks.ps1` checks all of these, the sentinel, and the ggml patches.
`build.cmd` runs it before every compile. Run it after any upstream sync; the
procedure is in `.claude/skills/upstream-sync/SKILL.md`, and `engine/UPSTREAM_SYNC`
records the last upstream commit synced.

## Lua plugin host

Solvers, schedulers, guidance modes and VAE-decode postprocessors are Lua 5.4
plugins. `PluginRegistry::init` (`lua-plugin-registry.h`) scans
`engine/plugins/{solvers,schedulers,guidance,postprocess}/` and then the same four
folders under the project root `plugins/`, once at startup. `ace-server` and
`ace-synth` both call it. `lua-plugin.h` runs each file in its own sandboxed VM,
passes float arrays without copying, reads each plugin's metadata and parameter
schema, and exposes the native `apg()` helper and the optional `post_step()` hook
to guidance plugins.

- `GET /plugins` returns the registry as JSON; the UI builds its sampler controls
  from it.
- A request picks plugins by name: `infer_method` (solver), `scheduler`,
  `guidance_mode`, and `postprocess_plugin` for VAE decode. Plugin parameters travel
  in `plugin_params` as `{"pluginName:key": value}`.
- Aliases: solver `ode` is `euler`, scheduler `karras` is `sgm_uniform`. A scheduler
  named `name:args` falls back to the plugin `name`.
- The DiT schedule, including the scheduler plugin and `custom_timesteps`, is built
  once in `ops_build_schedule` (`pipeline-synth-ops.cpp`). Cover's source preservation
  then picks its start step from that schedule, and the GGML and TRT samplers run it
  as given; they no longer rebuild it. Rescale keeps truncate's step count (#124).
- An unknown solver falls back to `euler` and an unknown guidance mode to `apg`,
  with an error line in the log.
- MM3's flow stage (`mm3-plugins.h`), YuE2's NAR stage and `/sa3-refine` use the same
  registry.

New plugins need no C++ rebuild. Writing one is covered in
[plugins-authoring.md](plugins-authoring.md).

## Adapters at runtime

ACE-Step DiT adapters (LoRA and LoKr, PEFT directories or single `.safetensors`
files) are found in `--adapters` or passed by path. The request's `adapters` array
is a stack; `adapter` and `adapter_scale` are the single-adapter form. How they are
applied depends on `adapter_mode`:

| Mode | Where | What happens |
|---|---|---|
| `merge` (default) | `adapter-merge.h` | Deltas are added to the base weights at load, before QKV fusion. The merged weight is kept in F32 unless `adapter_merge_lowvram` requantises it to the base type |
| `runtime` | `adapter-runtime.h` | Deltas are precomputed on the GPU and kept in VRAM next to the quantised base (`adapter_runtime_quant`: `bf16`, `q8_0`, `q4_k`) and added in the graph |
| `runtime_lowrank` | `adapter-runtime.h` | The raw low-rank factors stay in VRAM instead of the full delta |
| TensorRT DiT | `adapter-trt.h` | Merged weights are refit into the live engine through `IRefitter` |

Per-section masking (`adapter_sections`) and timestep gain curves (`gain_curve` on a
stack entry) force runtime mode, because a merged weight cannot vary per frame or
per step. `adapter_group_scales`, `rebase_source`/`rebase_beta` and the concept
steering vectors in `concepts` (`concept-steer.h`) are also sideband fields. The
long cold-start delta precompute can be cancelled through `adapter-cancel.h`, and
its progress shows as `adapter_progress` in `GET /job`.

Other adapter paths:

- The ACE-Step planner LM takes a runtime LoRA (`lm_adapter`, from `<adapters>/lm/`
  or a path) through `lm-adapter.h`, with artist tokens, trained KV prefixes and
  shared PiSSA residuals in `artist-token-runtime.h`, `lm-prefix-runtime.h` and
  `pissa-residual.h`.
- MM3 LM adapters are runtime or merge (`mm3-lm-adapter.h`, `mm3-lm-merge.h`,
  `mm3-lm-dora.h`); MM3 DiT LoRAs merge at load (`mm3-adapter.h`).
- YuE2 NAR LoRAs merge at load (`yue2-adapter.h`), chosen with
  `POST /yue2/select-model`. On a ConvRot INT8 base the LoRA factors are applied at
  runtime instead (`yue2-convrot-adapter.h`).

The `adapter-system` skill in `.claude/skills/` covers failure modes.

## TensorRT paths

Native TensorRT is the only non-GGML runtime; it has no switch of its own, the model
path picks it.

- Native TensorRT (raw NvInfer) for the ACE-Step DiT and LM, and for the MM3 DiT. It
  was chosen over ONNX Runtime's TensorRT provider because adapter switching needs
  `IRefitter`.
  - ACE-Step DiT: selecting an `.onnx` DiT routes through `dit-trt.h`. The engine is
    built on first use and cached next to the ONNX as `.engine`; weights are refit
    from the ONNX on each load.
  - ACE-Step LM: `lm-trt.h` is used when `lm_full.onnx` sits in the LM's directory.
    The TRT-LLM path (`lm-trtllm.h`) is compiled out unless
    `-DHOT_STEP_TRTLLM_ENABLE=ON`.
  - MM3 DiT: `"dit_backend": "tensorrt"` on `/mm3/synth` uses `mm3-dit-trt.h`, which
    builds a base engine per GPU under `<models>/mm3/mm3-trt-cache/` and refits it
    per DiT GGUF and adapter stack.
ONNX Runtime is gone from the engine: no build links it, and there is no CMake option
for it. What used to run on it:

- SA3 refine (StableStep) runs on GGML only. `backend=onnx` on `/sa3-refine` or
  `/pp-vae-reencode` returns 400.
- An ONNX DiT's text and condition encoders run on GGML. The condition encoder loads
  from the safetensors XL DiT folder the FSQ fallback finds; without one the request
  fails with a message naming the folder.
- ONNX VAEs and ONNX text encoders are no longer registered. `/synth` with an `.onnx`
  `vae_model` returns 400. Requests for the removed ACE preview pipeline also return 400.

TensorRT support is compiled in when the SDK is found in `engine/deps/tensorrt/`
(Windows) or installed system-wide (Linux), which defines `HOT_STEP_TRT`. On Windows,
`nvinfer_10.dll` and `nvonnxparser_10.dll` are linked with `/DELAYLOAD`. The DLLs are
large and not in the release archive; users fetch them through the Model Manager.
Delay-loading lets every binary start without them, and every TensorRT entry point
first asks `trt-runtime-probe.h` whether the DLLs can be loaded, so a missing runtime
fails the request with a message instead of crashing the process.

Timing breakdowns, benchmarking from logs and the list of what is done and planned
are in `.claude/skills/engine-performance/SKILL.md`.

## Training subcommands

`ace-train` links no `acestep-core`; every trainer is header-only under
`engine/src/train/`. It writes machine-readable JSONL on stdout with `--jsonl` and
human logs on stderr, and exits 0 on success, 1 on a runtime failure and 2 on bad
usage. `ace-train --help` prints the full option list.

| Subcommand | Purpose |
|---|---|
| `preprocess` | Build per-song tensor caches from an ACE-Step dataset |
| `train-lm` | Train an ACE-Step planner-LM LoRA |
| `train-dit` | Train an ACE-Step DiT LoRA or LoKr |
| `detok-table` | Dump the FSQ detokenizer output for every code |
| `mm3-encode` | MM3 DAV encode: raw audio to flow latents |
| `mm3-codes` | MM3 dataset to RVQ codes (LM training input) |
| `mm3-launder` | Cover-launder a dataset through the MM3 model before encoding codes |
| `mm3-retarget` | Shorten tracks over MM3's 6:00 cap by removing one repeated section |
| `mm3-lm-train` | Train an MM3 LM adapter |
| `mm3-lm-loss`, `mm3-lm-probe` | MM3 LM trainer diagnostics and wiring gate |
| `mm3-preprocess` | MM3 dataset to flow-DiT target latents |
| `mm3-condition` | MM3 AR rollout to a flow-DiT conditioning cache |
| `mm3-train-dit` | Train an MM3 flow-DiT LoRA |
| `rec7-selftest` | Parity gate for the rec7 state encoder |
| `yue2-preprocess` | Audio folder to cached YuE2 VAE latents and a manifest. Skips `*.engine.wav` (old server conversion-cache leftovers, duplicates of real tracks), as the dataset scanner does |
| `yue2-tokenize` | Fill a YuE2 manifest's codec ids |
| `yue2-align` | Fill word timings for the AR lyric-cursor loss |
| `yue2-sheet` | Fill the ABC lead sheet with SheetSage2; `--melody-only` serves cover inference, while training keeps full score by default |
| `yue2-nar-train` | Train a YuE2 NAR LoRA |
| `yue2-ar-train` | Train a YuE2 AR (composer) LoRA |
| `yue2-prepare-aitk`, `yue2-import-aitk-cache` | Build or import caches for joint training |
| `yue2-joint-train` | YuE2 AR and NAR joint training. `--eval-base-loss` trains nothing: it prints the base planner CE and decoder flow MSE per song (the Optimise phase) |
| `yue2-optim-check` | Optimizer self-check |
| `spike` | Phase-0 evidence runs |

The flash-attention training ops these trainers use come from the ggml patch stack
below. How the Training Studio drives these subcommands, and the design decisions
behind them, are in [training-internals.md](training-internals.md).

## The ggml patch stack

`engine/ggml` is a pristine submodule checkout. `engine/patches/*.patch` holds every
HOT-Step change to it, and CMake applies them in sorted order at configure time
(`HOT_STEP_APPLY_PATCHES`, on by default). A patch that already reverses cleanly is
skipped.

| Patch | What it changes |
|---|---|
| `alloc-free-blocks.patch` | Raises ggml-alloc's per-chunk free-block table from 256 to 1024 entries |
| `bf16-out-prod.patch` | CUDA `out_prod` accepts a BF16 `src0` |
| `cpy-q-occupancy.patch` | Fixes the launch geometry of CUDA quant-to-F32 copies |
| `cudagraph-log.patch` | Env-gated trace of CUDA graph decisions |
| `f16-f32-accumulate.patch` | CUDA F16 GEMMs accumulate and write in F32 on every architecture |
| `flash-attn-train.patch` | Adds `GGML_OP_FLASH_ATTN_TRAIN` and `_BACK`, with new `ggml-cuda/fattn-train.{cu,cuh}` |
| `metal-bin-threads.patch` | Raises the Metal binary-op threadgroup cap |
| `metal-im2col-ic.patch` | Adds a separate Metal im2col kernel for the VAE's 1D convolutions |
| `mm-backward.patch` | Env-gated dtype-agnostic `MUL_MAT` backward (`GGML_BACKWARD_MM=1`) |
| `quant-cpy-kquant.patch` | CUDA `CPY` reaches ggml's generic quant-to-F32 converter, so K-quants can be cast |
| `zz-yue2-convrot8.patch` | Adds the CUDA-only `GGML_OP_CONVROT8` and `_BACK` ops for YuE2 ConvRot training, with new `ggml-cuda/convrot8.{cu,cuh}` |
| `zzz-yue2-bf16-round.patch` | Adds the `GGML_UNARY_OP_BF16_ROUND` unary op |
| `sched-unplaced-log.patch` | Logs the node no backend accepts before the scheduler's `cur_backend_id != -1` assert |
| `zzzz-vulkan-train-ops.patch` | Vulkan `BF16_ROUND` and `FLASH_ATTN_TRAIN`/`_BACK`, with new `vulkan-shaders/fa_train_*` files; what the YuE2 joint trainer needs off CUDA |

`engine/patches/README.md` explains each one in depth, with the measurements behind
it.

Two rules that bite:

- `flash-attn-train` and `zz-yue2-convrot8` both add to the same enum in `ggml.h`,
  so once both are applied neither reverses on its own. CMake then logs "neither
  applies nor reverses" for `flash-attn-train` on every healthy build. That warning
  is expected. `verify-hooks.ps1` checks the symbols themselves and is the reliable
  answer.
- Never `git reset --hard` in the superproject. With `submodule.recurse=true` it also
  resets `engine/ggml`, the two patches that create files then refuse to reapply,
  and the build fails with hundreds of CUDA errors. Recovery steps are in
  [AGENTS.md](../../AGENTS.md#git-rules).

## HOT-Step endpoints

Routes `ace-server` adds on top of upstream's `/lm`, `/synth`, `/understand`,
`/job`, `/health`, `/props`, `/logs` and `/`. Request and response details are in
[the endpoint reference](../../engine/docs/ARCHITECTURE.md#endpoints).

| Route | Purpose |
|---|---|
| `POST /vae` | VAE encode (audio in, latents out) or decode (latents in, audio out) |
| `POST /codes-decode` | Render LM audio codes to audio (training audition) |
| `POST /warm` | Preload a DiT, VAE and adapter |
| `GET /jobs` | List every job in the table |
| `GET /plugins` | Lua plugin registry |
| `GET /vram` | GPU memory use (CUDA builds) |
| `GET /models/loaded` | Modules resident in the model store |
| `POST /models/unload` | Evict one module by label |
| `POST /models/restore-policy` | Undo a `?keep_loaded=1` switch |
| `POST /pp-vae-reencode` | Round-trip audio through the post-processing VAE |
| `POST /sa3-refine` | Stable Audio 3 SDEdit refine of a finished track |
| `POST /supersep/separate`, `GET /supersep/progress`, `GET /supersep/result`, `GET /supersep/serve`, `POST /supersep/release`, `POST /supersep/recombine` | Stem separation jobs |
| `POST /spectral-lifter` | Spectral Lifter clean-up (CPU) |
| `GET /mm3/props`, `POST /mm3/warm`, `POST /mm3/unload`, `POST /mm3/select-model` | MM3 status, residency and quant selection |
| `POST /mm3/synth`, `GET /mm3/job`, `GET /mm3/take`, `GET /mm3/stream` | MM3 generation, progress, ensemble takes, live audio |
| `POST /mm3/tokenize-check`, `POST /mm3/imatrix` | MM3 prompt token count, imatrix collection for `quantize` |
| `POST /mm3/voc-decode`, `/mm3/dit-forward`, `/mm3/flow-sample`, `/mm3/depth-frame`, `/mm3/cond-encode`, `/mm3/lm-plan`, `/mm3/synth-e2e` | MM3 bring-up and parity endpoints. They run GPU work outside the job queue; do not build features on them |
| `GET /yue2/props`, `POST /yue2/warm`, `POST /yue2/unload`, `POST /yue2/select-model` | YuE2 status, residency, quant and adapter selection |
| `POST /yue2/synth` | YuE2 generation (progress and result through `/job`) |
| `POST /yue2/tokenize-check`, `POST /yue2/imatrix`, `POST /yue2/align` | YuE2 prompt check, imatrix collection, lyric forced alignment |

## Related

- [engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md): request JSON, generation modes, CLI flags, endpoints
- [architecture.md](architecture.md): the three tiers and how requests flow between them
- [building.md](building.md): build scripts, toolchains, optional SDKs
- [plugins-authoring.md](plugins-authoring.md): writing Lua plugins
- [training-internals.md](training-internals.md): the training system
