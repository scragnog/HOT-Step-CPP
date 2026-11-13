# Engine guide

`engine/` is the C++17/GGML inference and training engine behind HOT-Step. It
started as a fork of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)
(ACE-Step 1.5 on GGML) and now also carries MiniMax-Music3
(MM3), YuE2, Stable Audio 3 (SA3) and HeartMuLa, plus a training toolchain, a Lua sampler-plugin
host and optional TensorRT paths.

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
| `engine/src/heartmula/` | HeartMuLa global/local LMs, tokenizer, LoRA, HeartCodec encoder/decoder, pipeline and HTTP routes |
| `engine/src/sa3-*.h` | Stable Audio 3: the DiT (`sa3-dit-ggml.h`), SAME autoencoder (`sa3-same-ggml.h`), T5Gemma text encoder (`sa3-t5gemma-enc.h`), the sampling loop behind `/sa3-refine` and `/sa3-synth` (`sa3-refine.h`), the checkpoint-preview resolver (`sa3-preview-resolver.h`) and the boot probe (`sa3-families.h`) |
| `engine/src/train/` | Trainers and dataset preprocessors used by `ace-train` |
| `engine/src/moss/` | MOSS-Music captioner used by `ace-caption` |
| `engine/src/flamingo/` | Music Flamingo (AF-Whisper encoder, rotary time embedding, Qwen2-7B) used by `ace-caption --family flamingo` |
| `engine/src/solvers/`, `schedulers/`, `guidance/` | Older C++ sampler registries. The live sampler resolves solvers, schedulers and guidance through the Lua plugin registry instead. Still in use: the `*-interface.h` types the Lua host passes (`SolverState` and the rest), `guidance/apg-core.h` for the native `apg()` bridge, and `scheduler_clamp()` from `schedulers/scheduler-registry.h` |
| `engine/tools/` | One `.cpp` per binary, a few headers they share (`beat-this.h`, `midi-grid.h`, `monitor-chain.h` and others), Python converters (`convert-*.py`) and fixture scripts |
| `engine/tests/` | Unit tests: the sources of the CMake test targets, and standalone files whose header gives the compile line |
| `engine/plugins/` | Bundled Lua plugins: `solvers/`, `schedulers/`, `guidance/` |
| `engine/ggml/` | ggml submodule, pinned to the HOT-ggml fork (see [The ggml fork](#the-ggml-fork)) |
| `engine/vendor/` | yyjson, LuaJIT 2.1, cpp-httplib, minimp3, pocketfft, VST3 SDK |
| `engine/deps/` | Optional vendored SDK: `tensorrt/` |

## Binaries

All targets are defined in `engine/CMakeLists.txt`.

| Binary | Source | What it does |
|---|---|---|
| `ace-server` | `tools/hot-step-server.cpp` | The HTTP server the app runs. ACE-Step `/lm`, `/synth`, `/understand` plus every HOT-Step endpoint, the `/mm3/*` and `/yue2/*` families and the SA3 routes |
| `ace-lm` | `tools/ace-lm.cpp` | ACE-Step 5 Hz LM from the command line: request JSON in, `request0.json`..`requestN-1.json` out |
| `ace-synth` | `tools/ace-synth.cpp` | ACE-Step text encoder, DiT and VAE from the command line: request JSONs in, MP3 or WAV out |
| `ace-understand` | `tools/ace-understand.cpp` | Reverse pipeline: audio in, caption, lyrics and metadata out |
| `ace-caption` | `tools/ace-caption.cpp` | Local captioning with MOSS-Music or Music Flamingo (`--family`, else picked by the files in `--models`): audio in, prose caption, MM3-format caption, YuE2 fields or section-tagged lyrics out. `--hint` (or a third `--src-list` column) is appended to every non-lyrics prompt. Decoding is greedy unless `--temperature` > 0, then nucleus sampling (`--top-p`, `--seed`). Weights come from `tools/convert-moss.py` / `tools/convert-flamingo.py` |
| `ace-midi` | `tools/ace-midi.cpp` | MuScriptor audio-to-MIDI transcription (`--transcribe`), plus parity modes against a Python oracle. With `--beat-model` it also runs Beat This! (see below) |
| `ace-train` | `tools/ace-train.cpp` | Training toolchain, one subcommand per job (see [Training subcommands](#training-subcommands)) |
| `neural-codec` | `tools/neural-codec.cpp` | ACE-Step VAE encode and decode, with optional Q8/Q4 latent compression |
| `mp3-codec` | `tools/mp3-codec.cpp` | Standalone MP3 encoder and decoder |
| `quantize` | `tools/quantize.cpp` | GGUF requantizer for ACE-Step, MM3, YuE2 and SA3 DiT files, with optional `--imatrix` |
| `mastering` | `tools/mastering.cpp` | Reference-based mastering (the matchering algorithm) |
| `vst-host` | `tools/vst-host.cpp` | VST3 host: `--scan`, `--gui`, `--process`, `--process-chain`, `--monitor`. Each plugin gets a 10 s warm-up pre-roll of the track's opening, fed time-reversed so plugin buffers hold the song's own start when the real pass begins. `--monitor` (Windows, WASAPI) plays a track through the chain live, driven by a control file the server rewrites atomically with an increasing `seq`; see the note below |
| `yue2-probe` | `tools/yue2-probe.cpp` | YuE2 bring-up and parity CLI (`--info`, `--load`, `--tokenize`, `--ar-parity` and more) |
| `heartmula-generate` | `tools/heartmula-generate.cpp` | HeartMuLa text-to-music CLI using the same pipeline as `/heartmula/synth`, with seed, cancellation and audio checks |
| `mulacover-check` | `tools/mulacover-check.cpp` | MuLaCover port checks: Qwen style vector against capture-layout fixtures, the converted pair through discovery and load, refusals of bad cover tensors, and conditioning, GACA and prefill/decode parity against the 2c captures ([MuLaCover](#mulacover-cover-model)) |
| `sa3-ggml-test`, `fattn-train-test`, `moss-ggml-test`, `bs-roformer-test`, `mdx23c-test` | `tools/*-test.cpp` | Parity tests against reference outputs |
| `yue2-train-gemm-precision-test` | `tools/yue2-train-gemm-precision-test.cpp` | The YuE2 trainer's BF16 GEMMs against `YUE2_AITK_PRECISION=strict-f32`, forward and backward, at YuE2 shapes on the first GPU |
| `yue2-convrot8-cpu-numeric-test`, `yue2-*-metal-*-test` | `tools/yue2-*.cpp` | YuE2 training-op gates: the CPU ConvRot8 kernels bit for bit against a scalar oracle, and each Metal training kernel against the CPU backend or the Metal kernel it replaced. The Metal ones skip without a Metal device |
| `yue2-rms-double-round-test`, `yue2-linear-lora-round-test`, `yue2-clip-non-finite-test`, `yue2-head-loss-device-test` | `tools/yue2-*.cpp` | YuE2 trainer checks: whether two graph simplifications are bit-exact, that a non-finite gradient still fails the clip, and that the device head loss matches the host mode |
| `yue2-nar-op-bench`, `yue2-mul-bf16-dispatch-profile` | `tools/yue2-*.cpp` | Timing diagnostics with no pass or fail: Metal kernel throughput at YuE2 NAR shapes, and the dispatch cost of the trainer's BF16 multiply |
| `backend-health-test` | `tools/backend-health-test.cpp` | The GH #242 recovery flow on the shared backend with an injected fault (`ACE_FAULT_BACKEND_ERROR=align`, compiled into this target only): `--no-probe` must fail, `--other-holder` must end on the clear error. It proves the control flow, not recovery from a real Metal fault |
| `yue2-vae-lifetime-test` | `tests/test-yue2-vae-lifetime.cpp` | YuE2 VAE decoder cache lifetime (see [YuE2](#yue2-plan-compose-render)) |
| `convrot-graph-probe` | `tools/convrot-graph-probe.cpp` | ConvRot8 under repeated CUDA graph capture and replay (CUDA only, see [ggml-fork.md](ggml-fork.md)) |

`-DBUILD_TESTING=ON` adds CTest targets the default build leaves out: `test-monitor-chain`,
the DoRA fixtures (`dora-common-test`, `dora-runtime-test`, `mm3-dit-dora-test`,
`sa3-dora-load-test`, `yue2-aitk-dora-test`), the SA3 training gates (`sa3-train-grad-test`
and an `ace-train sa3-train-check` run) and `yue2-recompose-test`. The two SA3 training
gates exit 77, which CTest counts as skipped, when their fixtures or the base DiT are absent.

`fattn-train-test --backend cpu\|cuda\|vulkan\|metal` (default `cpu`, both arms on CPU) resolves
the requested GPU backend by registry device name, never by "first GPU device found": with CUDA
and Vulkan both loaded, `--backend cuda` cannot land on Vulkan or vice versa. A requested
backend that is not loaded is a hard failure (prints the loaded device list), never a silent
CPU fallback. `--prec tf32` and `--prec bf16` stay CUDA-only; Vulkan and Metal run f32 at the
same 1e-4 tolerance as CPU. The timing and check modes (`--bench`, `--bench-lm`, `--bench-tr`,
`--bench-yue2`, `--dsw-check`, `--saved-check`) target CUDA (the tf32 arm is a tensor-core
question), or Metal with `--backend metal` for the Metal FA-train checks, and reject any other
`--backend` rather than silently ignoring it.

`vst-host --monitor` first asks WASAPI's shared mode to take the track's own rate. If the
device refuses, it renders at the device rate through a linear resampler
(`tools/monitor-resampler.h`). The resampler keeps every processed source frame it has not
yet passed, plus the fractional phase, across device callbacks, so the track advances by
exactly `n * track_rate / device_rate` frames per `n` device frames whatever the callback
sizes are. Seek, track change and stop drop the buffered frames; pause leaves them for the
resume. A track at a new rate re-prepares every plugin at that rate.
`engine/tests/test-monitor-resampler.cpp` checks continuity and frame accounting with
irregular block sizes (compile line in the file). The plugin chain loop itself lives in
`tools/monitor-chain.h` (`monitor_run_chain`): each block is cut into chunks of at most
`MAX_CHUNK` frames and every chunk runs the plugins in order, ping-ponging between two
scratch buffers, so a non-commutative chain keeps its order and the output may alias the
input. `engine/tests/test-monitor-chain.cpp` checks order, in-place output and multi-chunk
blocks without any plugin; it is opt-in (`-DBUILD_TESTING=ON`). On the Node side
(`server/src/routes/vst.ts`), monitor start, restart and stop run one at a time, so
overlapping requests cannot spawn two hosts on the same control and status files. Host
lifecycle lines go to `logs/<session>/vst_host.log`.

`tools/ace-server.cpp` is upstream's server, kept as a reference for syncs. It is
not compiled. The shipped `ace-server` is built from `hot-step-server.cpp`.

The Node server starts `ace-server` with `--models`, `--host` and `--port`, and adds
`--adapters`, `--keep-loaded`, `--noise-profile`, `--draft-lm`, `--vae-chunk` and
`--vae-overlap` when they are configured, plus `--user-plugins` when the app runs with a
data folder (`server/src/services/aceEngineProcess.ts`). The app's default port is
8085; the binary's own default is 8080.

### ace-midi speculative decoding

`ace-midi --draft-model <dir> [--speculative-k 1-4]` ports upstream MuScriptor PR #95.
The draft (small) proposes k tokens one step at a time; the target checks `[current,
p1..pk]` in one block and keeps the proposals up to the first disagreement, plus its own
token there. Rejected K/V rows are not undone: `n_past` moves back and the next write
overwrites them. The block runs through `ggml_soft_max_ext` with an explicit causal mask,
because `ggml_diag_mask_inf` is wrong on CUDA when `n_past > 0`. Models of different card
sizes pair fine (small 1393, large 1395); only the 1393 emittable tokens are compared.

Measured on a 4-minute track, RTX 5090: large 121.6 s scalar, 90.1 s at k=2, 79.4 s at
k=4; medium 60.9 / 66.1 / 59.6 s. The MIDI was byte-identical to scalar decoding in every
run, CPU included. Decode is launch-bound (about 0.21 ms per layer per token at any width,
so a 14-layer small step costs 2.9 ms against 10 ms for 48-layer large), which is why only
large gains. The app passes `--draft-model small --speculative-k 4` for large only
(`speculativeArgs` in `server/src/services/muscriptor.ts`).

Upstream PR #94 (preallocated KV cache without a NaN fill, native reshapes instead of
einops, one cache write per step) has no GGML counterpart to port: the cache is allocated
uninitialised already, and graph build plus allocation is about 6% of a decode step.

### ace-midi tempo, meter and bars

`ace-midi --beat-model <dir>` runs Beat This! (`tools/beat-this.h`, a GGML port of
CPJKU's beat and downbeat tracker, MIT weights `final0` as `model.safetensors`) on the
source audio at its own rate. `tools/midi-grid.h` then fits upstream MuScriptor v0.4's
grid to the beats: least-squares tempo, meter from downbeat spacing (written as n/4 when
90% of bars agree), an onset-delay correction, and a tempo map when the beats do not fit
one tempo within 5% of a beat. The MIDI gets a time signature, `set_tempo` in every track,
and a whole-bar delay so tick 0 is a bar line. Without `--beat-model` the output is the old
fixed 120 BPM file, byte for byte (`--validate-midi`).

| Flag | Effect |
|---|---|
| `--beats-out <file>` | Tracked beats plus the written grid in audio seconds (`beats`, `bars`, `audio_shift_s`: add it to a MIDI time to get the audio time) |
| `--bpm`, `--beats-per-bar`, `--first-downbeat` | Override the detected grid |
| `--tempo-map` / `--fixed-tempo` | Force a tempo map or one tempo (default: map only when unsteady) |
| `--beat-scale <2\|0.5\|…>` | Beats per detected beat, a power of two |
| `--quantize [--grid N] [--quantize-strength x]` | Snap notes to 1/N beat (default: the measured subdivision) |
| `--fill-gaps <beats>`, `--trim-leading-bars` | Hold notes over short rests; drop empty leading bars |
| `--export <events.jsonl> --beats-in <beats.json>` | Re-export a transcription with new options; no MuScriptor model loaded |
| `--abc-out <file> [--abc-ins <program>] [--abc-melody-only]` | Also write a YuE2 two-voice ABC lead sheet (`tools/midi-abc.h`) through SheetSage2's notation stage: voice notes → Vocal, one instrument's top line → Ins, a Krumhansl key estimate and half-bar template chords, notes pre-snapped to the 4-per-beat subbeat grid. Prints a `{"type":"abc",...}` summary line |
| `--track <audio>` | Beats and grid only (needs `--beat-model` and `--beats-out`) |
| `--self-test-grid` | Grid maths on synthetic beats with known answers |

## HeartMuLa

`heartmula/pipeline.h` runs tags and lyrics through the global language model and
local decoder, then decodes eight RVQ codes per frame through HeartCodec's flow
model and scalar waveform decoder. Output is stereo 48 kHz. Duration is capped
at 240 seconds, or 3000 frames at 12.5 Hz; EOS can finish earlier. A sampled code
8192 is reserved, outside the codec's 0..8191 range. The pipeline returns
`bad_codes` before retaining that frame, including for `codes_only` requests.
EOS tokens at 8193 or above still end generation normally.

The tokenizer, global/local LM pair, RVQ, flow and waveform decoder are separate
GGUF components. Discovery checks their metadata and dependencies; the LM pair
must share a base fingerprint. LoRA targets the global LM. The codec encoder
components are available for audio encoding but are not loaded for text generation.
See [the component and HTTP contract](../../engine/docs/HEARTMULA_CONTRACT.md).

LM and codec run in separate phases, with admission against free device memory
before each load and cancellation checks throughout. Without keep-loaded, each
phase unloads the other and completion frees both. Requested prompt and duration
determine the KV allocation. CUDA admission uses NVML where available, matching
the selected device by PCI identity, with a GGML fallback.

`heartmula/server.h` registers `/heartmula/*` after the shared job system and the
MM3/YuE2 unload handlers. `engine/verify-hooks.ps1` checks both this include and
the `heartmula_register_routes()` call. Synth jobs use the shared worker and
`/job` polling. `heartmula-generate` exercises the same pipeline from JSON;
`node tools/engine-smoke.mjs` includes a fixed-seed HeartMuLa render when its
models are installed.

### MuLaCover cover model

MuLaCover is the HeartMuLa 3B/300M stack fine-tuned for covers from a lead sheet.
Its weights are CC BY-NC 4.0 with a non-commercial model licence, and so are its
outputs. `engine/tools/convert-mulacover.py` converts the checkpoint into
`heartmula-mulacover-global-BF16.gguf` and `heartmula-mulacover-local-BF16.gguf`,
using the HeartMuLa component contract with the variant `MuLaCover`. Discovery and
the LM loader therefore treat it as one more variant. Tensors with two or more
dimensions stay BF16 bit for bit, and 1-D tensors widen to F32, so no value
changes. Nothing in the checkpoint is dropped. The converter takes the installed
`heartmula-tokenizer.gguf` and `heartcodec-rvq-F32.gguf` as dependencies, and
refuses unless the tokenizer is identical and the RVQ came from the HeartCodec
checkpoint MuLaCover pins.

The global file also holds the cover modules of backbone layers 20 to 27 (upstream
`peft_layers`; these are full modules, not LoRA factors). Each layer has a
symbolic adaptor for the melody, drum and chord rolls, gated adaptive cross
attention and a tag MLP. The file also has `qwen_linear` for the style slot.
`heartmula_cover_load` (`heartmula/model.h`) checks every one of these tensors'
names, shapes and types from the header before it allocates anything, and names
the first bad tensor. `heartmula_load_lm` loads them whenever the variant is
MuLaCover.

The style condition comes from Qwen3-Embedding-0.6B through the ACE text encoder
(`qwen3_style_vector` in `qwen3-enc.h`). The input is the lowercased, trimmed tag
line with no chat template, ending in the EOS token that upstream's tokenizer
also adds. The function pools the last hidden row, applies an L2 norm in float,
and refuses more than 8192 tokens. The parity reference is FP32, upstream's
default for this model: with the BF16 text encoder the vector is within 1e-3 of it.
The Q8_0 encoder is about 6e-3 away. Both registry files are tensor-identical to the
pinned Qwen checkpoint (the Q8_0 one after re-quantisation).

The conditioning path is split between two files. `heartmula/mulacover.h` builds
the prompt: tags, a style slot, a MuQ slot, then the lyrics, with upstream's
special ids. It also runs the context graph once per request. The input is four
rows on the source's sixteenth-note grid: the melody roll (onset and sustain per
pitch, 256 wide), the drum roll (128), the chord chroma (12) and a padding mask,
at most 5000 steps. Key positions are the row index, with no BPM scaling. Each
cover layer's adaptor projects the rolls to 24 wide each, fuses them to 512 and
runs two bidirectional Llama layers, then projects to 3072. Layer N's context is
the running sum of adaptor outputs 20 to N. The graph writes every layer's roped
cross-attention and gate keys and values into an F16 cache, which is about 940 MiB
at 5000 steps with CFG. `heartmula/lm-graph.h` note G adds the per-step half while
a cover is bound. Before each adapted layer, a tag MLP scales and shifts the hidden
state from the style vector. After the layer, gated cross attention (8 heads x 384,
gate one head x 12) reads that cache, with queries at the backbone positions. The
style slot gets `qwen_linear` of the style vector. On the unconditioned CFG row the
rolls and the tag-MLP input are all -1, and both slots take the unconditional
embedding. Without a bound cover the graph is the plain HeartMuLa graph.

`mulacover-check` checks this path against the five 2c capture cases, comparing
every captured key. Token ids, rolls, CFG rows, masks and positions match exactly.
Teacher-forced codes all fall inside the engine's top-250. The captures ran torch
under BF16 autocast, and torch itself in FP32 misses them by as much as the engine
does. So the float tensors are gated against the same run in torch FP32:
`fp32_ref.py`, beside the capture script, writes it to each case's `fp32/`
folder, and a case without one fails. The engine must stay within twice torch's
own FP32-to-BF16 drift on every float tensor, in relative L2 and in max abs. On
logits it must also flip no more argmaxes than torch BF16 does, and only on
near-ties. At the fed codes, its cross-entropy may sit no further from FP32's
than torch BF16's does. These are the rules `heartmula-probe` uses for the plain
model. The original rules are still evaluated and shown per key. A manifest that is
unreadable, has fewer than 98 keys or lacks a key the engine produces fails the
case.

## How ace-server is organised

One process, one port, one GPU worker thread. `POST /lm`, `/synth`, `/understand`,
`/vae`, `/codes-decode`, `/warm`, `/mm3/synth`, `/mm3/refine` and `/yue2/synth` create a job, push
it onto a FIFO queue and return `{"id":"..."}` straight away. Clients poll
`GET /job?id=` and fetch the result with `GET /job?id=&result=1`. ACE-Step, MM3 and
YuE2 jobs share that queue, so they never run on the GPU at the same time. The job
table keeps 32 entries and evicts the oldest finished ones first.

`POST /sa3-synth`, `/sa3-refine`, `/pp-vae-reencode` and `/yue2/align` answer in the
same request, but `on_worker()` in `hot-step-server.cpp` runs their body on the worker
thread while the HTTP thread waits. Every module shares one GGML backend, and two threads
computing on it at once abort the Metal driver or corrupt the model store (GH #226).
These routes therefore queue behind whatever job is running: a call made during a
long YuE2 batch waits for the whole batch, so a direct engine client needs a timeout
long enough to cover that wait. The MM3 bring-up endpoints, `POST /mm3/imatrix` and
`POST /yue2/imatrix` are not wrapped yet and still compute on their HTTP thread.

Metal reports a failed command buffer only at the next synchronize, and from then on
refuses every `graph_compute` until the backend is recreated (GH #242). `backend_healthy()`
in `backend.h` synchronizes the shared backend and runs a one-element graph. `/yue2/align`
calls it after the MMS_FA forward and answers 500 `MMS_FA compute failed on the GPU backend
(device error)`; it only try-locks the YuE2 job locks, so while the NAR lane renders it logs
`probe skipped` instead. A YuE2 job calls it again before the AR prefill. When the backend
is unhealthy the job drains the NAR lane, unloads YuE2 and reloads on a fresh backend. That
works only when YuE2 held the last backend ref: if another model still holds one, the job
fails with `GPU backend in error state, unload models to recover`, and unloading the other
models clears it. Under the default one-module policy the backend usually reaches zero refs
between jobs anyway, so the recovery matters mostly with `--keep-loaded`. The probe covers
the shared backend only, not the NAR lane's own `backend_nar`.

`POST /sa3-synth` and `/sa3-refine` cancel by connection: when the client closes it,
both SA3 sampling loops stop before the next step, and a full-loop Lua solver sees
`on_step` return true. `aceClient` then awaits `GET /worker-drain`, a no-op on the
worker, so Node releases the lane only after the engine has stopped. Limits: a Lua
`sample()` that ignores `on_step`'s return runs to its last step, and model load,
encode and decode are not checked; the drain waits for them.

Models are discovered by scanning `--models` at startup. ACE-Step files are
classified by their GGUF `general.architecture` into LM, text encoder, DiT and VAE
buckets (`model-registry.h`). MM3 looks in `<models>/mm3/` and then `<models>/`;
YuE2 looks in `<models>/yue2/` and then `<models>/`. The server still boots when only
MM3, YuE2 or SA3 weights are present (`hot-step-families.h`); the ACE-Step endpoints then
answer 501.

GPU residency is owned by `model-store.h`. The default policy keeps at most one
module in VRAM at a time. `--keep-loaded`, or `?keep_loaded=1` on `/lm`, `/synth` or
`/warm`, switches to keeping everything resident. `POST /models/restore-policy` undoes
the per-request switch; it refuses when `--keep-loaded` came from the command line.

## Multi-GPU workers (experimental)

`ACESTEPCPP_WORKER_GPUS` (see [Configuration](config.md)) starts one `ace-server` per listed
CUDA card, each pinned by UUID. The server binds each job's engine calls to one worker, so the
backends hosted by `ace-server` reach the same worker's port without a client argument. The
worker table and status are in `server/src/services/engineWorkers.ts`; `GET
/api/settings/engine-workers` serves the status. A worker restarts only while idle. Each
engine stays one process, one port, one GPU worker thread, so each model must fit in one
card. Server-side model caches and model-change routes still target `w0`. With two or more workers,
jobs are dispatched in parallel, one per worker; training, post-processing and previews hold every worker. Real card affinity has not been verified on a dual-GPU machine.

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

On Metal, the matrix-matrix kernel (used when a matmul has more than 8 columns, so
on every prefill) stages its f32 input as half. The 1.7B LM's SwiGLU output goes past
half's 65504, which would make `down_proj` return inf and fill the KV cache with NaN.
`qwen3_build_mlp` therefore takes a `down_scale`: `qw3lm_init_backend` sets it to 1/16
on Metal and to 1 everywhere else, and the MLP scales the `down_proj` input by it and
divides the output by it again. If a prefill still produces non-finite logits,
`pipeline-lm.cpp` logs `[LM] ERROR: prefill logits are not finite`. On the server
side, a `/lm` result that comes back without audio codes fails the generation with
"LM planner returned no audio codes" (`services/backends/ace/generate.ts`) instead of
rendering from the caption alone.

### MiniMax-Music3: planner LM, then DiT renderer

```
caption + lyrics -> prompt assembly and caption cleaning (mm3-request.h)
  -> tokenizer (mm3-tokenizer.h)
  -> AR planning loop (mm3-ar-loop.h): global LM (mm3-lm-graph.h) emits one semantic
     code per 25 fps frame; the RVQ depth decoder (mm3-depth-graph.h) adds 7 acoustic
     codebooks per frame -> frame hiddens
  -> per 200-frame window, 100-frame hop by default, `flow_hop_frames` up to 160 (mm3-pipeline.h):
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

### MM3 latent refiner: audio in, audio out

An optional post-process stage, `POST /mm3/refine`, runs the terminusresearch
minimax-music3-latent-refiner v0.10 on any finished render, not only MM3's. It is the
package's `MiniMaxMusic3RefinerPipeline` plus the loudness convention of the runner we
validated it with. Request and wire contract: [engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md#mm3-latent-refiner-request-post-mm3refine).

```
audio (any rate, mono or stereo) -> 44.1 kHz stereo, zero-padded to the 512-sample DAV hop
  -> windows (30 s, 2 s overlap by default; window 0 = the whole clip as one sequence)
  -> DAV encoder (mm3-dav-encode.h)               audio -> latents [128, L], 86.13 Hz
  -> MERT-v1-95M (mm3-refiner-mert.h)             13 hidden states, resampled to L frames
  -> refiner DiT bridge (mm3-refiner-dit.h)       32 deterministic Euler steps from the input latents
  -> linear-fade overlap stitch (mm3-refiner.h)
  -> DAV decoder (mm3-vocoder-graph.h)            latents -> 44.1 kHz stereo
  -> trim, match the input's integrated loudness, clip, back to the input rate, mix
```

`mm3-refiner.h` holds the pipeline (`mm3_refiner_run`), the window plan, the overlap-add and a
port of pyloudnorm's K-weighted integrated loudness; `mm3-refiner-job.h` is the route and job.
Windowing, hop rounding (python's half-to-even `round`), the `torch.linspace` fades and the
trim follow `pipeline.py` exactly. `mix` blends in the audio domain after the loudness match,
and `mix=0` returns the input untouched without loading anything (the route answers it
without queueing). A mono source (identical channels) comes back mono: the refiner is not
channel-symmetric, so its two outputs differ slightly.

**Residency.** All five modules go through `ModelStore` (`MODEL_MM3_ENC`,
`MODEL_MM3_REFINER_MERT`, `MODEL_MM3_REFINER_CLAP`, `MODEL_MM3_REFINER_DIT`,
`MODEL_MM3_REFINER_VOC`; labels `MM3-DAV-Enc`, `MM3-Refiner-MERT`, `MM3-Refiner-CLAP`,
`MM3-Refiner-DiT`, `MM3-Refiner-Voc`). Under `EVICT_STRICT` only one GPU module is resident, so
the run is phased: every window is encoded, then every window gets its MERT states, then its
CLAP style vector, then every window is bridged, then the stitched latents are decoded. Each module is required once and released before the next (RAII handles, so success,
error and cancel all unload; under keep-loaded they stay for the next job). Conditioning stays on
the host between phases: about 100 MB per 30 s window of MERT states, freed as each window's
bridge finishes. The decoder is `Mm3RefVoc`, the `voc.*` tensors loaded on their own with their
own graph state, so the 12 to 23 GB MM3 model need not be warm. All five files are opened
before any module loads (four when the style override below is set), so a missing or unreadable file fails fast and names its role.

**Precision.** The DAV encoder (`strict_f32`, set by the store; training leaves it off) and MERT
run their GEMMs with `GGML_HINT_STRICT_F32`: on ggml-cuda's default TF32 the DAV latents and
MERT states miss the frozen parity gates. The refiner DiT and the decoder run on the default.

**Cost.** A 30 s window peaks at about 2.3 GB of VRAM beyond the idle baseline, the DiT's dense
attention being most of it. The DiT attends over 2L tokens, so memory grows with the square of
the sequence L: 60 s peaks at about 6.3 GB. L is the whole clip in direct mode (`window=0`), the
window in windowed mode, or the clip when it is shorter than the window; it is limited to 5168
frames (60 s) in every case, answered with 413 before anything queues. Measured on an RTX 5090 with the GPU idle and all four modules cold-loaded
(F32 refiner and MERT, F16 encoder and decoder): a 34 s clip at 32 steps takes 6.5 s, a 60 s clip
(three windows) 8.9 s; the bridge is about two thirds of that.

**Style vector.** The DiT takes a 512-d CLAP vector per window, made by the CLAP audio tower
(`mm3-refiner-clap.h`, `mm3-refiner-clap-*.gguf`): channel mean, centre crop of at most 10 s,
48 kHz, mel, HTSAT, projection, L2 norm, as `pipeline.py` does. `mm3_refiner_run` computes all
windows' vectors in one CLAP phase after MERT (reported as `refine_condition`) and keeps them on
the host. A missing CLAP file answers 501 like any other pack file; nothing falls back to zeros
or a constant. For testing, `HOTSTEP_MM3_REFINE_TEST_STYLE=<raw float32 file of 512 values>`
supplies one vector for every window instead (`mm3_refiner_style_override()`; the CLAP file is
then not needed); the probe passes the torch fixtures' vectors per window through the same
`Mm3StyleFn`.

**Checks.** `mm3-refiner-probe` (`engine/tools/mm3-refiner-probe.cpp`) runs the whole pipeline
against the torch fixtures (window starts, overlap weights, DAV latents, stitched latents, raw and
final audio, loudness gain) and the behaviour tests; `tools/mm3-refiner/check-integration.ps1`
runs it for steps 1, 8 and 32 on all twelve clips, then drives `POST /mm3/refine` on throwaway
engines (400, 413, 501, bypass, phases, cancel, FIFO, 48 kHz, mono, VRAM growth, and the ten
native clips end to end with the real CLAP style against the torch `final.wav`). It needs the
ignored fixtures from `tools/mm3-refiner/README.md` and a `tools/mm3-refiner/artifacts/models/mm3/`
folder holding `mm3-refiner-refiner-f32`, `mm3-refiner-mert-f32`, `mm3-refiner-clap-f32`,
`mm3-enc-f16` and `mm3-voc-f16`.

Two header quirks: `mm3-refiner.h` hides the `MM3_DIT_MAX_NODES` macro of `mm3-dit-graph.h`
while it reads `mm3-refiner-dit.h`, which declares a constant of that name; and
`mm3-refiner-job.h` has its own error helper because `json_error()` is defined after its
include point in `hot-step-server.cpp`.

### YuE2: plan, compose, render

```
style + lyrics (+ optional ABC) -> tokenizer and prompt assembly (yue2-tokenizer.h)
  -> plan stage: ABC lead sheet, only when cot is "melody" or "full"
  -> semantic stage: the AR "composer" (yue2-lm-graph.h) samples codec ids
  -> NAR stage: mixture-of-transformers flow matching with a midpoint ODE solver
     (yue2-nar-graph.h), optionally chunked (nar_chunk_frames)
  -> Oobleck VAE decode, "standard", "legacy" or local "dec2x" variant (yue2-vae-graph.h)
  -> stitch; a track that peaks past full scale is scaled to -0.3 dBFS (yue2_peak_scale)
  -> 16-bit WAV (yue2-pipeline.h, yue2-job.h)
```

`codec_ids` on a `/yue2/synth` request supplies the composer's codec stream (raw ids,
0..32767), which is prefilled instead of sampled. With `continue_codec_ids: true` the
stream is treated as a prefix: it is prefilled under the unchanged prompt and score,
then the composer keeps sampling from frame `codec_ids.length`. Use it to keep a
render's codecs up to a section boundary and recompose the rest. The frame clock, caps,
`min_tokens` and the repetition window carry on from the prefix. Under guidance
(`cfg_scale` other than 1) the negative set is prefilled with the same prefix, because the
decode feeds both sets each sampled frame. The request accepts one to four songs
with a shared prefix and no `semantic_retries`, and the prefix must stay under both
the context and the stage cap. Each `songs` entry can supply its own explicit seed.
Matching prompts share the prefix prefill by copying KV rows and logits, including
the negative CFG rows; scheduled lyrics replay the prefix as a batch. Every row's
output must begin with the exact prefix, or the whole job fails. Reproduction by
seed requires the same batch width.
A lyric schedule also runs under guidance: the negative prompt hides the same score lines
at the same times (`yue2-lyric-schedule.h`), though the server's Lyric Timing still skips
itself at a CFG other than 1. The stage fails if the output does not begin with the exact
prefix. The NAR and VAE then render the whole stream again, prefix included, unless
`render_from_frame` starts them later. Prefix noise is the same for the same
`noise_seed`, but the NAR attends both ways across its chunk, so the prefix audio is
close to the original render without being identical. Render the whole song for
anything kept; do not splice old prefix audio onto the new suffix.

`render_from_frame` is what keeps late section takes cheap. Each NAR chunk attends
over the prompt and its own frames only (`yue2_seal_chunks`), so starting the chunks
at a later frame drops no context the NAR reads; the composer still prefills the
whole prefix. The VAE decodes the same window, and each track reports it as
`audio_from`. The window's audio is a slightly different render of the same codes
than a full render, because the chunk edges and the peak normalisation move with
it. On a 3.5-minute build a take went from 22.6 s to 7.7 s.
Every finished track in the `/job` status `tracks` array carries its `semantic_ids`, and
the server saves them beside the WAV as `<song>.semantic.json`, with the request that made
them as `<song>.yue2-request.json`, so a later request can resume the stream. Add
`preview_max_frames` to stop the composer at a section's end.
`yue2-probe --continue-check <request.json>` checks prefix identity, cancellation and
the context cap on any backend.

NAR plugins and plan conditioning. On a `/yue2/synth` request, `infer_method` (the NAR
solver) accepts `md_wasserstein_yue2`, `md_ancestral_yue2`, and the full-loop solvers
`md_latent_eq_yue2` and `md_tlie_yue2`; `scheduler` accepts `md_ht_scheduler V3`,
`md_plan_aware_yue2`, `md_plan_loose_yue2` and `md_plan_overdrive_yue2`. Any other name
fails the request.
Full-loop solvers run through `yue2_nar_run_loop_solver` (`engine/src/yue2/yue2-nar-loop.h`),
which turns a model error, a cancel or non-finite latents into a failed chunk;
`engine/tests/test-yue2-nar-loop.cpp` checks it. The Lua host has no instruction or
time budget, so a plugin that loops without calling `on_step` cannot be cancelled. A
Lua solver gets a seed mixed from the variation's noise seed and the chunk's start
frame, so each chunk draws its own solver noise. `postprocess_plugin` names a Lua
postprocess plugin that replaces the native tiled VAE decode; empty (the default) or
an unknown name keeps the native decode, and so does a plugin that fails.
These `plugin_params` keys tune the NAR stage; every default leaves it unchanged:

| Key | Default | Effect |
|-----|---------|--------|
| `yue2:cads_strength` | 0 | CADS noise on the conditioning, 0..1; 0 builds nothing |
| `yue2:cads_t1`, `yue2:cads_t2` | 0.6, 0.9 | CADS anneal window |
| `yue2:cads_rescale` | true | restore the clean conditioning's per-row mean and spread after noising |
| `yue2:plan_w_start`, `yue2:plan_w_end` | 1, 1 | plan (AR-prefix) attention weight across the steps, clamped to 0..4 |
| `yue2:plan_w_curve` | `linear` | `linear` or `cosine` ramp between them |
| `yue2:plan_cfg` | 1 | plan CFG scale, 1..8; above 1 adds a plan-masked pass per velocity call |
| `yue2:plan_cfg_t1`, `yue2:plan_cfg_t2` | 0, 1 | step window where plan CFG applies (swapped if reversed) |
| `yue2:plan_cfg_rescale` | 0.7 | CFG-rescale blend, 0..1 |
| `yue2:plan_cfg_level_match` | true | with plan CFG above 1, match the latent level to the no-CFG render |

A YuE2 NAR scheduler can also set per-step plan weights through the `plan_strength`
global, and Lua plugins on this path see the plan as `model_context.yue2`; both are
in [Plugin authoring](plugins-authoring.md).

YuE2 loads its weights itself, outside `model-store.h`. Under the default policy a
YuE2 job first evicts every unreferenced store module (`store_evict_all`), as MM3 does,
because the store keeps 0-byte entries resident after use and their real VRAM would
make the YuE2 load run out of memory (#160, #195; the ONNX Runtime SA3 refiner was the
case that hit it).

The YuE2 VAE decoder cache belongs to its model. Unloading YuE2 or switching
VAE variants destroys the decoder graph, scheduler compute buffers and derived
weights before releasing the VAE weights or backend. A resident model reuses its
cache between decodes. `yue2-vae-lifetime-test <models-dir>` checks both VAE
variants, repeated unload/reload and cache reuse with real decoder allocations;
set `GGML_BACKEND=CPU` or `GGML_BACKEND=CUDA0` to select the device.

The AR and NAR halves live in one `yue2-lm-<type>.gguf` and are resident together;
the VAE is a separate `yue2-vae-{standard,legacy}-<type>.gguf` (`yue2-model.h`). A third,
local-only variant, `dec2x` (`yue2-vae-dec2x-f32.gguf`), is a fine-tuned standard VAE whose
decoder has 9 residual units per block instead of 3. It is not distributed; the UI lists
it only when the engine finds the file. Its GGUF carries `yue2vae.dec_res_dilations` (the
decoder's own unit list; the encoder keeps `res_dilations`) and a 40-frame decode halo,
because the extra units widen the receptive field to 29 frames. Build one with
`tools/yue2-vae-train` (train, then `export.py`, then `convert-yue2.py --src-vae-dec2x`);
`yue2-probe --vae-roundtrip <in.f32> <out.f32> --vae <variant>` measures what any VAE
loses on real audio. The
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

Source guidance is the optional second path from the recording to the render.
`ace-train yue2-guide` runs MERT and the tokenizer head over the source with
TF32 off, which ace-server cannot promise, and keeps the head's top-K codes per
frame instead of its argmax. The head's argmax matches the code YuE2 itself drew
on only 27.6% of frames when re-reading one of its own renders; top-20 holds it
on 78.4% and top-64 on 91.4%. So the semantic stage mixes the LM's distribution
with the tokenizer's over those candidates (`guide_scale`) and can forbid
everything else (`guide_outside`), at the same seam as `end_bias`, before the
reference sampler chain. On a 20 s cot=off test the share of drawn codes inside
the top-64 went from 1.4% unguided to 37%, 83%, 96% and 100% at scales 0.25,
0.5, 0.75 and 1.0. The idea comes from comfyui_acetricks'
TokenizerGuideLogitsProcessor; its log-space blend was replaced by the mixture
because the log-space version barely moved at 0.5.

The post-render cover drift check matches sung lyric blocks to like-labelled
score sections in order, skipping instrumental tags and listing unmatched sung
blocks. A section is unscored when its opening or full-block median MMS_FA word
score is below 0.2, or when supplied mix and vocal-stem alignments disagree by
more than half a bar. The live route currently has mix audio only, so its result
states that stem agreement was not checked; offline calibration can supply both.
For the offline alignment gate, Whisper's word timestamps from a saved vocal
stem provide a separate decoder. Exact recognized words are matched to source
lyrics in order, and a section is unscored if its first matched word among the
first five source words differs from MMS_FA by more than half a score bar.
Whisper transcripts longer than twice the supplied lyric word count are
rejected as unreliable. The live route does not run this Whisper check.
`server/src/services/backends/yue2/scoreClock.ts` walks each Vocal bar
under its active `M:` and `Q:` fields; a later meter or tempo change therefore
changes the expected section time. ABC note and rest durations place the drift
reference at the first sounding Vocal note in each section. The result also
retains the section boundary; a section with no Vocal note is unscored. The
offline `server/scripts/yue2-c4-sidecar.ts`
uses that same clock to propose word-to-note matches from cached `cursor_words`
and ABC. It marks shared notes and timing gaps uncertain. The cache has no
original variable beat timestamps, so these matches need manual review before
use as training data. Drift results include a metric version in their saved
cache key; older results are recomputed when requested.

## HOT-Step hook files

Upstream files are kept as close to acestep.cpp as possible so syncs stay a copy.
Four upstream files carry HOT-Step `#include` hooks, and losing them is the main
risk of a sync:

| Upstream file | Hook | If a sync overwrites it |
|---|---|---|
| `pipeline-synth-ops.cpp` | `#include "hot-step-sampler.h"` in place of `dit-sampler.h` | Compiles, but every solver, scheduler, guidance mode, DCW and custom schedule goes dead. The link then fails on `hotstep_sampler_linked_`, a sentinel `hot-step-server.cpp` references on purpose |
| `model-store.h` | `hot-step-params.h` | Compile error |
| `dit.h` | `adapter-merge.h` and `adapter-runtime.h` | Compile error |
| `fsq-detok.h` (with `fsq-tok.h`) | `fsq-quant.h` in place of upstream's own FSQ quantizer copies | Silent if both FSQ headers are overwritten: upstream's quantizer comes back and matches only about 40% of the reference indices. Overwriting one of them is a compile error |

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
- `fsq-quant.h` holds the FSQ levels and index encode and decode, following the
  vector_quantize_pytorch ResidualFSQ reference (soft clamp, then hard clamp).
- `hot-step-families.h` lets the server boot when only MM3, YuE2 or SA3 weights exist.
  SA3's probe lives in `sa3-families.h`.
- `hot-step-build-flags.h` carries `HOT_STEP_DISABLE_FA` for the Volta build.

`hot-step-server.cpp` wires the model families in with one include and one
registration call each: `minimax/mm3-server.h`, `minimax/mm3-job.h` and
`minimax/mm3-refiner-job.h` (`mm3_register_routes`, `mm3_register_job_routes`,
`mm3_refiner_register_routes`) and `yue2/yue2-server.h` (`yue2_register_routes`). SA3 has no subtree hook; its routes are written in
`hot-step-server.cpp` itself.

`engine/verify-hooks.ps1` checks the four hooks, the sentinel, `hot-step-server.cpp`'s own
`hot-step-params.h` include, the family includes and registration calls, both boot gates'
`hot_step_any_family_weights_present()` calls, and the HOT-ggml changes (Hooks 7 to 18).
`build.cmd` runs it before every compile. Run it after any upstream sync; the
procedure is in `.claude/skills/upstream-sync/SKILL.md`, and `engine/UPSTREAM_SYNC`
records the last upstream commit synced.

## Lua plugin host

Solvers, schedulers, guidance modes and VAE-decode postprocessors are Lua
plugins, run by LuaJIT 2.1 (`engine/vendor/luajit`, an unmodified snapshot that
CMake copies into the build tree and builds with its own `msvcbuild.bat` or
`make`, with Lua 5.2 compatibility on). Float arrays reach a plugin as a view
that a host prelude in `lua-plugin.h` builds over an FFI pointer: element loops
compile to machine code (an md_wasserstein step on a 6000-frame YuE2 latent went
from about 135 ms to about 2 ms), while bounds and read-only checks stay and the
pointer itself is never reachable. `math.random` is a port of Lua 5.4's
xoshiro256** generator, so seeded plugins draw the same numbers as before. `PluginRegistry::init` (`lua-plugin-registry.h`) scans
`engine/plugins/{solvers,schedulers,guidance,postprocess}/` and then the same four
folders under the project root `plugins/`, once at startup. `ace-server` then
scans the root given by `--user-plugins`, which the app sets to
`<data folder>/plugins` whenever it runs with a data folder (a release always does). A user plugin replaces a shipped plugin of the same name
(the shipped Lua state is closed and the log names both files); any other duplicate
keeps the first one loaded. `ace-server` and `ace-synth` both call it. `lua-plugin.h` runs each file in its own sandboxed VM,
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
- MM3's flow stage (`mm3-plugins.h`), YuE2's NAR stage, `/sa3-refine` and `/sa3-synth`
  use the same registry.

New plugins need no C++ rebuild. Writing one is covered in
[plugins-authoring.md](plugins-authoring.md).

## Stable Audio 3 training

`ace-train sa3-train` trains Medium DoRA-rows adapters against the separate
Medium base checkpoint. SAME and T5Gemma pre-encode the prepared instrumental
snapshot. The base stays frozen, segmented backward graphs train the adapters,
and AdamW updates their F32 host masters. CUDA and CPU are supported by the
server runner; operation support is checked before graph execution.

Raw adapters and optimizer state stay in the run directory. Inference exports
contain row scales computed against the installed inference DiT and are
validated before entering `models/sa3-adapters`. `sa3-train-check` replays the
official-reference fixtures. See [training internals](training-internals.md#stable-audio-3-medium-training)
for the API, preparation and integrated checker, and the
[CLI reference](../../engine/docs/ARCHITECTURE.md#stable-audio-3-training-cli)
for standalone arguments.

## HeartMuLa training

`ace-train heartmula-prepare` and `ace-train heartmula-train` train a rank-64 LoRA
on the 28 backbone attention blocks of HeartMuLa 3B, with the local decoder, the
embeddings and the codec frozen. The spec, the loss definition, the tolerances and
the measured numbers are in
[HEARTMULA_TRAINING_CONTRACT.md](../../engine/docs/HEARTMULA_TRAINING_CONTRACT.md);
the code is `engine/src/train/heartmula-prepare.h` and
`engine/src/train/heartmula-train-{graph,run,state}.h`. The CLI contract, the
`HMTRAIN {json}` event stream and the exit codes (0 finished, 3 paused at a
checkpoint, 4 cancelled) are the header of `heartmula-train-run.h`.

`engine/tools/heartmula-train-reference.py` is the offline oracle. It is not part
of the app and nothing runs it at training time. It writes the hashed fixtures
(prompts, codes, logits, losses, gradients, AdamW steps) that `ace-train
heartmula-train --selftest <fixture dir>` checks against, and the Python step-time
and memory baseline the native trainer is compared with. Fixtures are large and live
outside the repo; the contract lists their paths and SHA-256 hashes.

Defaults the server passes: `--attn auto` takes the first of BF16 flash attention,
TF32 flash attention and exact attention that the device supports. `--keep-layers
auto` keeps the largest number of top blocks (28, 24, ... 0) that fits the free
VRAM without recomputing them, else recomputes every block. Losses are bit-identical
in every mode. `--lora-dropout` defaults to 0 (0.05 costs 8% step time and 1.3 GiB).
Contract section 8.1 has the per-setting table; on an RTX 5090 the default step on a
84.7-second clip, four micro-batches, takes 908 ms against the Python recipe's
1336 ms.

Two opt-in options exist for recipe experiments, and the server passes neither.
`--cb0-weighted` scales codebook k's CE by `w_k / 4.8`, with w = 2.0 for CB0 and
0.7 down to 0.1 for CB1-7. The step event's `loss` is then that weighted objective,
and `loss_mean` carries the plain mean; with equal CEs the two are the same.
`--eval base|<adapter.gguf>` (repeatable), with `--eval-samples id,id`, scores
instead of training. It runs the training forward pass only (no backward, optimizer
or dropout, nothing written) over whole songs. With `--eos` it adds the end frame
where training would, that is, only when it fits the context. It prints
`eval_sample` and `eval` events: mean CE per codebook, a frame-weighted mean per
model, `loss` and `loss_cb0w`. Repeated runs print identical numbers. The base
model, a run's `step-N.gguf` checkpoints and published adapters can all be scored
after the fact, for example on songs held out of training.

The converter `engine/tools/convert-heartmula.py --only encoder` builds the four
HeartCodec encoder files Prepare needs. Besides `HeartCodec-oss` and
`HeartCodec-oss-encoder` under `--src`, it also reads the root `tokenizer.json` and
`gen_config.json` (their hashes go into the manifest), so those two files must sit
next to the two HeartCodec directories. The resulting `heartcodec-rvq-F32.gguf` must
be byte-identical to the installed one.

The server side (runner, previews, routes) is in
[training internals](training-internals.md#heartmula-backbone-lora-training).

## Stable Audio 3 sampling

`POST /sa3-synth` generates from caption tokens using the same SA3 model and
adapter path as StableStep. The Node backend defaults to 8 steps and the
`md_pingpong_blend` Lua solver over the native `pingpong` wire sampler. Explicit
`pingpong` or `euler` selects a native loop; an opted-in solver plugin takes
precedence over the default Lua solver.

The optional `synth_model` query field picks the generation DiT by bare
filename, such as `sa3-dit-Q8_0.gguf`. A missing, empty, non-`sa3-dit-*.gguf` or
uninstalled name falls back to `sa3-dit-BF16.gguf`. The training base
`sa3-dit-base-*` (any case) and path-like names return 400. The `X-SA3-DiT`
response header names the DiT that ran. `/sa3-refine` always uses BF16. Full
contract: [ARCHITECTURE](../../engine/docs/ARCHITECTURE.md).

Checkpoint previews add `snapshot_run=<dataset>/<adapter>/<stamp>`,
`snapshot_step=N` and optional `snapshot_scale` (default 1, a finite number or
400). `sa3-preview-resolver.h` loads the run's `snapshots/step-N.gguf` as an extra
adapter stacked after any named ones. It is an internal path beside the public
adapter names: `models/sa3-adapters` parsing and the `/sa3/props` catalogue are
unchanged, and snapshots never appear there. The resolver checks the run and step
charsets (no separators beyond the two slashes), keeps the canonical file under
the training runs root, reads the step file once, hashes those bytes and loads an
immutable content-addressed copy from `sa3-medium/.preview-cache` (newest 16
kept), and requires a `dora-rows-v1` export whose inference base matches the DiT
the request loads. The content hash is part of the model key, so a rewritten step
file loads fresh. Failures are 400. `tools/sa3-preview-proof.mjs` checks the render
effect, the rejections and cache freshness.

Seed breeding is limited to noise-only synthesis. `seed_b` selects a second
noise stream and `seed_blend` mixes it into the initial latent and each pingpong
re-noise draw. Node resolves an omitted or blank blend to 0.5 when seed B is set;
explicit 0 disables breeding and 1 selects seed B's noise. The native loop
advances both streams together. The Lua solver uses `math.randomstream(seed_b)`
for an independent host-backed 64-bit stream, selected through the shared
`SolverState.seed_b` and `seed_blend` globals. Other
solver callers, ACE-Step included, leave those globals at -1 and 0; ACE-Step's own
`seed_b` breeding mixes only its initial noise (`pipeline-synth-ops.cpp`).
StableStep refinement does not enable breeding.

`engine/tests/test-sa3-seed-blend.cpp` exercises the real LuaJIT bridge without
models or GPU work. Its header gives the standalone compile command. It checks
blend 0/default/1, an analytic first-step noise mix, replay across reused plugin
states and seed-B equivalence, including odd latent lengths and seeds above
32 bits.

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

DoRA magnitudes (PEFT `lora_magnitude_vector`, LyCORIS `dora_scale`) are folded in by
merge mode. The runtime modes skip them with a warning unless the request sets
`adapter_runtime_dora`; the stack is then merged into private copies of the weights and
kept as merge-equivalent full deltas, in `runtime_lowrank` too.

Other adapter paths:

- The ACE-Step planner LM takes a runtime LoRA or DoRA (`lm_adapter`, from
  `<adapters>/lm/` or a path) through `lm-adapter.h`, with artist tokens, trained KV
  prefixes and shared PiSSA residuals in `artist-token-runtime.h`,
  `lm-prefix-runtime.h` and `pissa-residual.h`. A DoRA's column norms are computed
  once after load.
- MM3 LM adapters are runtime or merge (`mm3-lm-adapter.h`, `mm3-lm-merge.h`). A
  runtime DoRA gets a one-time norm pass (`mm3-lm-dora.h`), and `adapter_runtime_dora`
  re-normalises it for the request's scale dials. MM3 DiT LoRAs and DoRAs merge at
  load (`mm3-adapter.h`). A runtime MM3 LM adapter can change strength along the
  song (`lm_adapter_fade_*`, `MM3LmAdapterSchedule` in `mm3-lm-adapter.h`): the AR
  loop re-sets the scales between iterations, which rebuilds the decode graph and
  keeps the KV cache. The plan dial (`lm_adapter_plan` < 1) adds a base copy of every
  take's CFG pair (`MM3LmGraph::base_rows`), masks the adapter delta off on those rows
  with a resident `[1, 1, B]` tensor in the KV buffer, and mixes the two models' logits
  before the semantic draw (`mm3-ar-loop.h`).
- YuE2 adapters merge at load (`yue2-adapter.h`): AR (composer) and NAR files in
  LoRA, LoKr or DoRA form (`yue2-{ar,nar}-{lora,lokr,dora}-v1`), chosen with
  `POST /yue2/select-model`. On a ConvRot INT8 base the LoRA factors are applied at
  runtime instead (`yue2-convrot-adapter.h`).
- SA3 `dora-rows-v1` adapters from `models/sa3-adapters` merge at load
  (`sa3-dit-ggml.h`).

The `adapter-system` skill in `.claude/skills/` covers failure modes.

## TensorRT paths

Native TensorRT is the only non-GGML runtime. For ACE-Step the model path picks it; for
MM3 the request's `dit_backend` does.

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
    per DiT GGUF and adapter stack. With keep-loaded off the runtime is freed after
    every job, but the refitted plan's bytes (~4.5 GB) stay in host RAM
    (`g_mm3_trt_plan`), so the next job only deserializes (~0.5 s) instead of
    reading the file again. `POST /mm3/unload` (without `keep_ar_cache`), a model
    selection change and a switch to the GGML renderer drop them; `/mm3/props`
    reports the size as `dit_runtime.host_plan_mb`. The DiT GGUF is staged on the
    CPU only as a refit source: on a refit-cache miss `Runtime::refit` stages it
    for the refit and frees it after. The rest load stages the whole DiT only when
    `MM3_ADAPTER` is set, because the adapter merge patches that copy.
    FP8 (experimental, default off): request `"dit_precision": "fp8"` sets
    `MM3Model::dit_fp8` and builds a separate base engine with FP8 Q/DQ on the MLP
    GEMMs of blocks 2..33 (weight scales computed at refit, activation scales from
    the request's `dit_fp8_calib`, else `<models>/mm3/mm3-dit-trt.fp8.json`), with
    its own refit cache key. The server ships the calibration as
    `server/src/data/mm3-dit-trt.fp8.json` and sends its path; it is made by
    `tools/mm3-trt-export/fp8_lab.py calib`. A job whose precision or calibration
    differs from the loaded runtime drops it (`mm3-job.h`); the host plan slot keys
    on the refit-cache file, so switching costs one plan read (~0.5-2 s) and no
    refit. Dev env: `MM3_DIT_TRT_FP8=1` forces FP8 on for every TensorRT job;
    `MM3_DIT_TRT_DUMP=<dir>` writes every forward's input and velocity for the lab
    tool. Measurements: `mm3-backend` skill, `reference/performance.md`.

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

## Training roundtrip

`ace-train roundtrip --backend <backend> --audio <file> --models <dir> --out <x.wav>`
puts one track through the encode that backend's training prep runs, decodes the
result straight back and writes two WAVs. `x.wav` is the reconstruction: everything an
adapter can learn from that track. `x.source.wav` is what the encoder was fed, cut and
levelled the same way, so the two compare at matched level.

| Backend | Training stores | Decoded with | Encode follows |
|---|---|---|---|
| `ace` | VAE latents (`target_latents`) | the same VAE's decoder | `preprocess`, through the shared `pp_encode_audio`. `--vae` is required; `--max-duration 600`, `--normalize none`, `--target-db -1`, `--dtype f32`, `--vae-chunk 384`, `--vae-overlap 48` |
| `sa3` | SAME latents | the SAME decoder | the `sa3Stems.ts` conversion (`-t 384`, 44.1 kHz, 16-bit) and `sa3-train`'s clip prep |
| `heartmula` | HeartCodec codes, 8 codebooks | the codec's flow decoder, 10 steps, guidance 1.25, `--seed 42` | `heartmula-prepare` |
| `mm3` | DAV latents, the flow-DiT target | the vocoder | `mm3-preprocess`; `--enc` defaults to the newest `mm3-enc-*` |
| `yue2` | VAE latents, the NAR target | the VAE decoder | `yue2-preprocess`; `--vae standard`, `--loudness-lufs 0` (off) |

Pass the values the dataset was prepared with when they differ from these defaults.
Every backend but `ace` runs with TF32 off, as its prep does. SA3 trains on the
SuperSep instrumental stem, which this command does not separate: give it the prepared
`instrumental.wav` to hear exactly what training reads, or a full mix to hear the codec
on the vocals too.

Four of the five decode deterministically. HeartCodec's decoder starts from noise, so
its waveform does not line up with the source and another `--seed` gives another take;
judge it by ear, not by sample error. The token streams LM adapters train on (ACE 5 Hz
codes, MM3 RVQ codes, YuE2 semantic codes) need a generative model to become audio and
are not covered. The code is `engine/src/train/roundtrip-run.h`.

## Training subcommands

`ace-train` links no `acestep-core`. The trainers are headers under `engine/src/train/`;
the one compiled piece is `yue2-aitk-kernels` (`train/aitk-runtime.cmake`), the YuE2
joint trainer's runtime, which on a CUDA build adds the 8-bit AdamW, fused head-loss and
gradient-clip kernels, and the LM trainers' q8_0 to BF16 weight stage (`q8_bf16_cuda.cu`,
used by `--weights bf16` on a q8_0 base). It writes machine-readable JSONL on stdout with `--jsonl` and
human logs on stderr, and exits 0 on success, 1 on a runtime failure and 2 on bad
usage. `ace-train --help` prints the full option list.

| Subcommand | Purpose |
|---|---|
| `preprocess` | Build per-song tensor caches from an ACE-Step dataset |
| `train-lm` | Train an ACE-Step planner-LM adapter: LoRA or LoKr, with DoRA, HRA, HiRA, LoHa and PiSSA as options |
| `train-dit` | Train an ACE-Step DiT adapter: LoRA or LoKr, with DoRA, HRA, HiRA, LoHa and PiSSA as options |
| `detok-table` | Dump the FSQ detokenizer output for every code |
| `mm3-encode` | MM3 DAV encode: raw audio to flow latents |
| `mm3-codes` | MM3 dataset to RVQ codes (LM training input) |
| `mm3-launder` | Cover-launder a dataset through the MM3 model before encoding codes |
| `mm3-retarget` | Shorten tracks over MM3's 6:00 cap by removing one repeated section |
| `mm3-lm-train` | Train an MM3 LM adapter |
| `mm3-lm-loss`, `mm3-lm-probe` | MM3 LM trainer diagnostics and wiring gate |
| `mm3-preprocess` | MM3 dataset to flow-DiT target latents |
| `mm3-condition` | MM3 AR rollout to a flow-DiT conditioning cache |
| `mm3-train-dit` | Train an MM3 flow-DiT LoRA, or DoRA with `--dora`; CLI-only |
| `rec7-selftest` | Parity gate for the rec7 state encoder |
| `sa3-train` | Train a Stable Audio 3 Medium DoRA-rows adapter on CUDA or CPU (see [Stable Audio 3 training](#stable-audio-3-training)) |
| `sa3-train-check` | Replay the official SA3 trainer capture; exits 77 when the fixtures or the base DiT are absent |
| `yue2-preprocess` | Audio folder to cached YuE2 VAE latents and a manifest. Skips `*.engine.wav` (old server conversion-cache leftovers, duplicates of real tracks), as the dataset scanner does |
| `yue2-tokenize` | Fill a YuE2 manifest's codec ids. `--dump-mert` also writes `codes/<stem>.mert.f16`, the tokenizer head's exact instance-normed `[T25, 1024]` MERT input, for head-training experiments |
| `yue2-align` | Fill word timings for the AR lyric-cursor loss |
| `yue2-guide` | One audio file to a source guide (`--audio`, `--out`, `--topk 64`, `--lufs -14`): the real-audio tokenizer's top-K codes per 25 Hz frame, read by the YuE2 semantic stage through `guide_path` (`yue2/yue2-guide.h`) |
| `yue2-sheet` | Fill the ABC lead sheet with SheetSage2; `--melody-only` serves cover inference, while training keeps full score by default |
| `yue2-nar-train` | Train a YuE2 NAR LoRA. `--manifest` may repeat (all caches train as one set); `--companion` merges `nar_lora_joint_v9` under the NAR first, as the server does at render time, and the export records it in `companion` metadata. `--audio-loss <w>` adds an audio-domain term on low-noise steps, the clean-latent estimate decoded through the frozen VAE and scored against the source recording ([training internals](training-internals.md#yue2-nar-audio-domain-loss)) |
| `yue2-ar-train` | Train a YuE2 AR (composer) LoRA |
| `yue2-prepare-aitk`, `yue2-import-aitk-cache` | Build or import caches for joint training |
| `yue2-joint-train` | YuE2 AR and NAR joint training. `--eval-base-loss` trains nothing: it prints the base planner CE and decoder flow MSE per song (the Optimise phase). `--eval-adapted --resume <record> [--eval-dataset <held-out dataset.json>]` also trains nothing: it restores the checkpoint's adapters and prints base vs adapted planner CE and decoder MSE per song ([training internals](training-internals.md)). `--planner-gate-every N [--planner-gate-min-epochs 5 --planner-gate-patience 3 --planner-gate-epsilon 0.001]` freezes the planner on a held-out CE plateau (needs `--holdout-items` and `--nar-extra-steps`; [training internals](training-internals.md#held-out-planner-gate)). `--resume` restores optimizer state exactly; older 8-bit records load with a note ([training internals](training-internals.md)) |
| `yue2-optim-check` | Optimizer self-check |
| `yue2-batch-check` | CPU self-check of the AR sequence layouts the joint trainer builds, no model needed |
| `heartmula-prepare` | Sample list (audio, tags, lyrics) to HeartCodec codes and prompt ids, one sample per track, none cut ([HeartMuLa training](#heartmula-training)) |
| `heartmula-train` | Train a HeartMuLa backbone LoRA from a prepared cache; `--stop-after` and `--resume` drive checkpoint previews; `--eval` scores the base or adapters on held-out samples instead ([HeartMuLa training](#heartmula-training)) |
| `roundtrip` | One track through a backend's training encode and straight back to audio, to hear what an adapter can learn from it ([Training roundtrip](#training-roundtrip)) |
| `spike` | Phase-0 evidence runs |

`yue2-prepare-aitk --caption-dropout-trigger <text>` sets the exact style text
for caption-dropout prefixes, without an "in the style of" wrapper. The value
must be one line, at most 128 bytes; CR, LF and NUL are rejected. If omitted,
the dropout style is the bare `--trigger` value, or empty when neither is set.
Preparation tokenizes these prefixes; `yue2-joint-train --caption-dropout <p>`
selects them during training. HOT-Approved uses this preparation flag with the
trigger, falling back to the dataset slug. Older prepared datasets need
re-preparing; the server invalidates the old Fixed recipe preparation cache.

Every prepared track must fit the 24,576-token context in full, because the
planner trains on the whole song: the longer prompt prefix, plus the ABC sheet,
plus the frame count and three markers (or twice the 1500-frame decoder window
plus five, if that is larger). Preparation and the dataset reader apply the
same rule (`dataset_detail::context_tokens` in `yue2-aitk-dataset.h`).
Preparation leaves out a track that does not fit and prints
`[yue2-prepare] NOTE: <track> needs N tokens, over the 24576-token YuE2 context; skipped`.
The note goes on to break the total down (style and lyrics, score, audio length
and frames) and says to shorten the song or train it without its score.
A prompt token fault (an empty prefix, a token outside the vocabulary) fails the
dataset with `song <id>: <reason>`.
It fails only when no track fits. If the reader rejects the prepared manifest,
the error carries the reader's reason after `prepared manifest failed reader roundtrip:`.

The flash-attention training ops these trainers use come from the HOT-ggml fork
below. How the Training Studio drives these subcommands, and the design decisions
behind them, are in [training-internals.md](training-internals.md).

## The ggml fork

`engine/ggml` is a submodule pinned to the `hot-step-neutral` branch of HOT-ggml, a fork of
ggml-org/ggml. Every HOT-Step change to ggml is a commit on that branch, so a checkout and
build leave `engine/ggml` clean and nothing is applied at configure time.

| Change | What it changes |
|---|---|
| CPU fused snake | One CPU kernel for the mul → sin → sqr → mul → add snake chain |
| `alloc-free-blocks` | Raises ggml-alloc's per-chunk free-block table from 256 to 1024 entries |
| `bf16-out-prod` | CUDA `out_prod` accepts a BF16 `src0` |
| `bf16-mul-mat-grad` | A BF16 `MUL_MAT` weight operand may carry a gradient, which the YuE2 trainer's BF16 GEMMs need |
| `cudagraph-log` | Env-gated trace of CUDA graph decisions |
| `f16-f32-accumulate` | CUDA F16 GEMMs accumulate and write in F32 on every architecture |
| `flash-attn-train` | Adds `GGML_OP_FLASH_ATTN_TRAIN` and `_BACK`, with `ggml-cuda/fattn-train.{cu,cuh}` |
| CUDA BF16 flash-attn-train | BF16 Q/K/V kernels for the fused training attention on sm_80+, for an explicit `GGML_PREC_BF16` request |
| `metal-acc-set-cpy` | Metal non-inplace `ACC`/`SET` copy every column |
| `metal-bin-threads` | Raises the Metal binary-op threadgroup cap |
| `metal-im2col-ic` | Adds a separate Metal im2col kernel for the VAE's 1D convolutions |
| Metal training ops | Metal kernels for the ops YuE2 joint training runs (fused training attention, `CONVROT8`, `BF16_ROUND`, `RMS_NORM_BACK`, `REPEAT_BACK`, `OUT_PROD`, `SILU_BACK`), CPU `CONVROT8` kernels, and optional hints on the fused attention op |
| `metal-kquant-cpy` | Metal copies K-quants to F32 and F16, so a YuE2 LoRA trains on a K-quant base |
| `mm-backward` | Env-gated dtype-agnostic `MUL_MAT` backward (`GGML_BACKWARD_MM=1`) |
| `quant-cpy-kquant` | CUDA `CPY` reaches ggml's generic quant-to-F32 converter, so K-quants can be cast |
| `sched-unplaced-log` | Logs the node no backend accepts before the scheduler's `cur_backend_id != -1` assert |
| `zz-yue2-convrot8` | Adds the `GGML_OP_CONVROT8` and `_BACK` ops for YuE2 ConvRot training, with `ggml-cuda/convrot8.{cu,cuh}`; the Metal training ops add CPU and Metal kernels |
| `zzz-yue2-bf16-round` | Adds the `GGML_UNARY_OP_BF16_ROUND` unary op |
| `zzzz-vulkan-train-ops` | Vulkan `BF16_ROUND` and `FLASH_ATTN_TRAIN`/`_BACK`, with `vulkan-shaders/fa_train_*`; what the YuE2 joint trainer needs on Vulkan |
| BF16 training speed-ups | CUDA (`cuda-bf16-round-fuse`, `cuda-bf16-round-defer`, `cuda-bf16-storage`, `cuda-gemm-bf16-store`, `cuda-bf16-producer-store`, `bf16-round-elide`, `bf16-round-inplace`, `fa-train-cuda-hints`) and Metal (`metal-out-prod-smem`, `metal-cpy-transpose`, `metal-bf16-round-producer`, `metal-mul-mm-half-prepass`) work for the YuE2 trainer. Most are bit-identical, and most have an environment switch that turns them off |
| RPC patch-level check | The RPC client refuses a server without the fork's ops |

The quant-to-F32 copy launch fix that used to be `cpy-q-occupancy` is upstream now.
[ggml-fork.md](ggml-fork.md) explains each change in depth, with the measurements behind it,
and how to change ggml.

Two rules that bite:

- Never edit `engine/ggml` in place. A change there is not what CI or users build, and
  `verify-hooks.ps1` (Hook 17) refuses a dirty or off-pin submodule. Commit it to the fork
  and move the gitlink.
- A checkout from before the fork caches the old ggml-org URL in `.git/config`, so a plain
  `git submodule update` cannot find the pinned commit. Run
  `git submodule sync -- engine/ggml` first.

## HOT-Step endpoints

Routes `ace-server` adds on top of upstream's `/lm`, `/synth`, `/understand`, `/vae`
(VAE encode or decode), `/job`, `/health`, `/props`, `/logs` and `/`. Request and
response details are in [the endpoint reference](../../engine/docs/ARCHITECTURE.md#endpoints).

| Route | Purpose |
|---|---|
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
| `POST /sa3-synth` | Stable Audio 3 generation from noise (see [Stable Audio 3 sampling](#stable-audio-3-sampling)) |
| `GET /sa3/props` | SA3 readiness, sample rate, prompt token limit and adapter catalogue |
| `GET /worker-drain` | Answers once the worker queue ahead of it has finished |
| `POST /supersep/separate`, `GET /supersep/progress`, `GET /supersep/result`, `GET /supersep/serve`, `POST /supersep/release`, `POST /supersep/recombine` | Stem separation jobs |
| `POST /spectral-lifter` | Spectral Lifter clean-up (CPU) |
| `GET /mm3/props`, `POST /mm3/warm`, `POST /mm3/unload`, `POST /mm3/select-model` | MM3 status, residency and quant selection |
| `POST /mm3/synth`, `GET /mm3/job`, `GET /mm3/take`, `GET /mm3/stream` | MM3 generation, progress, ensemble takes, live audio |
| `POST /mm3/refine` | MM3 latent refiner on any audio (progress and result through `/job`; see [MM3 latent refiner](#mm3-latent-refiner-audio-in-audio-out)) |
| `POST /mm3/tokenize-check`, `POST /mm3/imatrix` | MM3 prompt token count, imatrix collection for `quantize` |
| `POST /mm3/voc-decode`, `/mm3/dit-forward`, `/mm3/flow-sample`, `/mm3/depth-frame`, `/mm3/cond-encode`, `/mm3/lm-plan`, `/mm3/synth-e2e` | MM3 bring-up and parity endpoints. They run GPU work outside the job queue; do not build features on them |
| `GET /yue2/props`, `POST /yue2/warm`, `POST /yue2/unload`, `POST /yue2/select-model` | YuE2 status, residency, quant and adapter selection |
| `POST /yue2/synth` | YuE2 generation (progress and result through `/job`) |
| `POST /yue2/tokenize-check`, `POST /yue2/imatrix`, `POST /yue2/align` | YuE2 prompt check, imatrix collection, lyric forced alignment |

## Headless smoke test

`node tools/engine-smoke.mjs` renders two fixed-seed instrumentals with
`ace-synth --check`, one 10 s and one 7.33 s (patch padding and ragged VAE tiles), and
exits 1 unless both pass, each WAV is fresh and exactly frames x 1920 samples long, and
the engine log yields per-stage times. No app, server or UI is
involved. It resolves the models folder through `server/scripts/data-paths.ts`
(override with `--models <dir>`), uses `acestep-v15-turbo-BF16.gguf` (override with
`--dit <file>`), and writes requests, WAVs and logs to a per-run folder under
`logs/engine-smoke/`, pruning run folders older than a day.
`--self-test` renders six tracks, corrupts five inside ace-synth (NaN, subnormal,
silence, DC, clip), forces a write failure, and passes only if every fault is caught.
The block-based monitor chain has its own test, `engine/tests/test-monitor-chain.cpp`
(odd block sizes, decaying tail, subnormal count); how to run it is in the
engine-dev-loop skill. The checks and thresholds are in
[engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md#ace-synth-reference).
Engine hand-offs carry its `[Check Batch0]`, `stages:` and `wav:` lines and the
`ENGINE SMOKE:` verdict as the render receipt.

## Related

- [engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md): request JSON, generation modes, CLI flags, endpoints
- [architecture.md](architecture.md): the three tiers and how requests flow between them
- [building.md](building.md): build scripts, toolchains, optional SDKs
- [plugins-authoring.md](plugins-authoring.md): writing Lua plugins
- [training-internals.md](training-internals.md): the training system

## Release data paths

Release engine executables and shipped plugins remain app-owned. The server passes the data folder plugin overlay through `--user-plugins` and supplies absolute model and tool paths. Downloaded runtime libraries live under `<data folder>/tools/runtime`. See [Data folder and upgrades](../user/data-folder.md).

## HeartCodec CPU parity probe

`heartcodec-probe` (`engine/tools/heartcodec-probe.cpp`) checks the HeartCodec
RVQ, flow transformer and Euler solver, scalar decoder, and assembled 48 kHz
stereo waveform against fixed-input NPY reference fixtures. It loads only the
GGML CPU backend. F16 model weights execute in F32 to match the reference
baseline; the shared RVQ remains F32. This probe does not add an app backend.

The decoder uses 372-code chunks, a 320-code hop, 104 latent frames of context
and a stereo crossfade. Short inputs repeat before decoding and trim to exactly
3840 samples per input code. Supplied noise makes cross-runtime parity possible
without requiring identical random generators.

Run `heartcodec-probe --self-test` for geometry, overlap, CFG, cancellation and
input guards. For numerical checks, use `--mode rvq`, `velocity`, `latent`,
`scalar` or `decode` with `--models <component-directory>` and
`--fixtures <case-directory>`. `--trace <output-directory>` saves and compares
each intermediate during decode. See the CLI reference in
[ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md).
