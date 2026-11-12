# HeartMuLa engine contract

The `/heartmula/*` routes in `hot-step-server`, the generate request,
progress and cancel behaviour, and the GGUF metadata every HeartMuLa component
carries. The server backend and UI build against this page while the engine is
ported; the engine slices implement it. Where an implementation has to differ,
change this page in the same commit.

Status: implemented by `engine/src/heartmula/pipeline.h` (catalogue, checks,
admission, generation) and `engine/src/heartmula/server.h` (routes), except
`POST /heartmula/encode` and `POST /heartmula/imatrix`, which answer 501 until
their slices land. `engine/tools/heartmula-generate.cpp` runs the same pipeline
from the command line; `tools/engine-smoke.mjs` uses it for its HeartMuLa case.

## Model

HeartMuLa is autoregressive. A 3B global transformer reads the prompt (tags,
one zero MuQ slot, lyrics) and predicts codebook 0 of each 80 ms audio frame; a
3-layer local decoder predicts codebooks 1 to 7 for that frame. HeartCodec turns
the 8 x 12.5 Hz codes into 48 kHz stereo: shared RVQ lookup, a flow-matching
transformer (Euler, 29.76 s chunks with overlap), then a scalar waveform
decoder. Generation stops at audio EOS (any codebook id 8193 or above) or at the
frame limit.

Components, each its own GGUF so each loads at its own quant:

| Role | Architecture (`general.architecture`) | Needed for text generation |
|---|---|---|
| `global` | `heartmula-global` | yes |
| `local` | `heartmula-local` | yes |
| `tokenizer` | `heartmula-tokenizer` | yes |
| `rvq` | `heartcodec-rvq` | yes |
| `flow` | `heartcodec-flow` | yes |
| `scalar-decoder` | `heartcodec-scalar-decoder` | yes |
| `scalar-encoder` | `heartcodec-scalar-encoder` | no, kept for later |
| `muencoder`, `wavlm`, `whisper`, `query` | `heartcodec-<role>` | no, `/heartmula/encode` only |

## Files and discovery

Files live in `<models>/heartmula/`, then `<models>/` (same search order as YuE2).
Names follow the converter:

```
heartmula-<variant>-<role>-<QUANT>.gguf   global, local       e.g. heartmula-3b-global-Q4_K_M.gguf
heartcodec-<role>-<QUANT>.gguf            codec and encoder   e.g. heartcodec-flow-Q8_0.gguf
heartmula-tokenizer.gguf
```

`<variant>` is the lower-cased `heartmula.variant` (`3b`, `3b-happy-new-year`).
`<QUANT>` is the text after the last `-` and is what the pickers show. Identity
always comes from the metadata below, never from the name: a renamed file is
still checked, and a file whose name and metadata disagree is listed with an
error and is not selectable.

Discovery runs at start-up and on `GET /heartmula/models?rescan=1`. It reads
headers only.

**Encoder pack (training Prepare).** The four files `heartcodec-muencoder-F16.gguf`,
`heartcodec-wavlm-F16.gguf`, `heartcodec-whisper-F16.gguf` and `heartcodec-query-F16.gguf`
are not needed to generate. They reach a machine the way the generation packs do, by running
the converter on the upstream weights:
`python engine/tools/convert-heartmula.py --src <checkpoint-root> --out <dir> --only encoder`
(needs `HeartCodec-oss` and `HeartCodec-oss-encoder` under `--src`, with `tokenizer.json` and
`gen_config.json` in the `--src` root, plus numpy, torch,
safetensors and `gguf==0.17.1`). Copy the four encoder files into `<models>/heartmula/`.
The run also writes `heartcodec-rvq-F32.gguf`; it is byte-identical to the installed one
(checked by SHA-256), and the encoder files pin it by hash. The training status route
reports a missing pack, with these names, before Prepare is offered.

## Routes

All bodies are JSON. Errors are `{"error": "<message>", "code": "<code>"}`
with the status below. Field names are snake_case.

| Route | Purpose |
|---|---|
| `GET /heartmula/props` | Backend state: availability, residency, current pick, VRAM |
| `GET /heartmula/models` | Component catalogue with compatibility per file |
| `POST /heartmula/select-model` | Set the variant, per-role quants and the adapter |
| `POST /heartmula/warm` | Load the current pick for the LM phase. Idempotent |
| `POST /heartmula/unload` | Free every HeartMuLa buffer. Idempotent |
| `POST /heartmula/synth` | Queue a generation; returns the shared job id |
| `POST /heartmula/cover` | Queue a MuLaCover cover from lead-sheet rolls; returns the shared job id |
| `GET /heartmula/job?id=` | HeartMuLa progress detail and failure reasons |
| `GET /job?id=`, `GET /job?id=&result=1` | Shared coarse status and the WAV result |
| `POST /job?id=&cancel=1` | Shared cancel |
| `POST /heartmula/encode` | Audio to codes (encoder pack). 501 until the encoder lands |
| `POST /heartmula/imatrix` | Arm, disarm and save imatrix collection, as `/yue2/imatrix`. 501 until the quant slice |

There is no HeartMuLa-only cancel route: cancel is the shared
`POST /job?id=&cancel=1`, the same as ACE, MM3 and YuE2, so the Node queue's
existing cancel path works unchanged.

### `GET /heartmula/props`

```json
{
  "backend": "heartmula",
  "available": true,
  "synth_ready": true,
  "max_duration_s": 240,
  "context_length": 8192,
  "cover_ready": true,
  "cover_error": "",
  "cover_max_duration_s": 300,
  "cover_max_steps": 5000,
  "defaults": { "temperature": 1.0, "topk": 50, "cfg_scale": 1.5,
                "codec_steps": 10, "codec_guidance": 1.25, "duration_s": 120 },
  "selected": { "variant": "3B", "global": "Q8_0", "local": "Q8_0",
                "flow": "Q8_0", "scalar_decoder": "F16" },
  "requested": { "variant": "3B", "global": "Q8_0", "local": "Q8_0",
                 "flow": "Q8_0", "scalar_decoder": "F16" },
  "adapter": { "requested": "", "loaded": "", "scale": 1.0, "in_force": false },
  "resident": { "global": false, "local": false, "flow": false,
                "scalar_decoder": false, "rvq": false, "tokenizer": true },
  "encode_ready": false,
  "vram": { "resident_bytes": 0, "free_bytes": 6979321856, "reserve_bytes": 1610612736 },
  "errors": []
}
```

- `synth_ready`: the tokenizer, RVQ and one compatible file for each of
  `global`, `local`, `flow` and `scalar-decoder` are present and pass the
  checks below. The Node backend reports the backend up from this key alone, as
  with YuE2 and MM3. It is not a VRAM claim.
- `selected` is what is in force after fallbacks; `requested` is what
  select-model was last told. They differ when a requested file is missing.
- `defaults` are heartlib's own defaults; the UI resets to them.
- `errors` lists discovery and compatibility problems as plain sentences.
- `cover_ready`: the MuLaCover global and local files and
  `Qwen3-Embedding-0.6B-BF16.gguf` (in the models root) are installed and resolve with
  the current codec pick. `cover_error` says what is missing, in the words a cover
  job fails with. `cover_max_duration_s` and `cover_max_steps` are the limits of
  `POST /heartmula/cover`.

### `GET /heartmula/models`

```json
{
  "variants": ["3B", "3B-happy-new-year"],
  "components": {
    "global": [ { "type": "Q4_K_M", "filename": "heartmula-3b-global-Q4_K_M.gguf",
                  "bytes": 2050000000, "variant": "3B", "file_type": 15,
                  "base_fingerprint": "<hex>", "compatible": true, "error": "" } ],
    "local": [], "flow": [], "scalar_decoder": [], "rvq": [], "tokenizer": [],
    "scalar_encoder": [], "muencoder": [], "wavlm": [], "whisper": [], "query": []
  }
}
```

Every file found is listed, compatible or not; `error` says why one is not.
`compatible` for `global` and `local` is judged against the tokenizer and RVQ
present; the pairing between global and local is judged at select time. Role
keys use `_` where the role name has `-`. Adapters are not listed here: the Node
server scans adapter folders, and select-model validates the file it is given.

### `POST /heartmula/select-model`

Every field is optional. An absent field leaves that pick alone; `""` or `null`
clears it back to the default (same rule as `/yue2/select-model`).

| Field | Type | Meaning |
|---|---|---|
| `variant` | string | `3B` or `3B-happy-new-year`. Never substituted: if the variant has no global and local file, the request is a 404. `MuLaCover` is refused with 409 `incompatible`: it renders covers only, through `POST /heartmula/cover`, and `/heartmula/models` leaves it out of `variants` (its files stay under `components`) |
| `global_type`, `local_type` | string | Quant token for each LM part, chosen independently |
| `flow_type`, `scalar_decoder_type` | string | Quant token for each codec part, chosen independently |
| `adapter` | string or null | Absolute path to a HeartMuLa LoRA GGUF. `""`/`null` removes it. One adapter in v1 |
| `adapter_scale` | number | 0 to 2, default 1.0. 0 must give exactly the base logits |

Default pick, when nothing was requested: variant `3B`; each role uses `Q8_0` if
present, else `F16`, else the most precise type found. This keeps the default
inside the VRAM left when another application holds most of the GPU; F16 LM is an
explicit choice.

The handler validates the whole resulting pick before storing any of it, so a
rejected request changes nothing. A pick that changes a resident component
unloads that component; the next warm or synth loads the new one. Response: the
`selected`, `requested` and `adapter` blocks of `props`.

Errors: 400 malformed value (`bad_request`), 404 no such variant or type
(`not_found`), 409 `incompatible` for files that exist but fail a
[load-time check](#load-time-checks), for example an LM pair whose
`rvq_sha256` differs from the RVQ file (the message names the two files and
the check), 503 while a job is
loading or running HeartMuLa weights (`busy`). The busy case matches YuE2:
callers retry, or put the pick on the synth request instead.

### `POST /heartmula/warm` and `POST /heartmula/unload`

Warm loads tokenizer, global, local and the adapter for the current pick, after
the same VRAM admission as a job (below); on refusal it returns 507 with the
admission object. Response `{"loaded": true, "already_loaded": bool,
"resident_bytes": N, "load_ms": N}`.

Unload waits for the GPU worker to be idle, frees every HeartMuLa buffer and
returns `{"unloaded": true}`. The tokenizer stays loaded; it is a few MB of host
memory.

### `POST /heartmula/synth`

Validates the request synchronously, then queues it on the single shared GPU
worker and returns `{"id": "<job id>"}`. Progress and results use the shared
job system plus `GET /heartmula/job`.

| Field | Default | Validation | Meaning |
|---|---|---|---|
| `tags` | required | non-empty string, at most 4096 bytes | Style tags. Lower-cased and wrapped in `<tag>...</tag>` unless already wrapped, as heartlib does |
| `lyrics` | `""` | string, at most 32768 bytes | Lower-cased, as heartlib does. Empty lyrics encode as BOS, EOS (heartlib itself fails on empty lyrics, so this case has no reference) |
| `duration_s` | 120 | number, 1 to 240 | Upper bound. Frames = floor(duration_s x 12.5). The song ends earlier if the model emits audio EOS |
| `seed` | random | non-negative integer, or a decimal string for values past 2^53 | Sampler seed for the LM |
| `noise_seed` | `seed` | as `seed` | Seed for the codec's flow noise |
| `temperature` | 1.0 | greater than 0, at most 4 | LM sampling temperature, all 8 codebooks |
| `topk` | 50 | integer, 1 to 8197 | Top-k before sampling, all 8 codebooks |
| `cfg_scale` | 1.5 | 1 to 10 | LM classifier-free guidance. 1.0 runs a single unguided branch (half the KV) |
| `codec_steps` | 10 | integer, 1 to 100 | Flow Euler steps per chunk |
| `codec_guidance` | 1.25 | 1 to 10 | Flow guidance scale |
| `variant`, `global_type`, `local_type`, `flow_type`, `scalar_decoder_type`, `adapter`, `adapter_scale` | absent | as select-model | The job's own pick, applied when the job starts. Absent keeps the current pick |
| `adapter_scale_attn`, `adapter_scale_mlp`, `adapter_scale_early`, `adapter_scale_mid`, `adapter_scale_late` | 1.0 | 0 to 2 | Per-group adapter dials, synth only. A site's scale is `adapter_scale` x (attn or mlp) x its depth third: layers [0, n/3), [n/3, 2n/3) and the rest (0-8, 9-17, 18-27 on 28 layers). A change rebuilds the LM graphs and drops the KV rows, nothing reloads |
| `adapter_rebase` | 0.75 | 0 to 1 | Basin re-base beta, used only by an adapter trained on another variant (below) |
| `infer_method`, `scheduler`, `guidance_mode` | `""` | plugin names, or null | Lua sampler plugins on the HeartCodec flow decode (`heartmula/codec-plugins.h`), ACE's field names. All empty runs the native Euler loop unchanged. Unknown names and full-loop solvers warn and fall back to native. Guidance plugins run only when `codec_guidance` is above 1 |
| `flow_shift` | 1.0 | 0.01 to 20 | Shift handed to a scheduler plugin |
| `apg_norm_threshold` | 2.5 | 0 to 100 | APG norm threshold for `apg` and plugins that call it |
| `plugin_params` | `{}` | object of `"<plugin>:<key>": value` | Declared plugin params, coerced to strings as on MM3 |
| `codes` | none | array of frames, each 8 integers 0 to 8191, 1 to 3000 frames | Skip the LM and decode these codes. Validation and round-trip use |
| `codes_only` | false | bool, not with `codes` | Stop after the LM; result is the codes as JSON, no codec load |
| `id` | `"heartmula"` | string | Request label for logs |

The ranges for `temperature`, `topk`, `cfg_scale`, `codec_steps` and
`codec_guidance` are engine bounds; heartlib accepts any value and exposes no
other sampler or codec knobs (top-p, repetition penalty and the flow noise
temperature are not part of the reference path, so they are left out).

Synchronous checks, each a 400 `bad_request` naming the field: missing or
wrongly typed field, value outside its range, unknown field (so a typo is not
silently ignored), `codes` with `codes_only`. Then the context check, which needs
the tokenizer: `len(tags ids) + 1 + len(lyrics ids) + frames + 1` must be at most
`context_length` (8192), else 400 `context_overflow` with `prompt_tokens`,
`frames` and `limit` in the body. At the 240 s cap there are 3000 frames, which
leaves 5191 prompt tokens. A pick given on the request is checked as
select-model would check it (404/409 as there). 503 `unavailable` when
`synth_ready` is false.

Result (`GET /job?id=&result=1`): `audio/wav`, 48 kHz stereo, 16-bit, the
same encoder YuE2 uses. With `codes_only`: `application/json`
`{"frames": N, "codebooks": 8, "frame_rate": 12.5, "codes": [[c0..c7], ...]}`.

Completed status (`GET /job?id=`) carries `end_reason`: `completed` (audio EOS),
`limit_hit` (reached the frame limit) or `cancelled`.

### `POST /heartmula/cover`

A MuLaCover cover (`heartmula/mulacover.h`) through the synth job path: same
queue, `GET /heartmula/job`, result and cancel. The body is the synth body plus
`rolls`, with these differences:

| Field | Default | Validation | Meaning |
|---|---|---|---|
| `rolls` | required | object, below | The lead sheet on its sixteenth-note grid |
| `tags` | required | as synth | The style line, e.g. `topic:[..]; genre:[..]; instrument:[..]; mood:[..]`. Passed to the prompt and to Qwen3-Embedding as text, as upstream does; an empty field such as `topic:[]` is only text, and all-empty tags give the zero style vector |
| `duration_s` | 300 | number, 1 to 300 | Upper bound, upstream's `max_audio_length_ms` default. The model ends the song with audio EOS, which need not fall at the end of the lead sheet |
| `topk` | 250 | as synth | Upstream's cover default; `temperature` 1.0 and `cfg_scale` 1.5 as synth |
| `variant` | `MuLaCover` | `MuLaCover` or absent | The pick is MuLaCover's own global and local (default quants unless `global_type` or `local_type` is given) with the current codec pick |
| `adapter` | none | absent or empty | HeartMuLa adapters do not apply to covers (400) |
| `codes`, `codes_only` | | refused | Unknown fields here |

`rolls`: `version` 1, `length` 1 to 5000 rows, optional `bpm` (positive, not used:
key positions are the row index) and `dtype` (`float32`), and four flat row-major
arrays of 0 or 1: `pianoroll` (length x 256, pitch x 2 onset, pitch x 2 + 1
sustain), `drum_pianoroll` (length x 128), `chord` (length x 12 chroma) and
`context_mask` (length, 1 = padding row). A wrong size or value, or a sheet that is
all padding, is a 400 naming the field.

Missing models: 503 `model_missing` at submit (and as the job error if files go
between submit and start) when the MuLaCover pair or the BF16 Qwen3 embedding is
not installed; the message names what to install. The Q8_0 embedding is not used:
it misses upstream's FP32 style vector by 6e-3, against 1e-3 for BF16.

The job first runs the tags through the BF16 Qwen3 embedding (stage `style`,
after its own admission for the file size, freed straight after), then the LM
phase. LM admission adds MuLaCover's context cache to the KV:
`8 x 2 x 2 x (384 x 8 + 12) x steps x cfg rows` bytes, 941 MiB at 5000 steps with
CFG. The context check is the synth one on the cover prompt (tags, style slot,
MuQ slot, lyrics); at 300 s there are 3750 frames.

### `POST /heartmula/encode`

Reserved for the encoder slice; 501 `not_implemented` until then. Shape:
`multipart/form-data` with `audio` (WAV or MP3, any rate, mono or stereo) and an
optional `request` JSON part with the encoder pack types (`muencoder_type`,
`wavlm_type`, `whisper_type`, `query_type`). It queues a job; the result is the
`codes_only` JSON above. The encoder pack is never loaded for `/synth`.

## Progress

The shared `GET /job?id=` keeps working by mapping HeartMuLa phases onto the
existing `JobPhase` names by role, as MM3 does, so no member is added to the
enum in `hot-step-server.cpp`:

| HeartMuLa phase (`stage`) | Shared `phase` | `phase_step` / `phase_total` |
|---|---|---|
| `queued` | `queued` | 0 / 0 |
| `style` (covers) | `loading_text_enc` | 0 / 1, then 1 / 1 |
| `admission` | `loading_dit` | 0 / 0 |
| `loading_lm` | `loading_dit` | components loaded / to load |
| `loading_adapter` | `loading_adapter` | 0 / 0 |
| `prefill` | `encoding_cond` | prompt positions done / prompt positions |
| `frames` | `dit_inference` | frames done / frame limit |
| `loading_codec` | `loading_vae` | components loaded / to load |
| `codec_decode` | `vae_decode` | flow steps done / (chunks x `codec_steps`), then scalar chunks |
| `encoding_output` | `encoding_output` | 0 / 0 |

`frames` counts against the limit, so a song that ends at EOS jumps to the next
phase before 100%. That is expected; the UI should not treat it as an error.

`GET /heartmula/job?id=` gives the detail:

```json
{
  "id": "<hex>", "status": "running", "phase": "dit_inference", "stage": "frames",
  "prompt_tokens": 412, "prefill_done": 412,
  "frames_done": 731, "max_frames": 3000,
  "codec_chunk": 0, "codec_chunks": 0, "codec_step": 0, "codec_steps": 10,
  "seed": "1234", "noise_seed": "1234",
  "pick": { "variant": "3B", "global": "Q8_0", "local": "Q8_0",
            "flow": "Q8_0", "scalar_decoder": "F16", "adapter": "", "adapter_scale": 1.0 },
  "admission": null,
  "timings_ms": { "admission": 2, "load": 4100, "prefill": 380, "frames": 51000, "codec": 0, "encode": 0 },
  "end_reason": "",
  "error": null
}
```

`seed` and `noise_seed` are decimal strings so 64-bit values survive JSON. On
failure `error` is `{"code": "...", "message": "..."}`.

## Cancel

Cancel is a flag the worker checks at the points below; the job then ends with
status `cancelled` and no result. A cancel that arrives before the job is
marked done always wins, including during output encoding.

| Phase | Checked | Worst-case latency | State after |
|---|---|---|---|
| queued | when the worker picks the job up | 0 | nothing loaded |
| admission, loading | between components | one component load | no partly loaded component; fully loaded ones follow the residency rule below |
| prefill | between prefill chunks (at most 512 positions) | one chunk | LM KV freed |
| frames | after every frame (one global step plus the local decoder) | one frame, tens of ms | LM KV freed; no partial audio in v1 |
| codec decode | between flow steps and between scalar-decoder chunks | one step | codec scratch freed |

Residency after a cancel or failure is the same as after a successful job:
without `--keep-loaded`, every HeartMuLa buffer is freed. A cancel is often a
user reclaiming VRAM, so nothing is kept beyond the normal rule.

## Residency and VRAM admission

The job runs in two phases on the shared worker, so HeartMuLa serialises with
ACE, MM3, YuE2 and SA3. The LM phase holds tokenizer, global, local, adapter and
the KV cache. The codec phase holds RVQ, flow and scalar decoder. By default the
LM parts are freed before the codec loads. Under `--keep-loaded` they stay only
if admission says both phases fit together.

Before each phase loads anything, admission computes:

- `need`: tensor bytes of the files about to load, plus the KV for this request
  (global: 28 layers x 8 KV heads x 128 x 2 (K, V) x 2 bytes x positions, times 2
  with CFG; local: 8 positions), plus `reserve_bytes`. Positions = prompt tokens
  + frame limit + 1.
- `free`: free device memory now, plus what the engine can reclaim from idle
  resident weights of other backends (the ACE store, idle MM3/YuE2/SA3 weights).
  Nothing with work in flight is evicted; the YuE2 render lane is drained first.

`reserve_bytes` covers scratch and activations. The default, 1536 MiB, is
reported in `props`; `HEARTMULA_VRAM_RESERVE_MB` overrides it. Measured on CUDA
with F16 files: the largest graph buffers are 208 MiB for a 512-position
prefill chunk, 194 MiB for a flow step and about 1.1 GiB for a scalar-decoder
time chunk (186 latent frames plus 67 of context; the whole 744-frame chunk
needed 3.1 GiB). Both chunk sizes are fixed, so the reserve does not grow with
the song. A 240 s request with F16 files peaked at 9.0 GB above the idle
baseline in the LM phase (weights 7.5 GiB, KV 0.71 GiB) and 5.0 GB in the codec
phase, CUDA context included. On a GPU the codec keeps linear matrices in their
stored type (F16 or a quant); norms, tables and convolution kernels are widened
to F32.

`free` and `free_bytes` are measured on the device HeartMuLa runs on. An NVIDIA
GPU is matched to NVML by its PCI bus id and gets NVML's figure (what nvidia-smi
reports). On Windows, `cudaMemGetInfo` counts other processes' idle VRAM as
free because WDDM can page it out to system memory, which is the spill admission
exists to prevent. The CPU backend, other GPUs and machines without NVML use the
ggml figure. `free_bytes` is 0 until a HeartMuLa job or warm has initialised
the device.

If `need > free`, nothing is evicted and nothing is loaded, and the job fails
with status `failed`:

```json
{ "code": "vram_insufficient",
  "message": "HeartMuLa needs 6.9 GB for this request; 4.1 GB is free and 0.3 GB can be reclaimed",
  "admission": { "phase": "lm", "need_bytes": 7400000000, "free_bytes": 4400000000,
                 "reclaimable_bytes": 320000000, "reserve_bytes": 1610612736,
                 "weights_bytes": 4190000000, "kv_bytes": 590000000,
                 "positions": 3413, "cfg": true } }
```

The same object appears as `admission` in `GET /heartmula/job`, and warm
returns it with status 507. The engine never falls back to a smaller quant, a
shorter song or CPU offload on its own; the server shows the numbers and the
user picks a smaller quant, a shorter duration or `cfg_scale` 1.0. When
admission passes and an allocation still fails, the job fails with
`vram_alloc_failed` and the same numbers, so the reserve can be corrected.

## GGUF metadata

Written by the converter, the tokenizer converter and the adapter converter;
copied unchanged by `quantize`. All keys are strings unless marked u32.

### Every component

| Key | Value |
|---|---|
| `general.architecture` | as in the component table |
| `general.file_type` | the usual GGUF file type; the quant ladder sets it |
| `heartmula.format_version` | u32, 1. The engine rejects versions it does not know |
| `heartmula.component` | role name: `global`, `local`, `flow`, `rvq`, `scalar-decoder`, `scalar-encoder`, `muencoder`, `wavlm`, `whisper`, `query`, `tokenizer`, `lora` |
| `heartmula.variant` | `3B` or `3B-happy-new-year` for global, local and lora; `shared` for everything else |
| `heartmula.base_fingerprint` | sha256 hex of the source checkpoint, defined below. Quant-independent |
| `heartmula.source` | JSON: each source file's bytes, sha256 and HF revision |
| `heartmula.config_json` | the source `config.json` text. The engine reads hyperparameters from it |
| `heartmula.config_sha256` | sha256 of that text |
| `heartmula.dependencies` | JSON `{"tokenizer": {"tokenizer.json": "<sha256>", "gen_config.json": "<sha256>"}, "rvq_sha256": "<sha256 or null>"}` |
| `heartmula.tensor_layout`, `heartmula.tensor_manifest` | converter records; informational, not checked at load |

`heartmula.base_fingerprint` = sha256 of the UTF-8 bytes of
`json.dumps({name: sha256 for each *.safetensors file in the source checkpoint},
sort_keys=True, separators=(",", ":"))`. Global and local from one checkpoint
share it; the two LM variants differ; codec components share the HeartCodec
checkpoint's value and encoder-pack components share the encoder checkpoint's.
It does not change when a file is re-quantised, so an adapter trained on the
base works over every quant of that base and over no other.

`rvq_sha256` is the sha256 of the whole `heartcodec-rvq` GGUF file the
component was converted against; null in the RVQ file itself. The RVQ file is
never quantised, so its hash is stable.

### Tokenizer (`heartmula-tokenizer`)

The vocabulary and merge storage is the tokenizer slice's choice. The identity
and id keys are fixed:

| Key | Value |
|---|---|
| `heartmula.tokenizer.source_sha256` | JSON with the same shape as `heartmula.dependencies.tokenizer`: `{"tokenizer.json": "<sha256>", "gen_config.json": "<sha256>"}` |
| `heartmula.token.text_bos` | u32, 128000 |
| `heartmula.token.text_eos` | u32, 128001 |
| `heartmula.token.audio_eos` | u32, 8193 |
| `heartmula.token.empty` | u32, 0 (the pad id in audio frames) |

### Quantised files

`quantize` keeps every `heartmula.*` key and adds `quantize.imatrix`,
`quantize.imatrix_file` and `quantize.imatrix_chunks` when an imatrix was used.
It does not embed hashes: the ladder manifest (below) holds the SHA-256 of the
F16 source, the imatrix and each output. `heartmula.tensor_manifest` describes
the F16 source, not this file.

The LM ladder is built per part (global and local separately) from the F16 pair
by `tools/heartmula-quant-ladder.mjs build`: Q8_0 down to IQ1_S plus MXFP4 and
NVFP4, every rung with that variant's imatrix, named
`heartmula-<variant>-<part>-<LEVEL>.gguf` (the level is the last dash segment,
so there is no `-imat` twin; `quantize.imatrix*` records it). Collect the
imatrix with `heartmula-quant --mode collect` on the F16 pair, score a rung with
`--mode ref` / `--mode eval`, both documented in `engine/tools/heartmula-quant.cpp`.
`quantize` leaves the 1-D norms, `unconditional_text_embedding` and `muq_linear`
alone, floors the embeddings, `codebook0_head`, `audio_head.*` and `projection` at
the embedding type, and bumps `mlp.w2` (and `attn.output_proj` in the L levels).
Codec files get Q8_0 on the 2-D `.weight` matrices only (the flow transformer);
the scalar decoder and RVQ have none, so they stay F16 and F32. The ladder
manifest, not the GGUF, holds the source and imatrix SHA-256.

### LoRA adapter (`heartmula-lora`)

| Key | Value |
|---|---|
| `heartmula.component` | `lora` |
| `heartmula.variant` | the base variant it was trained on |
| `heartmula.base_fingerprint` | the base global's fingerprint |
| `heartmula.lora.target_component` | `global` (the only target in v1) |
| `heartmula.lora.rank` | u32 |
| `heartmula.lora.alpha` | f32 |
| `heartmula.lora.targets` | JSON array of base tensor names: the q, k, v and output projections of each backbone layer, plus `mlp.w1`, `mlp.w2` and `mlp.w3` for adapters trained with `--lora-mlp` |

Tensors: `<base tensor name>.lora_a` with shape [rank, in] and
`<base tensor name>.lora_b` with shape [out, rank], in the GGUF's reversed
dimension order as usual. Applied at runtime as
`W x + (alpha / rank) x scale x B (A x)`, over any base quant, where scale is
`adapter_scale` times the site's group dials (request table above); the base
weights are never modified.

Basin re-base. An adapter whose variant or base fingerprint differs from the loaded
global is accepted when its home global is installed: a compatible global file with
the adapter's `heartmula.variant` and `heartmula.base_fingerprint`, the loaded global's
quant first, then F16, then any. The home weights S of the targeted projections load
with the factors (charged to admission), and each targeted projection computes
`(1 - adapter_rebase) W x + adapter_rebase S x` plus the LoRA term: ACE's
`W + beta (S - W)` nudge applied at runtime, so any quant of either base works.
Without an installed home global the pick is refused with the variant to install.

Tools. `engine/tools/convert-heartmula-lora.py --lora X.pt --alpha A --base <global GGUF> --out Y.gguf [--negatives DIR]`
converts a torchtune LoRA state dict (alpha is not stored in the `.pt`, so it is required) and checks every
target against the base file; `--negatives` also writes four refused files and a valid rank-1 file.
`engine/tools/heartmula-adapter-reference.py` freezes adapter-on PyTorch codebook-0 logits. The parity gate is
`heartmula-probe --lm FIXTURES --global G --local L --adapter Y.gguf --adapter-ref DIR [--negatives DIR]`
(run it with `GGML_BACKEND=CPU` while the GPU slot is held elsewhere); it also checks strength 0 against the
base bit for bit and that a switch, reload or new strength drops the built graphs and KV rows.

### Load-time checks

The engine runs these at discovery (to fill `compatible`), at select-model and
again when a job loads, since files can change on disk in between:

1. `general.architecture` matches the role and `heartmula.format_version` is
   known.
2. Global and local have the same `heartmula.variant` and
   `heartmula.base_fingerprint`. Mixed quants are fine; mixed variants are not.
3. Each LM file's `heartmula.dependencies.tokenizer` equals the tokenizer's
   `heartmula.tokenizer.source_sha256` (compared as parsed JSON objects), and
   both global and local have `rvq_sha256` equal to the sha256 of the RVQ file
   present. The LM's audio codes index that codebook, so an LM pair converted
   against a different RVQ would decode to the wrong audio even when the codec
   pack is consistent with itself.
4. Flow, scalar decoder and every encoder-pack file have `rvq_sha256` equal to
   the sha256 of the RVQ file present. Flow and scalar decoder have the same
   `heartmula.base_fingerprint`.
5. The adapter's `heartmula.base_fingerprint` equals the selected global's, its
   target component is `global`, and every target tensor exists in the global
   file with matching `in` and `out`.
6. Hyperparameters parsed from `heartmula.config_json` match the tensor shapes
   the graph expects (layer count, heads, KV heads, widths, vocab sizes).

A failed check names both files and the key, for example
`heartmula-3b-local-Q8_0.gguf: base_fingerprint differs from heartmula-3b-happy-new-year-global-F16.gguf`.
