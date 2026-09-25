# Engine request and CLI reference

This page is the reference for the engine's request JSON, generation modes, CLI
flags and HTTP endpoints. For how the engine is put together (binaries, pipelines,
hook files, plugins, adapters, TensorRT, training, the ggml patch stack), read the
engine guide: [docs/dev/engine.md](../../docs/dev/engine.md).

Building the engine: [docs/dev/building.md](../../docs/dev/building.md).
Model files and where to get them: [docs/user/models.md](../../docs/user/models.md).

Everything here was checked against `engine/src/request.cpp`,
`engine/tools/hot-step-server.cpp` and the CLI sources in `engine/tools/`. When the
code and this page disagree, the code wins; fix the page.

## Request JSON reference

The ACE-Step request (`AceRequest`, `engine/src/request.h`) is one JSON object. It
is read by `ace-lm`, `ace-synth` and `ace-understand`, and by ace-server's `/lm`,
`/synth`, `/understand`, `/vae` and `/codes-decode`. Missing fields keep their
defaults. `/lm` and `/understand` return the same shape, so their output can be fed
straight to `/synth` or `ace-synth`.

```json
{
    "caption":              "",
    "negative_prompt":      "",
    "lyrics":               "",
    "bpm":                  0,
    "duration":             0,
    "keyscale":             "",
    "timesignature":        "",
    "vocal_language":       "",
    "seed":                 -1,
    "lm_batch_size":        1,
    "synth_batch_size":     1,
    "lm_temperature":       0.85,
    "lm_cfg_scale":         2.0,
    "lm_cfg_cutoff_ratio":  1.0,
    "lm_top_p":             0.9,
    "lm_top_k":             0,
    "lm_rep_penalty":       1.0,
    "lm_rep_window":        64,
    "lm_rep_mode":          "presence",
    "lm_dry_base":          1.75,
    "lm_dry_min_len":       3,
    "lm_negative_prompt":   "",
    "lm_seed":              -1,
    "use_cot_caption":      true,
    "lm_mode":              "generate",
    "audio_codes":          "",
    "inference_steps":      0,
    "guidance_scale":       0.0,
    "shift":                0.0,
    "custom_timesteps":     "",
    "dit_sliding_window":   -1,
    "audio_cover_strength": 1.0,
    "cover_noise_strength": 0.0,
    "cover_noise_method":   "",
    "repainting_start":     0,
    "repainting_end":       -1,
    "task_type":            "text2music",
    "track":                "",
    "latent_shift":         0.0,
    "latent_rescale":       1.0,
    "lss_strength":         0.0,
    "lss_var_thresh":       0.15,
    "lss_dc_remove":        true,
    "output_format":        "mp3",
    "peak_clip":            10,
    "mp3_bitrate":          128,
    "synth_model":          "",
    "lm_model":             "",
    "vae":                  "",
    "adapter":              "",
    "adapter_scale":        1.0,
    "adapters":             [],
    "adapter_sections":     [],
    "lm_adapter":           "",
    "lm_adapter_scale":     1.0,
    "postprocess_plugin":   "",
    "pp_vae_reencode":      false,
    "get_lrc":              false,
    "use_ort_vae":          false,
    "stream_mode":          false,
    "stream_depth":         8,
    "stream_chunk_dir":     ""
}
```

`GET /props` returns this object with every default filled in, under `default`.

Two more fields are parsed but not used for dispatch: `solver` (default `"euler"`)
and `stork_substeps` (default 10). `solver` only appears in a log line. On ace-server
the solver comes from `infer_method` and `stork_substeps` from the sideband (see
[ace-server sideband fields](#ace-server-sideband-fields)). The `dcw_scaler`,
`dcw_high_scaler` and `dcw_mode` keys are also read into `AceRequest`, but the
sampler only acts on the sideband copy, which needs `dcw_enabled`.

### Text conditioning

**`caption`** (string). Style, mood, instruments. Fed to the LM and to the DiT text
encoder. `/lm` rejects an empty caption. `/synth` and `ace-synth` require one except
for `lego`, `extract` and `complete` (and, on `/synth`, `cover` and `repaint`).

**`negative_prompt`** (string, default `""`). When non-empty, it is encoded as the
DiT's unconditional branch for CFG.

**`lyrics`** (string, default `""`). Three states:

- `""`: the LM writes lyrics from the caption.
- `"[Instrumental]"`: no vocals. The parser also forces `vocal_language` to
  `"unknown"`, which is what the DiT was trained with.
- Anything else: used as written. The LM only fills missing metadata.

### Metadata (the LM fills what is unset)

**`bpm`** (int, `0` = unset).

**`duration`** (float seconds, `0` = unset). The LM's FSM constrains it to 10 to 600
seconds. If a request reaches the DiT with no duration, it renders 30 seconds.

**`keyscale`** (string, `""` = unset), for example `"C major"`, `"F# minor"`.

**`timesignature`** (string, `""` = unset). The numerator only: `"4"` for 4/4.

**`vocal_language`** (string, `""` = unset). `""` lets the LM detect it, `"unknown"`
means no specific language, any other code is used as given.

### Seeds and batching

**`seed`** (int64, `-1` = random). Seed for the DiT's Philox noise; the low 32 bits
are used. Resolved once per input request.

**`lm_seed`** (int64, `-1` = unset). Seed for the LM sampler. On ace-server `/lm`, an
unset `lm_seed` follows `seed`, and a random value is drawn if both are unset. Batch
item `b` of `/lm` gets `lm_seed + b`.
<!-- TODO(verify): ace-lm and ace-understand never resolve lm_seed = -1, so the LM
     sampler seeds from (uint32_t)-1 on every run. Check whether that is intended. -->

**`lm_batch_size`** (int, default 1). Number of LM variations. `ace-lm` clamps it to
1..9; ace-server clamps it to 1..`--max-batch`.

**`synth_batch_size`** (int, default 1). DiT variations per request, each with
`seed + i`. The total across a `/synth` batch is clamped to 9.

The rules:

1. Each input request runs as if it were the only one.
2. `seed = -1` resolves to one random value per input request.
3. `lm_batch_size = N` produces N LM outputs; `synth_batch_size = N` produces N DiT
   outputs with consecutive seeds.

### LM sampling (`ace-lm`, `/lm`)

**`lm_mode`** (string, default `"generate"`). `generate` fills metadata and lyrics
and writes audio codes. `inspire` turns a short query into metadata and lyrics, no
codes. `format` turns caption and lyrics into metadata and lyrics, no codes. Any
other value is rejected.

**`lm_temperature`** (float, default 0.85). Applies to both phases.

**`lm_cfg_scale`** (float, default 2.0). LM classifier-free guidance. Always active
in phase 2 (audio codes). Phase 1 turns CFG off whenever it writes free text (lyrics
or the CoT caption), so it only applies there when lyrics are given and
`use_cot_caption` is false. `1.0` disables it.

**`lm_cfg_cutoff_ratio`** (float, default 1.0). Fraction of code tokens that use CFG.
`0.5` runs CFG for the first half of the tokens only.

**`lm_top_p`** (float, default 0.9). Nucleus cutoff. `1.0` disables it.

**`lm_top_k`** (int, default 0). `0` disables top-k.

**`lm_rep_penalty`** (float, default 1.0 = off). Repetition penalty over recently
emitted audio codes, to break code loops.

**`lm_rep_window`** (int, default 64). Lookback in codes (64 codes is about 13 s at
5 Hz).

**`lm_rep_mode`** (string, default `"presence"`). How the penalty is applied:
`presence` penalises each distinct code in the window once, `frequency` raises the
penalty to the power of the occurrence count, `dry` penalises only codes that would
extend a verbatim repeat. Unknown values fall back to `presence`.

**`lm_dry_base`** (float, default 1.75) and **`lm_dry_min_len`** (int, default 3).
`dry` mode only: growth per extra matched code, and matched codes before any penalty.

**`lm_negative_prompt`** (string, default `""`). Negative caption for phase 2 CFG.
Empty uses a caption-less unconditional prompt.

**`use_cot_caption`** (bool, default true). When true the LM rewrites the caption
through CoT and that version goes to the DiT. When false the caption is kept as
written.

**`audio_codes`** (string, default `""`). Comma-separated 5 Hz FSQ code IDs. When set
in `generate` mode the LM is skipped and the DiT decodes these codes.

### DiT sampling (`ace-synth`, `/synth`)

**`inference_steps`** (int, `0` = auto). Auto is 8 on a turbo DiT and 50 otherwise.
Clamped to 300.

**`guidance_scale`** (float, `0` = auto). Auto is 1.0 (no CFG). A value above 1.0 on
a turbo model is kept, with a warning in the log.

**`shift`** (float, `0` = auto). Timestep shift, `t' = s*t / (1 + (s-1)*t)`. Auto is
3.0 on turbo and 1.0 otherwise. On ace-server, `-1` computes a shift from duration
and step count instead (between 1 and 6, centred on 3).

**`custom_timesteps`** (string, default `""`). Comma-separated descending timesteps,
for example `"0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0"`. Overrides
`inference_steps` and `shift`. The last value is the endpoint, so N values give N-1
steps.

**`dit_sliding_window`** (int, default -1). Window override, in tokens (12.5 per
second), for the DiT's sliding-window attention layers. `-1` keeps the model's value,
`0` makes every layer full attention, a positive value sets the window. Larger than
the trained window is out of distribution.

**`latent_shift`**, **`latent_rescale`** (float, defaults 0.0 and 1.0). Applied to the
DiT output before VAE decode: `latent * latent_rescale + latent_shift`.
<!-- TODO(verify): on the GGML path through ace-server these look applied twice, once
     in hot-step-sampler.h from the sideband copy and once in pipeline-synth-ops.cpp
     from the request copy. Confirm before relying on non-default values. -->

**`lss_strength`** (float, default 0 = off), **`lss_var_thresh`** (default 0.15),
**`lss_dc_remove`** (default true). Latent Spectral Suppressor: before VAE decode,
latent channels whose relative variance is under `lss_var_thresh` are attenuated
toward `1 - lss_strength`, with optional per-channel DC removal.

### Source audio and tasks

**`task_type`** (string, default `"text2music"`). One of `text2music`, `cover`,
`cover-nofsq`, `repaint`, `lego`, `extract`, `complete`. See
[Generation modes](#generation-modes).

**`track`** (string, default `""`). Stem name for `lego`, `extract` and `complete`.

**`audio_cover_strength`** (float, default 1.0). Fraction of DiT steps that see the
source as context. Below 1.0 the remaining steps switch to silence context and the
text2music instruction.

**`cover_noise_strength`** (float, default 0.0). Starts diffusion from a blend of
noise and the clean source latents instead of pure noise, when the task uses a
source context. `1.0` starts close to the source.

**`cover_noise_method`** (string, default `""`). How the schedule is shortened for
`cover_noise_strength`. `""` truncates the early steps. `"rescale"` keeps the full
step count and rescales the schedule into the reduced range.

**`repainting_start`**, **`repainting_end`** (float seconds, defaults 0 and -1).
Region for `repaint` and `lego`. A negative start outpaints before the source; an
end past the source outpaints after it; `-1` means the end of the source.

### Output

**`output_format`** (string, default `"mp3"`). `mp3`, `wav16`, `wav24` or `wav32`.
`ace-synth` uses it for the file extension and format. `/synth` uses it when the URL
has no `?format=`.

**`peak_clip`** (int, default 10). Output normalisation percentile: `0` is plain peak
normalisation, `10` clips the top 0.001 %, `999` clips the top 0.1 %.

**`mp3_bitrate`** (int, default 128). Used by `/vae` and `/codes-decode`. `/synth`
and `ace-synth` use the `--mp3-bitrate` flag instead.

**`get_lrc`** (bool, default false). Runs an extra DiT pass for cross-attention
alignment and produces LRC lyric timestamps. ace-server returns them in the
`X-LRC-Text` header of the `/job` result.

**`postprocess_plugin`** (string, default `""`). A Lua postprocess plugin to use for
VAE decode instead of the built-in tiled decoder.

**`pp_vae_reencode`** (bool, default false). Round-trips the audio through the
post-processing VAE when a PP-VAE model is installed.

**`use_ort_vae`** (bool, default false). Decodes through the ONNX Runtime VAE when
ace-server was started with `--onnx-dir` and a `vae_decoder.onnx` was found.

**`stream_mode`** (bool, default false), **`stream_depth`** (int, default 8),
**`stream_chunk_dir`** (string). Routes the DiT through the ring-buffer streaming
pipeline. Needs an ONNX DiT.

### Models and adapters

**`synth_model`** (string). DiT file name from the registry. Empty keeps the loaded
DiT on ace-server, or takes the first one.

**`lm_model`** (string). LM file name for `/lm` and `/understand`. Same fallback.

**`vae`** (string). VAE file name for `/vae` and `/codes-decode`. `/synth` selects its
VAE with the sideband field `vae_model` instead; `ace-synth` always uses the first VAE
in the registry.

**`adapter`** (string), **`adapter_scale`** (float, default 1.0). One DiT adapter: a
name from `--adapters`, or on ace-server an absolute path to a `.safetensors` file or
an adapter directory (`adapter_model.safetensors` or `lokr_weights.safetensors`).

**`adapters`** (array). A stack of DiT adapters; when non-empty it replaces `adapter`.
Each entry is a name or path string, or an object:

```json
{ "name": "my-adapter", "scale": 0.8, "gain_curve": [0.0, 1.0, 1.0], "gain_domain": "steps" }
```

`gain_curve` is an optional list of gains sampled uniformly over the run;
`gain_domain` is `"t"` (flow-matching time, the default) or `"steps"` (fraction of
steps). Curves are applied per step and never baked into weights, so changing them
does not reload the model.

**`adapter_sections`** (array). Per-section adapter masking. One object per lyric
section, `{ "weights": [...], "size": n }`: `weights` holds each stacked adapter's
scale for that section and `size` is a relative length hint. Active with two or more
adapters, or with any stack that uses gain curves. Forces `adapter_mode` to
`runtime`.

**`lm_adapter`** (string), **`lm_adapter_scale`** (float, default 1.0). A runtime LoRA
for the ACE-Step planner LM: a name from `<adapters>/lm/` or a path. `/lm` fails the
job if the name does not resolve.

### ace-server sideband fields

`/synth` reads these with a second parser (`parse_server_fields`). They are not part
of `AceRequest`, so `/lm` does not echo them back. When `/synth` receives an array,
only the first object's sideband fields are read.

| Field | Default | Meaning |
|---|---|---|
| `vae_model` | `""` | VAE from the registry. An `.onnx` VAE routes decode through ONNX Runtime and encodes with the first non-ONNX VAE |
| `emb_model` | `""` | Text encoder from the registry |
| `infer_method` | `"euler"` | Solver plugin name. Unknown names fall back to `euler` |
| `scheduler` | `""` | Scheduler plugin name. Empty uses the shift schedule |
| `guidance_mode` | `"apg"` | Guidance plugin name. Unknown names fall back to `apg` |
| `plugin_params` | `{}` | Plugin parameters, `{"pluginName:key": value}`. Values may be strings, numbers or booleans |
| `apg_momentum`, `apg_norm_threshold` | 0.75, 2.5 | APG tuning |
| `stork_substeps`, `beat_stability`, `frequency_damping`, `temporal_smoothing` | 10, 0.25, 0.4, 0.13 | Copied into the solver state |
| `cfg_cutoff_ratio` | 1.0 | Fraction of DiT steps that run CFG |
| `cache_ratio` | 0.0 | Fraction of middle steps that reuse the previous velocity instead of a forward pass |
| `custom_timesteps` | `""` | Same as the request field; the sampler reads this copy |
| `dcw_enabled`, `dcw_mode`, `dcw_scaler`, `dcw_high_scaler` | false, `"low"`, 0.1, 0.0 | Wavelet-domain sampler correction. `dcw_mode` is `low`, `high`, `double` or `pix` |
| `latent_shift`, `latent_rescale` | 0.0, 1.0 | Same as the request fields |
| `denoise_strength`, `denoise_smoothing`, `denoise_mix` | 0.0, 0.7, 0.25 | Post-VAE spectral denoiser. Uses the `--noise-profile` noise profile when one was loaded |
| `adapter_mode` | `"merge"` | `merge`, `runtime` or `runtime_lowrank` |
| `adapter_runtime_quant` | `"bf16"` | Runtime delta precision: `bf16`, `q8_0`, `q4_k` |
| `adapter_merge_lowvram` | false | Merge mode: requantise merged weights to the base type instead of F32 |
| `adapter_group_scales` | all 1.0 | Object with `self_attn`, `cross_attn`, `mlp`, `cond_embed`, `time_embed`, `proj_in` |
| `adapter_section_align_at` | 0.55 | Fraction of steps before per-section masks are rebuilt from cross-attention alignment. `<= 0` keeps the proportional map |
| `adapter_section_isolation` | 0.0 | 0 to 1 penalty on self-attention across section boundaries |
| `rebase_source`, `rebase_beta` | `""`, 0.0 | Nudge adapted weights toward the adapter's training base: a DiT name from the registry, and the strength. Merge mode only |
| `concepts` | `[]` | Concept steering vectors: `{"name", "path", "target": "dit" or "lm", "alpha", "layers": [...]}` |
| `concept_extract` | none | `{"out", "name", "positive", "negative", "target_class", "pairs", "null_ref", "top_k"}`. Turns the job into a paired harvest that writes a concept GGUF instead of audio. Needs `out`, `positive` and `negative` |
| `seed_strength` | 0.0 | Repaint only: bias the region's initial noise toward the `seed_latents` multipart part |
| `evict_lm` | false | Free the LM before loading the synth pipeline |
| `vae_chunk` | 0 | Per-request VAE tile size (`0` keeps the loaded value) |
| `batch_cfg` | -1 | `0` splits CFG into two forwards, `1` batches them, `-1` keeps the default |

<!-- TODO(verify): /lm never parses the sideband, so a concept with target "lm" only
     reaches the LM through whatever the last /synth left in g_hotstep_params. -->

### MiniMax-Music3 request (`POST /mm3/synth`)

Parsed by `mm3_parse_synth_request` in `engine/src/minimax/mm3-request.h`. Type errors
are reported, not ignored.

| Field | Default | Meaning |
|---|---|---|
| `caption` | required | Structured caption. Markdown formatting is cleaned before use |
| `lyrics` | `""` | Empty means instrumental |
| `duration` | required unless `max_frames` | Seconds. Rendered at 25 frames per second, capped at 9000 frames |
| `max_frames` | none | Frame count; wins over `duration` |
| `seed` | -1 | `-1` draws a random seed |
| `ar_seed` | -1 | Separate seed for the planner stage; `-1` ties it to `seed` |
| `steps` | checkpoint (30) | Flow steps, 1 to 1000 |
| `cfg_flow` | checkpoint (1.7) | Flow CFG, above 0 up to 100 |
| `flow_uncond_interval` | 1 | 1 to 16. Above 1, the unconditional branch runs only every Nth step |
| `get_wav_bits` | 16 | 16, 24 or 32 |
| `get_lrc` | false | Lyric timestamps from the LM's alignment heads |
| `get_ar_codes` | false | Keep the planner codes on the job (`GET /mm3/job?id=&ar=1` or `GET /mm3/take?...&ar=1`) |
| `dit_backend` | `"ggml"` | `ggml` or `tensorrt` |
| `depth_fused` | true | Fused depth-decoder graph with GPU sampling |
| `infer_method`, `scheduler`, `guidance_mode`, `plugin_params` | `""`, `""`, `""`, `{}` | Lua plugins for the flow stage. Empty keeps the native Euler path and plain CFG. `guidance_mode: "apg"` uses native APG |
| `flow_shift` | 1.0 | Above 0 up to 20. Only used with a scheduler plugin |
| `apg_norm_threshold` | 2.5 | 0 to 100 |
| `lm_temperature`, `lm_top_k`, `lm_top_p` | 1.0, 0, 0.0 | Planner sampling. `lm_top_k` 0 uses the checkpoint's value |
| `lm_rep_penalty`, `lm_rep_window`, `lm_rep_mode`, `lm_dry_base`, `lm_dry_min_len` | 1.0, 320, `"dry"`, 1.75, 15 | Planner repetition penalty, in 25 fps frames |
| `takes` | 1 | Songs from one planner pass, 1 to 8, clamped to the checkpoint's row budget. Take t uses `seed + t`; fetch with `GET /mm3/take` |
| `require_eos`, `stop_after_first_eos`, `eos_rounds`, `min_frames` | false, false, 4, 0 | Natural-ending controls. `eos_rounds` is 1 to 16; `min_frames` drops plans shorter than that |
| `reuse_ar` | false | Reuse the previous planner output when every planner input matches |
| `forced_frame_hiddens_file`, `save_frame_hiddens`, `frame_hiddens_save_path` | | Load or save the planner output as a `.mm3hiddens` file |
| `forced_semantic`, `forced_acoustic` | | Replay captured codes. Both or neither; `forced_acoustic` has 7 entries per semantic entry |
| `stream` | false | Emit audio per window for `GET /mm3/stream` |
| `lm_adapter` | `""` | Path to a PEFT LM adapter |
| `lm_adapter_mode` | `"runtime"` | `runtime` or `merge` |
| `lm_adapter_merge_gpu` | true | Merge mode: merge on the GPU |
| `lm_adapter_scale`, `lm_adapter_scale_attn`, `lm_adapter_scale_mlp`, `lm_adapter_scale_early`, `lm_adapter_scale_mid`, `lm_adapter_scale_late` | | Adapter scales: overall, by module group, and by depth third |
| `lm_soft_off` | false | Run the adapter's weight delta without its artist token or KV prefix |

### YuE2 request (`POST /yue2/synth`)

Parsed by `yue2_parse_request` in `engine/src/yue2/yue2-request.h`.

| Field | Default | Meaning |
|---|---|---|
| `style` | required | Style tags |
| `lyrics` | `""` | Lyrics |
| `cot` | `"off"` | `off`, `melody` or `full`. `melody` and `full` run the ABC plan stage |
| `abc` | none | ABC text to use instead of running the plan stage |
| `plan_only` | false | Stop after the plan stage and return the ABC score. Needs `cot` other than `off` |
| `semantic_only` | false | Stop after the semantic stage and return the codec ids |
| `semantic_retries` | 0 | Re-draw a song whose semantic stage hits its cap, up to this many times |
| `seed` | random | Song seed. Song i of a batch uses `seed + i` |
| `noise_seed` | `seed` | NAR noise seed. Variation j uses `noise_seed + j` |
| `lm_batch_size` | 1 | Songs per request, 1 to 4 |
| `synth_batch_size` | 1 | Noise variations per song, 1 to 9 |
| `plan_<field>`, `semantic_<field>` | checkpoint | Per-stage sampler overrides. `<field>` is `temperature`, `top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `min_tokens` or `max_tokens` |
| `cfg_scale` | resolved from the checkpoint | NAR CFG |
| `ode_steps` | checkpoint | NAR ODE steps |
| `ode_method` | `"midpoint"` | The only method implemented |
| `infer_method`, `scheduler`, `plugin_params` | `""`, `""`, `{}` | Lua solver and scheduler for the NAR stage. Empty keeps native midpoint and the uniform grid |
| `nar_cache_ratio` | 0.0 | Fraction of middle NAR steps that reuse the last velocity |
| `nar_chunk_frames` | 0 | NAR chunk cap in frames (25 per second). 0 renders a normal song in one chunk |
| `end_threshold` | 0.0 | 0 to 1. Stop the semantic stage when the sampler puts this much probability on the end token |
| `end_bias`, `end_bias_from_sec`, `end_bias_ramp_sec` | 0.0 | Additive end-token bias (-50 to 50), when it starts, and its ramp |
| `codec_ids` | none | Semantic codes (0 to 32767) to render instead of sampling. Single song, single variation |
| `preview_max_frames` | 0 | Bound on semantic frames, 0 to 9000 (training previews) |
| `vae_variant` | `"standard"` | `standard` or `legacy` |
| `noise_source`, `noise_fixture_path` | `"native"` | Validation only: `fixture` reads NAR noise from a raw f32 file |
| `id` | `"yue2"` | Request label |

Progress and results for `/yue2/synth` use the shared `GET /job`.

## Generation modes

These apply to ACE-Step requests. The engine reads what is in the JSON: an empty
field means "fill it", a filled field means "keep it". Which LM call runs before the
DiT is the client's choice; the DiT side routes on `task_type` alone.

**Caption only** (`lyrics` empty). The LM runs phase 1 with the inspire instruction
to write lyrics and metadata through CoT, then phase 2 for audio codes. CFG is off in
phase 1. With `lm_batch_size > 1` each item runs its own phase 1, so the songs
differ.

**Caption and lyrics.** Phase 1 fills missing metadata (and, with
`use_cot_caption`, rewrites the caption), then phase 2 writes codes. Given metadata
is never overwritten.

**Everything given** (caption, lyrics, bpm, duration, keyscale, timesignature, and
`use_cot_caption` false). Phase 1 is skipped and the LM writes codes directly.

**Instrumental** (`lyrics` is `"[Instrumental]"`). Treated as lyrics given, so no
lyrics are written.

**Passthrough** (`audio_codes` set). The LM is skipped and the DiT decodes the codes.

**Cover** (`task_type: "cover"` with source audio). The source is resampled to
48 kHz, VAE-encoded, and put through an FSQ round trip (25 Hz to 5 Hz and back),
which loses fine detail, so the DiT reinterprets it freely.
`audio_cover_strength` sets how many steps see the source. The duration comes from
the source.

**Cover without FSQ** (`task_type: "cover-nofsq"`). The same, but the DiT gets the
clean 25 Hz latents and stays close to the source. Pairing it with the same file as
the reference audio is the usual setup.

**Repaint** (`task_type: "repaint"` with source audio). Regenerates the region from
`repainting_start` to `repainting_end` and keeps the rest. Outside the region the
source latents are spliced back before decode, and on the audio path the original
samples are spliced back after decode.

```json
{
    "task_type": "repaint",
    "caption": "Smooth jazz guitar solo with reverb",
    "lyrics": "[Instrumental]",
    "repainting_start": 10.0,
    "repainting_end": 25.0
}
```

**Lego** (`task_type: "lego"`, source audio and `track`). Generates a new
instrument layer in the context of a backing track. Takes an optional region.

**Extract** (`task_type: "extract"`, source audio and `track`). Isolates one stem.

**Complete** (`task_type: "complete"`, source audio and `track`). Builds a full mix
around an isolated stem. `track` can be a preformatted list such as
`"VOCALS | DRUMS"`.

Standard `track` names: `vocals`, `backing_vocals`, `drums`, `bass`, `guitar`,
`keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`, `woodwinds`.

| Task | Turbo DiT | Base or SFT DiT |
|---|---|---|
| text2music | yes | yes |
| cover, cover-nofsq | yes | yes |
| repaint | yes | yes |
| lego, extract, complete | runs, with a warning that turbo output is incoherent | yes |

The instruction the DiT receives per task is defined in `engine/src/task-types.h`.

## ace-lm reference

```
Usage: ace-lm --models <dir> --request <json> [options]

Required:
  --models <dir>         Directory of GGUF model files
  --request <json>       Input request JSON (carries lm_model)

Debug:
  --max-seq <N>          KV cache size (default: 8192)
  --no-fsm               Disable FSM constrained decoding
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump-logits <path>   Dump prefill logits (binary f32)
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

The LM comes from the request's `lm_model`, resolved against `--models`; empty takes
the first LM found. The input is never modified. Output is always numbered:
`request.json` produces `request0.json` .. `requestN-1.json`, one per
`lm_batch_size` item. `lm_mode` is honoured.

## ace-synth reference

```
Usage: ace-synth --models <dir> --request <json...> [options]

Required:
  --models <dir>          Directory of GGUF model files
  --request <json...>     One or more request JSONs (from ace-lm --request)

Optional:
  --adapters <dir>        Directory of adapter files (enables JSON adapter field)
  --src-audio <file>      Source audio (WAV or MP3)
  --ref-audio <file>      Timbre reference audio (WAV or MP3)

Audio encoding:
  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split DiT CFG into two separate forwards
  --clamp-fp16            Clamp hidden states to FP16 range
  --dump <dir>            Dump intermediate tensors
```

The first request picks the models: `synth_model` for the DiT (empty takes the first),
`adapter` or `adapters` from `--adapters`, and `output_format` for the output. The
text encoder and VAE are always the first in their registry bucket. All requests run
as one GPU batch, with `synth_batch_size` expanding each. Output is
`<request basename><index>.mp3` (or `.wav`), so `request0.json` gives
`request00.mp3`.

`--src-audio` supplies the source for cover, repaint, lego, extract and complete.
`--ref-audio` supplies a timbre reference for any task. Both are resampled to 48 kHz
and VAE-encoded.

The CLI reads only `AceRequest` fields. The sideband fields in
[ace-server sideband fields](#ace-server-sideband-fields) are not available here,
so sampling uses the default solver (`euler`) and guidance (`apg`).

```bash
# LM: request.json -> request0.json (metadata, lyrics, codes)
./ace-lm --models models --request /tmp/request.json

# DiT + VAE: request0.json -> request00.mp3
./ace-synth --models models --request /tmp/request0.json

# Cover from an existing song, no LM
./ace-synth --models models --request /tmp/cover.json --src-audio song.wav
```

## ace-understand reference

Reverse pipeline: audio in, then VAE encode, FSQ tokenize and the LM's understand
prompt. The output JSON is a request that `ace-lm` or `ace-synth` can take.

```
Usage: ace-understand --models <dir> --src-audio <file> [--request <json>] [options]

Required:
  --models <dir>          Directory of GGUF model files
  --src-audio <file>      Source audio (WAV or MP3, any sample rate)

Optional:
  --request <json>        Request JSON carrying model selection and
                          sampling params (lm_model, synth_model,
                          lm_temperature, lm_top_p, lm_top_k)

When no --request is given, understand defaults apply
(temperature 0.3, top_p disabled).

Output:
  -o <json>               Output JSON (default: stdout summary)

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --max-seq <N>           KV cache size (default: 8192)
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --dump <dir>            Dump tok_latents + tok_codes (skip LM)
```

The DiT file supplies the FSQ tokenizer weights.

## ace-server reference

```
Usage: ace-server --models <dir> [options]

Required:
  --models <dir>          Directory of GGUF model files

Adapter:
  --adapters <dir>        Directory of adapters

Memory control:
  --keep-loaded           Keep models in VRAM between requests
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

ONNX/TensorRT:
  --onnx-dir <dir>        Directory with ONNX models (e.g. vae_decoder.onnx)

Speculative decoding:
  --draft-lm <path>       Path to 0.6B draft LM (auto-discovers if omitted)
  --no-draft              Disable draft model auto-discovery

Output:
  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)

Server:
  --host <addr>           Listen address (default: 127.0.0.1)
  --port <N>              Listen port (default: 8080)
  --max-batch <N>         LM batch limit (default: 1)
  --max-seq <N>           KV cache size (default: 8192)

Debug:
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split CFG into two separate forwards (LM + DiT)
  --clamp-fp16            Clamp hidden states to FP16 range
```

Notes the usage text does not cover:

- `--noise-profile <wav>` is also accepted. It loads a noise profile for the
  spectral denoiser (`denoise_strength`).
- `--draft-lm` auto-discovery is disabled in the code; only an explicit path enables
  speculative decoding.
- `--max-batch` is clamped to 1..9.
- `--models` is scanned at startup, and its `onnx/` subfolder too. `--adapters` is
  scanned for DiT adapters and its `lm/` subfolder for planner-LM adapters.
- `--onnx-dir` looks for `vae/vae_decoder.onnx`, then `vae_decoder.onnx`.
- The app starts ace-server on port 8085.

| Pipeline | Needs | Enables |
|---|---|---|
| LM | an LM GGUF | `/lm` |
| Synth | DiT, text encoder and VAE | `/synth`, `/warm` |
| Understand | LM, DiT and VAE | `/understand` |

Endpoints whose pipeline has no models return 501. With only MM3 or YuE2 weights
present, the server still starts and serves those routes.

### Endpoints

All compute endpoints are asynchronous. They return `{"id":"<hex>"}` at once and put
the work on the single GPU queue.

```
POST /lm[?keep_loaded=1]         LM job. Body: AceRequest JSON. lm_mode picks
                                 generate, inspire or format.
POST /synth[?format=mp3|wav16|wav24|wav32][&keep_loaded=1]
                                 Synth job. Body: AceRequest JSON, or an array of
                                 them, or multipart/form-data with parts:
                                   request      JSON text (required)
                                   audio        source audio (WAV or MP3)
                                   ref_audio    timbre reference audio
                                   src_latents  raw f32 [T*64], replaces audio
                                   ref_latents  raw f32 [T*64], replaces ref_audio
                                   seed_latents raw f32 [T*64], see seed_strength
                                 Without ?format= the first request's output_format
                                 is used.
POST /understand                 Understand job. multipart/form-data: audio
                                 (required), request (optional JSON).
POST /vae                        VAE job. multipart/form-data: audio (encode,
                                 returns raw f32 latents) or src_latents (decode,
                                 returns audio), plus optional request JSON (vae,
                                 output_format, peak_clip, mp3_bitrate).
POST /codes-decode               Codes to audio. Body: AceRequest JSON with
                                 audio_codes (required), synth_model, vae,
                                 output_format, peak_clip.
POST /warm[?keep_loaded=1]       Preload. Body: {"dit", "vae", "adapter",
                                 "adapter_scale"}.

GET  /job?id=N                   Status: {"status", "phase", "phase_step",
                                 "phase_total", "adapter_progress"}. status is
                                 running, done, failed or cancelled. YuE2 jobs add
                                 end_reason, stage_end_reasons, tracks and abc.
GET  /job?id=N&result=1          Result. /lm and /understand: JSON array of
                                 AceRequest. /synth: one audio file, or
                                 multipart/mixed for a batch; LRC text in the
                                 X-LRC-Text header when get_lrc was set.
GET  /job?id=N&latent=1          /synth: the first track's post-DiT latents, raw
                                 f32 [T*64].
POST /job?id=N&cancel=1          Cancel a job.
GET  /jobs                       Every job in the table.

GET  /health                     {"status":"ok"}
GET  /props                      version, models (lm, embedding, dit, vae),
                                 adapters, lm_adapters, cli (max_batch,
                                 mp3_bitrate), default (full AceRequest), presets
                                 (turbo, sft).
GET  /logs                       Server stderr as server-sent events.
GET  /plugins                    Lua plugin registry.
GET  /vram                       {"used_mb", "total_mb", "free_mb"} (zeros on
                                 non-CUDA builds).
GET  /models/loaded              Resident modules: {"loaded": [{"label", "mb",
                                 "in_use"}]}.
POST /models/unload              Body {"label"}. Evicts one module.
POST /models/restore-policy      Undo ?keep_loaded=1. 409 when --keep-loaded was
                                 given on the command line.
GET  /                           Embedded web UI (gzip).
```

Post-processing endpoints. These run synchronously on the HTTP thread, except
SuperSep, which has its own job table.

```
POST /pp-vae-reencode[?blend=0..1][&backend=onnx|gguf][&out_fmt=s16|s24|f32]
                                 Body: WAV. PP-VAE round trip. blend 0 is fully
                                 re-encoded, 1 is the original.
POST /sa3-refine?...             Body: WAV or MP3. Stable Audio 3 refine. Query
                                 parameters are documented above the handler in
                                 hot-step-server.cpp (tokens, n_tokens, strength,
                                 steps, sampler, seed, rms_match, env_match, mix,
                                 band_blend, band_freq, band_width, out_sr,
                                 backend, adapters, solver, scheduler,
                                 guidance_mode, guidance_scale, plugin_params).
POST /supersep/separate?level=0..4
                                 Body: audio. Starts stem separation, returns {"id"}.
GET  /supersep/progress?id=      Progress.
GET  /supersep/result?id=        Stem list.
GET  /supersep/serve?id=&stem=N  One stem as WAV.
POST /supersep/release?id=       Drop a job and free its stems.
POST /supersep/recombine         Body {"id", "stems": [{"index", "volume",
                                 "muted"}]}. Returns the mix as WAV.
POST /spectral-lifter?...        Body: WAV. Query: denoise_strength, noise_floor,
                                 hf_mix, transient_boost, shimmer_reduction.
```

`/pp-vae-reencode`, `/sa3-refine`, `/supersep/serve`, `/supersep/recombine` and
`/spectral-lifter` take `?out_fmt=s16|s24|f32` for the returned WAV; the default is
`s16`.

MiniMax-Music3 and YuE2 endpoints. Request fields for the two synth routes are in
[MiniMax-Music3 request](#minimax-music3-request-post-mm3synth) and
[YuE2 request](#yue2-request-post-yue2synth).

```
GET  /mm3/props                  MM3 files, config, what is loaded.
POST /mm3/warm, /mm3/unload      Load or free MM3 weights.
POST /mm3/select-model           Body {"lm", "depth", "cond", "dit", "voc"} quant
                                 tokens ("" = best available).
POST /mm3/synth                  MM3 job on the shared queue.
GET  /mm3/job?id=                MM3 progress: stage, window, seed, timings.
                                 &ar=1 returns the planner codes.
GET  /mm3/take?id=&take=N        Audio for one take of an ensemble render.
                                 &lrc=1 returns its LRC, &ar=1 its planner codes.
GET  /mm3/stream?id=             Live audio chunks of a running job (one reader).
POST /mm3/tokenize-check         Prompt token count (5000-token limit).
POST /mm3/imatrix                Activation statistics for quantize --imatrix.
POST /mm3/voc-decode, /mm3/dit-forward, /mm3/flow-sample, /mm3/depth-frame,
     /mm3/cond-encode, /mm3/lm-plan, /mm3/synth-e2e
                                 Bring-up and parity endpoints. They run outside
                                 the job queue.

GET  /yue2/props                 YuE2 files, config, what is loaded.
POST /yue2/warm, /yue2/unload    Load or free YuE2 weights.
POST /yue2/select-model          Body {"vae_variant", "lm_type", "lm_adapter",
                                 "lm_adapter_scale"}; absent keys are left alone.
POST /yue2/synth                 YuE2 job; poll with GET /job.
POST /yue2/tokenize-check        Prompt assembly and token count, no GPU work.
POST /yue2/imatrix               Activation statistics for quantize --imatrix.
POST /yue2/align                 multipart: audio + lyrics. Forced alignment.
```

Errors are JSON, `{"error":"message"}`, with a 4xx or 5xx status.

### Concurrency

One worker thread runs every GPU job in FIFO order: ACE-Step, MM3 and YuE2 alike.
The job table holds 32 entries; finished jobs are evicted oldest first and running
jobs never are. Each job has its own cancel flag, and a cancel during adapter
precompute takes effect between deltas.

By default the model store keeps one module in VRAM at a time. `--keep-loaded`
keeps everything resident for the life of the process; `?keep_loaded=1` does the
same from a request, until `POST /models/restore-policy`.

Request bodies are limited to 256 MB. Socket read and write timeouts are 600 s.

## neural-codec reference

ACE-Step VAE encoder and decoder as an audio codec: 48 kHz stereo to 64-channel
latents at 25 Hz and back.

```
Usage: neural-codec --vae <gguf> --encode|--decode -i <input> [-o <output>] [--q8|--q4]

Required:
  --vae <path>            VAE GGUF file
  --encode | --decode     Encode audio to latent, or decode latent to WAV
  -i <path>               Input (WAV/MP3 for encode, latent for decode)

Output:
  -o <path>               Output file (auto-named if omitted)
  --q8                    Quantize latent to int8 (~13 kbit/s)
  --q4                    Quantize latent to int4 (~6.8 kbit/s)
  --format <fmt>          WAV format: wav16, wav24, wav32 (default: wav16)

Output naming: song.wav -> song.latent (f32) or song.nac8 (Q8) or song.nac4 (Q4)
               song.latent -> song.wav

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Latent formats (decode auto-detects):
  f32:  flat [T, 64] f32, no header. ~51 kbit/s.
  NAC8: header + per-frame Q8. ~13 kbit/s.
  NAC4: header + per-frame Q4. ~6.8 kbit/s.
```

NAC8 and NAC4 files start with an 8-byte header: the 4-byte magic (`NAC8` or
`NAC4`) and a uint32 frame count. A NAC8 frame is 66 bytes (an f16 scale and 64
int8 values).

```bash
./neural-codec --vae models/vae-BF16.gguf --encode --q4 -i song.wav -o song.nac4
./neural-codec --vae models/vae-BF16.gguf --decode -i song.nac4 -o song_decoded.wav
```

## mp3-codec reference

Standalone MP3 encoder and decoder. The encoder in `engine/mp3/` is the one
`ace-synth` and ace-server use for MP3 output. The mode follows the output
extension.

```
Usage: mp3-codec -i <input> -o <output> [options]

  -i <path>     Input file (WAV or MP3)
  -o <path>     Output file (WAV or MP3)
  -b <kbps>     Bitrate for MP3 encoding (default: 128)
  --format <f>  WAV format: wav16, wav24, wav32 (default: wav16)
```

```bash
mp3-codec -i song.wav -o song.mp3 -b 192
mp3-codec -i song.mp3 -o song.wav --format wav32
```
