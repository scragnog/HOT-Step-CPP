# Engine request and CLI reference

This page is the reference for the engine's request JSON, generation modes, CLI
flags and HTTP endpoints. For how the engine is put together (binaries, pipelines,
hook files, plugins, adapters, TensorRT, training, the HOT-ggml fork), read the
engine guide: [docs/dev/engine.md](../../docs/dev/engine.md).

Building the engine: [docs/dev/building.md](../../docs/dev/building.md).
Model files and where to get them: [docs/user/models.md](../../docs/user/models.md).

Everything here was checked against `engine/src/request.cpp`,
`engine/tools/hot-step-server.cpp`, `engine/src/minimax/mm3-request.h`,
`engine/src/yue2/yue2-request.h` and the CLI sources in `engine/tools/`. When the
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
    "lm_min_p":             0.0,
    "lm_cfg_min_p":         0.0,
    "lm_cfg_logprob":       false,
    "lm_cfg_rescale":       0.0,
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
    "get_lrc":              false
}
```

`GET /props` returns every `AceRequest` field with its default under `default`,
including the seed-breeding, solver and DCW fields described below that this block
leaves out.

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

**`seed_b`** (int64, `-1` = off) and **`seed_blend`** (float 0 to 1, default 0). Seed
breeding: the initial DiT noise is a slerp from the `seed` noise toward the `seed_b`
noise (each batch item treated as one vector, normalized lerp when nearly parallel).
`0` gives `seed` exactly, `1` gives `seed_b` exactly. Synth batch items offset
`seed_b` by the same `+i` as `seed`. API field names on `/api/generate`: `seedB`,
`seedBlend`.

**`seed_blend_end`** (float 0 to 1, default `-1` = off). Seed morph: when set with
`seed_b`, the blend varies along the latent time axis. The noise is `[T, C]`
time-major; frame `f` is slerped on its own (per-frame vector across channels) at
`w(f) = seed_blend + (seed_blend_end - seed_blend) * smoothstep(f / (T-1))`, so a
song can start as one seed and end as the other. API field: `seedBlendEnd`.
Per-frame morph is not the same as whole-item breeding even at a constant blend (each frame is slerped on its own), so when `seed_blend_end` equals `seed_blend` the engine uses the whole-item breeding path.

**`lm_seed`** (int64, `-1` = unset). Seed for the LM sampler. On ace-server `/lm`, an
unset `lm_seed` follows `seed`, and a random value is drawn if both are unset. Batch
item `b` of `/lm` gets `lm_seed + b`.

**`lm_batch_size`** (int, default 1). Number of LM variations. `ace-lm` clamps it to
1..9; ace-server clamps it to 1..`--max-batch`.

**`synth_batch_size`** (int, default 1). DiT variations per request, each with
`seed + i`. The total across a `/synth` batch is clamped to 9.

The rules:

1. Each input request runs as if it were the only one.
2. `seed = -1` resolves to one random value per input request.
3. `lm_batch_size = N` produces N LM outputs; `synth_batch_size = N` produces N DiT
   outputs with consecutive seeds.
4. LM output `b` comes back as a standalone request (`lm_batch_size` 1) carrying
   `seed + b` (from a random base when `seed` is -1), `seed_b + b` when `seed_b` is
   set, and `lm_seed + b`.

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

**`lm_min_p`** (float, default 0 = off). Code phase only: after top-p, drop every code
whose probability is under `lm_min_p` times the top code's.

**`lm_cfg_min_p`**, **`lm_cfg_logprob`**, **`lm_cfg_rescale`** (defaults 0, false, 0).
Opt-in CFG refinements for the code phase (`engine/src/lm-cfg-extras.h`).
`lm_cfg_min_p` blends only codes the conditional pass gives at least that fraction of
its top probability; the rest keep their conditional logit. `lm_cfg_logprob` blends
log-softmax rows instead of raw logits. `lm_cfg_rescale` (0 to 1) pulls the blended
row's spread back toward the conditional row's. All off keeps the plain blend. The
speculative (draft model) path ignores `lm_min_p` and these three and logs a warning,
as it does for the repetition penalty.

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
`inference_steps` and `shift`, and a scheduler plugin. The last value is the
endpoint, so N values give N-1 steps. On ace-server the sideband copy of this field
wins, and it drops the last value only when that value is `0`, so end the list with
`0`.

**`dit_sliding_window`** (int, default -1). Window override, in tokens (12.5 per
second), for the DiT's sliding-window attention layers. `-1` keeps the model's value,
`0` makes every layer full attention, a positive value sets the window. Larger than
the trained window is out of distribution.

**`latent_shift`**, **`latent_rescale`** (float, defaults 0.0 and 1.0). Applied to the
DiT output before VAE decode: `latent * latent_rescale + latent_shift`.

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
`cover_noise_strength`. `""` truncates the early steps. `"rescale"` runs the same
number of steps as truncate, rebuilding the selected scheduler's schedule across the
reduced range. With `custom_timesteps` set, `"rescale"` truncates.

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
alignment and produces LRC lyric timestamps. ace-server returns the first track's
LRC, base64-encoded, in the `X-LRC-Text` header of the `/job` result.

**`postprocess_plugin`** (string, default `""`). A Lua postprocess plugin to use for
VAE decode instead of the built-in tiled decoder. An unknown name, or a plugin whose
`process()` fails, falls back to the built-in decoder.

**`pp_vae_reencode`** (bool, default false). Round-trips the audio through the
post-processing VAE when a PP-VAE model is installed.

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
| `vae_model` | `""` | VAE from the registry. Empty takes the first non-ONNX VAE. A name ending in `.onnx` is a 400: ONNX decoding was removed |
| `emb_model` | `""` | Text encoder from the registry |
| `infer_method` | `"euler"` | Solver plugin name. Unknown names fall back to `euler` |
| `scheduler` | `""` | Scheduler plugin name, or `composite:<a>+<b>:<crossover>:<split>` to blend two. Empty uses the shift schedule; an unknown name uses `linear` |
| `guidance_mode` | `"apg"` | Guidance plugin name. Unknown names fall back to `apg` |
| `plugin_params` | `{}` | Plugin parameters, `{"pluginName:key": value}`. Values may be strings, numbers or booleans |
| `apg_momentum`, `apg_norm_threshold` | 0.75, 2.5 | APG tuning |
| `stork_substeps`, `beat_stability`, `frequency_damping`, `temporal_smoothing` | 10, 0.25, 0.4, 0.13 | Copied into the solver state |
| `cfg_cutoff_ratio` | 1.0 | Fraction of DiT steps that run CFG |
| `cache_ratio` | 0.0 | Fraction of middle steps that reuse the previous velocity instead of a forward pass |
| `custom_timesteps` | `""` | Same as the request field; `ops_build_schedule` prefers this copy |
| `dcw_enabled`, `dcw_mode`, `dcw_scaler`, `dcw_high_scaler` | false, `"low"`, 0.1, 0.0 | Wavelet-domain sampler correction. `dcw_mode` is `low`, `high`, `double` or `pix` |
| `latent_shift`, `latent_rescale` | 0.0, 1.0 | Same as the request fields |
| `denoise_strength`, `denoise_smoothing`, `denoise_mix` | 0.0, 0.7, 0.25 | Post-VAE spectral denoiser. Uses the `--noise-profile` noise profile when one was loaded |
| `adapter_mode` | `"merge"` | `merge`, `runtime` or `runtime_lowrank` |
| `adapter_runtime_quant` | `"bf16"` | Runtime delta precision: `bf16`, `q8_0` or `q4_0` (`q4_k` is read as `q4_0`). A tensor whose input width is not a multiple of 32 stays BF16 |
| `adapter_runtime_dora` | false | Runtime modes: DoRA magnitude rescaling for DoRA adapters, including runtime paths forced by adapter windows or sections. Merge mode is unchanged. A non-boolean value is ignored. The Node field is `adapterRuntimeDora`, sent only when true |
| `adapter_merge_lowvram` | false | Merge mode: requantise merged weights to the base type instead of F32 |
| `adapter_group_scales` | all 1.0 | Object with `self_attn`, `cross_attn`, `mlp`, `cond_embed`, `time_embed`, `proj_in` |
| `adapter_section_align_at` | 0.55 | Fraction of steps before per-section masks are rebuilt from cross-attention alignment. `<= 0` keeps the proportional map |
| `adapter_section_isolation` | 0.0 | 0 to 1 penalty on self-attention across section boundaries |
| `rebase_source`, `rebase_beta` | `""`, 0.0 | Nudge adapted weights toward the adapter's training base: a DiT name from the registry, and the strength. Merge and runtime modes; skipped with per-section masking |
| `concepts` | `[]` | Concept steering vectors: `{"name", "path", "target": "dit" or "lm", "alpha", "layers": [...]}` |
| `concept_extract` | none | `{"out", "name", "positive", "negative", "target_class", "pairs", "null_ref", "top_k"}`. Turns the job into a paired harvest that writes a concept GGUF instead of audio. Needs `out`, `positive` and `negative` (each a string or an array of strings); `pairs` defaults to 24, `top_k` to 4 |
| `seed_strength` | 0.0 | Repaint only: bias the region's initial noise toward the `seed_latents` multipart part |
| `evict_lm` | false | Free the LM before loading the synth pipeline |
| `vae_chunk` | 0 | Per-request VAE tile size (`0` keeps the loaded value) |
| `batch_cfg` | -1 | `0` splits CFG into two forwards, `1` batches them, `-1` keeps the default |
| `stream_mode` | false | Removed with the ONNX streaming pipeline. `true` (or `"true"`, `"1"`) is a 400 |

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
| `lm_adapter_scale`, `lm_adapter_scale_attn`, `lm_adapter_scale_mlp`, `lm_adapter_scale_early`, `lm_adapter_scale_mid`, `lm_adapter_scale_late` | 1.0 each | Adapter scales, -4 to 4: overall, by module group, and by depth third |
| `lm_soft_off` | false | Run the adapter's weight delta without its artist token or KV prefix |
| `adapter_runtime_dora` | false | The same key ACE reads, here for the LM adapter: true recomputes a DoRA adapter's column norms at the dialled scale instead of its base scale |

Out-of-range planner sampling values are clamped, not refused. The ranges are
`lm_temperature` 0.05 to 4, `lm_top_k` 0 to 16385, `lm_top_p` 0 to 1,
`lm_rep_penalty` 1 to 2, `lm_rep_window` 0 to 4000, `lm_dry_base` 1.05 to 4 and
`lm_dry_min_len` 1 to 200.
The prompt is tokenized before the job is queued, so a prompt over the 5000-token
limit is a 400 that names the count. The response carries `id` (also as `job_id`),
the resolved `seed` (and `seed_str`, the same value as a decimal string, because a
64-bit seed does not survive a JavaScript number), the clamped `takes`, `ar_seed`
and the resolved sampling settings.

### YuE2 request (`POST /yue2/synth`)

Parsed by `yue2_parse_request` in `engine/src/yue2/yue2-request.h`.

| Field | Default | Meaning |
|---|---|---|
| `style` | required | Style tags |
| `lyrics` | `""` | Lyrics |
| `cot` | `"off"` | `off`, `melody` or `full`. `melody` and `full` run the ABC plan stage |
| `abc` | none | ABC text to use instead of running the plan stage |
| `plan_only` | false | Stop after the plan stage and return the ABC score. Needs `cot` other than `off`; cannot be combined with `abc` or `semantic_only` |
| `semantic_only` | false | Stop after the semantic stage and return the codec ids |
| `semantic_retries` | 0 | 0 to 20. Redraws per song whose semantic stage runs away, up to 4 side by side; the first to end cleanly wins |
| `plan_cutoff` | 1.5 | 0 to 10. A stream past this x its plan's length + 20 s is a runaway (`limit_hit`); 0 = only the stage cap |
| `seed` | random | Song seed, a non-negative integer or a decimal string (for values past 2^53). Song i of a batch uses `seed + i` |
| `noise_seed` | `seed` | NAR noise seed, same encoding. Variation j uses `noise_seed + j` |
| `lm_batch_size` | 1 | Songs per request, 1 to 4 |
| `songs` | none | A batch of different songs: an array of 1 to 4 objects (up to 8 with `plan_only`), each with its own `style` (required), `lyrics`, `abc`, `seed` and `noise_seed`. Everything else comes from the request. Sets the batch size and cannot be combined with `lm_batch_size` above 1. Song i without a `seed` uses the request seed plus i. Tracks come back song-major in array order |
| `synth_batch_size` | 1 | Noise variations per song, 1 to 9 |
| `plan_<field>`, `semantic_<field>` | checkpoint | Per-stage sampler overrides. `<field>` is `temperature`, `top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `min_tokens`, `max_tokens` or `min_p`. `min_p` (0 to 1, default 0 = off) is not a checkpoint field: after top-p it drops every token under that fraction of the top token's probability |
| `cfg_min_p`, `cfg_logprob`, `cfg_rescale` | 0, false, 0 | Opt-in refinements of the semantic stage's logit CFG, as `lm_cfg_*` above. All off keeps the reference BF16 blend |
| `cfg_scale` | 1.01 with `cot: "off"`, else 1.0 | NAR CFG |
| `ode_steps` | checkpoint | NAR ODE steps |
| `ode_method` | `"midpoint"` | The only method implemented; any other value is a 400 |
| `infer_method`, `scheduler`, `plugin_params` | `""`, `""`, `{}` | Lua solver and scheduler for the NAR stage. Empty keeps native midpoint and the uniform grid. Only an allowlist runs: solvers `md_wasserstein_yue2`, `md_ancestral_yue2`, `md_latent_eq_yue2` and `md_tlie_yue2`; schedulers `md_ht_scheduler V3`, `md_plan_aware_yue2`, `md_plan_loose_yue2` and `md_plan_overdrive_yue2`. Any other name is a 400. `plugin_params` values must be strings, numbers or booleans |
| `postprocess_plugin` | `""` | Lua postprocess plugin for the VAE decode; empty, unknown or failing keeps the native decode |
| `lm_adapter`, `lm_adapter_scale`, `lm_type` | absent | The job's own model pick, with the [`/yue2/select-model`](#endpoints) rules, applied when the job starts. Absent keeps the current pick |
| `lyric_schedule` | off | Hides lyric sections from the semantic stage until their time on the score clock: `{"mode": "bias" or "mask", "bias", "abc", "lead_sec", "behind", "sections": [{"start_sec", "lyric": [c0, c1], "abc": [[c0, c1], ...]}], "blind": [{"from_sec", "to_sec", "lyric", "abc"}]}`. Spans are codepoint ranges into the request's `lyrics` and `abc`. Mode `bias` adds `bias` (-60 to below 0, required) to the hidden prompt rows' attention; mode `mask` hides them completely. `abc: true` also hides each section's score lines, `lead_sec` (0 to 30) reveals a section that long before its first note, and `behind` (-1 to 256, -1 = never) also hides sections more than that many before the current one. A blind window hides its spans while the composer writes frames from `from_sec` up to `to_sec`. Up to 256 sections, in score order, and 16 blind windows; at least one of the two |
| `nar_cache_ratio` | 0.0 | Fraction of middle NAR steps that reuse the last velocity |
| `nar_chunk_frames` | 0 | NAR chunk cap in frames (25 per second). 0 renders a normal song in one chunk |
| `end_threshold` | 0.0 | 0 to 1. Stop the semantic stage when the sampler puts this much probability on the end token |
| `end_bias`, `end_bias_from_sec`, `end_bias_ramp_sec` | 0.0 | Additive end-token bias (-50 to 50), when it starts, and its ramp |
| `guide_path` | none | Source guide file from `ace-train yue2-guide` (absolute path). Turns on source guidance for the semantic stage: per 25 Hz frame, the real-audio tokenizer's top-K codes for a recording |
| `guide_scale` | 0.5 | 0 to 1. Mixes the LM's distribution with the tokenizer's (renormalised over the candidates): `(1 - s) p_LM + s p_tok`. Roughly the share of frames that follow the source |
| `guide_k` | 0 | Candidates used per frame, 0 = every one the file holds |
| `guide_outside` | 0.0 | -1000 to 0. Additive penalty on codes outside the candidates; -100 or below makes the candidates the only legal codes |
| `guide_offset_sec` | 0.0 | Shift the source against the output, -3600 to 3600 s |
| `guide_allow_end` | false | Allow the end token while a guide frame is active |
| `guide_schedule`, `guide_schedule_power` | `"flat"`, 1.0 | Scale envelope over the guide: `flat`, `up`, `down`, `sin` or `inv_sin`, raised to the power |
| `guide_lfo`, `guide_lfo_period_sec`, `guide_lfo_depth`, `guide_lfo_phase` | `"off"`, 4.0, 0.0, 0.0 | Periodic scale modulation: `off`, `sine`, `triangle`, `saw` or `square`. Depth 1 swings the scale between 0 and its value |
| `codec_ids` | none | Semantic codes (0 to 32767) to render instead of sampling. Without continuation, single song and single variation only |
| `continue_codec_ids` | false | Treat `codec_ids` as a shared prefix for 1 to 4 songs and keep sampling after it, at any `cfg_scale`. No `semantic_retries` |
| `preview_max_frames` | 0 | Bound on semantic frames, 0 to 9000 (training previews) |
| `render_from_frame` | 0 | 0 to 9000. The NAR and VAE start at this codec frame (clamped to the stream), so the audio covers only the frames from it to the end. The composer still reads every frame. Section takes send their lead-in |
| `vae_variant` | `"standard"` | `standard`, `legacy`, or `dec2x` (a local fine-tuned decoder; only when `yue2-vae-dec2x-*.gguf` is present) |
| `noise_source`, `noise_fixture_path` | `"native"` | Validation only: `fixture` reads NAR noise from a raw f32 file |
| `id` | `"yue2"` | Request label |

Progress and results for `/yue2/synth` use the shared `GET /job`.

For resumed batches, send `songs: [{style, lyrics, abc, seed}, ...]` with
`codec_ids`, `continue_codec_ids: true` and optionally `preview_max_frames`.
Each song needs its own explicit seed to identify its take; its `noise_seed`
defaults to that seed. Cover Studio uses one synthesis variation per song and
submits up to four songs at a time, so ten section takes use chunks of 4, 4 and 2.
The prefix, frame caps, repetition history and CFG state apply to every row.
Distinct prompts prefill separately; matching prompts copy the prefix KV rows
and logits. A lyric schedule replays the prefix in a batched decode.

Completed job status includes `tracks`, with one entry per audio part in part
order. Each entry has `song`, `variation`, `seed`, `noise_seed`, `frames`,
`audio_from` (the codec frame its audio starts at, 0 unless `render_from_frame` was set),
`end_reason` and `semantic_ids`. The audio result is a WAV for one track or
`multipart/mixed` for several tracks. Pair each part with its metadata, then
map by `song` and `seed`: a dropped song leaves a gap in the song indices.
Section takes reject the entire chunk if an expected track is missing or its
seed does not match. A row failure, including a prefix mismatch, fails the
whole engine job with one error. Seeds reproduce only at the same batch width;
song 0 at batch 4 can differ from the same seed at batch 1.

Under `--keep-loaded` the request's render half (NAR and VAE) runs on a second thread with
its own backend instance (`yue2-job.h`, the NAR lane) once the AR half has composed and
sealed the song, so the work thread can compose the next request meanwhile. The job's phase
reads `nar` from the handoff. The lane is skipped, and the render runs inline, when
`YUE2_NO_OVERLAP` is set, under ConvRot, with a Lua NAR solver or scheduler, or with less
than `YUE2_OVERLAP_MIN_FREE_GB` (default 4) of free VRAM. Warm, unload and select-model wait
for a render in flight before touching weights.

### Stable Audio 3 request (`POST /sa3-synth`)

`/sa3-synth` reads its parameters from the query string (or form fields), not a JSON
body. It runs synchronously on the GPU worker and returns a WAV. `/sa3-refine` shares
the plugin and adapter parameters and adds its refine-only ones, which are documented
above its handler in `hot-step-server.cpp`.

| Parameter | Default | Meaning |
|---|---|---|
| `tokens`, `n_tokens` | required | The prompt as comma-separated T5Gemma token ids (the Node server tokenizes) and the count of real tokens. A missing or zero `n_tokens` is a 400 |
| `duration` | 30 | Seconds, at least 1; anything above 600 is cut to 600 |
| `steps` | 8 | 1 to 64 |
| `sampler` | `pingpong` | `pingpong` or `euler` |
| `seed` | from the clock | Unsigned 64-bit seed |
| `seed_b`, `seed_blend` | -1, 0 | Seed breeding, below |
| `synth_model` | `sa3-dit-BF16.gguf` | Generation DiT, below |
| `adapters` | none | `name:strength,name:strength`. Each needs `<models>/sa3-adapters/<name>.gguf`; a missing file or a name with a path in it is a 400 |
| `snapshot_run`, `snapshot_step`, `snapshot_scale` | none, none, 1 | Checkpoint previews, below |
| `solver`, `scheduler`, `guidance_mode`, `guidance_scale`, `plugin_params` | none, none, none, 1, none | Lua plugins. The solver parameter is `solver` here, where `/synth` uses `infer_method`; `plugin_params` is a JSON object string. `guidance_scale` is clamped to 1 to 20, and guidance runs only above 1. A full-loop solver needs `sampler=euler` |
| `postprocess_plugin` | none | Lua postprocess plugin for the decode |
| `sa3_lua_renoise` | 1 | `0` turns off the pingpong re-noise the engine wraps around a per-step Lua solver |
| `out_sr`, `out_fmt` | 44100, `s16` | Output rate (8000 to 192000) and WAV sample format |
| `backend` | auto | `gguf` or auto; `onnx` is a 400. Without `sa3-dit-BF16.gguf` the request is a 501 |
| `debug_zero_noise`, `dump_latent` | off | Validation only: zero noise, and a bare file name under `<models>/sa3-latents/` for the final latent |

Seed breeding: the whole pure-noise latent is slerped from the `seed` noise toward
the `seed_b` noise (same std::mt19937_64 gaussian at `seed_b`); `seed_blend` 0 gives
`seed` exactly, 1 gives `seed_b`. `seed_b` -1 is off. No effect on `/sa3-refine`.

Generation DiT: `synth_model` is a bare filename in the models root (the same field
name ACE uses for its DiT), for example `sa3-dit-Q8_0.gguf`. An empty or absent
value uses `sa3-dit-BF16.gguf`; so does a value that does not match
`sa3-dit-*.gguf` or names a file that is not installed, with the reason in the log.
A name containing `/`, `\` or `..` is a 400, and so is the training base
`sa3-dit-base-*` in any letter case. The `sa3-dit-*.gguf` match is case-sensitive,
so a differently cased name falls back to BF16 instead of caching the same file
under a second key. The response header `X-SA3-DiT` names the DiT that ran. The
store keys the SA3 model on the DiT path, so each quant caches as its own entry.
`/sa3-refine` takes no DiT parameter and always loads `sa3-dit-BF16.gguf`.

Checkpoint previews: `snapshot_run` (`<dataset>/<adapter>/<stamp>`), `snapshot_step`
(positive decimal) and `snapshot_scale` (finite float, default 1) load the captured
training export `snapshots/step-N.gguf` under the SA3 training runs root as an extra
adapter. Bad charsets, a file outside the root, a non-`dora-rows-v1` GGUF, an
inference base that differs from the loaded DiT or an unparseable scale return 400.
The public adapter names and `/sa3/props` are unaffected. Details:
[engine reference](../../docs/dev/engine.md#stable-audio-3-sampling).

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
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split DiT CFG into two separate forwards
  --clamp-fp16            Clamp hidden states to FP16 range
  --dump <dir>            Dump intermediate tensors
  --check                 Check output for NaN/Inf, silence, DC and clipping;
                          exit 2 if any track fails
```

`--check` measures each track's raw float buffer before it is encoded (the WAV and
MP3 writers replace NaN/Inf with 0 and clamp to full scale, which would hide a
broken render) and prints one `[Check BatchN] PASS|FAIL` line with the values. A
track fails on any non-finite or subnormal sample, a peak below -60 dBFS, a per-channel mean
above 0.02, or more than 0.1% of samples past full scale. The env var
`HOTSTEP_CHECK_INJECT` (a comma list, one of `none`, `nan`, `subnormal`, `silence`,
`dc`, `clip` per track) corrupts tracks before the check so the checks themselves can be tested. ace-synth exits 1 when a
track produces no audio or its file fails to write, and 2 when a `--check` fails.
Setting `HOTSTEP_BCAST_TEST` or `HOTSTEP_KRON_TEST` makes ace-synth run a
model-free backend self-test (the per-section mask broadcast, or the LoKr
Kronecker apply) and exit with its result instead.

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
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --max-seq <N>           KV cache size (default: 8192)
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --dump <dir>            Dump tok_latents + tok_codes (skip LM)
```

The DiT file supplies the FSQ tokenizer weights. The understand defaults only apply
when no request is given: a `--request` file, like the `request` part of
`/understand`, starts from the normal `AceRequest` defaults (temperature 0.85,
top_p 0.9) for any field it leaves out.

## ace-server reference

```
Usage: ace-server --models <dir> [options]

Required:
  --models <dir>          Directory of GGUF model files

Adapter:
  --adapters <dir>        Directory of adapters

Plugins:
  --user-plugins <dir>    Extra plugin root, scanned after the shipped plugins

Memory control:
  --keep-loaded           Keep models in VRAM between requests
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

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
  --backend-check         Load GGML backends without models, print a JSON report, exit
```

Notes the usage text does not cover:

- `--backend-check` needs no `--models`. Its report lists each registered backend
  with its devices, the device the engine would pick, and every `ggml-*.dll` beside
  the exe with whether it registered; on Windows a DLL that did not register is
  named with its load error and the first import that could not be loaded. The
  release smoke test (`release/smoke-artifact.mjs`) decides pass or fail from it.
- `--noise-profile <wav>` is also accepted. It loads a noise profile for the
  spectral denoiser (`denoise_strength`).
- `--draft-lm` auto-discovery is disabled in the code; only an explicit path enables
  speculative decoding.
- `--max-batch` is clamped to 1..9.
- `--models` is scanned at startup, and its `onnx/` subfolder too. `--adapters` is
  scanned for DiT adapters and its `lm/` subfolder for planner-LM adapters.
- Lua plugins are scanned once at startup from `engine/plugins/` and then the
  project root's `plugins/`, each with `solvers/`, `schedulers/`, `guidance/` and
  `postprocess/` subfolders. `--user-plugins <dir>` is a third root with the same
  subfolders, scanned last; a user file that declares a shipped plugin's name
  replaces it, and any other duplicate keeps the first one loaded. The app passes
  `<data folder>/plugins` whenever it runs with a data folder, which a release
  always does; a dev checkout without a data folder passes nothing.
- The app starts ace-server on port 8085.

| Pipeline | Needs | Enables |
|---|---|---|
| LM | an LM GGUF | `/lm` |
| Synth | DiT, text encoder and VAE | `/synth`, `/warm` |
| Understand | LM, DiT and VAE | `/understand` |
| VAE | a VAE | `/vae` |
| Codes decode | DiT and VAE | `/codes-decode` |

Endpoints whose pipeline has no models return 501. With only MM3, YuE2 or Stable
Audio 3 weights present, the server still starts and serves those routes.

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
                                 "adapter_scale"}; adapter is a registry name.
                                 A no-op unless models are kept loaded: the job
                                 result is then {"warm":false,"reason"}.

GET  /job?id=N                   Status: {"status", "phase", "phase_step",
                                 "phase_total", "adapter_progress",
                                 "songs_done"}. status is running, done, failed
                                 or cancelled. songs_done is a YuE2 batch's bit
                                 mask of songs that finished the running stage,
                                 0 otherwise. YuE2 jobs add end_reason,
                                 stage_end_reasons, tracks and abc.
GET  /job?id=N&result=1          Result. /lm and /understand: JSON array of
                                 AceRequest. /synth: one audio file, or
                                 multipart/mixed for a batch; the first track's
                                 LRC, base64-encoded, in the X-LRC-Text header
                                 when get_lrc was set.
GET  /job?id=N&latent=1          /synth: the first track's post-DiT latents, raw
                                 f32 [T*64].
POST /job?id=N&cancel=1          Cancel a job.
POST /job?id=N&drop_songs=MASK   YuE2 batch: take songs out (bit b = song b);
                                 the rest carry on. {"songs_dropped":MASK}.
                                 Tracks keep their batch song index.
GET  /jobs                       Every job in the table.

GET  /health                     {"status":"ok"}
GET  /worker-drain               {"status":"ok"} once everything queued on the
                                 worker before it has finished.
GET  /props                      version, models (lm, embedding, dit, vae),
                                 adapters, lm_adapters, cli (max_batch,
                                 mp3_bitrate), default (full AceRequest), presets
                                 (turbo, sft).
GET  /logs                       Server stderr as server-sent events, starting
                                 with up to the last 512 lines.
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

Post-processing and Stable Audio 3 endpoints. `/pp-vae-reencode`, `/sa3-refine` and
`/sa3-synth` answer synchronously, but their work runs on the GPU worker queue
behind any job already queued. `/spectral-lifter` runs on the HTTP thread. SuperSep
queues on the same worker and keeps its own job table, which has no eviction:
release a job when you are done with it. The `/sa3-synth` parameters are in
[Stable Audio 3 request](#stable-audio-3-request-post-sa3-synth).

```
POST /pp-vae-reencode[?blend=0..1][&backend=gguf][&out_fmt=s16|s24|f32]
                                 Body: WAV. PP-VAE round trip on GGML. blend 0 is
                                 fully re-encoded, 1 is the original.
                                 backend=onnx is retired and returns 400.
POST /sa3-refine?...             Body: WAV or MP3. Stable Audio 3 refine. Query
                                 parameters are documented above the handler in
                                 hot-step-server.cpp (tokens, n_tokens, strength,
                                 steps, sampler, seed, rms_match, env_match, mix,
                                 band_blend, band_freq, band_width, out_sr,
                                 backend, adapters, solver, scheduler,
                                 guidance_mode, guidance_scale, plugin_params,
                                 postprocess_plugin, sa3_lua_renoise,
                                 debug_zero_noise). backend is gguf or auto (both
                                 GGML); backend=onnx returns 400.
GET  /sa3/props                  SA3 availability (sa3-dit-BF16.gguf and the
                                 tokenizer), sample rate, prompt token limit and
                                 the adapters in <models>/sa3-adapters/.
POST /sa3-synth?...              Stable Audio 3 text to audio; returns WAV and an
                                 X-SA3-DiT header.
POST /supersep/separate?level=0..5
                                 Body: audio. Starts stem separation, returns {"id"}.
                                 Level 3 is retired and runs as 2.
GET  /supersep/progress?id=      Progress.
GET  /supersep/result?id=        Stem list.
GET  /supersep/serve?id=&stem=N  One stem as WAV.
POST /supersep/release?id=       Drop a job and free its stems.
POST /supersep/recombine         Body {"id", "stems": [{"index", "volume",
                                 "muted"}]}. Returns the mix as WAV.
POST /spectral-lifter?...        Body: WAV. Query: denoise_strength, noise_floor,
                                 hf_mix, transient_boost, shimmer_reduction.
```

`/pp-vae-reencode`, `/sa3-refine`, `/sa3-synth`, `/supersep/serve`,
`/supersep/recombine` and `/spectral-lifter` take `?out_fmt=s16|s24|f32` for the
returned WAV; the default is `s16`.

MiniMax-Music3 and YuE2 endpoints. Request fields for the two synth routes are in
[MiniMax-Music3 request](#minimax-music3-request-post-mm3synth) and
[YuE2 request](#yue2-request-post-yue2synth).

```
GET  /mm3/props                  MM3 files, config, what is loaded.
POST /mm3/warm, /mm3/unload      Load or free MM3 weights.
POST /mm3/select-model           Body {"lm", "depth", "cond", "dit", "voc"} quant
                                 tokens ("" = best available). A legacy "synth"
                                 key sets the four non-LM roles at once.
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
POST /yue2/align                 multipart: audio + lyrics. Forced alignment, run
                                 on the GPU worker.
```

Errors are JSON, `{"error":"message"}`, with a 4xx or 5xx status.

### Concurrency

One worker thread runs every GPU job in FIFO order: ACE-Step, MM3 and YuE2 jobs,
SuperSep separations, and the synchronous worker routes (`/pp-vae-reencode`,
`/sa3-refine`, `/sa3-synth`, `/yue2/align`, `/worker-drain`). The other `/mm3/*`
and `/yue2/*` routes, warm, unload and imatrix included, run on the HTTP thread. The
job table holds 32 entries; finished jobs are evicted oldest first and running
jobs never are. Each job has its own cancel flag, and a cancel during adapter
precompute takes effect between deltas.

`/sa3-synth` and `/sa3-refine` have no job id. They cancel when the client closes
the connection: both sampling loops check before every step, and a full-loop Lua
solver sees it as `on_step` returning true. A client that drops one of these calls
awaits `GET /worker-drain` to know the engine has stopped. Limits: a Lua `sample()`
that ignores `on_step`'s return runs to its last step, and model load, encode and
decode are not checked, so the drain waits for them.

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
int8 values); a NAC4 frame is 34 bytes (an f16 scale and 32 bytes of packed 4-bit
values). `--input` and `--output` are accepted as long forms of `-i` and `-o`.

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

## quantize reference

GGUF requantizer for ACE-Step, MiniMax-Music3 and YuE2 files. It reads a BF16 or
F16 GGUF and writes the quantized file one tensor at a time.

```
Usage: quantize <input.gguf> <output.gguf> <type> [--imatrix <file.gguf>]
Types: Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0 NVFP4 MXFP4
       IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ3_XXS IQ4_XS
```

The type name is matched case-insensitively, and `-m` is a short form of
`--imatrix`. Most types raise the important tensors (`v_proj`, `down_proj` and their
equivalents) to a higher type in some or all layers; `Q3_K_S`, `Q5_K_S`, `Q6_K`,
`Q8_0`, `NVFP4`, `MXFP4` and `IQ4_XS` do not. The embedding has its own type:
`Q6_K` under the K-quants and `IQ4_XS`, `Q8_0` under `Q8_0`, `NVFP4` and `MXFP4`,
and `Q2_K` to `Q5_K` under the other i-quants. MM3 files follow the converter's
choice: F16 tensors are quantized and F32 tensors are left alone. `IQ1_S`, `IQ2_XXS` and
`IQ2_XS` cannot be built without an importance matrix and stop before reading the
input. Collect one with `POST /mm3/imatrix` or `POST /yue2/imatrix`; the file
format is described in `engine/src/minimax/mm3-imatrix.h`.

## ace-train reference

`ace-train <subcommand> [options]` runs one training job per subcommand; the list is
in [the engine guide](../../docs/dev/engine.md#training-subcommands), and
`ace-train --help` prints the full option list. With `--jsonl` a trainer writes
machine-readable events on stdout and human logs on stderr. The sections below give
the flags of a few subcommands in detail.

`ace-train mm3-train-dit --dora` trains the MM3 flow-DiT adapter as DoRA instead of
LoRA (`MM3TrainArgs.dora`). It takes the same `--cache`, `--models` and `--out`
arguments as LoRA and needs the outputs of both `mm3-preprocess` and
`mm3-condition`. This trainer is CLI-only: it is not an app training job or a DiT
adapter capability.

## Stable Audio 3 training CLI

```text
ace-train sa3-train --models <models> --base-dit <medium-base.gguf> --inference-dit <medium.gguf> --manifest <train-manifest.json> --out <new-run> --steps 1000 --rank 16 --alpha 16 --lr 0.0001 --duration 12 --seed 42 --checkpoint-every 100 --export-every 0 --bwd mm --jsonl
ace-train sa3-train-check --models <models> --base-dit <medium-base.gguf> --fixtures <reference-directory>
```

`--manifest`, `--models`, `--base-dit` and `--out` are required for `sa3-train`;
`--fixtures`, `--base-dit` and `--models` for `sa3-train-check`. `--duration` sets
the maximum crop in seconds; the 12-second example is a smoke configuration, while
the trainer and server default is 380 seconds. `--cache` overrides the pre-encoding
cache, otherwise `<out>/cache`. `--inference-dit` defaults to
`<models>/sa3-dit-BF16.gguf`. `--resume` reads `<out>/state.gguf` and checks the
saved run fingerprint; without `--resume`, an `--out` that already holds
`state.gguf` is refused. Total `--steps` includes completed updates.
`--checkpoint-every` sets the state-save interval; `--export-every 0` disables
periodic inference snapshots. Final raw, state and inference files are separate. The server's checkpoint previews run the
trainer in segments: an absolute `--steps` bound, then `--resume` with a larger one in the
same `--out`, with `--export-every 0`; `<out>/state.gguf` records the completed step.

`--bwd mm` is the default backward path; `--bwd outprod` selects OUT_PROD.
`--tf32 on|off` defaults off. The standalone `--train-seconds` option also trains
the seconds embedder, but the server deliberately excludes it from published
adapters because that adapter site has no inference loading contract.
Writing `<out>/STOP` requests a checkpoint stop at the next training step.
`--jsonl` emits start, resume, data, progress, log, step, checkpoint, export,
stopped, done and error events (the `type` field). A stopped run is resumable when
state exists; it is not a completed publication. See [training internals](../../docs/dev/training-internals.md#stable-audio-3-medium-training).

## ace-train YuE2 preparation reference

`ace-train yue2-prepare-aitk` builds native joint-training caches from a Legacy
manifest. See [training internals](../../docs/dev/training-internals.md) for the
full preparation command and workflow.

```
ace-train yue2-prepare-aitk --legacy-manifest FILE --checkpoint FILE
    --tokenizer GGUF_OR_DIR --output NEW_DIR --model vae=FILE
    --model semantic=FILE --model sheetsage=FILE [--trigger WORD]
    [--caption-dropout-trigger TEXT] [--lyric-timing 0|1]
```

| Flag | Default | Meaning |
|---|---|---|
| `--trigger <text>` | empty | Inject a trigger into full captions. With a non-empty caption, the style becomes `<text>, in the style of <text>. <caption>`; with an empty caption, it is just `<text>`. One line, at most 128 bytes |
| `--caption-dropout-trigger <text>` | `--trigger`, or empty | Exact style text for the caption-dropout prefixes, with no wrapper. One line, at most 128 bytes; CR, LF and NUL are rejected |
| `--lyric-timing 0\|1` | 0 | Attach the word timings the AR lyric-cursor loss trains on. With 1, every source with lyrics needs a `cursor_words` file. The server passes 1 unless the preparation turns lyric timing off, which the Fixed recipe requires |

The fallback dropout style is also bare trigger text. Preparation tokenizes the
dropout prefixes with the same lyrics and sheet conditioning as the full-caption
prefixes. `ace-train yue2-joint-train --caption-dropout <p>` selects them during
training; `--caption-dropout-trigger` belongs on the preparation command.

HOT-Approved, the Fixed recipe preset, passes `--caption-dropout-trigger` with
the trigger, or the dataset slug if no trigger is set. Older prepared datasets
need re-preparing; the server invalidates the old Fixed recipe preparation cache.

## ace-train YuE2 hold-out evaluation reference

These flags belong to `ace-train yue2-joint-train`. See
[training internals](../../docs/dev/training-internals.md#yue2-preparation-operations)
for the evaluation output and sampling procedure.

| Flag | Default | Meaning |
|---|---|---|
| `--holdout-items I,J,...` | none | Exclude distinct zero-based prepared-dataset indices from training; at least one item must remain. With adapted evaluation and no separate evaluation dataset, score only these items |
| `--eval-adapted --resume <optimizer.resume>` | off | Restore a checkpoint and compare base versus adapted planner CE and decoder MSE without updating weights. Pass the run's training flags and total `--steps` |
| `--eval-dataset <dataset.json>` | training dataset or its hold-out split | Score every item of this separate prepared dataset; takes precedence over the in-dataset split for evaluation |
| `--eval-nar-window N` | `--nar-crop-frames` | Adapted-evaluation window in frames; `0` means the whole song |
| `--eval-seed N` | `--seed` | Unsigned seed for adapted-evaluation draws and planner-gate readings |
| `--planner-gate-every N` | 0 (off) | Score the planner on `--holdout-items` every N updates; a plateau freezes it. Needs `--holdout-items` and `--nar-extra-steps`; refuses `--eval-dataset`, `--lr-schedule wsd` and the scoring modes |
| `--planner-gate-min-epochs N` | 5 | Training epochs (song draws over training songs only) before the planner may freeze |
| `--planner-gate-patience N` | 3 | Readings in a row not below the best by the epsilon before it freezes; at least 1 |
| `--planner-gate-epsilon F` | 0.001 | Absolute CE a reading must beat the best by |

Resume records bind the hold-out split as a set. Supply the same indices on
training and evaluation resumes. The saved order must contain every non-held-out
item exactly once and its cursor must index that order. A legacy record with no
saved split can resume only without a hold-out; a new split requires a new run.
The summary reports `held_out:true` when using a saved split or a separate
evaluation dataset, and `false` when scoring the unsplit training dataset.

The planner gate keeps its settings, best, best step and stale count in the
resume record. A resume must pass the same gate flags and `--eval-seed`
(otherwise "resume planner gate mismatch"), frozen or not. Each reading prints a
`planner_gate` line and a freeze prints `planner_frozen` with reason `kl`,
`planner_ce` or `manual`. See
[training internals](../../docs/dev/training-internals.md#held-out-planner-gate).
These flags do not add a training early-stop rule.

## fattn-train-test reference

Parity harness for `GGML_OP_FLASH_ATTN_TRAIN{,_BACK}`: compares the fused op
against the autodiff'd reference attention chain. See
[flash-attn-training skill](../../.claude/skills/flash-attn-training/SKILL.md)
for the full ops/patch background.

```
Usage: fattn-train-test [--backend cpu|cuda|vulkan|metal] [--prec f32|tf32|bf16]
                         [--extra] [--large] [--cheap] [--quick] [--fwd-only]
                         [--threads N] [--bench] [--bench-iters N]
                         [--bench-tr] [--bench-lm] [--bench-yue2]
                         [--dsw-check] [--saved-check]

  --backend   cpu (default), cuda, vulkan or metal. Resolved by registry
              device name, never by the first GPU device found, so a CUDA
              request cannot land on Vulkan or the reverse. A requested
              backend that is not loaded is a hard failure (it prints the
              loaded devices), never a silent CPU fallback.
  --prec      f32 (default), tf32 or bf16. tf32 and bf16 need --backend
              cuda; asking for them anywhere else is an error, not an
              ignore.
  --bench     Times fused vs manual attention on CUDA (the tf32 arm is a
              CUDA tensor-core question). Defaults to CUDA with no
              --backend; --backend metal runs the Metal checks; any other
              explicit --backend is rejected.
```

`--bench-lm` times the ACE-Step LM trainer's attention geometry instead of the
DiT's (`--lm-S`, `--lm-small`, `--lm-nh`, `--lm-nkv`, `--lm-gq`, `--lm-layers`,
`--lm-only manual|blocked|fused` shape it). `--bench-tr` (`--tr-s`, `--tr-enc`,
`--tr-real`, `--tr-only`, `--tr-fused`), `--bench-yue2` (`--yue2-only`,
`--yue2-scale`, `--yue2-sites`), `--dsw-check` and `--saved-check` are further
timing and Metal checks. The comment block at the top of
`tools/fattn-train-test.cpp` describes the parity grid, `--bench` and `--bench-lm`.
The tool exits 0 only when every parity case passes; the bench modes exit 0 unless
the backend is missing or a compute call fails.

```bash
fattn-train-test --backend cuda
fattn-train-test --backend vulkan --prec f32
fattn-train-test --bench
```

## HeartCodec parity probe

The standalone `heartcodec-probe` uses the GGML CPU backend and S1-style NPY
fixtures. It accepts F16/F32 codec components, with F32 execution of the rounded
F16 weights. Q8 codec loading and GPU execution are not part of this probe.

```text
heartcodec-probe --self-test
heartcodec-probe --mode rvq|velocity|latent|scalar|decode --models DIR --fixtures CASE
  [--index N] [--out DIR] [--trace DIR] [--threads N] [--backend-dir DIR]
  [--atol NUMBER] [--rtol NUMBER]
```

`--index` selects an intermediate, default 0. `--threads` defaults to 8.
`--backend-dir` defaults to the executable's directory. `--out` writes NPY
results and, for scalar/decode, a float32 48 kHz stereo WAV. `--trace` writes
and checks RVQ vectors, every flow velocity, solved latents and scalar outputs
during full decode. Both output directories must be outside the reference case.
C- and Fortran-order little-endian float32, int32 and int64 fixtures are accepted.

Both maximum absolute error and relative L2 must pass. Component defaults are
0.001 and 0.0001. Full waveform defaults are 0.01 and 0.002, below one tenth of
the reference's measured multi-chunk F32-to-F16 waveform drift. Isolated scalar
checks retain the component bounds; scalar outputs from integrated native
latents use waveform bounds because the round-to-ninth quantizer is discontinuous.
The final comparison can be tightened with `--atol` and `--rtol`. Exit 0 means
parity passed, 1 means a comparison failed, and 2 means invalid input or a load
error. Codec dependencies, configuration hashes and supported geometry are
validated before execution.
