# Generation backends

HOT-Step CPP runs three music models, which the app calls backends: ACE-Step 1.5,
MiniMax-Music3 and YuE2. All three run inside the same C++/GGML engine process, so
switching between them changes which model's weights are loaded, not which program is
running. Each one turns a style description and lyrics into a stereo song, but they
build it differently, accept different inputs, and work with different parts of the app.
ACE-Step 1.5 is the default and the only backend that supports every studio. The other
two are text-to-music only.

For which quant and settings give the best output on each backend, see
[Getting higher quality output](quality.md), which has a per-backend cheat sheet. This
page covers what each backend is and what it can do.

## Switching backends

The backend picker is in the global bar at the top of the app, a small button labelled
with the active backend's name. Click it and choose another backend from the list. The picker
appears whenever two or more backends are registered with the server, and is hidden
when there is only one.

- The choice is stored on the server, so it survives a restart. A new install starts on
  ACE-Step 1.5.
- Switching tells the outgoing backend to release its models from VRAM, in the
  background, so two model families do not sit on the GPU together. The next generation
  loads the new backend's models first.
- The global bar reshapes to the active backend. The Models, Adapters, Generation and
  LM / Thinking clusters show that backend's own controls, and controls that do nothing
  on it are hidden. Cover Studio, Repaint, Stem Builder and STORM show a "not supported
  by the active backend" notice that points you back to ACE-Step 1.5.
- If the active backend's model files are not installed, the bar says so and shows a
  Get models button that opens the [Model Manager](studios/model-manager.md).

On YuE2, hovering the picker shows the model's licence notice (see [YuE2](#yue2)).

## ACE-Step 1.5

The original model the app was built around, and the one HOT-Step's engine started as a
port of. A generation runs in three stages:

1. A planner language model (0.6B, 1.7B or 4B) fills in whatever you left blank (BPM,
   key, time signature, duration, language, an enriched caption) and writes a plan of
   audio codes. You can switch this stage off from the LM / Thinking cluster.
2. A diffusion transformer (DiT) renders the audio latents from the plan, the caption
   and the lyrics. It comes in Standard and XL sizes, each in several variants.
3. A VAE decoder turns the latents into 48 kHz stereo audio.

A small text encoder model conditions the DiT on the caption.

What sets it apart from the other two:

- It is the only backend with covers and reference audio (Cover Studio), repaint
  (Repaint), stem generation (Stem Builder) and continuous streaming (STORM).
- BPM, key, time signature and duration are real input fields. Duration is a target the
  planner aims for, from 10 to 600 seconds.
- It runs the full set of Lua [plugins](plugins.md): solvers, schedulers, guidance modes
  and the postprocess plugins that replace the VAE decode.
- It takes DiT adapters (LoRA and LoKr) in the Adapters cluster, with merge and runtime
  modes, per-group strengths and per-section masking, plus planner LM adapters in the
  LM / Thinking cluster. See [Adapters](adapters.md).
- Batches of up to 9 songs per request (the engine caps `--max-batch` at 9).
- Lyric timestamps (LRC) come from the DiT's own attention during the render.

Models: a DiT, a planner LM, the text encoder and a VAE. The Model Manager's ACE-Step
packs bundle these (the packs also include the CUDA runtime files):

| Pack | Contents | Download |
|---|---|---|
| Quick Start | Standard Turbo DiT Q8_0, 4B LM Q8_0 | ~9.7 GB |
| Minimal Setup | Standard Turbo DiT Q4_K_M, 1.7B LM Q8_0 | ~5.8 GB |
| XL Quality | XL Turbo DiT Q8_0, 4B LM Q8_0 (needs about 12 GB VRAM) | ~12.4 GB |
| Blackwell Optimized | XL Turbo DiT MXFP4 for RTX 50-series cards, 4B LM Q8_0 | ~9.8 GB |

Every pack adds the text encoder, the standard VAE, ScragVAE and the PP-VAE. The full
list of files is on the [Models](models.md) page.

## MiniMax-Music3

A native C++/GGML port of MiniMax-Music3. Its pipeline has five model components, run in three steps:

1. An 8B planner language model writes the song frame by frame (25 frames per second)
   and decides where it ends by emitting a stop token.
2. A depth decoder adds the acoustic detail for each planned frame.
3. A condition encoder, a flow-matching DiT and a vocoder render the plan to 44.1 kHz
   stereo in overlapping windows, which are stitched together.

What sets it apart:

- The caption is a Structured Caption with three sections in order: Global Metadata
  (genre, tempo, key, mood, production), Vocal Details (or the lead instrument for an
  instrumental) and Arrangement (a section-by-section timeline). Tempo and key live in
  that prose; there are no BPM or key fields. Lyrics stay out of the caption, and the
  caption and lyrics share a 5,000-token budget. The Compose Caption button under the
  caption box turns a plain-English description into a Structured Caption with no AI
  model involved.
- The song decides its own length. Duration is a ceiling (up to 300 seconds), not a
  target.
- Play While Rendering (Generation cluster, off by default) starts playback a few
  seconds in, while later windows are still being rendered. The full file is saved as
  usual.
- Variations Per Render (1 to 4) renders several different songs from one prompt in a
  single pass, sharing the planner. Each request is otherwise one take.
- Flow Steps (default 30, range 2 to 60) is the main speed control. Below 30, Low-Step
  Compensation reshapes the schedule so short renders keep their low end and stereo
  image.
- The Lua solver, scheduler and guidance plugins can drive its flow DiT, behind the
  Sampler Plugins toggle, which is off by default and marked experimental. Postprocess
  plugins do not apply.
- Planner LM adapters load from the LM Adapter picker in the Adapters cluster, with a
  strength dial and separate Attention and MLP dials. They are applied per generation,
  so changing a dial takes effect on the next render with no reload. Put the adapter's
  trigger word in the caption.
- Lyric timestamps (LRC) come from the planner's attention, at line level.
- The planner's output can be saved and replayed, so you can re-render a plan with
  different flow settings without planning again. By default, changing only flow-stage
  settings reuses the last plan.
- Renderer (in the Models cluster, under the Flow DiT picker) switches the DiT between
  GGML and TensorRT. TensorRT needs the MM3 TensorRT DiT pack plus one builder DLL for
  your GPU, and the first render builds an engine once, which takes a few minutes.

Models: five GGUFs (planner LM, depth decoder, condition encoder, flow DiT, vocoder), each
selectable at its own quantisation in the Models cluster. Picking a quant loads it into
the engine and the choice is remembered across restarts. The packs:

| Pack | Precision | Download |
|---|---|---|
| MiniMax-Music3 (Q4_K_M) | LM and DiT Q4_K_M, depth Q8_0 | ~7.6 GB |
| MiniMax-Music3 (NVFP4) | LM and DiT NVFP4, for RTX 50-series cards | ~7.7 GB |
| MiniMax-Music3 (Q6_K) | LM and DiT Q6_K, depth Q8_0 | ~9.9 GB |
| MiniMax-Music3 (Balanced mix) | LM Q8_0, DiT Q4_K_M, depth Q8_0 | ~12.2 GB |
| MiniMax-Music3 (Q8_0) | Q8_0 throughout, recommended | ~13.4 GB |
| MiniMax-Music3 (F16) | Full precision | ~23.6 GB |

Training an adapter also needs the MiniMax-Music3 Training pack (~2.8 GB) on top of a
generation pack. See [Training MiniMax-Music3 adapters](training/minimax-music3.md).

The model's licence requires the name "MiniMax-Music3" to be shown as-is, which is why
the picker never shortens it.

## YuE2

A native C++/GGML port of YuE2. Its language model has two halves that share no weights:

1. The AR (autoregressive) half composes. It first writes a lead sheet in ABC notation
   (structure, chords, melody), then the song itself as codec frames, 25 per second,
   and decides where the song ends.
2. The NAR half renders those frames into audio latents with a flow-matching solver.

A VAE decoder then produces 48 kHz stereo. Every GGUF generation also applies a small
companion decoder adapter to the NAR half, which ships in each YuE2 pack.

What sets it apart:

- The prompt is freeform: a plain style description plus lyrics. BPM and key are folded
  into the style sentence rather than sent as fields. The language comes from the
  lyrics.
- The song decides its own length, up to six minutes, and there is no duration control.
- Chain of Thought (Generation cluster) sets how much the planner writes before audio:
  Full (the default), Melody only, or Off. Off skips the lead sheet but runs guidance,
  which makes it the slowest mode, not the fastest.
- Preview the score first (off by default) plans the lead sheet only, shows it with
  playback, and lets you continue, re-plan with a new seed, or cancel before any audio
  is rendered. You can also paste your own ABC lead sheet to render from.
- Without the preview, the app plans first anyway and redraws the seed when the lead sheet
  is a runaway, has no vocal line, or hit its cap, up to the attempt count in Settings. If
  every attempt fails, the song renders with Chain of Thought off (straight from the lyric)
  rather than from a broken score, and the generation log says so.
- Batch Size renders several songs in one pass, up to a limit the engine reports for your
  setup, and Noise Variations renders the same composed song from different noise.
- Batch Queued Songs (on by default) does the same across the queue: when several YuE2
  jobs are waiting with the same settings and adapters, the one that reaches the engine
  takes the others with it and composes them together, up to that same limit. An album
  queued from Lyric Studio is the typical case. Songs per minute go up; each song finishes
  when its batch does rather than as soon as its own turn would have. A job that needs a
  different adapter, guidance or sampler setting waits for its own turn. A batch that
  fails hands its passengers back to the queue to render alone. Batched decoding rounds
  differently from solo decoding, so a seed rendered in a batch can differ from the same
  seed rendered alone; the Batch Size setting has always had the same property.
- It has its own NAR solver and scheduler choices (Midpoint or Wasserstein Flow, Uniform
  or HT V3). The shared Lua plugins do not run on it.
- Adapters load into two slots in the Adapters cluster, one for the AR half and one for
  the NAR half, each with its own strengths. An adapter is merged into the loaded model,
  so choosing one or changing a dial means pressing Apply, and the next generation
  reloads the model. Lead the style prompt with the adapter's trigger word.
- Lyric timestamps come from a forced aligner run after the render, switched on per
  render in the Post-Processing cluster. It needs the `mms-fa-f32.gguf` model.

Models: a language model GGUF and a VAE GGUF, selected separately in the Models cluster.
The packs:

| Pack | LM precision | Download |
|---|---|---|
| YuE2 Tiny (imatrix) | Q2_K-imat, audibly degraded | ~2.4 GB |
| YuE2 Compact (imatrix) | Q4_K_M-imat, for limited VRAM | ~3.1 GB |
| YuE2 Recommended | Q8_0 | ~4.6 GB |
| YuE2 Full Precision | BF16 | ~7.9 GB |
| YuE2 INT8 ConvRot (CUDA) | ConvRot checkpoint plus the Q8_0 GGUF; needs an NVIDIA GPU with BF16 support (Ampere or newer) | ~8.5 GB |

Training packs (Artist (AR) Training, Joint Training, Minted Regulariser) are separate.
See [Training YuE2 adapters](training/yue2.md).

YuE2's weights are licensed CC BY-NC 4.0. The upstream authors have clarified that
individual creators, musicians and researchers may use the model and its outputs
freely, including commercially, and that only companies need a commercial licence
(contact gezhang@umich.edu). Adapters trained on YuE2 carry the same terms.

## Capability matrix

"Yes" and "No" come from each backend's capability manifest in the server and the
controls the UI shows for it. "?" means not yet confirmed.

| | ACE-Step 1.5 | MiniMax-Music3 | YuE2 |
|---|---|---|---|
| Text to music | Yes | Yes | Yes |
| Lyrics | Yes | Yes | Yes |
| Caption format | Free text, plus BPM, key and time signature fields | Structured Caption | Free text; BPM and key folded into the style |
| Duration | Set by you, 10 to 600 s | Model-ended, ceiling 300 s | Model-ended, up to 360 s, no control |
| Songs per request | Up to 9 | 1, or up to 4 with Variations Per Render | Batch Size, capped by the engine |
| Covers and reference audio | Yes (Cover Studio) | No | No |
| Repaint | Yes (Repaint) | No | No |
| Extend | ? | No | No |
| Adapters at generation | DiT LoRA and LoKr, planner LM LoRA | Planner LM LoRA, applied per render | AR and NAR LoRA, merged at load |
| Training in Training Studio | Yes | Yes | Yes |
| Streaming playback | Continuous streaming in STORM; no per-render preview | Play While Rendering | No |
| Lua plugins (solvers, schedulers, guidance) | Yes, plus postprocess plugins | Yes, opt-in and experimental; no postprocess plugins | No (own NAR solver and scheduler) |
| Lyric timestamps (LRC) | Yes, from the DiT | Yes, from the planner | Yes, forced alignment after the render |
| Post-processing (VST chain, mastering, StableStep, Whisper) | Yes | Yes | Yes |
| PP-VAE and Spectral Lifter | Yes | No | No |
| Output | 48 kHz stereo | 44.1 kHz stereo | 48 kHz stereo |
| Typical VRAM | ? | ? | ? |
| Typical time per song | ? | ? | ? |

<!-- TODO(verify): ACE-Step extend. The engine supports outpaint (engine/docs/ARCHITECTURE.md), but no extend control was found in the UI. -->
<!-- TODO(verify): typical VRAM per backend at the recommended pack. The only figure in the code is "XL Turbo needs about 12 GB" (model-registry.json, XL Quality pack). -->
<!-- TODO(verify): typical wall-clock time per song per backend, on a named GPU, at default settings. -->
<!-- TODO(verify): MM3 LoKr planner adapters and flow-DiT adapters at generation time. FEATURES.md and README claim both; the generation path in server/src/services/backends/minimax/generate.ts only shows the LM adapter. -->
<!-- TODO(verify): YuE2 training. The manifest reports training: false, but the Training Studio has YuE2 AR, NAR and Joint Training cards; the matrix follows the UI. -->
<!-- TODO(verify): whether the YuE2 Batch Size cap is above 1 on a typical install (it comes from the engine's max_lm_batch). -->

## Related

- [Generating a song](generation.md)
- [Getting higher quality output](quality.md)
- [Models](models.md)
- [Model Manager](studios/model-manager.md)
- [Adapters](adapters.md)
- [Plugins](plugins.md)
- [Training Studio](studios/training-studio.md)
- [Training ACE-Step adapters](training/ace-step.md)
- [Training MiniMax-Music3 adapters](training/minimax-music3.md)
- [Training YuE2 adapters](training/yue2.md)
