# Features

HOT-Step CPP is a desktop app for local AI music generation. It started as a fork of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) and now runs three music models, which the app calls backends: ACE-Step 1.5, MiniMax-Music3 and YuE2. All three run natively in one C++/GGML engine. Generation, training, stem separation, captioning and transcription all happen on your own machine; the only network calls are model downloads, Genius lyric lookups and any external LLM provider you choose to configure.

This page is the catalogue: one line per feature, grouped in sidebar order. Each group links to the page that explains how to use it. The full doc index is in [docs/README.md](docs/README.md).

## Backends

Pick the backend from a button in the global bar. The choice survives a restart, switching frees the outgoing backend's VRAM, controls a backend cannot use are hidden, and studios it cannot run point you back to ACE-Step 1.5. Details: [Backends](docs/user/backends.md), [Models](docs/user/models.md), [Getting higher quality output](docs/user/quality.md).

### ACE-Step 1.5

| Feature | What it does |
|---|---|
| Three-stage pipeline | Planner LM (0.6B, 1.7B or 4B), DiT (Standard or XL) and VAE, producing 48 kHz stereo. |
| Explicit music fields | BPM, key, time signature and a duration target from 10 to 600 seconds. |
| Every studio | The only backend with covers, repaint, stem generation, Song Builder and STORM streaming. |
| Full plugin support | Every Lua solver, scheduler, guidance mode and postprocess plugin. |
| DiT and planner adapters | LoRA and LoKr on the DiT with merge, runtime and low-rank modes, plus a planner LM adapter. |
| Batch size | Several takes from one request, each saved as its own song. |
| LRC from the DiT | Lyric timestamps read from the DiT's own attention during the render. |
| ScragVAE and PP-VAE | An alternative fine-tuned VAE decoder, and an optional second autoencoder pass that cleans up fizz. |

### MiniMax-Music3

| Feature | What it does |
|---|---|
| Native port | 8B planner LM, depth decoder, condition encoder, flow-matching DiT and vocoder, all in C++/GGML, producing 44.1 kHz stereo. |
| Split model format | Five GGUFs, each selectable at its own quantisation, so a high-precision planner can pair with a compact DiT. |
| Importance-matrix quants | Planner quants below Q8 are built against an importance matrix measured from the full-precision model. |
| Structured Caption | Global Metadata, Vocal Details and Arrangement sections, with tempo and key in the prose. Compose Caption builds one from a plain description with no AI model. |
| Natural endings | The planner decides where the song stops (ceiling 300 s), planning several candidates and keeping the first to end. |
| Variations Per Render | One to four different songs from one prompt in a single planner pass, each its own queue entry. |
| Play While Rendering | Playback starts a few seconds in while later windows are still rendering. The full file saves as usual. |
| Plan cache | Changing only flow-stage settings replays the saved plan instead of planning again. |
| Low-step compensation | Below 30 flow steps the schedule reshapes so short renders keep their low end and stereo image. |
| Planner adapters | LoRA adapters with Strength, Attention, MLP and depth-thirds dials, in runtime or merge mode. |
| LRC from the planner | Line-level lyric timestamps read from the planner's attention. |
| TensorRT renderer | Optional TensorRT flow DiT that refits weights from the selected GGUF. |
| Opt-in sampler plugins | The Lua solvers, schedulers and guidance modes can drive the flow DiT, behind an experimental toggle. |
| Guidance-distilled planners | Depth-pruned, guidance-distilled planner checkpoints load and run. |

### YuE2

| Feature | What it does |
|---|---|
| Native port | AR composer and NAR renderer halves plus a VAE decoder, in C++/GGML, producing 48 kHz stereo. |
| Freeform prompt | A plain style sentence plus lyrics. BPM and key go in the sentence; language comes from the lyrics. |
| Model-ended songs | The composer decides the length, up to six minutes. |
| Chain of Thought | Full lead sheet (default), Melody only, or Off. Off skips the lead sheet but runs guidance, so it is the slowest mode. |
| Lead sheet preview | Plans the score only and shows it as staff notation with playback and a health verdict; continue, re-plan or cancel. You can also paste your own ABC lead sheet. |
| Batch Size and Noise Variations | Several songs in one pass, or the same composed song rendered from different noise. |
| Own NAR sampler | Midpoint or Wasserstein Flow solvers with Uniform or HT V3 schedules. |
| Two adapter slots | Separate AR (composer) and NAR (renderer) adapters, merged into the loaded model with Apply. |
| Companion decoder | A small decoder adapter that ships in every YuE2 pack is applied to every GGUF generation. |
| Forced-alignment LRC | Lyric timestamps from an aligner run after the render. |
| Quant ladder | Imatrix-guided packs from Q2_K up to BF16, plus an INT8 ConvRot checkpoint for Ampere or newer NVIDIA cards. |

## Generation

The global bar holds every setting that applies to a render, whichever studio starts it. Details: [Generation](docs/user/generation.md), [Adapters](docs/user/adapters.md), [Plugins](docs/user/plugins.md).

| Feature | What it does |
|---|---|
| Global parameter bar | Models, backend, adapters, generation, LM, post-processing, profiles and VRAM in pinnable sections with one-line summaries. |
| Plugin pickers | Solver, schedule and guidance mode come from Lua plugins, each with its own settings panel. |
| Built-in schedules | Beta (Custom), Power and Composite (2-Stage), plus an Auto shift derived from duration and step count. |
| Performance trades | CFG Cutoff, LM CFG Cutoff and Step Cache for speed at some cost in adherence. |
| Timbre reference | A reference track fed to the DiT to guide tone and texture. |
| DCW correction | Wavelet-domain correction during sampling, low band, high band, both, or latent space. |
| Auto-trim | Renders past the requested length, cuts at a natural ending, and fades out only if none is found. |
| Latent post-processing | Latent shift, latent rescale and custom timestep lists. |
| Denoiser and LSS | An in-engine spectral gate after decode and a latent spectral suppressor before it. |
| Seed Manager | Save, star, search and reload seeds; the seed actually used is stored with each song. |
| LM seed lock | The LM seed follows the DiT seed, or runs independently with its own seed manager. |
| LM controls | Chain-of-thought caption, temperature, CFG, top-k, top-p, negative prompt, and Presence, Frequency or DRY repetition penalties. |
| LM codes strength | How many DiT steps the LM's codes condition. Repeat renders with the same seed can reuse cached codes. |
| Profiles | Save, apply, rename, export and import named snapshots of every bar setting, optionally with caption and lyrics. |
| VRAM indicator | GPU memory in use; hover to see loaded models and unload them. |
| Job queue | One queue for every studio, with retry once on failure, stall detection, a generation timeout, and Resume or Discard after a restart. |

### Post-processing chain

Everything runs on a copy of the render, in a fixed order, under one master switch. The original is kept, and the chain can run later from the Library.

| Feature | What it does |
|---|---|
| StableStep | Re-renders the instrumental through Stable Audio 3 and remixes the untouched vocal at its original balance. Optional adapters, crossover or mix blending, and its own sampler. |
| PP-VAE re-encode | Second autoencoder pass against fizz, with an original-blend slider and an ONNX/TensorRT path. |
| Spectral Lifter | Denoise, noise floor, high-frequency extension, transient boost and shimmer reduction. |
| Vocal Naturalizer | Experimental five-stage DSP pass against robotic vocal artefacts, skipped on instrumentals. |
| VST3 chain | Your own VST3 plugins, reorderable, with native editor windows, presets, live monitoring and a pre-chain gain offset. |
| Mastering | Matches level, EQ and dynamics to a reference track, which can double as the timbre reference. |
| Final normalizer | LUFS targets for streaming, broadcast or club playback with a look-ahead peak limiter. |
| Whisper lyrics | Word-level lyric transcription, guided by your lyrics, optionally on an isolated vocal. |
| Cover art | 1024 x 1024 cover with FLUX.2-klein-4B from the song's subject or lyrics, after or alongside the chain. |
| Quality evaluator | Spectral scoring of the unprocessed or processed audio, shown as a badge in the Library. |
| Tiled decoder | A postprocess plugin can replace the built-in VAE decode. |

## Auto-Gen

Genre-first song creation that fills in everything else. Details: [Auto-Gen](docs/user/studios/insta-gen.md)

| Feature | What it does |
|---|---|
| Genre picker | Searchable, categorised genre taxonomy, with a Random button that picks two to four. |
| Vocal modes | Instrumental, Lyrics from the built-in LM, or Lyrics + AI from an external LLM and your subject, or a random one. |
| Custom system prompt | Edit, save and reset the system prompt sent to the external LLM. |
| Caption rewrite toggle | Let the built-in LM enrich the caption, or send it as typed. |
| Lyric preview | Review and edit the lyrics, caption and metadata before committing. |
| Refine in Custom-Gen | Sends the previewed lyrics, caption and metadata to Custom-Gen for hand tuning. |
| Cover art override | Change the cover art subject for one song. |
| Own queue | Auto-Gen jobs run one at a time with a filtered generations list beside the panel. |

## Custom-Gen

Full manual control of caption, lyrics and music parameters. Details: [Custom-Gen](docs/user/studios/create.md)

| Feature | What it does |
|---|---|
| Style and lyrics | Caption, section-tagged lyrics, negative prompt, instrumental toggle and song info. |
| Wildcards | `{a\|b\|c}` syntax in caption and lyrics, expanded in place or at generate time from the seed. |
| LoRA trigger box | Prepends a trigger word for adapters that do not record one. |
| Beat I/O | Asks for a clean percussive intro and outro of 1 to 8 bars for DJ mixing. |
| Generate with AI | A configured LLM writes caption, lyrics, title and metadata from a genre and subject. |
| Compose Caption | MiniMax-Music3 Structured Caption from a plain description, with genre routing and warnings. |
| Caption source | On MiniMax-Music3 and YuE2, borrow a training dataset's caption: nearest tempo, a named track, or your own. |
| Latent import | Continue from a saved `.latent` file, with its embedded metadata. |
| Backend-aware fields | BPM, duration, key, time signature, vocal gender and language show only where the active backend uses them. |
| YuE2 lead sheet preview | Opens before render when "Preview the score first" is on. |

## Library, player and playlists

Every render and import lands here. Details: [Library](docs/user/studios/library.md)

| Feature | What it does |
|---|---|
| Three views | Grid with cover art, list, or table, remembered per browser. |
| Source filters | Tabs per studio plus Imported, with counts. |
| Table column picker | Opt-in columns for generation details such as seed, CFG, solver and adapter, with resizable widths. |
| Multi-select | Bulk download or delete. |
| Track menu | Post-processing, edit in Custom-Gen, playlist, download, send to Cover Studio, metadata, Export Params (JSON preset), Retranscribe Lyrics, cover art, A/B slots, delete. |
| Upload into Library | Imports WAV, MP3, FLAC, M4A, MP4, AAC, OGG, Opus, WebM and AIFF, with optional post-processing. |
| Metadata editor | Title, artist, album, year, genre, BPM, key, comment, lyrics and cover image; some fields are embed-only. |
| Editable cover-art prompt | Generate or regenerate cover art from a pre-filled, editable prompt. |
| A/B compare | Pin two tracks, play them against each other, and open a parameter diff. |
| Song details | Backend chip, per-backend "How it was made" breakdown, prompt and lyrics. |
| Streaming rows | A MiniMax-Music3 render in progress shows in the list and plays before its file exists. |
| Tagged downloads | Metadata and cover art embedded in downloaded files, in the format chosen in Settings. |
| Global player | Transport, repeat modes, spectrum analyzer, disco mode and volume, visible in every view. |
| Rate and pitch preview | Playback speed from 0.5x to 2x, and a 48k/44.1k toggle that hears a render clocked at 44.1 kHz. |
| Variant switch | No Adapter, Unmastered and Mastered versions, each with its own download. |
| Waveform tools | Trim and crop in and out points, section markers and a synced LRC lyrics bar. |
| Playlist queue | One browser-local play queue with reorder, Play All and Download All. |

## Lyric Studio

AI-assisted lyric writing from a learned style profile, plus per-album render presets. Details: [Lyric Studio](docs/user/studios/lyric-studio.md)

| Feature | What it does |
|---|---|
| Genius fetch | Pull an artist's or album's lyrics, or add artists, albums and songs by hand. |
| Style profiles | An LLM profiles themes, tone and structure; rhyme, meter and vocabulary statistics are computed locally without one. |
| Lyric generation | Streams new songs from a profile, with an optional subject and count, avoiding subjects, keys and titles used before. |
| Refine | An LLM revises lyrics and title, saved as a linked new generation. |
| Per-backend captions | ACE-Step/YuE2 caption, MiniMax-Music3 Structured Caption and YuE2 planner caption per song. |
| Caption source picker | Borrow a caption from the album's captioned training tracks. |
| Generate Audio | Queue a render through the active backend with the album's preset. |
| Send to Custom-Gen | Opens Custom-Gen with lyrics, caption, metadata and adapter preset filled in. |
| Album presets | DiT and planner adapters with group scales and a reference track on ACE-Step; one LM adapter on MiniMax-Music3; AR and NAR adapters on YuE2. |
| Editable system prompts | Override the generation, metadata, profiler and refinement prompts per provider. |
| Per-role providers | Separate provider and model for profiling, generation and refinement. |
| Duration from lyrics | Compute duration from lyrics and BPM instead of the LLM's estimate. |
| Randomize timbre | Pick a random reference track per render. |
| Bulk operations | Batch Genius fetches, profile builds, lyric generation with fill-to-target, audio renders and preset assignment. |
| Agent access | Drive Lyric Studio from Claude Code or Codex through the [MCP server](tools/mcp-lyricstudio/README.md). |

## Cover Studio

Re-style an existing recording. Requires the ACE-Step backend. Details: [Cover Studio](docs/user/studios/cover-studio.md)

| Feature | What it does |
|---|---|
| Source upload | MP3, WAV, FLAC, OGG, M4A, Opus or AAC, or send a track from the Library. |
| Latent import | Use a `.latent` or `.hslat` file and its embedded lyrics, caption, BPM and key. |
| Essentia analysis | Local BPM and key detection, cached per browser. |
| BPM and key fixes | Halve, double or type the tempo; override the detected key. |
| Genius lyrics search | Find the source's lyrics, paste your own, or go instrumental. |
| Target artist | Fills the style description from an artist profile, or drafts one with an LLM, and loads a matching album adapter preset. |
| Structure Fidelity | How closely the output follows the source arrangement. |
| Source Preservation and noise method | How much of the source survives, with Classic or Full Denoise noise methods. |
| NoFSQ mode | Skips FSQ quantisation for a result closer to the source. |
| Tempo and pitch | 0.5x to 2x tempo and -12 to +12 semitones, with the target key shown. |
| Timbre reference | A second reference track for the DiT's timbre conditioning. |
| Stem mix | Split the source with SuperSep and mute or lower stems before generating. |
| Serial cover queue | Queue more covers while one renders. |

## Repaint Studio

Regenerate one region of a track in place. Requires the ACE-Step backend, carries a work-in-progress banner, and does not extend audio past either end. Details: [Repaint Studio](docs/user/studios/repaint-studio.md)

| Feature | What it does |
|---|---|
| Source picker | Upload a file or pick a song from the Library. |
| Region selection | Waveform handles, a range slider or exact start and end times, with region playback. |
| Region lyrics | Line-by-line editor synced to the region when an LRC file exists, plain text otherwise. |
| Style reuse | Leave the style blank to reuse the source's own caption. |

## Stem Separator (Stem Studio)

Split a track into stems, or have the DiT regenerate parts. Details: [Stem Separator](docs/user/studios/stem-studio.md)

| Feature | What it does |
|---|---|
| SuperSep | BS-RoFormer, Mel-Band RoFormer and MDX23C in native GGML. |
| Separation levels | Basic (6 stems), Vocal Split (8), Full (12, with drum sub-stems), BS-RoFormer 2-stem, and Leap Xe 2-stem with a dedicated model per side. |
| Extract (DiT) | Regenerates up to 12 instrument tracks from the mix with a base DiT, with an optional style hint and lyrics. |
| Stem mixer | Synced preview with mute, solo and 0 to 200% volume per stem. |
| Downloads | One stem as WAV, or every stem as a ZIP. |
| Recent extractions | Reload or delete past jobs from either mode. |
| Library picker | Pick a source by studio, in its raw or mastered version. |

## Stem Builder

Generate a new instrument stem over a backing track. Requires the ACE-Step backend and a base DiT. Details: [Stem Builder](docs/user/studios/stem-builder.md)

| Feature | What it does |
|---|---|
| Target track | Pick one of 12 instruments, with an optional style hint. |
| Raw stems | Output skips mastering, the post-processing chain and adapters, and matches the source length. |
| Synced preview | Plays source and stem together with separate volume and mute. |
| Layer stack | Carry the newest layer forward as the next source to build an arrangement. |
| Recent builds | Upload dropzone plus the last 20 builds from the Library, each reusable as a source. |

## Song Builder

Build a song section by section with variant auditioning. Requires the ACE-Step backend. Details: [Song Builder](docs/user/studios/song-builder.md)

| Feature | What it does |
|---|---|
| Projects | A saved song with shared caption, BPM, key and time signature. |
| Section variants | Four candidates per section, streamed in one at a time; audition and pick one. |
| Append and prepend | Each new section is a repaint extension of the real audio built so far. |
| Transition blend | Up to 10 seconds of the seam regenerated as a transition. |
| Clip point | Attach the next section at any point set from the playhead. |
| Match a section's feel | Experimental bias toward an earlier section's harmonic shape. |
| Length in bars | Bars or seconds, with an estimate from the lyric line count. |
| Lyric corrections | Edit a committed section's lyrics to match what was actually sung. |
| Stop and keep | Cancel remaining variants and pick from the finished ones. |
| Fast variants | Mastering and heavy post-processing off by default while building. |

## STORM

Live streaming performance with crossfaded slots. Requires the ACE-Step backend. Details: [STORM](docs/user/studios/storm.md)

| Feature | What it does |
|---|---|
| Continuous mode | Renders back-to-back slots that crossfade into one endless stream. |
| Live controls | Change style, lyrics, seed, BPM, guidance, steps and length for the next slot without stopping. |
| Sticky fields | Pin a style or lyric change so it stays for every following slot. |
| Lyric advance | Loop, cycle or shuffle through lyric sections as slots play. |
| AI continuation | An external LLM continues the lyrics every 1, 2 or 4 slots. |
| Stream sampler overrides | Per-stream solver, scheduler and guider, changeable live. |
| Buffering | Crossfade length in beats and a maximum look-ahead buffer. |
| DJ mode | Two decks with a crossfader, Camelot key compatibility, cuts, nudges and beat quantize. |
| Sequential mode | Queues one song through the normal queue into the Library. |
| Record | Captures the stream to a `.webm` file in the browser. |
| Slot timeline | Each played slot shows its seed, detected key and BPM, and settings. |

## MIDI Studio

Audio-to-MIDI transcription on its own native engine. Weights are CC BY-NC 4.0, and MIDI made here inherits that. Details: [MIDI Studio](docs/user/studios/midi-studio.md)

| Feature | What it does |
|---|---|
| Native transcription | `ace-midi`, a C++/GGML port of MuScriptor, on any backend, from a library track or a WAV or MP3. |
| Multi-instrument output | A `.mid` with separate parts for drums, bass, guitar, keys and the rest. |
| Model sizes | Small (CPU), medium and large. |
| Gated weights in-app | Save a Hugging Face token and download weights with live progress. |
| Live piano roll | Notes appear as chunks complete. |
| Preview | Plays original and MIDI together with a crossfade and per-instrument mute and solo. |
| Persistent jobs | Transcriptions survive restarts. |

## Training Studio

Build datasets from your own audio and train adapters on your GPU, for all three backends. Details: [Training Studio](docs/user/studios/training-studio.md), [ACE-Step training](docs/user/training/ace-step.md), [MiniMax-Music3 training](docs/user/training/minimax-music3.md), [YuE2 training](docs/user/training/yue2.md), [Training internals](docs/dev/training-internals.md)

| Feature | What it does |
|---|---|
| Dataset wizard | Scan a folder, set a trigger word and language, preview before creating. |
| Labeling | Essentia BPM and key, Genius lyrics, and captions from local MOSS or a cloud LLM. |
| Local captioning | MOSS-Music-8B-Instruct in native GGML (`ace-caption`), in ACE-Step or MM3 caption formats. |
| Review grid | Spreadsheet editing of every track with bulk actions and scoped re-labels. |
| Dataset build | Writes a Side-Step-compatible `dataset.json` with trigger placement and genre ratio. |
| Preprocess and codes | Encodes audio for ACE-Step, or RVQ codes for MiniMax-Music3 with an optional cover-laundering pass. |
| ACE-Step adapters | Planner LM LoRA (0.6B, 1.7B, 4B) and DiT LoRA. |
| MiniMax-Music3 adapters | Planner LM LoRA trained from RVQ codes. |
| YuE2 adapters | Five-stage chain with Perform all stages, and Joint Training of the AR and NAR pair with Prodigy, AdamW or Muon. |
| Quality presets | Fast, Balanced and Thorough presets on the training forms. |
| Adapter methods | DoRA, rsLoRA, LoRA+, HiRA, LoHa, PiSSA and HRA, plus learned artist tokens and a trainable KV prefix on the LM trainers. |
| Small PiSSA adapters | The base model's own directions ship once, so each adapter carries only what it learned. |
| Quantized-base training | Train against a K-quant or MXFP4 base for a much lower VRAM floor. |
| Flash-attention training | Fused attention that makes memory linear in sequence length; on by default for the DiT, opt-in for the LM trainers. |
| Target-loss stopping | Stop on a loss target instead of a step count, and continue a stopped run. |
| Live monitoring | Loss chart, stats and milestone checkpoint badges. |
| Audition | Same-seed A/B of base and adapter planner, optionally rendered through the DiT. |
| Adapter scoring | Evaluation passes put an artist-match score in the planner adapter picker. |
| YuE2 Refine and Review | Push a planner adapter up KL rungs with a preview per rung, score the ladders by ear, and let a scoreboard rank the rungs by likeness, corruption and re-plans. |
| Batch pipeline | Import multiple folders and run label, build, preprocess and train unattended; YuE2's Train multiple chains its stages over several datasets. |
| Send to Lyric Studio | Export a dataset's artist and album with its trained adapters as a preset. |

## AI Assistant

An LLM chat sidebar that sees your current settings. Details: [Assistant](docs/user/studios/assistant.md)

| Feature | What it does |
|---|---|
| Streaming chat | Resizable panel with a collapsible thought-process block when the model provides one. |
| Settings snapshot | Every message carries a JSON snapshot of the current parameters and content fields. |
| Suggested changes | Proposed setting changes as from-to rows, applied one at a time or all at once. |
| Content edits | Writes or rewrites caption, lyrics and other content fields in Custom-Gen. |
| Shared providers | Same provider registry as Lyric Studio, with remembered provider and model. |
| Knowledge base | Covers engine parameters, plugins, adapters, post-processing and lyric formatting. |

## Model Manager

Browse, download and remove every model file in the app. Details: [Model Manager](docs/user/studios/model-manager.md), [Models](docs/user/models.md)

| Feature | What it does |
|---|---|
| Family tabs | ACE-Step 1.5, MiniMax-Music3, YuE2 and Shared, with role sub-tabs and installed counts. |
| Starter packs | One-click bundles per backend, plus shared runtime, separation, StableStep, Whisper and captioning packs. |
| Concurrent downloads | Several files at once with percent, speed and ETA, and resume after an interruption. |
| Verification | Size and header checks before a downloaded file is kept. |
| Delete | Remove installed files to free space. |
| StableStep licence gate | Licence acceptance and an optional Hugging Face token. |
| TensorRT builder match | Marks the builder DLL that matches your GPU. |
| First-launch prompt | Opens itself when no models are found. |

## Settings

Configuration for engine, server, providers, downloads and storage. Details: [Settings](docs/user/studios/settings.md)

| Feature | What it does |
|---|---|
| Environment editor | Edits `.env` with only changed keys written, and a Restart now button when a key needs it. |
| GPU device picker | Lists NVIDIA cards by name and VRAM and stores a stable GPU UUID. |
| Keep models in VRAM | Engine flag that keeps DiT, adapter and VAE resident between renders, plus a per-request toggle that skips DiT/VAE swaps. |
| Warm on startup | `.env` settings that preload a DiT, VAE and adapter after the engine boots. |
| Pipeline parallelism | Run Whisper, quality evaluation and cover art alongside post-processing. |
| Generation timeout | 10 minutes to 6 hours, per track. |
| AI services | API keys and endpoints for Gemini, OpenAI, Anthropic, Ollama, LM Studio, llama.cpp, Unsloth and OpenAI-compatible servers, applied live. |
| Download defaults | WAV, FLAC, Opus or MP3, bitrate, mastered or original, and optional latent file. |
| Filename trigger | Use an adapter's filename as its trigger word, prepended, appended or replacing the caption. |
| Display language and theme | Interface language, light or dark theme, and generic artwork sets. |
| Storage cleanup | Stem storage counter and separate nukes for generations, written lyrics and profiles. |
| Terminal, Restart and Quit | Live engine log with search and a VRAM badge; sidebar restart and clean shutdown. |

## Engine and plugins

The C++17 engine behind every backend. Details: [Plugins](docs/user/plugins.md), [Plugin authoring](docs/dev/plugins-authoring.md), [Engine](docs/dev/engine.md), [Architecture](docs/dev/architecture.md)

| Feature | What it does |
|---|---|
| Portable builds | Windows CUDA, Vulkan and CPU, Linux, and macOS Metal releases. |
| Engine binaries | `ace-server`, `ace-lm`, `ace-synth`, `ace-understand`, `ace-train`, `ace-caption`, `ace-midi`, `neural-codec`, `mp3-codec` and `quantize`. |
| Lua plugins | Solvers, schedulers, guidance modes and postprocess plugins load from `engine/plugins/` at launch with no rebuild. |
| Plugin UI | Each plugin declares its own sliders, toggles and dropdowns, rendered by the UI. |
| Native bridge | `apg()` for momentum, projection and norm thresholding, and `post_step()` for extra forward passes. |
| Loop-owning solvers | A solver can take over the whole denoising loop for adaptive stepping. |
| Model formats | GGUF, safetensors directories and ONNX, detected by path. |
| Quantize tool | K-quants, IQ quants, MXFP4 and NVFP4, with importance-matrix support. |
| TensorRT paths | MiniMax-Music3 flow DiT, PP-VAE and StableStep through TensorRT on NVIDIA. |
| Flash-attention training ops | Custom fused attention forward and backward in the ggml CUDA backend. |
| Upstream hooks | Fork hooks and a patch stack checked by `engine/verify-hooks.ps1` before every build. |

## Developer tooling

For contributors and agents. Details: [Building](docs/dev/building.md), [HTTP API](docs/dev/api.md), [Architecture](docs/dev/architecture.md), [Releasing](docs/dev/releasing.md), [Writing the docs](docs/dev/docs-contributing.md)

| Feature | What it does |
|---|---|
| HTTP API | Everything the UI does is JSON over HTTP on port 3001, indexed per route. |
| Dev loop | `dev.bat` runs Vite with hot reload and the Node server in watch mode. |
| Safe engine rebuild | `dev-rebuild.bat` shuts the app down cleanly before rebuilding the engine. |
| Release gate | [`tools/release-gate`](tools/release-gate/README.md) drives the app through its API and stages renders for an ear test. |
| Release prerequisites | `check-release-prereqs.mjs` confirms every model and data file is reachable by a user. |
| Docs checks | `tools/docs/build-docs.mjs` regenerates tables; `check-docs.mjs` catches drift and broken links. |
| Lyric Studio MCP server | [`tools/mcp-lyricstudio`](tools/mcp-lyricstudio/README.md) exposes Lyric Studio to Claude Code and Codex. |
