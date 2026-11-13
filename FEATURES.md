# Features

HOT-Step CPP is a desktop app for local AI music generation. It started as a fork of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) and now runs four music models, which the app calls backends: ACE-Step 1.5, MiniMax-Music3, YuE2 and Stable Audio 3. All four run natively in one C++/GGML engine. Generation, training, stem separation, captioning and transcription all happen on your own machine; the only network calls are model downloads, Genius lyric lookups, any external LLM provider or second training PC you choose to configure, and a check for a newer version on GitHub at most once a day.

This page is the catalogue: one line per feature, with the studios in sidebar order. Each group links to the page that explains how to use it. The full doc index is in [docs/README.md](docs/README.md).

Studio lists and status views reuse recent server results when you return to a
view. Active jobs refresh while running; returning to a view never submits a
new job. See [Generation](docs/user/generation.md), [MIDI Studio](docs/user/studios/midi-studio.md)
and [Model Manager](docs/user/studios/model-manager.md).

## Backends

Pick the backend from the Generation engine box at the top of the sidebar; it appears once two or more backends are registered. The choice survives a restart, switching frees the outgoing backend's VRAM, controls a backend cannot use are hidden, and the sidebar lists only the studios the active backend can run, moving you to Custom-Gen if the open one drops out. Details: [Backends](docs/user/backends.md), [Models](docs/user/models.md), [Getting higher quality output](docs/user/quality.md).

### ACE-Step 1.5

| Feature | What it does |
|---|---|
| Three-stage pipeline | Planner LM (0.6B, 1.7B or 4B), DiT (Standard or XL) and VAE, producing 48 kHz stereo. |
| Explicit music fields | BPM, key, time signature and a duration target from 10 to 600 seconds. |
| Every studio | Supports all studios, including audio-conditioned covers, repaint, stem generation, Song Builder and STORM streaming. |
| Full plugin support | Every Lua solver, scheduler, guidance mode and postprocess plugin. |
| DiT and planner adapters | LoRA, LoKr and DoRA on the DiT with merge, runtime and low-rank modes, plus a planner LM adapter. Runtime mode applies DoRA magnitudes only with Runtime DoRA magnitudes on. |
| Batch size | Several takes from one request, each saved as its own song. |
| LRC from the DiT | Lyric timestamps read from the DiT's own attention during the render. |
| ScragVAE and PP-VAE | An alternative fine-tuned VAE decoder, and an optional second autoencoder pass that cleans up fizz. |

### MiniMax-Music3

| Feature | What it does |
|---|---|
| Native port | 8B planner LM, depth decoder, condition encoder, flow-matching DiT and vocoder, all in C++/GGML, producing 44.1 kHz stereo. |
| Split model format | Five GGUFs, each selectable at its own quantisation, so a high-precision planner can pair with a compact DiT. Unpicked, the planner runs at Q8_0 and the depth decoder at NVFP4 on RTX 50-series cards. |
| Importance-matrix quants | Planner quants below Q8 are built against an importance matrix measured from the full-precision model. |
| Structured Caption | Global Metadata, Vocal Details and Arrangement sections, with tempo and key in the prose. Compose Caption builds one from a plain description with no AI model. |
| Natural endings | The planner decides where the song stops (ceiling 300 s), planning several candidates and keeping the first to end. |
| Variations Per Render | One to four different songs from one prompt in a single planner pass, each its own queue entry. |
| Play While Rendering | Playback starts a few seconds in while later windows are still rendering. The full file saves as usual. |
| Plan cache | Changing only flow-stage settings replays the saved plan instead of planning again. |
| Saved plans | Save AR Plan (Plank) keeps a render's planner codes so flow-stage settings can be compared on the same plan. Save Plan To Disk keeps the planner output as a file that skips the planner on replay and survives a restart. |
| Fast defaults | 20 flow steps, a 150-frame window hop and TensorRT FP8, each ear-tested against the reference settings with no audible loss. [Details](docs/user/backends.md) |
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
| Freeform prompt | A plain style sentence plus lyrics; language comes from the lyrics. Include BPM/key in caption (on by default) ends the sentence with the trained BPM and key tail, taken from Song Info or the dataset caption. |
| Model-ended songs | The composer decides the length, up to six minutes. |
| Chain of Thought | Full lead sheet (default), Melody only, or Off. Off skips the lead sheet but runs guidance, so it is the slowest mode. |
| Lead sheet preview | Plans the score only and shows it as staff notation with playback and a health verdict; continue, re-plan or cancel. You can also paste your own ABC lead sheet. |
| Cover Studio | Transcribes a source into a reviewed ABC lead sheet, or accepts pasted ABC or imported MusicXML with a lyric verse fallback, then composes from that score. SheetSage2 is optional when you provide the score. |
| Batch Size and Noise Variations | Several songs in one pass, or the same composed song rendered from different noise. |
| Composer sampling | Composer temperature, Top-P, Top-K, Min-P and repetition controls, plus CFG Plausibility, CFG in Log-Probabilities and CFG Rescale. Composer Top-K 0 (unlimited) with a temperature above 1.0 is clamped to 1.0, with a warning. |
| Own NAR sampler | NAR Solver: Midpoint (stock), or the experimental Wasserstein Flow, Ancestral Euler, Latent Channel EQ and Timestep Lie. NAR Scheduler: Uniform (stock), or the experimental HT V3, Plan-Aware, Plan-Loose and Plan Overdrive. |
| Plan controls | Experimental Plan Strength (start, end and curve), Plan CFG with rescale and level match, and Plan Conditioning Noise (CADS) change how closely the NAR renderer follows the plan. |
| Plan Play presets | Four importable profiles that vary how strongly the NAR renderer follows the plan. See [YuE2 Plan Play presets](docs/user/presets/yue2-plan-play/README.md). |
| What changes what | Composer settings shape the song plan. NAR settings shape its rendering, and experimental plan controls can change adherence to the plan. See [YuE2](docs/user/backends.md#what-changes-what-on-yue2). |
| Two adapter slots | Separate AR (composer) and NAR (renderer) adapters merged into the loaded model, with Strength, Attention, MLP and depth-thirds dials that save when you let go of them. The AR adapter has its own Planner Strength for the score, and an adapter that records no trigger word gets a LoRA trigger word box. |
| ComfyUI adapter import | Splits a single-file YuE2 adapter from ComfyUI or ai-toolkit into the AR and NAR pair and selects both. See [Adapters](docs/user/adapters.md#importing-comfyui-adapters). |
| Companion decoder | A small decoder adapter that ships in every YuE2 pack is applied to every GGUF generation. |
| Forced-alignment LRC | Lyric timestamps from an aligner run after the render. |
| Quant ladder | Imatrix-guided packs from Q2_K up to BF16, plus an INT8 ConvRot checkpoint for Ampere or newer NVIDIA cards. |

### Stable Audio 3

| Feature | What it does |
|---|---|
| Text to audio | Stable Audio 3 Medium in C++/GGML renders from the caption alone, producing 44.1 kHz stereo. Lyrics and a negative prompt are not used. |
| Fixed length | The requested duration, up to 600 seconds, is the output length; the model does not end the song itself. |
| Sampler | 8 steps by default (1 to 64) with MD Pingpong Blend, or the native PingPong or Euler. Sampler Plugins switches the denoise to the shared Lua solvers, schedulers and guidance modes. |
| Loudness Normalize | Optional integrated LUFS target (-30 to -6, default -14) with a -1 dBFS peak ceiling, applied before post-processing. |
| Generation DiT | BF16 by default, with an experimental quant ladder down to IQ1_S in the Model Manager. The choice is saved. |
| DoRA adapters | Adapters in the `sa3-adapters` models folder, the StableStep adapters and those trained in Training Studio, merge at load. |
| Post-processing | The chain and Whisper lyrics run as on the other backends; StableStep is off, since it would refine Stable Audio 3 with itself. |

## Generation

The global bar holds every setting that applies to a render, whichever studio starts it. Details: [Generation](docs/user/generation.md), [Adapters](docs/user/adapters.md), [Plugins](docs/user/plugins.md).

| Feature | What it does |
|---|---|
| Global parameter bar | Models, adapters, generation, LM and post-processing in pinnable sections with one-line summaries, and Profiles at the right end. On a narrow screen a Settings button in the header opens the same sections in a sheet. |
| Plugin pickers | Solver, schedule and guidance mode come from Lua plugins, each with its own settings panel. |
| Built-in schedules | Beta (Custom), Power and Composite (2-Stage), plus an Auto shift derived from duration and step count. |
| Performance trades | CFG Cutoff, LM CFG Cutoff and Step Cache for speed at some cost in adherence. |
| Timbre reference | A reference track fed to the DiT to guide tone and texture. |
| DCW correction | Wavelet-domain correction during sampling, low band, high band, both, or latent space. |
| Auto-trim | Renders past the requested length, cuts at a natural ending, and fades out only if none is found. |
| Latent post-processing | Latent shift, latent rescale and custom timestep lists. |
| Denoiser and LSS | An in-engine spectral gate after decode and a latent spectral suppressor before it. |
| Seed Manager | Save, star, search and reload seeds; the seed actually used is stored with each song. |
| Seed breeding | Blend the initial noise of two seeds (`seedB`, `seedBlend`) to get children between two keeper takes. API only for now. |
| Seed morph | Time-varying seed blend (`seedBlendEnd`): the noise morphs from `seedBlend` at the start to `seedBlendEnd` at the end, slerped per latent frame. API only for now. |
| SA3 seed breeding | The same `seedB` and `seedBlend` fields for Stable Audio 3 text-to-audio; a `seedB` with no blend breeds at 0.5. API only. |
| LM seed lock | The LM seed follows the DiT seed, or runs independently with its own seed manager. |
| LM controls | Chain-of-thought caption, temperature, CFG, top-k, top-p, Min-P, CFG Plausibility, CFG Rescale, negative prompt, and Presence, Frequency or DRY repetition penalties. |
| LM codes strength | How many DiT steps the LM's codes condition. Repeat renders with the same seed can reuse cached codes. |
| Profiles | Save, apply, rename, export and import named snapshots of the bar's settings, optionally with caption and lyrics. Each backend keeps its own list, and a profile can carry that backend's own knobs. |
| VRAM indicator | GPU memory in use, at the top of the Terminal box; hover to see loaded models and unload them. |
| Job queue | One queue for every studio, with retry once on failure, stall detection, a generation timeout, and Resume or Discard after a restart. |
| Job progress | The Custom-Gen sidebar entry, the phone header and the browser tab title show the running count and the active job's percentage. An optional notification fires when a job finishes in a hidden tab; clicking it opens the song. |

### Post-processing chain

Everything runs on a copy of the render, in a fixed order, under one master switch. The original is kept, and the chain can run later from the Library.

| Feature | What it does |
|---|---|
| StableStep | Re-renders the instrumental through Stable Audio 3 and remixes the untouched vocal at its original balance. Optional adapters, crossover or mix blending, and its own sampler. |
| MM3 Latent Refiner | Optional, default-off whole-mix refinement with steps, window/direct mode, overlap and dry/wet mix. Model pack unlisted pending licensing and publication. [Controls and limits](docs/user/generation.md#mm3-latent-refiner). |
| PP-VAE re-encode | Second autoencoder pass against fizz, with an original-blend slider. |
| Spectral Lifter | Denoise, noise floor, high-frequency extension, transient boost and shimmer reduction. |
| Vocal Naturalizer | Experimental five-stage DSP pass against robotic vocal artefacts, skipped on instrumentals. |
| VST3 chain | Your own VST3 plugins, reorderable, with native editor windows, presets, live monitoring and a pre-chain gain offset. Named chain presets save with server revisions; an older browser preset set is imported once and kept as a fallback. |
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
| Duration | Uses the duration set in Custom-Gen; with that on auto, the LM's estimate decides. |
| Server-side jobs | Previews and direct renders run as server workflow jobs, and their audio goes through the shared queue. A Generations list beside the panel shows only Auto-Gen songs. |

## Custom-Gen

Full manual control of caption, lyrics and music parameters. ACE-Step songs play after rendering; MiniMax-Music3 can play while rendering. Details: [Custom-Gen](docs/user/studios/create.md)

| Feature | What it does |
|---|---|
| Style and lyrics | Caption, section-tagged lyrics, negative prompt, instrumental toggle and song info. |
| Wildcards | `{a\|b\|c}` syntax in caption and lyrics, expanded in place or at generate time from the seed. |
| Style presets | Twelve one-sentence ensemble descriptions, each instrumentation plus mood, written straight into the caption from the Presets button by the Style Description. |
| LoRA trigger box | Prepends a trigger word for adapters that do not record one, without repeating it on reuse. |
| Beat I/O | Asks for a clean percussive intro and outro of 1 to 8 bars for DJ mixing. |
| Generate with AI | A configured LLM writes caption, lyrics, title and metadata from a genre and subject. |
| Compose Caption | MiniMax-Music3 Structured Caption from a plain description, with genre routing and warnings. |
| Caption source | On MiniMax-Music3 and YuE2, borrow a training dataset's caption: nearest tempo, a named track, or your own. |
| Latent import | Continue from a saved `.latent` file, with its embedded metadata. |
| Backend-aware fields | BPM, duration, key, time signature, vocal gender and language show only where the active backend uses them. |
| YuE2 lead sheet preview | Opens before render when "Preview the score first" is on, and the score can be downloaded as `.abc`. |
| Drafts | The form is mirrored to saved drafts on the server; load one from the Drafts fold under Generate. On YuE2 a draft can also be saved under a name. |

## Library, player and playlists

Every render and import lands here. Details: [Library](docs/user/studios/library.md)

| Feature | What it does |
|---|---|
| Three views | Grid with cover art, list, or table, remembered per browser. |
| Parameter badges | Backend-specific saved parameters and Cover Studio tags on cards and rows; source identity and cover choices in "How it was made". |
| Source filters | Tabs per studio plus Imported, with counts. |
| Table column picker | Opt-in columns for generation details such as seed, CFG, solver and adapter, with resizable widths. |
| Multi-select | Bulk download, zip export (up to 256 MB), add to playlist, open two or more in Compare, or delete. Shift+click range select, Ctrl/Cmd+click toggle, long-press range select on touch. |
| Installable web app | Web app manifest, icons and an app-shell service worker (production builds only, never caches /api or audio), so the UI can be added to a phone home screen and run standalone. |
| Keyboard shortcuts | Desktop keys in every view except Compare and text fields: Space/K play-pause, J/L seek 10 s, and in the Library Up/Down/Enter to move and play a focused row, Delete, Esc to clear; `?` opens a cheat sheet. |
| Command palette | Desktop Ctrl+K / Cmd+K fuzzy-searches studios and library song titles; arrows and Enter to jump or play. |
| Phone bottom sheets | On phones a song's actions menu opens as a bottom sheet; swipe down or press Esc to close it. |
| Full-screen phone player | Tap the mini player on a phone for a viewport-sized dialog with big cover art, a scrubbable waveform and large transport buttons; swipe down, the chevron or the Android back button closes it. |
| Lock screen controls | Media Session support: track info and cover on the lock screen and OS overlay, with play/pause, next/previous and seek. |
| Track menu | Run or remove post-processing, edit in Custom-Gen, playlist, download, the ABC lead sheet of a YuE2 track, send to Cover Studio, metadata, Export Params (JSON preset), Retranscribe Lyrics, cover art, A/B slots, delete. |
| Upload into Library | Imports WAV, MP3, FLAC, M4A, MP4, AAC, OGG, Opus, WebM and AIFF, with optional post-processing. |
| Metadata editor | Title, artist, album, year, genre, BPM, key, comment, lyrics and cover image; some fields are embed-only. |
| Editable cover-art prompt | Generate or regenerate cover art from a pre-filled, editable prompt. |
| A/B compare | Pin two tracks, play them against each other, and open a parameter diff. |
| Song details | Backend chip, per-backend "How it was made" breakdown, prompt and lyrics. |
| Cover lyric drift | Measures sung section timing against vocal notes; sections without lyrics or vocal notes show Instrumental / N/A and leave the scoring count. |
| Streaming rows | A MiniMax-Music3 render in progress shows in the list and plays before its file exists. |
| Tagged downloads | Metadata and cover art embedded in downloaded files, in the format chosen in Settings. |
| Synced lyric export | Download .LRC saves a track's timings as a sidecar named like the audio, optionally with enhanced word stamps; a folder-mode audio download writes it too. |
| Global player | Transport, repeat modes, spectrum analyzer, disco mode and volume, visible in every view. |
| Rate and pitch preview | Playback speed from 0.5x to 2x, and a 48k/44.1k toggle that hears a render clocked at 44.1 kHz. |
| Variant switch | No Adapter, Unmastered and Mastered versions, each with its own download. |
| Waveform tools | Trim and crop in and out points, section markers, and a synced LRC lyrics bar that highlights each Whisper word with a tick per syllable on the waveform. An Edit button on the open strip opens a lyric editor modal: transport, speed, volume, the waveform, a 30 to 600 px/s zoom, one row per line with a draggable chip per word, word nudge, loop and split, whole-track nudge, undo and redo, optional autosave, and revision-checked saves that rebuild the `.lrc`. Unsaved edits can be discarded on close. |
| Playlist queue | One browser-local play queue with reorder, Play All and Download All. Ordered playlist snapshots save with server revisions. A browser playlist can be imported explicitly; a failed save keeps the local edit visible for reapplication. |

## Lyric Studio

AI-assisted lyric writing from a learned style profile, plus per-album render presets. Details: [Lyric Studio](docs/user/studios/lyric-studio.md)

| Feature | What it does |
|---|---|
| Genius fetch | Pull an artist's or album's lyrics, or add artists, albums and songs by hand. |
| Style profiles | An LLM profiles themes, tone and structure; rhyme, meter and vocabulary statistics are computed locally without one. |
| Lyric generation | Streams new songs from a profile, with an optional subject and count, avoiding subjects, keys and titles used before. |
| Write Lyrics | Local Ollama writes a complete song from a theme, style, structure or starter lines, without source songs or a learned profile, and saves it under Generated Lyrics. |
| Refine | An LLM revises lyrics and title, saved as a linked new generation. |
| Per-backend captions | ACE-Step/YuE2 caption, MiniMax-Music3 Structured Caption and YuE2 planner caption per song. |
| Caption source picker | Borrow a caption from the album's captioned training tracks. |
| Generate Audio | Queue a render through the active backend with the album's preset. |
| Send to Custom-Gen | Opens Custom-Gen with lyrics, caption, metadata and adapter preset filled in. |
| Render as / Render from | Render a song written for one album with another album's preset and caption source; the recording is listed under that album. |
| Album presets | DiT and planner adapters with group scales and a reference track on ACE-Step; one LM adapter on MiniMax-Music3; AR and NAR adapters on YuE2. |
| Editable system prompts | Edit System Prompts saves your own versions of the generation, metadata, profiler and refinement prompts. |
| Per-role providers | Separate provider and model for profiling, generation and refinement. |
| Duration from lyrics | Compute duration from lyrics and BPM instead of the LLM's estimate. |
| Randomize timbre | Pick a random reference track per render. |
| Master to Album Track | Masters every render to a random track from the album's training dataset, with Mastering on and the VST chain skipped. |
| Bulk operations | Batch Genius fetches, profile builds, lyric generation with fill-to-target, audio renders and preset assignment. Profile builds and lyric runs can process several artists at once. |
| Agent access | Drive Lyric Studio from Claude Code or Codex through the [MCP server](tools/mcp-lyricstudio/README.md). |

<!-- TODO(verify): Editable system prompts. The saved prompt_<name> settings are only read back by GET /api/lireek/prompts (server/src/routes/lireek.ts); no generation path reads them, so an edit has no effect yet. Reported as a bug. -->

## Cover Studio

Re-style a recording with ACE-Step's audio-conditioned cover or YuE2's reviewed
lead-sheet workflow, reusing dataset metadata and scores when available.
Stable Audio 3 audio transform with caption, Transform strength, seed and an optional DoRA adapter; instrumental output from the full source.
MiniMax-Music3 covers show target artists and albums with assigned MM3 adapters, with track captions or a Custom caption.

HeartMuLa's MuLaCover mode shares Prepare and saves four style tag fields and
a seed. Rendering and its separate CC BY-NC model pack are pending.

Details: [Cover Studio](docs/user/studios/cover-studio.md)

| Feature | What it does |
|---|---|
| Source upload | MP3, WAV, FLAC, OGG, M4A, Opus or AAC, or send a track from the Library. A source over 100 MB is converted to a cached MP3 for transcription, stems and alignment. |
| Latent import (ACE-Step) | Use a `.latent` or `.hslat` file and its embedded lyrics, caption, BPM and key. |
| Essentia analysis | Local BPM and key detection, cached per browser. |
| BPM and key fixes | Halve, double or type the source tempo and override its key; optionally ignore automatic BPM/key results, in which case a new source resets to 120 BPM / C major. Key help explains ACE conditioning and YuE2 score behavior. |
| Genius lyrics search | Find the source's lyrics, paste your own, or go instrumental. |
| Target artist (ACE-Step) | Fills the style description from an artist profile, or drafts one with an LLM, and loads a matching album adapter preset. |
| Structure Fidelity (ACE-Step) | Controls how closely the output follows the source arrangement. |
| Source Preservation and noise method (ACE-Step) | Starts from source audio with Classic or Full Denoise noise methods. |
| MiniMax-Music3 cover | Launders the source into a reusable plank with steps, seed and source caption controls, then covers its semantic codes under a target caption. Launder settings appear in song facts. |
| NoFSQ mode (ACE-Step) | Skips FSQ quantisation in the cover path. |
| Tempo and pitch (ACE-Step) | 0.5x to 2x tempo and -12 to +12 semitones, with the target key shown. |
| Timbre reference (ACE-Step) | A second reference track for the DiT's timbre conditioning. |
| Stem mix | Split the source with SuperSep and mute or lower stems before generating. |
| Lead-sheet review (YuE2) | Transcribe the full melody and chord score with optional SheetSage2, or paste ABC without it. Prepare it in two confirmed steps before rendering. A 2/4 score converts to 4/4 with a preview, an optional pickup bar and undo, and A beat earlier or A beat later moves every bar line by a beat; every note keeps its pitch and timing. The score player's Bar click marks beat 1 of every bar, and MIDI downloads the score with vocal, instrumental and chord tracks. The ABC/Chart switch shows the score as a song chart with bar cards per section, where chords and tempo are edited in place and a click on a bar plays from it. |
| Prepare: fix structure, assign lyrics (YuE2 and HeartMuLa) | Step 1 fixes the score: merge, split, rename or move section boundaries, split it into sections from the tagged lyrics with Sections from lyrics, regroup bars (such as 3/4 + 4/4 + 1/4 into two 4/4 bars, with preview and clear refusals), hear a selection from the score or, approximately, the recording, then confirm. Step 2 assigns pasted lyric blocks to the locked sections by hand or from tag-and-order suggestions, with opt-in recording alignment and chorus reuse as independent copies, and re-paste that keeps unchanged assignments. Generate and Build use the confirmed pair, and the server checks it. Older drafts migrate without re-transcribing. [Details](docs/user/studios/cover-studio.md#prepare-fix-the-structure-then-assign-lyrics) |
| Section and lyric aid (YuE2) | Prepare shows each score section with its label, bars and duration, and Assign lyrics shows the exact tagged lyrics a render will receive, in score order. Edits stay in the cover draft; dataset files are not changed. |
| Build by section (YuE2) | Compare up to ten takes in groups of four with shared progress. Cancel stops the current group and skips the rest. A successful render starts a fresh build; a failed one keeps it. Cover has its own AR strength (0 to 10, default 0.6). A settings row on the Build page changes voices, tempo, pitch, condition strength and AR strength until the first keep; after that, a change starts a fresh build. Take previews pause, resume, seek by tapping or dragging along the waveform, and stop on keep, delete or undo. A source bar shows the score and a song bar, empty at first, shows the output: add any section, repeat it, reorder or remove planned ones, add 2 to 16 bars the model makes up without the score (+ Improv), mark where the song ends; until then each take runs a few seconds into the section that follows it in the source; built sections stay locked until undone, and the plan survives a reload. Each chosen section's lyrics are editable beside the takes, for that section only, so a repeated chorus can sing new words; Prepare and the dataset keep theirs, and old takes can no longer be kept. A take is cropped with End at playhead before it is kept, and the last kept take can be cropped with an undo. A cut inside a section asks whether the next take composes that section again from the cut or moves on; either answer keeps the take, and composing again queues new takes from the cut with that take's caption. Each section's caption starts as the build's and can be edited for that section: it changes what is composed, while previews and the final render use one caption for the sound. Render Final Song appears in the Takes box once every section is built. Delete build (Build actions) removes the whole build and its takes from disk, refused while takes or a Save song render are in flight. |
| Lead-sheet sources (YuE2 and HeartMuLa) | Shared Prepare picker for SheetSage2, SheetSage2-Refined, MuScriptor and YourMT3+/ChordNet, with readiness per source. Drafts keep their choice; changing source or audio clears approval. MuScriptor chords are labelled as estimates. Native models are not installed yet and have no download entry. [Details](docs/user/studios/cover-studio.md) |
| Score conditioning (YuE2) | Render both vocal and instrumental score lines by default, or choose vocal melody only. Chord symbols are off by default; the source recording does not condition the audio renderer. |
| Instrumental melody mode (YuE2) | With Instrumental on, choose Vocal wins (the melody moves to Ins and cuts overlapping Ins notes) or Resting Ins bars only (the melody moves only into bars where Ins is silent, as in the YuE2 instrumental ComfyUI node). Vocal is muted either way. |
| Follow source (YuE2) | Optionally let the recording steer the composer through the real-audio tokenizer's codes, with an adherence dial, candidate count, hard restriction, an adherence curve over the song and a periodic wobble. Off while the pitch is shifted or a BPM is set. [Details](docs/user/studios/cover-studio.md) |
| Adapter pair (YuE2) | Render with the base model or an explicit AR/NAR pair captured when you queue the cover. Custom starts from the model picker's pair, and Base notes when it ignores a picked adapter. |
| HeartMuLa lead-sheet cover | Approved Prepare ABC, explicit topic/genre/instrument/mood tags, section-marked lyrics and seed; 240-second ceiling. MuLaCover weights and outputs are CC BY-NC 4.0. Engine support and downloads pending. |
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

## Compare

A/B listening sessions for songs already in the Library, on every backend: shared playhead, instant take switching, loop regions, level match, loudness and spectral metrics, a difference trace and 1 to 5 scoring. Details: [Compare](docs/user/studios/compare-studio.md)

| Feature | What it does |
|---|---|
| Sessions | Pick two or more songs from a Library search, or select them in the Library and press Compare. Sessions can be named and grouped, and archive themselves once every take is scored. |
| Shared playhead | Every take plays from the same position; Space plays or pauses and 1 to 9 switch takes. |
| A/B switch | Set A and Set B put two takes in quick-switch slots, and B or Tab flips between them. |
| Loop | Drag across a waveform to set a loop, with In, Out and Clear loop. |
| Level match | Turns every take down to the loudness of the quietest one. |
| Analysis | Integrated LUFS, short-term maximum, true peak, crest, spectral centroid, flatness and stereo correlation per take. |
| Difference trace | Loudness difference in LU and log-spectral distance in dB against the baseline take, over time. |
| Scoring | Character and Quality from 1 to 5, a Keep toggle and a note per take. |

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

Live streaming performance with crossfaded slots on ACE-Step and YuE2. Details: [STORM](docs/user/studios/storm.md)
<!-- TODO(verify): STORM also has a Stable Audio 3 path (StormPage.tsx isSa3, the server slot builder), but with Stable Audio 3 active the sidebar hides STORM and moves an open /storm to Custom-Gen (Sidebar.tsx STUDIO_NEEDS storm reads caps.features.streaming; backends/sa3/index.ts declares streaming: false). Reported as a bug; add Stable Audio 3 back here, and to the Stop row, once it is reachable. -->

| Feature | What it does |
|---|---|
| Continuous mode | Renders back-to-back slots that crossfade into one endless stream. |
| Directed Drift | Steer caption tags and backend-specific parameter targets with Pull, Rate and Jitter, parameter locks, seed evolution, macros and saved tag sets. |
| Stop | No next slot starts and the current slot is cancelled. Output is discarded, and the GPU lane stays held until the engine finishes. |
| Live controls | Change style, lyrics, seed, BPM, guidance, steps and length for the next slot without stopping. |
| Sticky fields | Pin a style or lyric change so it stays for every following slot. |
| Lyric advance | Loop, cycle or shuffle through lyric sections as slots play. |
| AI continuation | An external LLM continues the lyrics every 1, 2 or 4 slots. Named style/lyric direction presets and the prompt template save with server revisions; an older browser preset set is imported once and kept as a fallback. |
| Stream sampler overrides | Per-stream solver, scheduler and guider, changeable live. |
| Buffering | Crossfade length in beats and a maximum look-ahead buffer. |
| DJ mode | Two decks with a crossfader, Camelot key compatibility, cuts, nudges and beat quantize. |
| Sequential mode | Queues one song through the normal queue into the Library. |
| Record | Captures the stream to a `.webm` file in the browser. |
| Slot timeline | Each played slot shows its seed, detected key and BPM, and settings. |

## MIDI Studio

Audio-to-MIDI transcription on its own native engine, with the song's tempo, time signature and bar lines when the optional Beat This! model is downloaded, plus re-export with tempo corrections and quantize. Weights are CC BY-NC 4.0, and MIDI made here inherits that. Details: [MIDI Studio](docs/user/studios/midi-studio.md)

| Feature | What it does |
|---|---|
| Native transcription | `ace-midi`, a C++/GGML port of MuScriptor, on any backend, from a library track or a WAV or MP3. |
| Multi-instrument output | A `.mid` with separate parts for drums, bass, guitar, keys and the rest. |
| Model sizes | Small (CPU), medium and large. |
| Gated weights in-app | Save a Hugging Face token and download weights with live progress. |
| Live piano roll | Notes appear as chunks complete. |
| Preview | Plays original and MIDI together with a crossfade and per-instrument mute and solo. |
| Tempo & bars | With the optional Beat This! model (80 MB), the `.mid` carries the song's tempo, time signature and bar lines, and the piano roll shows numbered bars. Re-export with corrected BPM, beats per bar, half or double time, quantize, gap fill and a leading-bar trim. |
| YuE2 lead sheet | Writes the transcription as the two-voice ABC score YuE2 covers use, with an estimated key and optional chords, to copy or save for Cover Studio's ABC box. |
| Lyrics in the MIDI | A track with a `.lrc` file beside it gets each lyric line written into the `.mid` as a lyric event. |
| Persistent jobs | Transcriptions survive restarts, and a failed job that saved part of its MIDI keeps that part's preview and download. |

## Training Studio

Build datasets from your own audio and train adapters on your GPU, for ACE-Step, MiniMax-Music3, YuE2, HeartMuLa and Stable Audio 3. Details: [Training Studio](docs/user/studios/training-studio.md), [ACE-Step training](docs/user/training/ace-step.md), [MiniMax-Music3 training](docs/user/training/minimax-music3.md), [YuE2 training](docs/user/training/yue2.md), [HeartMuLa training](docs/user/training/heartmula.md), [Stable Audio 3 training](docs/user/training/sa3.md), [Training internals](docs/dev/training-internals.md)

| Feature | What it does |
|---|---|
| Dataset wizard | Scan a folder, set a trigger word and language, preview before creating. |
| Labeling | Essentia BPM and key, Genius lyrics with local Music Flamingo transcription where Genius has none, section tags normalised to one unnumbered set (and recovered from the audio for untagged lyrics), and captions from local MOSS, local Music Flamingo or a cloud LLM. |
| Local captioning | MOSS-Music-8B-Instruct and NVIDIA Music Flamingo in native GGML (`ace-caption`), in ACE-Step or MM3 caption formats, plus section-tagged lyric transcription. |
| Review grid | Spreadsheet editing of every track with bulk actions and scoped re-labels. |
| Dataset build | Writes a Side-Step-compatible `dataset.json` with trigger placement and genre ratio. |
| Training backend | The active global backend drives the phase tabs, dataset chips, Train page and checkpoint previews; a running job keeps its target. [Details](docs/user/studios/training-studio.md) |
| Preprocess and codes | Encodes audio for ACE-Step, or RVQ codes for MiniMax-Music3 with an optional cover-laundering pass. |
| ACE-Step adapters | Planner LM adapters (0.6B, 1.7B, 4B) and DiT adapters, as LoRA or LoKr. There are no automatic checkpoint previews; Audition compares adapters and saved milestones. |
| MiniMax-Music3 adapters | Planner LM LoRA trained from RVQ codes, at a BF16 training precision about 2.6x faster than FP32 by default. |
| HeartMuLa adapters | A native backbone LoRA trained on the 3B model, with HeartCodec preparation, recipe presets, a default 30-epoch budget capped at 480 updates, a durable multi-dataset queue, Prepare-then-Train chains, scoped cache reset and checkpoint Review. Needs a CUDA build and about 12 GB of VRAM. [Details](docs/user/training/heartmula.md) |
| HeartMuLa checkpoint previews | Training pauses every N steps, renders a full song from the saved checkpoint, with the tags and lyrics taken from the dataset, then resumes; the base model and the final adapter are rendered too. [Details](docs/user/training/heartmula.md#checkpoint-previews) |
| MiniMax-Music3 checkpoint previews | Render previews during training pauses the trainer at checkpoints to render samples beside the base model, then resumes it. The previews sit in the same checkpoint ladder as YuE2's, kept after a reload, with YuE2's blind rungs, rung and album scores, Render more, and Use this checkpoint, which records the dataset's pick, sets the LM adapter and any existing album presets to that step, and offers cleanup even without an album preset. [Details](docs/user/training/minimax-music3.md#picking-a-checkpoint) |
| YuE2 adapters | Five-stage chain with Perform all stages, Joint Training of the AR and NAR pair as LoKr (the default), LoRA or DoRA with Prodigy, AdamW or Muon, checkpoint losses averaged over the last 20 logged steps, a visible live loss graph, and Review entries for unscored runs even without previews. |
| YuE2 run review | Train keeps completed charts without replaying them again; finished and approved runs leave "Scored or finished". Delete run confirms removal of the joint run folder, checkpoints, previews and MP3 cache; archived logs and checkpoints saved outside that folder stay. Live runs cannot be deleted. Ladder playback uses cached MP3, with WAV fallback. [Details](docs/user/studios/training-studio.md) |
| YuE2 training GPU | Choose an NVIDIA card per Joint Training job, or use the GPU selected in Settings. [Details](docs/user/training/yue2.md#recommended-settings) |
| YuE2 held-out planner gate | Optional in Joint Training: hold a few songs out, score the planner on them as it trains, and freeze it when that loss stops improving so the decoder can train on. The card shows the held-out songs before Start. Off by default; the settings are starting points, not validated ones. [Details](docs/user/training/yue2.md#recommended-settings) |
| Optimise phase (YuE2) | Measure a prepared album before training, starting with the base model's loss per song; results saved in the dataset folder. [Details](docs/user/training/yue2.md#optimise-optional) |
| Training on another PC | Send YuE2 or MiniMax-Music3 batches to a second PC's GPU, then follow, listen, score and fetch the adapters from this one. Details: [YuE2](docs/user/training/yue2.md#training-on-another-pc), [MiniMax-Music3](docs/user/training/minimax-music3.md#training-on-another-pc) |
| Quality presets | Fast, Balanced and Thorough presets on the training forms. |
| Stable Audio 3 training | Independent SA3 Medium DoRA-rows target with SuperSep level 5 vocal stripping, saved-run resume, adapter picker and checkpoint previews rendered during training (the YuE2 preview view), a four-chip dataset card and fixed-seed SA3 auditions regardless of the selected backend. [Details](docs/user/training/sa3.md) |
| YuE2 checkpoint previews | Adjustable music adapter scale, default 0.6, with sound adapter strength fixed at 1.0 for training and manual previews. [Details](docs/user/training/yue2.md#recommended-settings) |
| YuE2 saved presets | Named training settings load with server-resolved legacy defaults while current dataset and output paths stay in place. [Details](docs/user/training/yue2.md#recommended-settings) |
| Adapter methods | DoRA, rsLoRA, LoRA+, HiRA, LoHa, PiSSA and HRA, plus learned artist tokens and a trainable KV prefix on the LM trainers. |
| Small PiSSA adapters | The base model's own directions ship once, so each adapter carries only what it learned. |
| Quantized-base training | Train against a K-quant or MXFP4 base for a much lower VRAM floor. |
| Flash-attention training | Fused attention that makes memory linear in sequence length; on by default for the ACE-Step DiT and LM trainers and the MiniMax-Music3 LM trainer. |
| Target-loss stopping | Stop on a loss target instead of a step count, and continue a stopped run. |
| Live monitoring | Loss chart, stats and milestone checkpoint badges. Finished YuE2 joint charts recover from disk after a restart; missing history shows an unavailable message. |
| YuE2 training charts | Separate AR planner and NAR decoder panels, metric explanations, More series controls and run-configured stop readings across joint, separate and Refine runs. [Details](docs/user/training/yue2.md#reading-the-training-charts) |
| Audition | Same-seed A/B of base and adapter planner, optionally rendered through the DiT. |
| Adapter scoring | Evaluation passes put an artist-match score in the planner adapter picker. |
| YuE2 Review | The YuE2 phases are Dataset, Prepare, Optimise, Train and Review. Review lists the checkpoint ladders awaiting your scores and opens faster with one run catalogue read per request for all datasets; score them by ear and a scoreboard ranks the rungs by likeness, corruption and re-plans. Refine, which pushed a planner adapter up KL rungs, is no longer a tab and opens only from Review for ladders made with the earlier recipe. |
| Batch pipeline | Import multiple folders and run label, build, preprocess and train unattended; Train multiple (YuE2 and MiniMax-Music3) chains each dataset's stages with one recipe. |
| Send to Lyric Studio | Export a dataset's artist and album with its trained adapters as a preset. |

## Settings

Configuration for engine, server, providers, downloads and storage, with album-preset adapter cleanup and data-folder move, repoint, empty-library and previous-install import actions. Details: [Settings](docs/user/studios/settings.md)

| Feature | What it does |
|---|---|
| Environment editor | Edits `.env` with only changed keys written, and a Restart now button when a key needs it. |
| GPU device picker | Lists NVIDIA cards by name and VRAM and stores a stable GPU UUID. |
| Multi-GPU workers (experimental) | Opt-in per-card engines for machines with two or more NVIDIA cards, plus a worker status list. With two or more cards ticked, jobs run in parallel, one per card. [Details](docs/user/studios/settings.md) |
| Keep models in VRAM | Engine flag that keeps DiT, adapter and VAE resident between renders, plus a per-request toggle that skips DiT/VAE swaps. |
| Warm on startup | `.env` settings that preload a DiT, VAE and adapter after the engine boots. |
| Pipeline parallelism | Run Whisper, quality evaluation and cover art alongside post-processing. |
| Generation timeout | 10 minutes to 6 hours, per track. |
| AI services | API keys and endpoints for Gemini, OpenAI, Anthropic, Ollama, LM Studio, llama.cpp, Unsloth and OpenAI-compatible servers, applied live. |
| Download defaults | WAV, FLAC, Opus or MP3, bitrate, mastered or original, and optional latent file. A Download folder saves downloaded tracks there instead of the browser's download location; zip exports still go through the browser. |
| Filename trigger | Use an adapter's filename as its trigger word, prepended, appended or replacing the caption. |
| Display language and theme | Interface language, light or dark theme, and generic artwork sets. |
| Settings and Lyric Studio backup | Export versioned JSON and preview a validated replacement before confirming import. [Details](docs/user/settings-backup.md). |
| Storage cleanup | Stem storage counter and separate nukes for generations, written lyrics and profiles. |
| Data folder | Release data persists outside the app across versions; the Data folder panel moves it, points at another folder, starts an empty library or imports a previous install. A development checkout can move and repoint its data too. See [Data folder and upgrades](docs/user/data-folder.md). |
| Updates | Shows the installed build and compares it with the Release or Master feed on GitHub, at startup once a day or with Check now. It only tells you; the app does not update itself. |
| Terminal, Restart and Quit | Live engine log with search and a VRAM badge; sidebar restart and clean shutdown. |

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

Browse, download and remove every model file in the app. It is not in the sidebar: open it with Get More Models in the global bar's Models section. Details: [Model Manager](docs/user/studios/model-manager.md), [Models](docs/user/models.md)

| Feature | What it does |
|---|---|
| Family tabs | ACE-Step 1.5, MiniMax-Music3, YuE2 and Shared, with role sub-tabs and installed counts. Stable Audio 3 files are under Shared, on the StableStep tab. |
| Starter packs | One-click bundles per backend, training packs among them, plus shared CUDA runtime, stem separation, StableStep separation, captioning and Stable Audio 3 training packs. |
| Concurrent downloads | Several files at once with percent, speed and ETA, and resume after an interruption. |
| Verification | Size and header checks before a downloaded file is kept. A file whose registry entry carries a sha256 is re-downloaded when the copy on disk does not match. |
| Delete | Remove installed files to free space. |
| StableStep licence gate | Licence acceptance and an optional Hugging Face token. |
| TensorRT builder match | Marks the builder DLL that matches your GPU. |
| First-launch prompt | Opens itself when no models are found, and the engine starts by itself once the downloads finish. |

## Engine and plugins

The C++17 engine behind every backend. Details: [Plugins](docs/user/plugins.md), [Plugin authoring](docs/dev/plugins-authoring.md), [Engine](docs/dev/engine.md), [Architecture](docs/dev/architecture.md)

| Feature | What it does |
|---|---|
| Portable builds | Windows (CUDA, Vulkan, CPU), Linux (CUDA, Vulkan, ROCm, CPU) and macOS Metal releases. |
| Google Colab notebook | Installs a release on a Colab GPU, with model packs, Drive persistence and a link to the UI. See [Colab](docs/user/colab.md). |
| Engine binaries | `ace-server`, `ace-lm`, `ace-synth`, `ace-understand`, `ace-train`, `ace-caption`, `ace-midi`, `neural-codec`, `mp3-codec`, `quantize`, `mastering` and `vst-host`. |
| Lua plugins | Solvers, schedulers, guidance modes and postprocess plugins load at launch with no rebuild, from `engine/plugins/`, the community `plugins/` folder and, in a release, the data folder's `plugins/`, where a plugin replaces a shipped one of the same name. |
| Plugin UI | Each plugin declares its own sliders, toggles and dropdowns, rendered by the UI. |
| Native bridge | `apg()` for momentum, projection and norm thresholding, and `post_step()` for extra forward passes. |
| Loop-owning solvers | A solver can take over the whole denoising loop for adaptive stepping. |
| Model formats | GGUF, safetensors directories and ONNX, detected by path. |
| Quantize tool | K-quants, IQ quants, MXFP4 and NVFP4, with importance-matrix support. |
| TensorRT paths | The MiniMax-Music3 flow DiT through TensorRT on NVIDIA, in the Windows CUDA 13.1 build. An ACE-Step DiT or LM exported to ONNX runs through TensorRT in that build too. |
| Flash-attention training ops | Custom fused attention forward and backward in the ggml CUDA backend. |
| Upstream hooks | Fork hooks and the HOT-ggml fork checked by `engine/verify-hooks.ps1` before every build. |

## Developer tooling

For contributors and agents. Details: [Building](docs/dev/building.md), [HTTP API](docs/dev/api.md), [Architecture](docs/dev/architecture.md), [Releasing](docs/dev/releasing.md), [Writing the docs](docs/dev/docs-contributing.md)

| Feature | What it does |
|---|---|
| HTTP API | Everything the UI does is JSON over HTTP on port 3001, indexed per route. |
| Dev loop | `dev.bat` (or `make dev` on Linux and WSL) runs Vite with hot reload and the Node server in watch mode. |
| Safe engine rebuild | `dev-rebuild.bat` shuts the app down cleanly before rebuilding the engine. |
| Release gate | [`tools/release-gate`](tools/release-gate/README.md) drives the app through its API and stages renders for an ear test. |
| Release prerequisites | `check-release-prereqs.mjs` confirms every model and data file is reachable by a user. |
| Docs checks | `tools/docs/build-docs.mjs` regenerates tables; `check-docs.mjs` catches drift and broken links. |
| Engine smoke test | `node tools/engine-smoke.mjs` renders a fixed-seed song headlessly and fails on NaN, silence, DC offset or clipping. |
| Lyric Studio MCP server | [`tools/mcp-lyricstudio`](tools/mcp-lyricstudio/README.md) exposes Lyric Studio to Claude Code and Codex, with shared discussion rooms and full-conversation PDF export. |
| HOT-Step MCP server | [`tools/mcp-hotstep`](tools/mcp-hotstep/README.md) exposes generation and training over the app's HTTP API. See [MCP](docs/user/mcp.md). |
| DAW facade | `/api/v1/daw` lets a DAW client generate and analyse audio against the running app. See [DAW facade notes](docs/dev/api.md#daw-facade-notes-apiv1daw). |
| RunPod target | A `runpod` Docker target for a RunPod GPU Pod; the published image arrives with the next release. See [RunPod Pod template](docs/dev/runpod.md). |
