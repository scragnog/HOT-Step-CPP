# HOT-Step CPP — Features

Everything HOT-Step CPP adds on top of the base [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) engine.

---

## C++ Inference Engine

Built on acestep.cpp (GGML/CUDA), with extensive modifications to the sampling, scheduling, and guidance systems:

### Lua Plugin Architecture

All solvers, schedulers, and guidance modes are implemented as hot-loadable Lua plugins. Drop a `.lua` file into the appropriate `engine/plugins/` subdirectory and it appears in the UI at next launch — no C++ rebuild required. Each plugin can declare its own user-facing parameters (sliders, toggles, dropdowns) via a schema table, which the UI renders dynamically.

The engine provides a native C++ bridge for performance-critical operations (APG momentum smoothing, perpendicular projection, norm thresholding) that Lua plugins can call via the `apg()` function. Advanced plugins can also declare a `post_step()` hook that receives model evaluation callbacks for techniques requiring extra forward passes at arbitrary latent positions. See the **[Plugin Authoring Guide](PLUGINS.md)** for the full API reference.

#### Solvers (17)

ODE/SDE solvers for the flow matching sampling loop:

| Plugin | Description |
|--------|-------------|
| **Euler** | 1st-order Euler method (1 NFE) |
| **Heun** | 2nd-order Heun's method (2 NFE) |
| **RK4** | Classic 4th-order Runge-Kutta (4 NFE) |
| **RK5** | 5th-order Runge-Kutta (6 NFE) |
| **GL2s** | Gauss-Legendre 2-stage implicit Runge-Kutta (2 NFE) |
| **RF-Solver** | 2nd-order rectified flow solver (2 NFE) |
| **DPM++ 2M** | DPM-Solver++ multistep 2nd-order (1 NFE) |
| **DPM++ 2M Adaptive** | Adaptive step-size variant of DPM++ 2M |
| **DPM++ 3M** | DPM-Solver++ multistep 3rd-order (1 NFE) |
| **UniPC** | Unified predictor-corrector (1 NFE) |
| **UniPC-P** | UniPC with p-corrector (1 NFE) |
| **JKASS Quality** | Multi-evaluation adaptive solver (4 NFE) |
| **JKASS Fast** | Single-evaluation JKASS variant (1 NFE) |
| **AFLOPS / AFLOPS-2** | Adaptive flow ODE solver with error estimation |
| **DOPRI5 / DOP853** | Dormand-Prince adaptive solvers (5th/8th order) |
| **SDE** | Stochastic differential equation solver with Philox RNG |
| **STORK-2 / STORK-4** | Stochastic Taylor Runge-Kutta solvers |

#### Schedulers (9)

Noise schedule curves for the denoising trajectory:

| Plugin | Description |
|--------|-------------|
| **Linear** | Uniform timestep spacing |
| **Cosine** | Cosine-annealed schedule |
| **Power** | Polynomial schedule with configurable exponent |
| **SGM Uniform** | Score-based generative model uniform schedule |
| **DDIM Uniform** | DDIM-style uniform schedule |
| **Linear-Quadratic** | Linear start transitioning to quadratic |
| **Beta (5,7)** | Beta distribution schedule |
| **Bong Tangent** | Tangent-based custom schedule |
| **Beta Math** | Generalised beta distribution with configurable α/β |

#### Guidance Modes (7)

Classifier-free guidance strategies, all routed through the native APG bridge:

| Plugin | Description |
|--------|-------------|
| **APG** | Analytical Perpendicular Guidance — momentum smoothing, perpendicular projection, norm thresholding |
| **Dynamic CFG** | Adaptive guidance scale that varies across timesteps — high early for structure, low late for detail |
| **CFG++** | Manifold-constrained guidance for few-step models |
| **Rescaled CFG** | Standard-deviation-based rescaling to prevent oversaturation |
| **CFG-Zero⋆** | Zero-init guidance — zeroes early ODE steps where CFG predictions are counterproductive (Fan et al. 2025) |
| **SMC-CFG** | Sliding Mode Control guidance — control-theoretic correction for stability at high scales (Han et al. 2025) |
| **CFG-MP** | Manifold Projection — iterative post-step projection using extra model evaluations to reduce the prediction gap (Su et al. 2025). Uses the `post_step()` hook for model callbacks |

### Other Engine Features

| Feature | Description |
|---------|-------------|
| **Composite 2-Stage Scheduler** | Blend two scheduler curves across the denoising trajectory for fine-grained noise control. |
| **Auto-Shift** | Adaptive noise shift scaling that adjusts based on track duration and step count. |
| **DCW Sampling** | Differential Correction in Wavelet domain — an alternative sampling technique calibrated for the GGML engine. |
| **Sideband Parameter Channel** | Extension layer for passing HOT-Step-specific parameters without modifying upstream function signatures, keeping the acestep.cpp sync path clean. |
| **Latent Post-Processing** | Latent shift, latent rescale, and custom timestep scheduling — expose the latent space for experimentation. |
| **LM Seed Locking** | Ties the LM seed to the DiT seed — locking the seed locks both, randomising randomises both. |
| **Upstream Sync Infrastructure** | Marker-based system for tracking acestep.cpp divergence and cleanly merging upstream changes. |

---

## LoRA / Adapter System

| Feature | Description |
|---------|-------------|
| **Per-Group Adapter Scales** | Independent scale control for self_attn, cross_attn, mlp, and cond_embed weight groups. |
| **K-Quant Adapter Support** | Custom CUDA copy kernels for GPU-accelerated merge of Q4_K_M, Q5_K_M, and Q6_K quantised adapters. |
| **Threaded CPU Dequant** | Multi-threaded CPU fallback with AVX512 fast-path for K-quant adapter merge when GPU copy isn't available. |
| **Runtime LoRA Mode** | Apply LoRA deltas in the forward pass graph at inference time, instead of permanently merging weights. Switchable per-generation. |
| **Adapter Browser** | File browser modal with scan endpoints, trigger word injection, and support for absolute paths outside the registry. |
| **Adapter Scale Override Presets** | Predefined scale profiles (e.g. "vocals up", "instruments up") selectable from the sidebar. |
| **Merge Model Detection** | Correctly identifies SFT-turbo blend models and skips inappropriate guidance clamping that would degrade output. |

---

## Audio Processing Pipeline

| Feature | Description |
|---------|-------------|
| **Lossless WAV Pipeline** | Engine outputs WAV16; 32-bit float WAV used throughout the processing chain to preserve dynamic range. |
| **Matchering Mastering Engine** | Integrated loudness, EQ, and dynamics matching to a user-supplied reference track. |
| **Mastered / Unmastered Toggle** | Instant A/B comparison via dual WaveSurfer instances with synced playback position. |
| **Spectral Denoiser** | Wiener-filter spectral subtraction for post-generation artifact removal (evolved from initial spectral gating approach). |
| **Profile-Based Denoiser** | Learns a noise profile from a reference sample for targeted, surgical artifact removal. |
| **Spectral Lifter** | Post-processing pipeline with tunable parameters for spectral shaping. Ported to native C++ (originally a Python subprocess). |
| **VST3 Host** | Scans, loads, and runs VST3 plugins for offline audio processing — 40+ plugins detected from standard install paths. |
| **VST3 Chain in Pipeline** | Wire a VST3 processing chain directly into the generation output — mastering, EQ, compression, etc. from your existing plugin collection. |
| **Real-Time Monitor (WASAPI)** | Low-latency audio preview with seek and transport controls for auditioning output before committing. |
| **Duration Buffer + Auto-Trim** | Generates slightly longer than requested, then detects natural song endings and trims cleanly — no more abrupt cuts. |
| **Configurable Fade-Out** | Slider-controlled fade duration; automatically skipped when auto-trim detects a clean ending. |
| **Download with Format Conversion** | Export as WAV, MP3, or FLAC with configurable defaults in Settings. |
| **48kHz Native Processing** | Mastering pipeline operates at the native 48kHz sample rate — no lossy resample round-trip. |
| **PP-VAE Neural Audio Polish** | Post-processing VAE that runs generated audio through an encode→decode round-trip to smooth spectral artifacts and improve tonal coherence. Optional wet/dry blend slider. F32 recommended for best quality. |
| **ScragVAE Decoder** | Fine-tuned VAE decoder with +38% high-frequency energy and +29dB dynamic range improvement. Drop-in replacement for the standard decoder — selectable at runtime from the Models dropdown. |

---

## UI / UX

| Feature | Description |
|---------|-------------|
| **Full React + Tailwind UI** | Purpose-built dark-themed interface, ported and extended from the Python-based HOT-Step 9000. |
| **WaveSurfer.js Waveform Player** | Bars-mode waveform visualisation with hover plugin; animated collapse/expand on pause. |
| **Spectrum Analyzer** | audioMotion-analyzer integration with mirrored bar mode and configurable density. |
| **Global Parameter Top Bar** | All engine settings extracted into a persistent, colour-coded top bar with collapsible section dropdowns. |
| **VRAM Indicator** | Real-time GPU memory usage display in the top bar. |
| **Terminal Panel** | Verbose generation progress streamed via SSE with batched UI updates for performance. |
| **Generation Queue** | Queue additional generations while one is running; completed/cancelled jobs auto-dismiss after 3 seconds. |
| **JSON Preset Export / Import** | Save and load complete generation parameter sets as JSON files. |
| **Global Playlist Sidebar** | Persistent, resizable playlist replacing per-page floating players. |
| **Inline Song Rename** | Pencil icon on any track for quick title editing. |
| **Bulk Select & Delete** | Multi-select tracks in the library for batch deletion. |
| **Human-Readable Model Labels** | Friendly names for GGUF model files with enriched badge summaries showing quantisation and size. |
| **Toggle Switches** | All boolean controls use styled toggle switches instead of plain checkboxes. |
| **Persistent UI State** | Accordion states, sidebar collapse, scroll positions, and panel sizes all persist across navigation. |
| **Per-Track Download Buttons** | Download individual tracks directly from the playlist sidebar. |
| **Artist-Title Metadata** | Create page fields for artist name, subject description, and key signature — embedded in generation metadata. |

---

## Lyric Studio

A complete AI-powered lyrics and music generation workspace, powered by the Lireek backend:

| Feature | Description |
|---------|-------------|
| **Lireek Backend** | Full server-side lyric engine with SQLite database for artists, albums, profiles, and generations. |
| **LLM Orchestration** | 6 LLM provider integrations (Gemini, LM Studio, OpenAI-compatible, etc.) with real-time SSE streaming. |
| **Artist Profiles** | Per-artist configuration with adapter presets, reference tracks, style summaries, and computed generation statistics. |
| **Lyric Profiler** | Statistical analysis engine — contraction rates, rhyme schemes, meter patterns, perspective tracking — computed locally without LLM calls. |
| **Streaming Generation** | Real-time SSE streaming of lyrics with live UI updates as the LLM writes. |
| **Audio Generation Queue** | Integrated music generation from lyrics with full parameter parity to the Create page. |
| **Bulk Operations** | "Fill to N" mode — auto-calculates how many generations each profile needs to reach a target count, with progress badges. |
| **Send to Create** | Transfers artist context, adapter path, reference track, key signature, and all metadata to the Create page in one click. |
| **Artist Sidebar** | Persistent sidebar with artist list, scroll position memory, and per-artist song counts. |
| **Album Pages** | Browse by album with header bars, generated songs tab, and inline audio playback. |
| **Database Migration** | Import tool for migrating from HOT-Step 9000’s `hotstep_lyrics.db` — artists, profiles, and generations. |
| **Dynamic LLM Model List** | Fetches available models from provider APIs instead of using a hardcoded list. |
| **Profile Stats Recalculation** | One-click re-run of all local statistical analysis without making any LLM calls. |
| **LRC Synced Lyrics** | Timestamped lyric display synced to audio playback with seeking support. |
| **Track Cropping** | Destructive IN/OUT point editing for trimming generated tracks to clean boundaries. |

---

## Cover Studio

Full-featured cover generation workspace with audio analysis, stem manipulation, and artist-specific generation:

| Feature | Description |
|---------|-------------|
| **Audio Analysis** | Essentia-based extraction of BPM, key, energy, and timbre characteristics from source tracks. |
| **Source Upload** | Upload and analyse reference audio for style-matched cover generation. Drag-and-drop with format auto-detection. |
| **BPM Correction** | ÷2 / Detected / ×2 buttons to fix Essentia’s common tempo halving/doubling errors. |
| **Key Override** | Manual key correction dropdown when Essentia’s detection is wrong — shows both detected and overridden keys. |
| **Style Description** | Editable caption field for describing the target style. Auto-filled from artist profile when available, freely editable. |
| **Artist-Optional Generation** | Generate covers using just a style description — no artist or adapter required. |
| **Pitch Shift** | ±12 semitone slider with real-time key transposition preview (e.g. “+3 st → F Major”). |
| **Tempo Scale** | 0.5x–2.0x tempo slider with computed BPM preview. |
| **Structure Fidelity** | Controls how closely the output follows the source’s arrangement and structure. |
| **Source Timbre** | Controls how much of the original artist’s sonic character is preserved in the output. |
| **Timbre Reference Conditioning** | Uses the target artist’s reference track as a DiT timbre conditioner to influence sonic character. |
| **Stem Separation + Recombination** | Advanced mode: split source into stems via SuperSep, configure the stem mix, then generate from the recombined audio. |
| **Album Adapter Presets** | Per-album adapter presets with bound reference tracks — select an album to auto-load the matching adapter and reference. |
| **Cover Generation UI** | Full workspace with metadata extraction, artist grid, cover-specific sliders, progress tracking, and recent covers list. |
| **Persistent State** | All settings, selections, and analysis results persist across navigation and reloads. |

---

## Stem Studio

Neural audio source separation with a 4-stage ONNX pipeline and interactive stem mixer:

| Feature | Description |
|---------|-------------|
| **SuperSep Pipeline** | 4-stage cascaded separation using specialised ONNX models for different instrument groups. |
| **Stage 1: BS-RoFormer** | Primary 6-stem split (Vocals, Drums, Bass, Guitar, Piano, Other) using Band-Split RoFormer with full-track chunking. |
| **Stage 2: Mel-Band RoFormer** | Vocal sub-separation into Lead Vocals and Backing Vocals. Full-track processing with late-vocal detection. |
| **Stage 3: MDX23C** | Drum sub-separation into Kick, Snare, Toms, Hi-Hat, Cymbals, and Other Percussion via STFT-based MDX processing. |
| **Stage 4: HTDemucs** | Hybrid transformer for “Other” refinement — dual-input model taking both STFT spectrograms and raw waveforms, with dual-output combination. |
| **4 Separation Levels** | Basic (6 stems), Vocal Split (+ lead/backing), Full (+ drum sub-stems), Maximum (+ other sub-stems). |
| **Interactive Stem Mixer** | Multi-solo, mute, and per-stem volume sliders with real-time Web Audio playback. |
| **Chunking + Overlap-Add** | Full-length audio processing with 1-second crossfade windows for seamless chunk boundaries. |
| **Sequential VRAM Management** | Models loaded and released strictly sequentially — peak GPU usage stays under 3 GB. |
| **Per-Stage WAV Exports** | All stages generate raw WAVs in `stage-N/` directories for diagnostics, regardless of downstream routing. |
| **Hidden Intermediate Stems** | Debug stems (e.g. raw Vocals before lead/backing split) saved to disk but filtered from the UI mixer. |
| **MDX STFT Preprocessing** | Generic engine function for MDX23C and HTDemucs models with stripped STFT layers — handles the [1,4,dim_f,T] tensor layout. |
| **Source Library Browser** | Pick source audio from the song library with search, source filtering, and mastered/unmastered toggle. |
| **ZIP Download** | Download all stems as a single ZIP archive, or download individual stems. |
| **Persistent Source Selection** | Source audio URL and filename persist across sessions via localStorage. |

---

## Stem Builder

Generatively create new instrument stems for source tracks using the DiT engine:

| Feature | Description |
|---------|-------------|
| **Generative Stem Creation** | Select a source audio file and generate new AI-created instrument layers (vocals, drums, bass, guitar, piano) that complement the original track. |
| **Instrument Layer Selection** | Choose which stems to generate — add missing instruments or create alternative takes for existing ones. |
| **Per-Stem Preview** | Real-time audio preview of generated stems alongside the source track with per-stem volume controls. |
| **Source Audio Browser** | Browse source audio from the song library with search and mastered/unmastered toggle. |
| **Iterative Layering** | Build up arrangements by generating stems one at a time — each new layer is created in the context of the existing mix. |
| **Full Pipeline Integration** | Generated stems pass through the complete engine pipeline including post-processing, mastering, and format export. |

---

## AI Assistant

In-app LLM-powered assistant with full context awareness:

| Feature | Description |
|---------|-------------|
| **Streaming Chat Sidebar** | Toggleable chat panel with SSE-streamed responses, markdown rendering, and thinking/response separation. |
| **Full Settings Awareness** | Every message includes a JSON snapshot of all engine parameters, content fields (lyrics, caption, BPM, duration, key, time signature, language), and active mode. |
| **Mode-Aware Guidance** | Automatically detects which studio the user is in (Create, Lyric Studio, Cover Studio, Stem Studio, Stem Builder) and tailors advice to that workflow. |
| **Actionable Suggestions** | LLM responses can include structured action blocks that the user can preview as diffs and apply individually or in bulk — settings update reactively. |
| **Content Editing** | Can write, rewrite, or update lyrics, style descriptions, and other content fields directly via action blocks with one-click apply. |
| **Per-Action Apply** | Each suggested change has its own Apply button — cherry-pick individual settings without accepting the full batch. Applied items show a checkmark and dim out. |
| **Thinking Separation** | LLM chain-of-thought is separated from the response and displayed in a collapsible "💭 Thought process" block — visible but visually distinct. |
| **Multi-Provider Support** | Uses the same LLM provider registry as Lyric Studio — supports Gemini, LM Studio, OpenAI-compatible endpoints, etc. Provider and model selection persisted independently. |
| **Knowledge Base** | Static knowledge base covering all engine parameters, solvers, schedulers, guidance modes, adapters, post-processing, troubleshooting, and lyric formatting rules. |
| **Markdown Rendering** | Lightweight built-in renderer for headers, bold, italic, inline code, fenced code blocks, lists, and horizontal rules — no external dependencies. |

---

## Timbre & Audio Conditioning

| Feature | Description |
|---------|-------------|
| **Timbre Reference** | Use a reference track as a DiT timbre conditioner to influence the sonic character of generations. |
| **FLAC Decoding** | Native dr_flac support for FLAC reference files alongside WAV and MP3. |
| **LM Code Cache** | Cache LM-generated audio codes for deterministic re-generation with consistent structure. |
| **LM Codes Strength** | Slider controlling how strongly cached LM codes influence the generation — from subtle guidance to exact reproduction. |
| **Co-Resident Models** | Run DiT and VAE from different model files simultaneously (e.g. turbo DiT with full VAE). |

---

## Settings & Configuration

| Feature | Description |
|---------|-------------|
| **Settings Page** | Central configuration hub for models, adapters, mastering references, and download preferences. |
| **Smart Defaults** | Works out of the box without a `.env` file — auto-discovers engine binary and model paths. |
| **Selectable VAE Decoder** | Choose between standard and alternative VAE decoders at runtime. |
| **LM / Thinking Toggle** | Skip or enable the LM inference phase entirely — useful for speed when you don't need metadata generation. |
| **Nuke Generations** | One-click wipe of all generated content and database entries. |
| **Configurable Download Defaults** | Set preferred export format, filename prefix, and download behaviour. |
| **Environment Editor** | Read and edit the server's `.env` file directly from the Settings UI with categorised sections, masked API keys, and save confirmation. |
| **Runtime Config Reload** | Hot-reload LLM provider settings and API keys without restarting the server. Engine-level changes show a restart notification. |

---

## Model Manager

| Feature | Description |
|---------|-------------|
| **In-App Model Downloads** | Browse and download 100+ GGUF models directly from the app — no manual file management needed. |
| **Curated Starter Packs** | 4 pre-configured bundles (Quick Start, Minimal, XL Quality, Blackwell Optimized) with one-click download of the full pipeline. |
| **Tabbed Model Catalogue** | Browse all available models organised by role (DiT, LM, Text Encoder, VAE, PP-VAE) with descriptions and quantisation badges. |
| **Concurrent Resumable Downloads** | Multiple simultaneous downloads with HTTP Range-based resumption — interruptions resume from where they left off. |
| **Real-Time Progress** | SSE-streamed download progress with speed, ETA, and per-file status tracking. |
| **Installed Status Tracking** | The catalogue shows which models you already have installed, with per-pack completion indicators. |
| **Model Deletion** | Remove installed models directly from the UI with confirmation prompts. |
| **5 HuggingFace Repos** | Models sourced from Serveurperso/ACE-Step-1.5-GGUF, scragnog/ace-step-1.5-gguf-merge-models, scragnog/Ace-Step-1.5-MXFP4-Quants, scragnog/Ace-Step-1.5-ScragVAE, and scragnog/HOT-Step-CPP-PP-VAE. |

---

## Build & Developer Tools

| Feature | Description |
|---------|-------------|
| **dev-rebuild.bat** | Graceful HTTP shutdown of the running app before engine rebuild — prevents the supervisor's auto-restart from causing a respawn loop. |
| **MSVC Build Compatibility** | Automatic Visual Studio discovery via vswhere, Ninja binary fallback, and Node.js version guard. |
| **File-Based Logging** | Structured logging system mirroring HOT-Step 9000 patterns for consistent debugging. |
| **Quantize Tool** | Experimental GGUF quantisation with IQ, NVFP4, MXFP4, and ternary format support. |
| **Quant Benchmark** | Automated inference benchmarking with peak VRAM tracking and results logging. |
| **MXFP4 Tensor Core Tests** | Blackwell GPU stress tests demonstrating 22–33% speedup with MXFP4 quantisation. |
| **Graceful Shutdown** | Proper Windows process cleanup with a "you can close this page" confirmation screen. |
