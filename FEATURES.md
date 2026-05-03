# HOT-Step CPP — Features

Everything HOT-Step CPP adds on top of the base [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) engine.

---

## C++ Inference Engine

Built on acestep.cpp (GGML/CUDA), with extensive modifications to the sampling, scheduling, and guidance systems:

| Feature | Description |
|---------|-------------|
| **Modular Solver Plugin System** | 14 ODE/SDE solvers — Euler, RK4, Heun, Gauss-Legendre 2s (implicit Runge-Kutta), RF-Solver (2nd-order rectified flow), JKASS Quality, and more. Each solver exposes its own sub-parameters. |
| **Modular Scheduler Plugin System** | 8 noise schedulers with conditional sub-parameters (power exponent, beta range, composite blend). |
| **Composite 2-Stage Scheduler** | Blend two scheduler curves across the denoising trajectory for fine-grained noise control. |
| **Modular Guidance Mode Plugin System** | 4 guidance modes including Dynamic CFG with adaptive scaling. |
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
| **Database Migration** | Import tool for migrating from HOT-Step 9000's `hotstep_lyrics.db` — artists, profiles, and generations. |
| **Dynamic LLM Model List** | Fetches available models from provider APIs instead of using a hardcoded list. |
| **Profile Stats Recalculation** | One-click re-run of all local statistical analysis without making any LLM calls. |

---

## Cover Studio

| Feature | Description |
|---------|-------------|
| **Audio Analysis** | Essentia-based extraction of BPM, key, energy, and timbre characteristics from source tracks. |
| **Source Upload** | Upload and analyse reference audio for style-matched cover generation. |
| **Cover Generation UI** | Full workspace with metadata extraction, artist selection, cover-specific settings, and recent covers list. |

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
