# HOT-Step CPP

> **⚠️ ALPHA — ACTIVE DEVELOPMENT ⚠️**
>
> This project is in **early alpha** and under very active development. Features may be incomplete, unstable, or change without notice.
>
> **For a more complete experience right now, use [HOT-Step 9000](https://github.com/scragnog/HOT-Step-9000)** — the Python-based version with a full feature set and wider platform support. This C++ version is being developed as a faster, lighter alternative and will mature over time.

A feature-rich UI for [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) — local AI music generation powered by GGML.

Describe a song with a text caption and lyrics, and get stereo 48kHz audio generated entirely on your local hardware. No cloud, no API keys, no subscriptions.

## Highlights

HOT-Step CPP extends the base acestep.cpp engine with 80+ features across inference, audio processing, and creative tooling. Here are the big ones:

🎛️ **14 Solvers, 8 Schedulers, 4 Guidance Modes** — Modular plugin systems for ODE/SDE solvers (Euler, RK4, Gauss-Legendre, RF-Solver, and more), noise schedulers (with composite 2-stage blending), and guidance modes (including Dynamic CFG). Each exposes its own sub-parameters.

🎸 **LoRA Adapters with Runtime Mode** — Per-group scale controls (self_attn, cross_attn, mlp, cond_embed), K-quant GPU support via custom CUDA kernels, and a runtime LoRA mode that applies deltas in the forward pass without permanently merging weights.

🎚️ **Matchering Mastering Engine** — Loudness, EQ, and dynamics matching to a reference track with instant mastered/unmastered A/B toggle. Operates at native 48kHz — no resample round-trip.

🔌 **VST3 Host** — Scan, load, and run your existing VST3 plugins directly in the generation pipeline. Offline processing and real-time WASAPI monitor mode with transport controls.

✍️ **Lyric Studio** — A complete AI-powered lyrics and music workspace. 6 LLM providers (Gemini, LM Studio, OpenAI-compatible), artist profiles with adapter presets, statistical lyric analysis, bulk generation with "Fill to N" mode, and full parameter parity with the Create page.

🎤 **Cover Studio** — Upload a reference track, get Essentia-based analysis (BPM, key, energy, timbre), and generate style-matched covers with artist-specific settings.

🔊 **Audio Post-Processing** — Spectral denoiser (Wiener-filter), Spectral Lifter (native C++), PP-VAE neural audio polish, duration buffer with auto-trim for clean endings, and configurable fade-out.

🧪 **Latent Space Controls** — Latent shift, latent rescale, custom timestep scheduling, DCW (Differential Correction in Wavelet domain) sampling, and auto-shift for adaptive noise scaling.

📦 **Lossless Pipeline** — WAV32 throughout the processing chain, with export to WAV, MP3, or FLAC.

📥 **In-App Model Manager** — Browse 100+ GGUF models across 5 HuggingFace repos, download with curated starter packs, and manage your model library without leaving the app. Concurrent resumable downloads with real-time progress.

🧬 **PP-VAE & ScragVAE** — Two custom VAE models. PP-VAE runs a neural encode→decode polish pass on generated audio to smooth spectral artifacts. ScragVAE is a fine-tuned decoder with improved high-frequency energy and dynamic range — both selectable at runtime.

👉 **[See the full feature list →](FEATURES.md)**

## Architecture

HOT-Step CPP is three components working together:

| Component | Tech | Purpose |
|-----------|------|---------|
| **Engine** | C++ / CUDA / GGML | The acestep.cpp inference engine — runs the AI models |
| **Server** | Node.js / TypeScript | Orchestrates the engine, manages songs, serves the UI |
| **UI** | React / Vite / TypeScript | The browser-based frontend |

## Prerequisites

You'll need these installed before building:

| Requirement | Version | Notes |
|-------------|---------|-------|
| [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | 2022 | Select "Desktop development with C++" workload |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 12.x+ | For NVIDIA GPU acceleration. **Select "Visual Studio Integration" during install.** |
| [CMake](https://cmake.org/download/) | 3.14+ | Usually included with VS Build Tools |
| [Node.js](https://nodejs.org/) | 18–22 LTS | **Node 24+ is not supported** — use nvm to install 22 LTS if needed |
| [Git](https://git-scm.com/) | Any | For cloning |

## Quick Start (Windows + NVIDIA)

### 1. Clone the repo

```cmd
git clone --recursive https://github.com/scragnog/HOT-Step-CPP.git
cd HOT-Step-CPP
```

> **Already cloned without `--recursive`?** Run `git submodule update --init --recursive` to fetch the ggml and vst3sdk submodules.

### 2. Build the engine

The easiest way:

```cmd
engine\build.cmd
```

This automatically finds your Visual Studio installation (any edition) and builds with CUDA.

Alternatively, open a **Developer Command Prompt for VS 2022** and build manually:

```cmd
cd engine
mkdir build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
cd ..\..
```

> **Note:** If you use **Ninja** as your CMake generator (`-G Ninja`), binaries will be placed directly in `engine/build/` rather than `engine/build/Release/`. The server auto-detects both locations.

### 3. Download models

Download four GGUF model files from [Hugging Face](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/tree/main) and place them in a `models/` directory at the repo root:

```
HOT-Step-CPP/
├── models/                          ← create this, put GGUFs here
│   ├── acestep-5Hz-lm-4B-Q8_0.gguf
│   ├── Qwen3-Embedding-0.6B-Q8_0.gguf
│   ├── acestep-v15-turbo-Q8_0.gguf
│   └── vae-BF16.gguf
├── engine/
├── server/
└── ui/
```

| Type | Recommended File | Size |
|------|-----------------|------|
| LM | `acestep-5Hz-lm-4B-Q8_0.gguf` | 4.2 GB |
| Text Encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 748 MB |
| DiT | `acestep-v15-turbo-Q8_0.gguf` | 2.4 GB |
| VAE | `vae-BF16.gguf` | 322 MB |

Smaller LM variants available: 0.6B (fast) and 1.7B (balanced).

#### Optional (recommended)

| Type | File | Size | Source |
|------|------|------|--------|
| ScragVAE | `scragvae-BF16.gguf` | 322 MB | [scragnog/Ace-Step-1.5-ScragVAE](https://huggingface.co/scragnog/Ace-Step-1.5-ScragVAE) |
| PP-VAE | `pp-vae-F32.gguf` | 644 MB | [scragnog/HOT-Step-CPP-PP-VAE](https://huggingface.co/scragnog/HOT-Step-CPP-PP-VAE) |

**ScragVAE** is a fine-tuned VAE decoder with improved high-frequency energy and dynamic range — drop-in replacement for the standard VAE. **PP-VAE** enables neural audio polish via an encode→decode round-trip in the post-processing chain.

> **💡 Tip:** You can also download models directly from the app! Click **Models → Get More Models** to browse 100+ models across 5 HuggingFace repos, with curated starter packs for quick setup.

### 4. Install UI & server dependencies

```cmd
install.bat
```

Or manually (PowerShell):

```powershell
cd server; npm install; cd ..
cd ui; npm install; cd ..
```

### 5. Run

```cmd
LAUNCH.bat
```

Open `http://localhost:3001` in your browser. That's it!

> **No `.env` file needed** for the standard setup. The server automatically finds the engine binary (checks `engine/build/Release/`, `engine/build/`, and `engine/build/Debug/`) and models at `models/`. See `.env.example` if you need to override paths for a custom setup.

**Development mode** (with hot-reload):
```cmd
dev.bat
```
Then open `http://localhost:3000`.

## Platform Support

| Platform | Status |
|----------|--------|
| Windows + NVIDIA (CUDA) | ✅ Primary target |
| Windows + AMD/Intel (Vulkan) | 🔧 Engine supports it, UI untested |
| Linux | 🔧 Engine supports it, UI/server scripts TBD |
| macOS (Metal) | 🔧 Engine supports it, UI/server scripts TBD |

## Troubleshooting

<details>
<summary><b>MSVC error C2589: illegal token on right side of '::'</b></summary>

This happens when `Windows.h` defines `min`/`max` as macros, which collide with `std::min`/`std::max`. The CMakeLists.txt should already define `NOMINMAX` — if you're seeing this, pull the latest version.

If building manually, add `-DCMAKE_CXX_FLAGS="/DNOMINMAX /DWIN32_LEAN_AND_MEAN"` to your cmake command.
</details>

<details>
<summary><b>npm install fails on Node.js 24+</b></summary>

Node.js 24 is too new for some dependencies. Use Node.js 22 LTS:

```cmd
nvm install 22
nvm use 22
```
</details>

<details>
<summary><b>build.cmd can't find vcvars64.bat</b></summary>

The build script uses `vswhere.exe` to find Visual Studio automatically. If it fails:

1. Make sure you have **Visual Studio 2022** (any edition) or **Build Tools** installed
2. Ensure the **"Desktop development with C++"** workload is selected
3. As a fallback, open a **Developer Command Prompt for VS 2022** and build manually (see Build the Engine above)
</details>

<details>
<summary><b>"ace-server.exe not found" after building with Ninja</b></summary>

Ninja is a single-config generator — binaries go directly in `engine/build/` instead of `engine/build/Release/`. The server auto-detects both locations. If you still see this error, pull the latest version or set `ACESTEPCPP_EXE` in your `.env` file to point to the binary.
</details>

<details>
<summary><b>CUDA error: "The CUDA Toolkit directory does not exist"</b></summary>

MSBuild can't find the CUDA Toolkit. Check:

1. The `CUDA_PATH` environment variable is set (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)
2. You selected **"Visual Studio Integration"** during the CUDA Toolkit install — without this, MSBuild has no `$(CudaToolkitDir)` macro
3. Restart your terminal after installing or modifying CUDA paths
</details>

<details>
<summary><b>"The input line is too long" when running build.cmd</b></summary>

Running `build.cmd` multiple times in the same terminal causes `vcvars64.bat` to append duplicate entries to `%PATH%` until it exceeds the Windows 8,192-character limit.

**Fix:** Close the terminal and open a fresh one. The build scripts now guard against this, but older versions don't — pull latest.
</details>

<details>
<summary><b>Build errors persist after fixing environment</b></summary>

If you changed CUDA versions, VS editions, or environment variables, the CMake cache may contain stale configuration:

```cmd
rd /s /q engine\build
engine\build.cmd
```

The `CMakeCache.txt` is only generated once — `build.cmd` skips reconfiguration if it already exists.
</details>

## Credits

- **[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)** — The AI music generation model by ACE Studio and StepFun
- **[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)** — The C++ GGML inference engine by ServeurpersoCom
- **[HOT-Step 9000](https://github.com/scragnog/HOT-Step-9000)** — The Python-based sister project with full feature support

## License

The engine component (`engine/`) is licensed under MIT. See [engine/LICENSE](engine/LICENSE) for details.
