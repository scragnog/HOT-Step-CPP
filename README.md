# HOT-Step CPP

> **⚠️ ALPHA — ACTIVE DEVELOPMENT ⚠️**
>
> This project is in **early alpha** and under very active development. Features may be incomplete, unstable, or change without notice.
>
> **For a more complete experience right now, use [HOT-Step 9000](https://github.com/scragnog/HOT-Step-9000)** — the Python-based version with a full feature set and wider platform support. This C++ version is being developed as a faster, lighter alternative and will mature over time.

A feature-rich UI for [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) — local AI music generation powered by GGML.

Describe a song with a text caption and lyrics, and get stereo 48kHz audio generated entirely on your local hardware. No cloud, no API keys, no subscriptions.

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
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 12.x+ | For NVIDIA GPU acceleration |
| [CMake](https://cmake.org/download/) | 3.14+ | Usually included with VS Build Tools |
| [Node.js](https://nodejs.org/) | 18+ | LTS recommended |
| [Git](https://git-scm.com/) | Any | For cloning |

## Quick Start (Windows + NVIDIA)

### 1. Clone the repo

```cmd
git clone https://github.com/scragnog/HOT-Step-CPP.git
cd HOT-Step-CPP
```

### 2. Build the engine

Open a **Developer Command Prompt for VS 2022** (or run `vcvars64.bat` manually), then:

```cmd
cd engine
mkdir build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
cd ..\..
```

> **Note:** The included `engine\buildcuda.cmd` script does the same thing but assumes Visual Studio Build Tools is installed at the default path. If you have the full Visual Studio IDE instead, use the commands above from a Developer Command Prompt.

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

### 4. Install UI & server dependencies

```cmd
install.bat
```

Or manually:

```cmd
cd server && npm install && cd ..
cd ui && npm install && cd ..
```

### 5. Run

```cmd
LAUNCH.bat
```

Open `http://localhost:3001` in your browser. That's it!

> **No `.env` file needed** for the standard setup. The server automatically finds the engine at `engine/build/Release/ace-server.exe` and models at `models/`. See `.env.example` if you need to override paths for a custom setup.

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

## Credits

- **[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)** — The AI music generation model by ACE Studio and StepFun
- **[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)** — The C++ GGML inference engine by ServeurpersoCom
- **[HOT-Step 9000](https://github.com/scragnog/HOT-Step-9000)** — The Python-based sister project with full feature support

## License

The engine component (`engine/`) is licensed under MIT. See [engine/LICENSE](engine/LICENSE) for details.
