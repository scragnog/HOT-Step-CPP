# HOT-Step CPP

> **⚠️ ALPHA — ACTIVE DEVELOPMENT ⚠️**
>
> This project is in **early alpha** and under very active development. Features may be incomplete, unstable, or change without notice.
>
> **For a more complete experience right now, use [HOT-Step 9000](https://github.com/scragnog/HOT-Step-9000)** — the Python-based version with a full feature set and wider platform support. This C++ version is being developed as a faster, lighter alternative and will mature over time.

A premium UI for [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) — local AI music generation powered by GGML.

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

You need four GGUF model files. Download from [Hugging Face](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/tree/main) and place them in a `models/` directory:

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

### 5. Configure

Copy `.env.example` to `.env` and edit the paths:

```ini
# Point to your built engine executable
ACESTEPCPP_EXE=D:\path\to\HOT-Step-CPP\engine\build\Release\ace-server.exe

# Point to your downloaded models
ACESTEPCPP_MODELS=D:\path\to\models

# Point to adapters directory (optional, for LoRA)
ACESTEPCPP_ADAPTERS=D:\path\to\adapters
```

### 6. Run

**Production mode:**
```cmd
LAUNCH.bat
```

**Development mode** (with hot-reload):
```cmd
dev.bat
```

Open `http://localhost:3000` (dev) or `http://localhost:3001` (production) in your browser.

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
