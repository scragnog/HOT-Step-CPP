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
| [Node.js](https://nodejs.org/) | 18–22 LTS | **Node 24+ is not supported** — use nvm to install 22 LTS if needed |
| [Git](https://git-scm.com/) | Any | For cloning |

## Quick Start (Windows + NVIDIA)

### 1. Clone the repo

```cmd
git clone https://github.com/scragnog/HOT-Step-CPP.git
cd HOT-Step-CPP
```

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

## Credits

- **[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)** — The AI music generation model by ACE Studio and StepFun
- **[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)** — The C++ GGML inference engine by ServeurpersoCom
- **[HOT-Step 9000](https://github.com/scragnog/HOT-Step-9000)** — The Python-based sister project with full feature support

## License

The engine component (`engine/`) is licensed under MIT. See [engine/LICENSE](engine/LICENSE) for details.
