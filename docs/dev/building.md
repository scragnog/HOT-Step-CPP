# Building from source

Most users should download a release instead. This page is for contributors: how to build
the engine, install the server and UI, run the dev loop, rebuild the engine safely, and
package a portable release.

## Prerequisites

Every platform needs Git and Node.js 18 to 22 LTS. Node 24 and later is not supported:
`server/package.json` and `ui/package.json` both declare `"node": ">=18.0.0 <24.0.0"`, and
native dependencies such as better-sqlite3 break on it. CMake 3.21 or later is required
(`engine/CMakeLists.txt`).

### Windows

| Requirement | Notes |
|---|---|
| [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | "Desktop development with C++" workload. CI pins VS 2022 because newer MSVC versions are rejected by CUDA 12.8 and 13.1 `nvcc` |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x or 13.x | For NVIDIA GPUs. Select "Visual Studio Integration" during install |
| CMake 3.21+ | Usually included with the VS Build Tools |
| Node.js 18 to 22 LTS | Use nvm to install 22 LTS if your system Node is newer |
| Git | Must be on `PATH`; CMake uses it to apply the ggml patches |
| Python with pip (optional) | `engine\build.cmd` uses it to fetch cuDNN 9 for CUDA-accelerated SuperSep |
| Vulkan SDK (optional) | Only for Vulkan builds |

### macOS (Apple Silicon)

| Requirement | Notes |
|---|---|
| Xcode Command Line Tools 16+ | `xcode-select --install`. Provides the Metal SDK and compiler |
| CMake 3.21+ | `brew install cmake` |
| Node.js 18 to 22 LTS | `brew install node@22` |
| Essentia (optional) | BPM and key detection. `bash tools/essentia/build-macos.sh Essentia`, or set `ESSENTIA_BIN` |

### Linux (x86_64)

| Requirement | Notes |
|---|---|
| GCC 11+ or Clang | `sudo apt install build-essential` |
| CMake 3.21+ | |
| Node.js 18 to 22 LTS | |
| CUDA Toolkit 12.x+ (optional) | NVIDIA GPUs |
| Vulkan SDK (optional) | AMD, Intel or NVIDIA through Vulkan |
| ROCm 6.1+ (optional) | AMD through HIP, built with `engine/buildhip.sh` |
| Essentia (optional) | `bash tools/essentia/build-linux.sh Essentia`, or set `ESSENTIA_BIN` |

## Clone

```
git clone --recursive https://github.com/scragnog/HOT-Step-CPP.git
cd HOT-Step-CPP
```

If you cloned without `--recursive`, run `git submodule update --init --recursive` to fetch
the `engine/ggml` and `engine/vendor/vst3sdk` submodules.

## Build the engine

CMake applies the patches in `engine/patches/` to the ggml submodule at configure time, so
a normal build needs nothing beyond Git on `PATH`. See
[The ggml patch stack](#the-ggml-patch-stack) if configure or the build complains.

### Windows

```
engine\build.cmd
```

`build.cmd` does the following, in order:

1. Finds `vcvars64.bat` through `vswhere` and sources it, unless `VSCMD_VER` shows the VS
   environment is already loaded.
2. Downloads the ONNX Runtime GPU SDK (1.25.1) into `engine/deps/onnxruntime` if it is
   missing. Set `ONNXRUNTIME_ROOT` to use your own copy. Without ORT, SuperSep is not built.
3. Installs `nvidia-cudnn-cu12` with pip and copies `cudnn64_9.dll` into
   `engine/build/Release/` if it is missing.
4. Runs `cmake ..` only when `engine/build/CMakeCache.txt` does not exist. The default flags
   are `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120a"
   -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON`. `HOT_STEP_CMAKE_FLAGS`
   replaces the backend flags when set (`update.bat` sets it for auto-detected backends).
5. Runs `engine\verify-hooks.ps1` and stops if a fork hook or ggml patch is missing.
6. Builds with `cmake --build . --config Release`.

Binaries land in `engine/build/Release/`. With the Ninja generator they land in
`engine/build/` instead; the server looks in both.

Other Windows scripts in `engine/`:

| Script | Configure | Use |
|---|---|---|
| `buildcuda.cmd` | `cmake .. -DGGML_CUDA=ON`, every run | Rarely; `build.cmd` supersedes it |
| `buildvulkan.cmd` | `cmake .. -DGGML_VULKAN=ON`, every run | Vulkan backend work |
| `buildall.cmd` | CUDA, Vulkan and all CPU variants, every run, plus the ORT download | Release-style build. `install.bat` calls this one |

All of them build into the same `engine/build/` folder. The scripts that reconfigure on
every run change the CMake cache, and switching between them can force a large part of the
CUDA code to recompile. Pick one and stay on it; for day-to-day work that is `build.cmd`
through `dev-rebuild.bat`.

### macOS

```
cd engine
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j $(sysctl -n hw.ncpu)
cd ../..
```

The Metal shader library is embedded in the binary, so no `.metallib` file is needed at
runtime. CI adds `-DGGML_BACKEND_DL=OFF` for the macOS build.

### Linux

```
cd engine
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release     # NVIDIA
# cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release # Vulkan
# cmake .. -DCMAKE_BUILD_TYPE=Release                  # CPU only
cmake --build . -j $(nproc)
cd ../..
```

`engine/` also holds `buildcuda.sh`, `buildvulkan.sh`, `buildcpu.sh`, `buildall.sh`,
`buildhip.sh` (ROCm) and `build-mac.sh`.

## Get the models

Put GGUF files in `models/` at the repo root, or download them in the app from
Models, Get More Models (the Model Manager). The minimum ACE-Step set is one file of each
type from [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/tree/main):

| Type | Example file |
|---|---|
| LM | `acestep-5Hz-lm-4B-Q8_0.gguf` |
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` |
| DiT | `acestep-v15-turbo-Q8_0.gguf` |
| VAE | `vae-BF16.gguf` |

Every file the Model Manager knows about is listed in
`server/src/data/model-registry.json`.

## Install server and UI dependencies

Windows: `install.bat` runs `npm install` in `server/` and `ui/`, rebuilds better-sqlite3,
and then builds the engine with `engine\buildall.cmd`. If you already built with
`build.cmd`, install the dependencies by hand instead so the build cache is not
reconfigured:

```
cd server; npm install; cd ..
cd ui; npm install; cd ..
```

macOS and Linux: `install.sh`, or the same two `npm install` commands.

## Run

| Command | Mode | What starts |
|---|---|---|
| `LAUNCH.bat` | Production (Windows) | Builds `ui/dist` if it is missing, then runs `npx tsx src/index.ts` in `server/` in a restart loop. Open `http://localhost:3001` |
| `dev.bat` | Development (Windows) | Two minimised windows: the server through `server/restart-loop.cmd` (`npx tsx watch src/index.ts`, with `HOT_STEP_DEV=1`), and Vite with `npx vite --port 3000 --host`. Open `http://localhost:3000` |
| `launch.sh` | Production (macOS, Linux) | The Node server, using a bundled, Homebrew or system Node in that order |
| `dev.sh` | Development (macOS) | Vite HMR plus `tsx watch` |

No `.env` is needed for a standard setup. The server finds the engine binary and
`models/` on its own. [config.md](config.md) lists everything you can override.

In dev mode, Vite serves the UI with hot module reload and proxies `/api`, `/audio` and
`/references` to `127.0.0.1:3001`. `tsx watch` restarts the Node server whenever a file
under `server/src` changes. That restart kills any running generation or training job, so
check that the app is idle before editing server code while it runs.

`HOT_STEP_DEV=1` tells `POST /api/shutdown` that the Vite process on port 3000 belongs to
the app. Shutdown kills it only after confirming it is a Node process running Vite.

## Type-checking

Do not run `npm run build` during development. Type-check instead:

| Folder | Command |
|---|---|
| `server/` | `npx tsc --noEmit` |
| `ui/` | `npx tsc --noEmit -p tsconfig.app.json` (or `npx tsc -b`) |

A bare `npx tsc --noEmit` in `ui/` checks nothing and exits 0. `ui/tsconfig.json` has
`"files": []` and only references `tsconfig.app.json` and `tsconfig.node.json`, so the root
project has no inputs.

## Engine rebuild rules

The Node server respawns `ace-server` whenever it exits abnormally. If you run
`engine\build.cmd` while the app is up, the linker cannot replace a running
`ace-server.exe`, and anything that kills it makes Node start it again. The result is a
respawn and file-lock loop.

- **Rebuild with `dev-rebuild.bat` at the repo root, not `engine\build.cmd`.** It posts to
  `http://localhost:3001/api/shutdown` so Node stops the engine, Vite and itself; waits for
  `ace-server.exe` to exit, force-killing it after 10 seconds and giving up after 15; then
  calls `engine\build.cmd`. It is safe when the app is not running.
- **It does not restart the app.** Start it again with `dev.bat` (the shutdown also killed
  Vite) or `LAUNCH.bat`.
- **Check the build yourself.** Look for `error C` or `error LNK` lines in the output and
  confirm `engine\build\Release\ace-server.exe` has a fresh timestamp. The exit code is not
  a reliable signal.
- **Run it from a real console.** `dev-rebuild.bat` waits with `timeout /t`, which fails
  when stdin is redirected. From an agent tool call or a piped shell the script can fall
  through, skip both the shutdown and the build, and still exit 0. In that case run the
  three steps by hand:

  ```powershell
  try { Invoke-RestMethod -Method Post -Uri http://localhost:3001/api/shutdown -TimeoutSec 10 | Out-Null } catch {}
  $n = 0; while ((Get-Process ace-server -ErrorAction SilentlyContinue) -and $n -lt 20) { Start-Sleep 1; $n++ }
  & cmd.exe /c "D:\path\to\HOT-Step-CPP\engine\build.cmd"
  ```

  Call `.bat` and `.cmd` files by absolute path. Some agent shells will not resolve a batch
  file from the working directory.
- **Rebuild straight after every edit** under `engine/src/` or `engine/tools/`, so a broken
  build points at one change.
- **Never run `cmake --build . --clean-first`** unless the GGML or CUDA layer itself
  changed. Recompiling the CUDA kernels takes over 20 minutes.
- **Stale object files:** if a header change seems to be ignored, delete only the core
  library and rebuild:

  ```powershell
  Remove-Item -Recurse -Force engine\build\acestep-core.dir
  Remove-Item -Force engine\build\Release\acestep-core.lib
  ```

- **Stale binary:** the server prefers a flat `engine/ace-server.exe` (the portable layout)
  over `engine/build/Release/`. If the app seems to run old code, check for a leftover flat
  binary or an `ACESTEPCPP_EXE` override in `.env`.

Lua plugins (solvers, schedulers, guidance, postprocess) need no rebuild. They are loaded
from `engine/plugins/` and the repo-root `plugins/` at engine start. See
[plugins-authoring.md](plugins-authoring.md).

## The ggml patch stack

`engine/ggml` is a git submodule kept at upstream. HOT-Step's changes to it live as patch
files in `engine/patches/` (training ops, BF16 and quant copies, F32 accumulation for F16
GEMMs, the fused flash-attention training ops, YuE2 ops and others).
`engine/patches/README.md` explains each one.

- **At configure time**, `engine/CMakeLists.txt` applies every `engine/patches/*.patch` in
  sorted order (option `HOT_STEP_APPLY_PATCHES`, on by default). A patch that reverses
  cleanly is treated as already applied.
- **That check is not reliable.** `flash-attn-train.patch` and `zz-yue2-convrot8.patch`
  add to the same enum in `ggml.h`, so once both are in, neither reverses on its own.
  CMake logs "neither applies nor reverses cleanly" for `flash-attn-train` on every healthy
  build. That warning is expected.
- **`engine/verify-hooks.ps1` is the check to trust.** It greps for the symbols and marker
  comments themselves: the three upstream include hooks (`pipeline-synth-ops.cpp` includes
  `hot-step-sampler.h`, `model-store.h` includes `hot-step-params.h`, `dit.h` includes
  `adapter-merge.h` and `adapter-runtime.h`), the MiniMax-Music3 and YuE2 route hooks in
  `hot-step-server.cpp`, the `hotstep_sampler_linked_` linker sentinel, and one marker per
  ggml patch. Exit 0 means all present. `build.cmd` runs it before every compile.

  ```
  powershell -File engine\verify-hooks.ps1
  ```

- **Patches need LF sources.** A tree checked out with `core.autocrlf=true` makes every
  hunk fail. Build scratch trees with
  `git -c core.autocrlf=false -c core.eol=lf archive HEAD`.
- **Never `git reset --hard`.** This checkout sets `submodule.recurse=true`, so a hard reset
  also resets `engine/ggml` and removes the patches. The two patches that create new files
  then fail on the next configure with "already exists in working directory", and the CUDA
  build fails on undefined `GGML_OP_CONVROT8` or `ggml_flash_attn_train_*`. To recover:

  ```
  del engine\ggml\src\ggml-cuda\convrot8.* engine\ggml\src\ggml-cuda\fattn-train.*
  git apply --ignore-whitespace engine\patches\flash-attn-train.patch
  git apply --ignore-whitespace engine\patches\zz-yue2-convrot8.patch
  powershell -File engine\verify-hooks.ps1
  ```

  To undo a working-tree change, use `git restore <path>` on explicit paths.

## What CI builds

| Workflow | Trigger | What it does |
|---|---|---|
| `release.yml` | Push of any `v*` tag | Builds every platform, packages portable archives, smoke-tests the packaged engine (`/health`, `/props`, `/plugins`), and drafts a GitHub Release |
| `cache-warm.yml` | Push to `master` touching `engine/ggml`, `engine/CMakeLists.txt` or `engine/patches/`, or manual | Builds the engine on `master` so tag builds can reuse the CMake cache. No packaging |
| `rocm-build.yml` | Push or PR touching `engine/`, or manual | Compile-only ROCm/HIP check. Marked `continue-on-error`, so check the job, not the run colour |
| `essentia.yml` | Called by `release.yml`, or manual | Builds the Essentia bundle for Linux and macOS |

`release.yml` build matrix:

| Platform | Runner | Variants |
|---|---|---|
| Windows | `windows-2022` | `cuda13.1`, `cuda12.8`, `cuda12-volta` (sm_70, cuBLAS forced, flash attention off), `vulkan`, `cpu` |
| Linux | `ubuntu-22.04` | `cuda13.1`, `cuda12.8`, `cuda12-volta`, `vulkan`, `rocm`, `cpu` |
| macOS | `macos-15` | Metal |

<!-- TODO(verify): rocm-build.yml says ROCm builds are not shipped, but release.yml's Linux matrix has a rocm variant and the release job collects every build artifact. Confirm whether a ROCm archive is published. -->

Each build job applies the patch stack with the same glob loop
(`for p in engine/patches/*.patch; do git apply --verbose "$p"; done`) before it builds.
Windows and Linux build with Ninja; macOS uses the default generator. Any pushed `v*` tag starts a full release build, so use a `-CI-Test` suffix for
throwaway compile checks. [releasing.md](releasing.md) is the full runbook.

## Portable releases

A portable release bundles the engine, a Node runtime, the server and the built UI, so users
extract and run with no prerequisites.

Layout of a Windows package, as assembled by `release.yml`:

| Path | Contents |
|---|---|
| `runtime/node.exe` | Portable Node 22 |
| `server/server.mjs` | The server bundled with esbuild (`release/esbuild.config.mjs`) |
| `server/ffmpeg.exe` | From the `ffmpeg-static` package |
| `server/data/` | Everything from `server/src/data/`, copied whole |
| `engine/` | `ace-server`, `ace-train`, `ace-caption`, `ace-midi`, `mastering`, `mp3-codec`, `neural-codec`, `quantize`, `vst-host`, the ggml DLLs for the variant, ONNX Runtime DLLs, and `engine/plugins/` |
| `plugins/` | The repo-root plugin overlay |
| `VERSION` | The release version, shown by the app |
| `.env.example` | Copied from the repo root |
| `HOT-Step.bat` | Launcher. Sets `HOT_STEP_ROOT` and runs `runtime\node.exe server\server.mjs` |

The package step fails the build if `ace-server`, `ace-train` or `ace-caption` is missing.
A new engine tool the app runs must be added to both the copy list and that required list
in `release.yml`. Any new file the app reads at runtime must reach users: runtime data goes
in `server/src/data/`, and model weights need an entry in
`server/src/data/model-registry.json` plus an upload to Hugging Face. Run
`node server/scripts/check-release-prereqs.mjs` before pushing to `master` or tagging.

To build a package locally:

| Platform | Command |
|---|---|
| Windows | `.\release\build-release.ps1 [-Version "1.5.0"] [-Variant cuda\|vulkan\|cpu] [-SkipEngine] [-SkipUI]`, output in `release/out/` |
| macOS | `./package-release.sh [--skip-build] [--version=1.2.0]`, output `HOT-Step-CPP-v<version>-macOS-arm64.tar.gz` |

## Build problems

- **MSVC error C2589, "illegal token on right side of '::'".** `Windows.h` is defining
  `min` and `max` as macros. `engine/CMakeLists.txt` defines `NOMINMAX` and
  `WIN32_LEAN_AND_MEAN`, so pull the latest source. For a hand-rolled configure, add
  `-DCMAKE_CXX_FLAGS="/DNOMINMAX /DWIN32_LEAN_AND_MEAN"`.
- **`build.cmd` cannot find `vcvars64.bat`.** Check that Visual Studio 2022 or the Build
  Tools are installed with the "Desktop development with C++" workload. As a fallback, open
  a Developer Command Prompt for VS 2022 and build by hand:

  ```
  cd engine
  mkdir build
  cd build
  cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
  cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
  ```

- **"The CUDA Toolkit directory does not exist".** MSBuild cannot find CUDA. Check that
  `CUDA_PATH` is set (for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`),
  that "Visual Studio Integration" was selected when installing the toolkit (without it
  MSBuild has no `$(CudaToolkitDir)`), and restart the terminal after changing either.
- **"The input line is too long".** Each `vcvars64.bat` run appends to `PATH` until it
  passes the 8,192-character limit. Open a fresh terminal. `build.cmd` skips `vcvars64.bat`
  when `VSCMD_VER` is already set, so this only hits older copies of the script.
- **`ace-server.exe` not found after a Ninja build.** The server checks
  `engine/build/Release/`, `engine/build/` and `engine/build/Debug/`. If it still misses the
  binary, set `ACESTEPCPP_EXE` in `.env`.
- **Errors persist after fixing the environment.** `build.cmd` configures only when
  `CMakeCache.txt` is missing, so a changed CUDA version or VS edition stays cached. Delete
  the build folder and build again: `rd /s /q engine\build`, then `engine\build.cmd`. This
  is a full rebuild, CUDA kernels included.
- **macOS: Metal compilation errors.** Install full Xcode, not only the Command Line Tools,
  and run `sudo xcodebuild -runFirstLaunch`. Errors about the Metal Toolchain can usually be
  ignored, since `-DGGML_METAL_EMBED_LIBRARY=ON` does not need a separate Metal Toolchain
  download.

## Related

- [architecture.md](architecture.md)
- [config.md](config.md)
- [releasing.md](releasing.md)
- [plugins-authoring.md](plugins-authoring.md)
- `engine/patches/README.md`
