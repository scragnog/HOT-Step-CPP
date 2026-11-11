# Building from source

Resolve runtime paths first using the [data folder configuration](config.md#dev-checkouts).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Most users should download a release instead. This page is for contributors: how to build
the engine, install the server and UI, run the dev loop, rebuild the engine safely, and
package a portable release.

For a GPU Pod deployment, see the [RunPod template guide](runpod.md). It covers
building the `runpod` Docker target and publishing it to your own registry.

## Prerequisites

For a stable copy alongside active development, see
[Stable production beside development](#stable-production-beside-development).

Every platform needs Git and Node.js 20 to 24 LTS. Node 24 is recommended:
`server/package.json` and `ui/package.json` both declare `"node": ">=20.0.0 <25.0.0"`.
CMake 3.21 or later is required (`engine/CMakeLists.txt`).

### Windows

| Requirement | Notes |
|---|---|
| [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | "Desktop development with C++" workload. CI pins VS 2022 because newer MSVC versions are rejected by CUDA 12.8 and 13.1 `nvcc` |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x or 13.x | For NVIDIA GPUs. Select "Visual Studio Integration" during install |
| CMake 3.21+ | Usually included with the VS Build Tools |
| Node.js 20 to 24 LTS | Use nvm to install 24 LTS |
| Git | Must be on `PATH` to fetch the submodules |
| Python with pip (optional) | `engine\build.cmd` uses it to fetch cuDNN 9 for CUDA-accelerated SuperSep |
| Vulkan SDK (optional) | Only for Vulkan builds |

### macOS (Apple Silicon)

| Requirement | Notes |
|---|---|
| Xcode Command Line Tools 16+ | `xcode-select --install`. Provides the Metal SDK and compiler |
| CMake 3.21+ | `brew install cmake` |
| PowerShell 7 (`pwsh`) | `brew install powershell`. `engine/build-mac.sh` runs the hook check `engine/verify-hooks.ps1` with it and stops if `pwsh` is missing |
| Ninja (optional) | `brew install ninja`. `engine/build-mac.sh` uses it when it is installed, and Unix Makefiles otherwise |
| Node.js 20 to 24 LTS | `brew install node@24` |
| Essentia (optional) | BPM and key detection. `bash tools/essentia/build-macos.sh Essentia`, or set `ESSENTIA_BIN` |

### Linux (x86_64)

| Requirement | Notes |
|---|---|
| GCC 11+ or Clang | `sudo apt install build-essential` |
| CMake 3.21+ | |
| Node.js 20 to 24 LTS | |
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

## The Makefile wrapper (Linux and WSL)

The root `Makefile` is the day-to-day entry point on Linux and WSL. It wraps
`cmake`, `npm install` and the launcher for you:

```
make setup       # apt: build-essential, cmake, git (use nvm for Node 20-24)
make submodules  # git submodule update --init --recursive
make build       # build the C++ engine
make install     # npm install in server/ and ui/
make run         # ./launch.sh, production mode
```

`make dev` runs `dev.sh` (Vite on :3000 and the server under `tsx watch`), `make dev-ui`
runs Vite alone, `make build-ui` rebuilds `ui/dist`, and `make open` opens the Vite UI when
it is running and :3001 otherwise (`OPEN_URL=` overrides). `make check-config`,
`make check-node` and `make clean CONFIRM=1` are also available; `make help` lists every
target. The Makefile picks the backend on its own: CUDA when a
driver is visible (the CUDA bridge under `/usr/lib/wsl/lib` on WSL, `/usr/local/cuda`
on native Linux), Vulkan if `glslc` is present, CPU otherwise. Force one with
`BACKEND=cuda`, `BACKEND=vulkan` or `BACKEND=cpu`. The steps below are the same work run
by hand, and remain the path to use on Windows, where the Makefile does not apply.

## Build the engine

`engine/ggml` is pinned to the HOT-ggml fork, which carries HOT-Step's ggml changes as
commits, so a normal build needs no extra step. See [The ggml fork](#the-ggml-fork) if
configure or the build complains.

### Windows

```
engine\build.cmd
```

`build.cmd` does the following, in order:

1. Finds `vcvars64.bat` through `vswhere` and sources it, unless `VSCMD_VER` shows the VS
   environment is already loaded.
2. Installs `nvidia-cudnn-cu12` with pip and copies `cudnn64_9.dll` into
   `engine/build/Release/` if it is missing.
3. Runs `cmake ..` only when `engine/build/CMakeCache.txt` does not exist. The default flags
   are `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120a"
   -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON`. `HOT_STEP_CMAKE_FLAGS`
   replaces the backend flags when set (`update.bat` sets it for auto-detected backends).
4. Runs `engine\verify-hooks.ps1` and stops if a fork hook or a HOT-ggml change is missing,
   or `engine/ggml` is dirty or off its pinned commit.
5. Builds with `cmake --build . --config Release`.

Binaries land in `engine/build/Release/`. With the Ninja generator they land in
`engine/build/` instead; the server looks in both.

Other Windows scripts in `engine/`:

| Script | Configure | Use |
|---|---|---|
| `buildcuda.cmd` | `cmake .. -DGGML_CUDA=ON`, every run | Rarely; `build.cmd` supersedes it |
| `buildvulkan.cmd` | `cmake .. -DGGML_VULKAN=ON`, every run | Vulkan backend work |
| `buildall.cmd [all\|cuda\|vulkan\|cpu]` | Defaults to CUDA, Vulkan and all CPU variants | Release-style build. A single backend disables the other GPU backend; all CPU variants remain enabled. `install.bat` uses the default |

All of them build into the same `engine/build/` folder. The scripts that reconfigure on
every run change the CMake cache, and switching between them can force a large part of the
CUDA code to recompile. Pick one and stay on it; for day-to-day work that is `build.cmd`
through `dev-rebuild.bat`.

### macOS

```
bash engine/build-mac.sh          # Release
bash engine/build-mac.sh Debug    # Debug
```

`build-mac.sh` runs `engine/verify-hooks.ps1` through `pwsh` and stops before CMake if a
hook is missing, then configures `engine/build/` with `-DGGML_METAL=ON
-DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BACKEND_DL=OFF` (Ninja when installed) and builds on
every core. The Metal shader library is embedded in the binary, so no `.metallib` file is
needed at runtime. CI's macOS build uses the same Metal flags.

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
`buildhip.sh` (ROCm) and `build-mac.sh`. `buildhip.sh` deletes `engine/build/` and
rebuilds from scratch on every run, because a cache from another backend breaks a HIP
configure.

## Get the models

Put GGUF files in `<data root>/models/`, or download them in the app from
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

Agents always run `dev.bat`, detached, and open the app at `http://localhost:3000`. The prod
launchers below are the end-user path.

| Command | Mode | What starts |
|---|---|---|
| `LAUNCH.bat` | Production (Windows) | Builds `ui/dist` if it is missing, then runs `npx tsx src/index.ts` in `server/` in a restart loop. Open `http://localhost:3001`. End-user path — agents use `dev.bat` |
| `dev.bat` | Development (Windows) | Two minimised windows: the server through `server/restart-loop.cmd` (`npx tsx watch src/index.ts`, with `HOT_STEP_DEV=1`), and Vite with `npx vite --port 3000 --host`. Then `open-browser-if-needed.mjs` opens `http://localhost:3000` unless a tab is already connected; `dev.bat --no-browser` (what `tools/detach.ps1` passes) skips it |
| `launch.sh` | Production (macOS, Linux) | The Node server in a restart loop, using a bundled, Homebrew or system Node in that order. Installs server dependencies if missing, and builds `ui/dist` if missing when UI dependencies are installed |
| `dev.sh` | Development (macOS, Linux) | Vite with `npx vite --port 3000 --host` and the server with `tsx watch src/index.ts` (`HOT_STEP_DEV=1`), in a restart loop like `dev.bat`'s. Installs missing dependencies and rebuilds `better-sqlite3` when it does not load under the current Node. Open `http://localhost:3000` |

No `.env` is needed for a standard setup. The server finds the engine binary and
`<data root>/models/` on its own. [config.md](config.md) lists everything you can override.

In dev mode, Vite serves the UI with hot module reload and proxies `/api`, `/audio` and
`/references` to `127.0.0.1:3001`. `tsx watch` restarts the Node server whenever a file
under `server/src` changes. That restart kills any running generation or training job, so
check that the app is idle before editing server code while it runs.

`HOT_STEP_DEV=1` tells `POST /api/shutdown` on Windows that the Vite process on port 3000
belongs to the app. Shutdown kills it only after confirming it is a Node process running
Vite.

On macOS and Linux, `dev.sh` does both jobs itself. `tsx watch` keeps running after the server
exits, so `dev.sh` watches the server process. When the server exits after writing
`.restart-requested` (in-app Restart, or a finished worker update), `dev.sh` stops Vite
and `tsx`, then starts both again from the current checkout. After Quit, Ctrl-C or
`SIGTERM` it stops them and exits. A server crash with neither waits for a file change, as
`tsx watch` always has. It stops only its own child processes. To start it over ssh so it
survives the session closing:

```
nohup ./dev.sh > logs/dev.log 2>&1 < /dev/null &
```

`logs/dev.pid` holds the PID; `kill $(cat logs/dev.pid)` stops the stack.

## Type-checking

Do not run `npm run build` during development. Type-check instead:

| Folder | Command |
|---|---|
| `server/` | `npx tsc --noEmit` (or `npm run typecheck`) |
| `server/scripts/` | `npx tsc --noEmit -p scripts/tsconfig.json`, run from `server/` |
| `ui/` | `npx tsc --noEmit -p tsconfig.app.json` (or `npx tsc -b`) |

The `server/` check covers only `server/src/`: `server/tsconfig.json` includes
`src/**/*`, so a changed script is not checked unless you run the `scripts/` command too.

A bare `npx tsc --noEmit` in `ui/` checks nothing and exits 0. `ui/tsconfig.json` has
`"files": []` and only references `tsconfig.app.json` and `tsconfig.node.json`, so the root
project has no inputs.

## Engine rebuild rules

The Node server respawns `ace-server` whenever it exits abnormally. If you run
`engine\build.cmd` while the app is up, the linker cannot replace a running
`ace-server.exe`, and anything that kills it makes Node start it again. The result is a
respawn and file-lock loop.

One abnormal exit is not counted as a crash: an empty models folder. `ace-server` prints
`ERROR: no models found` and exits 1; the server matches that stderr line (after stderr drains),
spends none of the three-strike budget, and reports `engine.noModels` on `/api/health`. The UI
shows a banner pointing at the Model Manager. The engine starts by itself when a download queue
drains, or when a `.gguf` appears in the models folder (checked every 5 s).

- **Rebuild with `dev-rebuild.bat` at the repo root, not `engine\build.cmd`.** It posts to
  `http://localhost:3001/api/shutdown` so Node stops the engine, Vite and itself; waits for
  `ace-server.exe` to exit, force-killing it after 10 seconds and giving up after 15; then
  calls `engine\build.cmd`. It is safe when the app is not running.
- **Work-channel reservations are optional.** The rebuild runs without joining a channel by
  default, even when `data/collaboration.db` exists at the checkout root. If you are using the
  HOT-Step work channel, set `HOTSTEP_WORK_CHANNEL=1` for the rebuild to reserve `app-server`,
  `gpu` and `engine-build` through `work-run.ps1`; it does so only when that database exists. See the [work-channel guide](../../tools/mcp-lyricstudio/README.md#resource-reservations-and-guarded-commands).
- **It does not restart the app.** Start it again with `dev.bat`, detached (the shutdown also
  killed Vite); `LAUNCH.bat` is the end-user prod launcher.
- **Check the build yourself.** Look for `error C` or `error LNK` lines in the output and
  confirm `engine\build\Release\ace-server.exe` has a fresh timestamp. The exit code is not
  a reliable signal.
- **Run it from a real console.** `dev-rebuild.bat` waits with `timeout /t`, which fails
  when stdin is redirected. From an agent tool call or a piped shell the script can fall
  through, skip both the shutdown and the build, and still exit 0. In that case run the
  three steps by hand:

  ```powershell
  try { Invoke-RestMethod -Method Post -Uri http://localhost:3000/api/shutdown -TimeoutSec 10 | Out-Null } catch {}
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

## The ggml fork

`engine/ggml` is a git submodule pinned to the `hot-step-0.29` branch of HOT-ggml, a fork of
ggml-org/ggml that carries HOT-Step's changes as commits (training ops, BF16 and quant
copies, F32 accumulation for F16 GEMMs, the fused flash-attention training ops, YuE2 ops and
others). [ggml-fork.md](ggml-fork.md) lists each one and explains why it exists.

- **A build leaves `engine/ggml` clean.** Nothing is applied at configure time. Change ggml
  on the fork, as a commit, and move the gitlink here.
- **`engine/CMakeLists.txt` refuses a stock ggml.** If `engine/ggml/include/ggml.h` has no
  `GGML_OP_FLASH_ATTN_TRAIN`, configure stops and prints the fix.
- **`engine/verify-hooks.ps1` is the check to trust.** It greps for the symbols and marker
  comments themselves: the three upstream include hooks (`pipeline-synth-ops.cpp` includes
  `hot-step-sampler.h`, `model-store.h` includes `hot-step-params.h`, `dit.h` includes
  `adapter-merge.h` and `adapter-runtime.h`), the `hot-step-server.cpp` hooks (its
  `hot-step-params.h` include, the MiniMax-Music3 and YuE2 route hooks, and the model-family
  boot gate), that `fsq-detok.h` and `fsq-tok.h` use the shared FSQ quantizer in
  `fsq-quant.h`, the `hotstep_sampler_linked_` linker sentinel, one check per HOT-ggml change (hooks 7 to 16
  and 18), and that `engine/ggml` is clean at the pinned commit (Hook 17). Exit 0 means all
  present. `build.cmd` and `build-mac.sh` run it before every compile; `buildall.cmd`,
  `buildcuda.cmd` and `buildvulkan.cmd` do not.

  ```
  powershell -File engine\verify-hooks.ps1        # Windows
  pwsh -NoProfile -File engine/verify-hooks.ps1   # macOS
  ```

- **A stock or stale `engine/ggml` has one fix.** The submodule URL is cached in
  `.git/config` at first init, so a checkout from before the fork keeps fetching ggml-org:

  ```
  git submodule sync -- engine/ggml
  git submodule update --init engine/ggml
  powershell -File engine\verify-hooks.ps1
  ```

  To undo a working-tree change, use `git restore <path>` on explicit paths.

## What CI builds

| Workflow | Trigger | What it does |
|---|---|---|
| `release.yml` | Push of any `v*` tag, or manual preflight | Tag pushes build every platform and draft a release after the Windows archive gate passes. Manual preflight builds and tests all five Windows variants and uploads artifacts only |
| `cache-warm.yml` | Push to `master` touching `engine/ggml`, `engine/CMakeLists.txt` or the workflow itself, or manual | Builds the engine on `master` so tag builds can reuse the CMake cache. No packaging |
| `rocm-build.yml` | Push or PR touching `engine/` or the workflow itself, or manual | Compile-only ROCm/HIP check. Marked `continue-on-error`, so check the job, not the run colour |
| `essentia.yml` | Called by `release.yml`, or manual | Builds the Essentia bundle for Linux and macOS |
| `docs.yml` | Push to `master`, or any PR | Runs `node tools/docs/check-docs.mjs` on Node 24 |
| `runpod-image.yml` | A published GitHub release, or manual on a `v*` tag | Builds the `runpod` Docker target and pushes it to GHCR. See the [RunPod guide](runpod.md) |

`release.yml` build matrix:

| Platform | Runner | Variants |
|---|---|---|
| Windows | `windows-2022` | `cuda13.1`, `cuda12.8`, `cuda12-volta` (sm_70, cuBLAS forced, flash attention off), `vulkan`, `cpu` |
| Linux | `ubuntu-22.04` | `cuda13.1`, `cuda12.8`, `cuda12-volta`, `vulkan`, `rocm`, `cpu` |
| macOS | `macos-15` | Metal |

The Linux `rocm` variant is packaged and published with the other archives
(`HOT-Step-CPP-v<version>-linux-x64-rocm.tar.gz`). It covers RDNA2 to RDNA4
(`gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201`) and does not bundle the ROCm runtime,
so users install ROCm themselves. `rocm-build.yml` is a separate compile check on engine
pushes and pull requests.

Each build job checks out `engine/ggml` at the pinned HOT-ggml commit and builds it as is.
Windows and Linux build with Ninja; macOS uses the default generator. Any pushed `v*` tag
starts a full release build. For a Windows check before tagging, manually dispatch
`release.yml` on the intended commit's branch. It uses the same Windows build and test
jobs, names the version `0.0.0-preflight-<12-character commit SHA>`, and skips Linux,
macOS, Essentia and release notes/creation. Dispatching it creates no tag or release.
Pushes and dispatches need the maintainer's approval. [releasing.md](releasing.md) is the
full runbook.

### Windows archive gate

`test-windows-artifacts` downloads each `release-win-<variant>` artifact onto a fresh
`windows-2022` VM, verifies the zip against its `.sha256` sidecar, then runs
`release/smoke-artifact.mjs`. The zip is the same immutable Actions artifact later
collected for the draft release. A failure in any of the five test jobs blocks the draft.

The helper extracts outside the checkout into `HOT Step smoke`, validates the package
and resolves its payload in either the flat or versioned Windows layout. It boots the
payload's `server/server.mjs` with its `runtime/node.exe`, and checks its engine's `/health`,
`/props` and `/plugins` using stub models. Its child processes search only Windows system
directories and the explicitly provisioned CUDA runtime; no toolkit or build tree is on
their `PATH`. The model-free `ace-server --backend-check` must initialise CPU and account
for the expected GPU backend, including an explicitly identified absent driver/device.
Missing package/runtime DLLs, unexplained loader failures and crashes fail the gate.
Removing the package backend DLL and, on CUDA, a provisioned runtime DLL must also fail.
These checks do not establish working GPU inference.

CUDA runtime downloads use the packaged registry/bootstrap contract and an `actions/cache`
entry keyed on the selected runtime pack's registry entries. CUDA 12.8 and Volta share
the CUDA 12 cache. The helper reports downloaded bytes and elapsed time on a miss.
Each test job uploads `windows-smoke-<variant>` evidence containing logs and `result.json`
on success or failure; the extraction and runtime cache are excluded from that upload.
The draft collects only `release-*` artifacts, so evidence never becomes a release asset.

Local helper checks, without starting the app or compiling:

```powershell
node --test release/smoke-artifact.test.mjs
node tools/docs/check-docs.mjs
```

For the archive command and backend acceptance rules, see
[the helper reference](../../release/smoke-artifact.md).

### Optional hosted CPU generation trial

The manual preflight input `cpu_generation_trial` defaults to `false`. Enabling it
adds a separate `windows-2022` job that downloads and verifies the CPU archive's
checksum, then extracts it under `HOT Step CPU trial`. The usual five Windows builds
and archive gates still run. The trial uses `continue-on-error`, and release creation
does not depend on it. Inspect its evidence even when the overall workflow is green.

The trial downloads `lm-0.6b-q80`, `dit-std-base-q4km`, `embedding-q8` and `vae-bf16`
from the registry, checks their file sizes, and requires the total to remain
3,277,122,752 bytes. Downloads have a separate ten-minute budget and no model cache.
It starts the packaged `ace-server` on `127.0.0.1:18085` with `--no-fa`, waits up to
30 seconds for it to report healthy, then sends the request to `/lm` and the LM's
output to `/synth`, with a shared five-minute inference budget that includes model
loading. The request is ten seconds, the LM's minimum supported duration, with one
DiT step. The server uses the packaged CPU backends and Windows system paths.

The `cpu-generation-trial` artifact contains `download.json`, `inference.json`, CPU
details, request JSON, process logs and any output WAV. Download timing is separate
from inference timing. RAM is the maximum process peak working set observed across
the sequential LM and synthesis phases, not total VM memory. Phase timings and peaks
are also recorded. A timeout or failed process preserves the available evidence.

Success requires a complete, nonempty stereo 48 kHz PCM16 WAV with a nonzero sample
and duration within 0.25 seconds of the requested ten seconds. One-step synthesis
proves execution only: the base DiT normally needs 60-100 steps, and audio quality
still needs a listening check. Hosted speed and memory use must be measured before
considering a generation gate. See the [trial dispatch instructions](releasing.md#optional-cpu-generation-measurement).

## Portable releases

A portable release bundles the engine, a Node runtime, the server and the built UI, so users
extract and run with no prerequisites.

Windows packages use either the flat protocol-0 layout or the versioned protocol-1
layout. See [Windows updater layout](releasing.md#windows-updater-layout-protocol-1)
for the install root, stable launcher, payload and pointer contract. The payload
bundles the Node runtime, server and native modules, engine tools and backend DLLs,
built UI, plugins and runtime data. `VERSION` and `build-info.json` identify that
payload. The archive gate resolves and validates both layouts before starting it.

The launcher must not spawn PowerShell. It used to start the browser helper with `powershell -ExecutionPolicy Bypass`, and behaviour-based antivirus quarantined the app's scripts for it. The helper is now `open-browser-if-needed.mjs` (it replaced the `.ps1`), `release/build-release.ps1` and `release.yml` copy it into the package, and `release/check-layout.mjs` fails if a portable launcher mentions PowerShell.

The package step fails the build if `ace-server`, `ace-train` or `ace-caption` is missing.
A new engine tool the app runs must be added to both the copy list and that required list
in `release.yml`. Any new file the app reads at runtime must reach users: runtime data goes
in `server/src/data/`, and model weights need an entry in
`server/src/data/model-registry.json` plus an upload to Hugging Face. Run
`node server/scripts/check-release-prereqs.mjs` before pushing to `master` or tagging.

To build a package locally:

| Platform | Command |
|---|---|
| Windows | `.\release\build-release.ps1 [-Version "1.5.0"] [-Variant cuda -CudaAsset cuda13.1\|cuda12.8\|cuda12-volta \| -Variant vulkan\|cpu] [-BuildBackend all\|cuda\|vulkan\|cpu] [-SkipEngine] [-SkipUI] [-NodeVersion 24.18.0] [-UpdaterProtocol 1]`, output in `release/out/` |
| macOS | `./package-release.sh [--skip-build] [--version=1.2.0]`, output `HOT-Step-CPP-v<version>-macOS-arm64.tar.gz`. It bundles Node 22.22.2, where CI bundles 24.18.0 |

`-BuildBackend` defaults to `all`, preserving the local build of CUDA, Vulkan and CPU
backends. `-Variant` chooses the archive name and packaged GPU DLL. For a local CPU-only
release test, use `-Variant cpu -BuildBackend cpu`; use matching `cuda` or `vulkan`
values for those backends. A single build backend must match `-Variant`, including
with `-SkipEngine`, and mismatches fail before staging is changed. `-Variant cuda` also
needs `-CudaAsset`, the CI asset the build stands in for, because `build-info.json` records
an exact variant and the script will not guess one from the toolkit. `-UpdaterProtocol 1`
builds the versioned layout the Windows updater installs (stable launcher root plus
`app/<id>/`), for updater test fixtures; releases stay protocol 0 and flat. See
[Windows updater layout](releasing.md#windows-updater-layout-protocol-1).

| Build backend | GGML_CUDA | GGML_VULKAN | Packaged GPU DLL |
|---|---|---|---|
| `all` | ON | ON | `ggml-cuda.dll`, `ggml-vulkan.dll` or neither, selected by `-Variant` |
| `cuda` | ON | OFF | `ggml-cuda.dll` |
| `vulkan` | OFF | ON | `ggml-vulkan.dll` |
| `cpu` | OFF | OFF | None |

All selections enable `GGML_CPU_ALL_VARIANTS`, `GGML_BACKEND_DL` and the static MSVC
runtime. Packages include the shared ggml DLLs and `ggml-cpu-*.dll` variants, and copy
only the selected GPU DLL even if older build outputs remain. Cache validation checks
both enabled and disabled backends; a mismatch clears `CMakeCache.txt` before configure.
The batch script is called by absolute path, and configure or build failures stop packaging.
`-SkipEngine` bypasses build and cache checks and requires matching existing binaries.
CUDA selections use the installed toolkit. They do not choose CI's `cuda13.1`, `cuda12.8`
or `cuda12-volta` matrix entries. Local builds still need separate approval.

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
- [ggml-fork.md](ggml-fork.md)

## Release assembly and data paths

CI and `release/build-release.ps1` bundle the server as `server/server.mjs`, with immutable resources copied wholesale from `server/src/data/` into app-owned `server/data/`. The local macOS `package-release.sh` uses the same bundle, flat `engine/` binaries and `runtime/bin/node` layout as CI. Both local packagers include `ace-train`, `ace-midi` and `ace-caption`.

Every package ships `release/.env.example` as `.env.example`, the shared launcher, `profile-path.mjs` and `README.txt`. No assembler precreates app-root models or adapters. Bootstrap creates the sibling data layout. The release template leaves managed paths unset; the checkout template retains its existing values. Launchers use `profile-path.mjs` to resolve the stable profile restart marker, matching `dataroot/paths.ts`, including `HOT_STEP_PROFILE_DIR` fixtures. Dev wrappers retain the checkout `.restart-requested` marker.

Run `node --test release/data-root-launcher.test.mjs` to check the launcher contract (own `HOT_STEP_ROOT` only, shared profile marker); the data-folder reuse fixtures are in `server/src/dataroot/dataroot.test.ts`.

Run `node server/scripts/check-release-prereqs.mjs` before shipping. It checks all three CI platform sections and both local assemblers, then runs disposable layout fixtures. Fixtures cover profile defaults for each platform, release reuse and unchanged dev selection; they do not run native Linux or macOS binaries. See [Data folder and upgrades](../user/data-folder.md).

## Stable production beside development

On Windows, `tools/prod/promote.mjs` prepares a separate clone of local master
at `D:\Ace-Step-Latest\hot-step-prod`. It serves the built UI on
`http://localhost:3101`, with a separate engine on 8186. Development keeps its
existing ports and binaries. No push or remote pull happens during promotion.

Scratch is the default data mode and requires an explicit `--scratch` root
under the dev checkout's `.tmp` directory. Shared mode requires `--data-root`
instead. There is no fallback between modes. Models in scratch mode are read
from the shared models directory. Startup warming and
keep-loaded are disabled. Do not submit generation or training while another
job holds the GPU.

From the development checkout, using PowerShell 7 and Node 20 to 24:

```powershell
node tools/prod/promote.mjs --scratch D:\Ace-Step-Latest\hot-step-cpp\.tmp\prod-scratch
node tools/prod/promote.mjs --data-root "N:\HOT-Step Data"
node tools/prod/promote.mjs --stop
node tools/prod/promote.mjs --start
```

Shared mode reads dev's own read-only data-path resolver, including its
checkout-specific pointer and `settings/.env` overrides. The requested root
must match that pointer. Production's private profile points at that same root,
with the same library, training, adapters and models paths without copying
them. This includes adapters on a
separate drive when configured in dev. All selected directories must already
exist. A pending move, unreadable pointer, root mismatch or missing directory
refuses promotion. The resolved paths print before preparation and startup.
Recorded-artifact restarts recheck the dev selection and refuse changed paths.
Promote again after changing dev's data paths.

For a stand-in validation, `--dev-profile <absolute-profile-directory>` selects
an isolated dev profile whose checkout-specific pointer names the stand-in
root. Use it only with `--data-root`. This does not edit the normal dev pointer.
Promotion writes only the production checkout's private profile pointer.
Do not hand-edit the live server's pointer to switch modes. Use a
new promotion with `--scratch` to return to isolation.

Stop both copies before an incompatible migration. Shared mode opens the same
database and settings, so a schema or data-format change that the other copy
cannot read must never run while that copy is active. Back up shared data
before its first activation. Shared-data and GPU concurrency still require
their separate validation before real shared-data use.

Promotion clones one recorded local master commit into a staging clone,
installs server and UI dependencies with `npm ci`, and builds both packages.
It checks that the engine tree is clean and matches that commit, and rejects
an executable older than any tracked C/C++/CUDA/CMake input, including
submodules. This freshness check supports existing prebuilt binaries; it is
not a reproducible-build attestation. Rebuild first after any engine input
change. Engine executables and DLLs are copied together and checked with
SHA-256 before and after copying. `.prod/artifact.json` records the commit,
engine source tree, executable timestamp and binary hashes.

Windows `dev-rebuild.bat`, `update.bat` and promotion share an exclusive file
handle for the entire operation. Promotion refuses during a rebuild, and
rebuilds refuse during promotion. Never call the engine build script directly
or bypass this lock. Source edits during promotion are also unsafe; leave the
engine tree alone until the snapshot finishes.

Before preparation and again before replacement, promotion checks
`/api/settings/data-folder` and refuses if its blockers list is nonempty,
missing, or a data move is pending. The managed launcher preloads a small
HTTP admission gate. Just before the final idle check, promotion pauses new
write requests and waits for previously admitted writes to finish. Read
requests, including preview audio, can continue. Writes receive HTTP 503
during the swap. If production is busy, the gate resumes it without shutdown.
The gate requires a private token and expires after 60 seconds if the
promoter disappears. A write still in progress after 15 seconds also aborts
and resumes admission. This gate covers HTTP submissions; the application's
busy checks must still cover every background job.

Only after all preparation succeeds does promotion shut down the verified
production PID through its own API. It waits for that process and its engine
to exit, renames the previous clone aside, installs the staged clone, and
starts it through `tools/detach.ps1`. It checks `/api/health` for the recorded
commit. Startup failure restores and starts the previous clone. Previous and
failed clones are retained for inspection; no automatic recursive cleanup
runs. Review these folders before removing them.

The launcher pins ports and the selected write paths in the process environment.
This takes precedence over data-root `settings/.env` values. Both modes use
private production profiles. This also avoids the server guard against
multiple checkout pointers to one root inside a single profile.
Other writable path overrides in the scratch `.env` must resolve inside that
scratch root. Generated relative paths are accepted on subsequent promotions.
Do not launch this clone with `LAUNCH.bat` or its in-app updater: use promotion
to keep the server, UI and engine together.

The service runs outside the agent's process lifetime through Task Scheduler,
with best-effort survival and no automatic restart supervisor. Logs are
`.prod/server.stdout.log`, `.prod/server.stderr.log`, and `logs/detached/`
inside the production clone. `.prod/process.json` records the PID and start
time. Stop with the command above; it verifies PID identity and checkout
ownership. An unowned listener on either production port aborts the operation.

Check process isolation without stopping development or using the GPU:

```powershell
& D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat --dry-run
node --test tools/prod/prod.test.mjs
```

The test creates an inert second process named `ace-server.exe` outside the
checkout and confirms that a rebuild dry run leaves it alive. It also checks
that the artifact lock excludes a concurrent rebuild and that busy or
unknown production state refuses promotion. These checks do not replace the
later two-instance shared-data and playback-through-rebuild validation.
