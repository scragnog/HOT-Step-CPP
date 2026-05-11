# Linux Release Support

**Status:** Implemented (bc07059)  
**Date:** 2026-05-11  
**Priority:** Medium — user demand confirmed (community forks already running Linux)

## Overview

Add Linux x64 portable releases to the CI pipeline. The engine already compiles on Linux (cross-platform CMake, VST3 Linux sources present). The macOS job serves as the template — Linux is simpler (no code signing, no Gatekeeper).

## Prerequisites Already Met

- **VST3 Linux sources** — `CMakeLists.txt` lines 362-367 already have the `else()` clause with `module_linux.cpp` + `threadchecker_linux.cpp`
- **Shutdown/restart** — `shutdown.ts` already has Linux code paths using `pgrep`/`SIGTERM`
- **GGML backends** — cross-platform by design
- **macOS job** — exists as a near-identical template

## Variants

| Variant | GPU Backend | Notes |
|---------|------------|-------|
| `cuda` | NVIDIA CUDA | Most common for ML workloads |
| `vulkan` | AMD/Intel/NVIDIA Vulkan | Broader GPU support |
| `cpu` | None | Fallback, significantly slower |

## Implementation

### 1. CI Job (`release.yml`)

Add a `build-linux` job using matrix strategy (same pattern as `build-windows`):

```yaml
build-linux:
  runs-on: ubuntu-24.04
  strategy:
    fail-fast: false
    matrix:
      variant: [cuda, vulkan, cpu]
      include:
        - variant: cuda
          cmake_flags: >-
            -DGGML_CUDA=ON -DGGML_VULKAN=OFF
            -DGGML_BACKEND_DL=ON
            -DGGML_CPU_ALL_VARIANTS=ON
        - variant: vulkan
          cmake_flags: >-
            -DGGML_CUDA=OFF -DGGML_VULKAN=ON
            -DGGML_BACKEND_DL=ON
            -DGGML_CPU_ALL_VARIANTS=ON
        - variant: cpu
          cmake_flags: >-
            -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
            -DGGML_BACKEND_DL=OFF
```

### 2. Build Dependencies (apt)

```yaml
- name: Install build dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build
    # Vulkan variant only:
    # sudo apt-get install -y libvulkan-dev
```

### 3. SDK Setup

- **CUDA:** `Jimver/cuda-toolkit@v0.2.35` (works on Linux, same as Windows)
- **Vulkan:** `sudo apt-get install -y libvulkan-dev` (headers only; user provides runtime driver)
- **sccache:** `mozilla-actions/sccache-action` (works on Linux)
- **Ninja:** `lukka/get-cmake` or apt's `ninja-build`

### 4. Engine Build

Same as Windows Ninja build but with GCC/Clang:

```bash
mkdir -p engine/build && cd engine/build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  ${CMAKE_FLAGS} \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache
cmake --build . -j $(nproc)
```

### 5. Native Addons

```bash
# better-sqlite3 — prebuilt for linux/x64
cd server/node_modules/better-sqlite3
npx prebuild-install --runtime node --target ${NODE_VERSION} --arch x64 --platform linux \
  || npm run build-release

# ffmpeg-static — auto-downloads Linux binary
cd server/node_modules/ffmpeg-static && node install.js
```

### 6. ONNX Runtime (SuperSep)

Download Linux GPU variant:
```
https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz
```

Copy `libonnxruntime.so` (+ CUDA provider .so files for CUDA variant) into the engine directory.

### 7. Portable Node.js

```bash
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz"
curl -sL "$NODE_URL" | tar -xz -C /tmp/node-portable --strip-components=1
```

Copy `bin/node` to `runtime/bin/node`.

### 8. Launch Script (`HOT-Step.sh`)

Based on macOS script, with restart loop:

```bash
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export HOT_STEP_ROOT="$DIR"

# Open browser after delay
(sleep 5 && xdg-open "http://localhost:3001" 2>/dev/null) &

# Restart loop
while true; do
    echo "Starting server..."
    "$DIR/runtime/bin/node" "$DIR/server/server.mjs"
    
    if [ -f "$DIR/.restart-requested" ]; then
        rm "$DIR/.restart-requested"
        echo "[HOT-Step] Restarting..."
        continue
    fi
    
    echo "[HOT-Step] Server stopped."
    break
done
```

### 9. Assembly

Clone of macOS assembly logic:
- Engine binaries (no `.exe` extension)
- Shared libraries: `*.so` (not `*.dylib`)
- No rpath fixing needed (Linux uses `LD_LIBRARY_PATH` or `RPATH`)
  - Set `CMAKE_INSTALL_RPATH=$ORIGIN` at configure time
- Variant marker: `echo "cuda" > engine/.variant`

### 10. Packaging

```bash
tar -czf "HOT-Step-CPP-v${VERSION}-linux-x64-${VARIANT}.tar.gz" -C release/staging .
```

### 11. Release Job Update

Add `build-linux` to the `needs` array:

```yaml
release:
  needs: [build-windows, build-macos, build-linux]
```

Download and include Linux artifacts alongside Windows and macOS.

## Differences from macOS

| Aspect | macOS | Linux |
|--------|-------|-------|
| Runner | `macos-15` | `ubuntu-24.04` |
| Package manager | `brew install` | `apt-get install` |
| Shared lib ext | `.dylib` | `.so` |
| Rpath fix | `install_name_tool` | `CMAKE_INSTALL_RPATH=$ORIGIN` |
| GPU backends | Metal | CUDA / Vulkan |
| Variants | single (Metal) | matrix (cuda/vulkan/cpu) |
| Node.js arch | `darwin-arm64` | `linux-x64` |
| ORT arch | N/A (no SuperSep yet) | `linux-x64-gpu` |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| glibc version mismatch (user's distro too old) | Build on Ubuntu 22.04 for wider compat, or document minimum glibc |
| CUDA driver version requirements | Document minimum driver version in README |
| Vulkan driver availability varies by distro | Document that user must install GPU drivers |
| better-sqlite3 prebuilt unavailable | Fallback: `npm run build-release` (compiles from source) |
| ORT .so dependencies (libcudnn etc.) | Bundle or document requirements |

## Estimated Effort

~2-3 hours — mostly copy-paste from macOS/Windows jobs with Linux-specific paths. The engine already compiles, so no source changes expected.
