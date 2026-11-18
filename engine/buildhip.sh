#!/bin/bash
# HOT-Step CPP — AMD ROCm/HIP build script.
#
# Community-contributed backend (PR #115). Requires ROCm 6.1+ (hard floor in
# ggml/src/ggml-hip/CMakeLists.txt). No CI coverage builds this — see
# .github/workflows/rocm-build.yml for the compile-only guard.

set -e

# Anchor to engine/. Without this, running the script from the repo root makes
# the wipe below delete the WRONG build/ and points `cmake ..` at the wrong tree.
cd "$(dirname "$0")"

# The wipe is intentional: every backend script configures into the shared
# engine/build/, and a stale CMakeCache from another backend poisons a HIP
# configure. This is always a full rebuild.
rm -rf build
mkdir build
cd build

if command -v rocminfo >/dev/null 2>&1; then
    GFX_NAME=$(rocminfo | awk '/ *Name: +gfx[1-9]/ {print $2; exit}')
else
    echo "rocminfo missing!"
fi

if [ -z "${GFX_NAME}" ]; then
    echo "Warn: Couldn't detect AMD GPU for HIP! Using fallback value (gfx1030)."
    GFX_NAME="gfx1030"
else
    echo "Building for GPU arch: ${GFX_NAME}"
fi

cmake .. -DGGML_HIP=ON -DGPU_TARGETS="${GFX_NAME}" \
    -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j "$(nproc --ignore=1)"
