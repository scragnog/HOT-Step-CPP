#!/bin/bash
# HOT-Step CPP — macOS build script (Apple Silicon + Metal)
#
# Prerequisites:
#   - Xcode (full install, not just command line tools — needed for Metal)
#   - CMake 3.20+   (brew install cmake)
#   - Ninja          (brew install ninja)  — optional but faster
#
# Usage:
#   ./build-mac.sh          # Release build with Metal
#   ./build-mac.sh Debug    # Debug build
#
# ONNX Runtime (SuperSep):
#   To enable stem separation, download the macOS ONNX Runtime package:
#     brew install onnxruntime
#   Or download from: https://github.com/microsoft/onnxruntime/releases
#   Place in engine/deps/onnxruntime-osx-arm64/ (or set ORT_ROOT)

set -e

cd "$(dirname "$0")"

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build"

echo "╔══════════════════════════════════════════╗"
echo "║       HOT-Step CPP — macOS Build         ║"
echo "║        Apple Silicon + Metal             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Build type: ${BUILD_TYPE}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Detect Ninja
GENERATOR=""
if command -v ninja &>/dev/null; then
    GENERATOR="-G Ninja"
    echo "Generator: Ninja"
else
    echo "Generator: Unix Makefiles (install ninja for faster builds: brew install ninja)"
fi

# Auto-detect ONNX Runtime
ORT_FLAG=""
if [ -n "${ORT_ROOT}" ]; then
    echo "ONNX Runtime: ${ORT_ROOT}"
elif [ -d "../deps/onnxruntime-osx-arm64" ]; then
    export ORT_ROOT="../deps/onnxruntime-osx-arm64"
    echo "ONNX Runtime: auto-detected at ${ORT_ROOT}"
else
    echo "ONNX Runtime: not found (SuperSep will be disabled)"
    echo "  → To enable: brew install onnxruntime, or download manually"
fi

cmake .. \
    ${GENERATOR} \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BACKEND_DL=OFF

# Build using all available cores
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --config "${BUILD_TYPE}" -j "${NCPU}"

echo ""
echo "═══════════════════════════════════════════"
echo "  Build complete! Binaries in: $(pwd)/"
echo ""
echo "  Targets built:"
[ -f ace-server ] && echo "    ✓ ace-server"
[ -f mastering ] && echo "    ✓ mastering"
[ -f mp3-codec ] && echo "    ✓ mp3-codec"
[ -f vst-host ] && echo "    ✓ vst-host"
echo ""
echo "  Next: cd .. && ./launch.sh"
echo "═══════════════════════════════════════════"
