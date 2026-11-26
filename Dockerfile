# ============================================================================
# HOT-Step 9000 CPP — Docker Build
# Multi-stage: Engine (CUDA) → UI (Vite) → Server deps → Runtime
#
# Usage:
#   docker compose build                              # Dev build (Blackwell only)
#   docker compose build --build-arg CUDA_ARCHS="75;80;86;89;90;120a"  # Distribution
# ============================================================================

# ── Stage 1: Engine Builder ─────────────────────────────────────────
# CUDA devel image: has nvcc, CUDA headers, cuDNN for building
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS engine-builder

# Default to Blackwell (sm_120a) for fast dev builds.
# Override with --build-arg for multi-arch distribution.
ARG CUDA_ARCHS="120a"
ARG ENABLE_TRT=1
ARG BUILD_JOBS

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build build-essential git curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# TensorRT SDK for DiT/LM acceleration (native TRT API)
RUN if [ "$ENABLE_TRT" = 1 ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Stage the Linux x64 ONNX Runtime GPU SDK from NuGet.
# NuGet ships the unversioned file. The Linux loader uses the ELF SONAME.
ARG ORT_VERSION=1.25.1
RUN ort_tmp="$(mktemp -d)" \
    && curl -fSL --retry 3 -o "$ort_tmp/ort.nupkg" \
        "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Linux/${ORT_VERSION}" \
    && python3 -m zipfile -e "$ort_tmp/ort.nupkg" "$ort_tmp/sdk" \
    && mkdir -p /opt/onnxruntime \
    && mv "$ort_tmp/sdk/buildTransitive/native/include" /opt/onnxruntime/include \
    && mv "$ort_tmp/sdk/runtimes/linux-x64/native" /opt/onnxruntime/lib \
    && test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h \
    && test -f /opt/onnxruntime/lib/libonnxruntime.so \
    && test -f /opt/onnxruntime/lib/libonnxruntime_providers_cuda.so \
    && test -f /opt/onnxruntime/lib/libonnxruntime_providers_shared.so \
    && ln -s libonnxruntime.so /opt/onnxruntime/lib/libonnxruntime.so.1 \
    && rm -rf "$ort_tmp"

WORKDIR /build/engine
COPY engine/ .
# The CPU numeric test includes this oracle relative to engine/tools/.
COPY tools/yue2-aitk-reference/convrot_cpu.h /build/tools/yue2-aitk-reference/convrot_cpu.h
RUN test -f ggml/CMakeLists.txt && test -f vendor/vst3sdk/CMakeLists.txt || \
    (echo "Initialize submodules first: git submodule update --init --recursive" >&2; exit 1)

# Build the C++ engine with CUDA + ONNX Runtime
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DL=ON \
    -DORT_ROOT=/opt/onnxruntime \
    && cmake --build build --config Release -j"${BUILD_JOBS:-$(nproc)}"

# Stage all binaries + shared libs into /staging for clean COPY
RUN mkdir -p /staging/engine \
    && for bin in ace-server ace-synth ace-lm ace-understand ace-train ace-caption ace-midi neural-codec quantize mastering mp3-codec vst-host; do \
         test -f "build/${bin}" || { echo "Missing engine binary: ${bin}" >&2; exit 1; }; \
         cp "build/${bin}" /staging/engine/; \
       done \
    && find build/ -maxdepth 1 -name '*.so' -exec cp {} /staging/engine/ \; \
    && find build/ -maxdepth 1 -name '*.so.*' -exec cp {} /staging/engine/ \; \
    && cp -P /opt/onnxruntime/lib/libonnxruntime*.so* /staging/engine/


# ── Stage 2: Server Dependencies ────────────────────────────────────
# Official Node binaries; native modules build on the runtime's Ubuntu/glibc.
FROM node:22 AS node-runtime
FROM ubuntu:22.04 AS server-deps
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/server
COPY server/package*.json ./
# Include dev dependencies so the locked tsx runtime is retained despite npm
# deduplicating its optional/dev entries. No unlocked follow-up install.
RUN npm ci --include=dev


# ── Stage 3: UI Builder ─────────────────────────────────────────────
FROM node:22-slim AS ui-builder

WORKDIR /build/ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ .
# Browser runtime imports need these sibling sources. Copy contracts/ whole so a
# new contract can't break the image; they import zod/v4, so retain the server's
# locked package and its normal resolution path (their service imports are type-only).
COPY server/src/contracts/ /build/server/src/contracts/
COPY server/src/services/streamSessions/analysis.ts /build/server/src/services/streamSessions/analysis.ts
COPY --from=server-deps /build/server/node_modules/zod/ /build/server/node_modules/zod/
RUN npx vite build


# ── Stage 4: Runtime ────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime-base
ARG ENABLE_TRT=1

# Install Node.js 22 (LTS) + runtime libraries the engine needs
# TensorRT 11 runtime libraries for native DiT/LM acceleration (dit-trt.h)
# Note: ORT TRT EP needs TRT 10 (libnvinfer.so.10) but segfaults due to
# version mismatch with CUDA 12.8. ORT falls back to CUDA EP gracefully
# which is fine for the small text/cond encoders. Native TRT 11 handles DiT.
RUN if [ "$ENABLE_TRT" = 1 ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer11 libnvinfer-plugin11 libnvonnxparsers11 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# Engine binaries + shared libraries (GGML backends, ORT, cuDNN)
COPY --from=engine-builder /staging/engine/ /app/engine/
# NOTE: Lua plugins are bind-mounted at runtime via docker-compose.yml

# Server source code + production dependencies
COPY server/ /app/server/
COPY --from=server-deps /build/server/node_modules/ /app/server/node_modules/

# UI static files (production build)
COPY --from=ui-builder /build/ui/dist/ /app/ui/dist/

# Noise samples (small WAV files for noise profiling, baked into image)
COPY noise_samples/ /app/noise_samples/

# Entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# GGML backends + ORT need to find their .so files
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/app/engine:${LD_LIBRARY_PATH}
ENV NODE_ENV=production

EXPOSE 3001 8085

ENTRYPOINT ["/app/entrypoint.sh"]

# RunPod persists its models, adapters, library and logs on the mounted volume.
FROM runtime-base AS runpod
COPY engine/plugins/ /app/engine/plugins/
COPY plugins/ /app/plugins/
ENV HOT_STEP_DATA_ROOT=/workspace \
    ACESTEPCPP_EXE=/app/engine/ace-server \
    ACESTEPCPP_MODELS=/workspace/models \
    ACESTEPCPP_ADAPTERS=/workspace/adapters \
    ACESTEPCPP_HOST=127.0.0.1 \
    ACESTEPCPP_PORT=8085 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=3001 \
    DATA_DIR=/workspace/data \
    GGML_CUDA_GRAPH_OPT=1
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${SERVER_PORT}/api/health >/dev/null || exit 1

# Keep the default build and Compose environment unchanged.
FROM runtime-base AS runtime
COPY .env.docker /app/.env
