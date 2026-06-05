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

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# TensorRT SDK for DiT/LM acceleration (native TRT API)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime GPU SDK (Linux x64) for SuperSep stem separation
ARG ORT_VERSION=1.25.1
RUN curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
    | tar xz -C /opt \
    && mv "/opt/onnxruntime-linux-x64-gpu-${ORT_VERSION}" /opt/onnxruntime

WORKDIR /build
COPY engine/ .

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
    && cmake --build build --config Release -j"$(nproc)"

# Stage all binaries + shared libs into /staging for clean COPY
RUN mkdir -p /staging/engine \
    && for bin in ace-server mastering mp3-codec vst-host; do \
         [ -f "build/${bin}" ] && cp "build/${bin}" /staging/engine/; \
       done \
    && find build/ -maxdepth 1 -name '*.so' -exec cp {} /staging/engine/ \; \
    && find build/ -maxdepth 1 -name '*.so.*' -exec cp {} /staging/engine/ \; \
    && cp /opt/onnxruntime/lib/libonnxruntime*.so* /staging/engine/ 2>/dev/null || true


# ── Stage 2: UI Builder ─────────────────────────────────────────────
FROM node:22-slim AS ui-builder

WORKDIR /build/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ .
RUN npx vite build


# ── Stage 3: Server Dependencies ────────────────────────────────────
# Full node image (not slim) — better-sqlite3 needs Python + g++ for native build
FROM node:22 AS server-deps

WORKDIR /build/server
COPY server/package*.json ./
# Install production deps. tsx is in both devDependencies and optionalDependencies,
# but npm --omit=dev deduplicates and skips it. Install explicitly.
RUN npm install --omit=dev && npm install tsx


# ── Stage 4: Runtime ────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Install Node.js 22 (LTS) + runtime libraries the engine needs
# TensorRT runtime libraries for DiT/LM acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

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

# Docker-specific environment config
COPY .env.docker /app/.env

# Entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# GGML backends + ORT need to find their .so files
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/app/engine:${LD_LIBRARY_PATH}
ENV NODE_ENV=production

EXPOSE 3001 8085

ENTRYPOINT ["/app/entrypoint.sh"]
