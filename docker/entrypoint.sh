#!/bin/bash
# ============================================================================
# HOT-Step 9000 CPP — Docker Entrypoint
# Verifies GPU access and starts the Node.js server (which spawns ace-server)
# ============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     HOT-Step 9000 ⚡ Docker             ║"
echo "║    High-Performance Music Generation     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Verify GPU access ───────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:    ${GPU_NAME} (${GPU_MEM})"
    echo "  Driver: ${GPU_DRIVER}"
else
    echo "  ⚠ nvidia-smi not found — GPU may not be available"
    echo "  Check that Docker has GPU access (--gpus=all or deploy.resources in compose)"
fi

# ── Verify engine binary exists ─────────────────────────────────────
if [ -f /app/engine/ace-server ]; then
    echo "  Engine: /app/engine/ace-server ✓"
else
    echo "  ⚠ ace-server binary not found at /app/engine/"
    echo "  The build may have failed — check Docker build logs"
fi

# ── Ensure bind-mount directories exist (in case host dirs are empty) ──
mkdir -p /app/models /app/adapters /app/server/data

# ── Check for models ────────────────────────────────────────────────
MODEL_COUNT=$(find /app/models -name '*.gguf' 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "  Models: ${MODEL_COUNT} GGUF file(s) found"
else
    echo "  ⚠ No .gguf models found in /app/models"
    echo "  Place model files in the ./models/ directory on the host"
fi

echo ""
echo "  Server:  http://localhost:${SERVER_PORT:-3001}"
echo "  Engine:  http://localhost:${ACESTEPCPP_PORT:-8085}"
echo ""
echo "  Starting server..."
echo ""

# ── Start the Node.js server ────────────────────────────────────────
# The server spawns ace-server as a child process automatically.
# Use exec to replace the shell — proper signal handling for graceful shutdown.
cd /app/server
exec node --import tsx/esm src/index.ts
