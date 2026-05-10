#!/bin/bash
# HOT-Step CPP — macOS Development Mode
#
# Starts Vite dev server (hot-reload UI) + Node.js server (tsx watch).
# Automatically finds a compatible Node.js.
#
# Usage:
#   ./dev.sh

set -e

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     HOT-Step 9000 ⚡ DEV MODE           ║"
echo "║       Vite HMR + tsx watch               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Find Node.js ────────────────────────────────────────────────────
find_node() {
    if [ -x "${ROOT_DIR}/runtime/node" ]; then
        echo "${ROOT_DIR}/runtime"; return 0
    fi
    for bp in "/opt/homebrew/opt/node@22/bin" "/usr/local/opt/node@22/bin"; do
        if [ -x "${bp}/node" ]; then echo "$bp"; return 0; fi
    done
    if command -v node &>/dev/null; then
        local ver; ver="$(node --version 2>/dev/null | tr -d 'v')"
        local major="${ver%%.*}"
        if [ "$major" -ge 18 ] && [ "$major" -lt 24 ] 2>/dev/null; then
            echo "$(dirname "$(command -v node)")"; return 0
        fi
    fi
    return 1
}

NODE_DIR=""
if NODE_DIR=$(find_node); then
    export PATH="${NODE_DIR}:${PATH}"
    echo "  Node.js: $(node --version)"
else
    echo "❌ No compatible Node.js found (need 18-22)."
    echo "   Fix: brew install node@22"
    exit 1
fi

# ── Install deps if needed ───────────────────────────────────────────
if [ ! -d "server/node_modules" ]; then
    echo "📦 Installing server dependencies..."
    cd server && npm install --loglevel=warn && cd ..
fi
if [ ! -d "ui/node_modules" ]; then
    echo "📦 Installing UI dependencies..."
    cd ui && npm install --loglevel=warn && cd ..
fi

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[dev.sh] Shutting down..."
    if [ -n "${VITE_PID}" ] && kill -0 "${VITE_PID}" 2>/dev/null; then
        kill "${VITE_PID}" 2>/dev/null || true
        echo "[dev.sh] Vite stopped"
    fi
}
trap cleanup EXIT INT TERM

# ── Start Vite (background) ─────────────────────────────────────────
echo "🚀 Starting Vite dev server..."
cd ui
npx vite &
VITE_PID=$!
cd ..

sleep 2

# ── Start Node server (foreground) ──────────────────────────────────
echo "🚀 Starting Node.js server..."
echo "   UI:   http://localhost:3000 (Vite HMR)"
echo "   API:  http://localhost:3001"
echo ""
cd server
npx tsx watch src/index.ts
