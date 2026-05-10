#!/bin/bash
# HOT-Step CPP — macOS Launcher (Production)
#
# Starts the Node.js server, which in turn spawns ace-server as a child process.
# Automatically finds a compatible Node.js (bundled > brew > system).
#
# Usage:
#   ./launch.sh                           # Normal launch
#   HOT_STEP_ROOT=$(pwd) ./launch.sh      # Portable mode

set -e

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         HOT-Step 9000 ⚡ CPP            ║"
echo "║    High-Performance Music Generation     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Find Node.js ────────────────────────────────────────────────────
# Priority: bundled runtime > brew node@22 > system node (if compatible)
find_node() {
    # 1. Bundled Node (portable release)
    if [ -x "${ROOT_DIR}/runtime/node" ]; then
        echo "${ROOT_DIR}/runtime"
        return 0
    fi

    # 2. Brew node@22
    local brew_paths=(
        "/opt/homebrew/opt/node@22/bin"
        "/usr/local/opt/node@22/bin"
    )
    for bp in "${brew_paths[@]}"; do
        if [ -x "${bp}/node" ]; then
            echo "$bp"
            return 0
        fi
    done

    # 3. System node (with version check)
    if command -v node &>/dev/null; then
        local ver
        ver="$(node --version 2>/dev/null | tr -d 'v')"
        local major="${ver%%.*}"
        if [ "$major" -ge 18 ] && [ "$major" -lt 24 ] 2>/dev/null; then
            echo "$(dirname "$(command -v node)")"
            return 0
        fi
    fi

    return 1
}

NODE_DIR=""
if NODE_DIR=$(find_node); then
    export PATH="${NODE_DIR}:${PATH}"
    echo "  Node.js: $(node --version) ($(which node))"
else
    echo "❌ No compatible Node.js found (need 18-22)."
    echo ""
    echo "   Your system Node: $(node --version 2>/dev/null || echo 'not installed')"
    echo ""
    echo "   Fix: brew install node@22"
    echo "   Or:  Run ./install.sh to set up everything"
    echo ""
    exit 1
fi

# ── Check engine binary ─────────────────────────────────────────────
ENGINE_BIN=""
for candidate in \
    "engine/build/ace-server" \
    "engine/ace-server" \
    "engine/build/Release/ace-server"; do
    if [ -f "$candidate" ]; then
        ENGINE_BIN="$candidate"
        break
    fi
done

if [ -n "$ENGINE_BIN" ]; then
    echo "  Engine:  ${ENGINE_BIN}"
else
    echo "  ⚠️  ace-server not found — the UI will start but generation won't work."
    echo "     Build it: cd engine && ./build-mac.sh"
    echo "     Or run:   ./install.sh"
    echo ""
fi

# ── Check server dependencies ────────────────────────────────────────
if [ ! -d "server/node_modules" ]; then
    echo ""
    echo "📦 Installing server dependencies (first run)..."
    cd server && npm install --loglevel=warn && cd ..
fi

# ── Build UI if needed ───────────────────────────────────────────────
if [ ! -d "ui/dist" ]; then
    if [ -d "ui/node_modules" ]; then
        echo ""
        echo "🔨 Building UI (first run)..."
        cd ui && npx vite build 2>&1 | tail -3 && cd ..
    else
        echo "  ⚠️  UI not built. Run ./install.sh or: cd ui && npm install && npx vite build"
    fi
fi

# ── Gatekeeper hint ──────────────────────────────────────────────────
if [ -n "$ENGINE_BIN" ] && xattr -l "$ENGINE_BIN" 2>/dev/null | grep -q "com.apple.quarantine" 2>/dev/null; then
    echo ""
    echo "  🔒 macOS is blocking unsigned binaries. Fix with:"
    echo "     xattr -cr ${ROOT_DIR}"
    echo ""
fi

# ── Open browser after delay ─────────────────────────────────────────
(sleep 4 && open "http://localhost:3001" 2>/dev/null) &

# ── Start the server (with restart loop) ─────────────────────────────
MARKER="${ROOT_DIR}/.restart-requested"
rm -f "$MARKER"

start_server() {
    cd "${ROOT_DIR}/server"
    if [ -d "node_modules/tsx" ]; then
        node --import tsx/esm src/index.ts
    else
        npx tsx src/index.ts
    fi
}

while true; do
    echo ""
    echo "  🎵 Starting server..."
    echo ""
    start_server
    EXIT_CODE=$?

    # Check for restart marker
    if [ -f "$MARKER" ]; then
        rm -f "$MARKER"
        echo ""
        echo "  🔄 Restarting server..."
        sleep 1
        continue
    fi

    # Normal exit
    echo ""
    echo "  Server exited (code $EXIT_CODE)."
    break
done

