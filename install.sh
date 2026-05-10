#!/bin/bash
# HOT-Step CPP — macOS Build-from-Source Installer
#
# One command to go from fresh clone to working app:
#   ./install.sh
#
# What this does:
#   1. Checks prerequisites (Xcode, Metal Toolchain, Node.js, CMake)
#   2. Initializes git submodules
#   3. Builds the C++ engine with Metal GPU acceleration
#   4. Installs server + UI dependencies
#   5. Builds the production UI
#
# After install completes, run ./launch.sh to start the app.

set -e

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; exit 1; }

TOTAL_STEPS=6

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     HOT-Step CPP — macOS Installer       ║${NC}"
echo -e "${BOLD}║       Build from Source (Apple Silicon)   ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Prerequisites ─────────────────────────────────────
step 1 "Checking prerequisites..."

# macOS check
if [[ "$(uname)" != "Darwin" ]]; then
    fail "This script is for macOS only."
fi
ok "macOS $(sw_vers -productVersion) ($(uname -m))"

# Xcode or Command Line Tools
if xcode-select -p &>/dev/null; then
    XCODE_PATH="$(xcode-select -p)"
    if [[ "$XCODE_PATH" == *"Xcode.app"* ]]; then
        ok "Xcode installed at ${XCODE_PATH}"
    else
        ok "Command Line Tools installed"
    fi
else
    fail "Xcode or Command Line Tools required.\n  Install: xcode-select --install"
fi

# Metal compiler
if xcrun metal --version &>/dev/null; then
    ok "Metal compiler available"
else
    warn "Metal compiler not found."
    echo "      This is required for GPU acceleration."
    echo ""
    echo "      If you have Xcode installed, download the Metal Toolchain:"
    echo "        sudo xcodebuild -runFirstLaunch"
    echo "        xcodebuild -downloadComponent MetalToolchain"
    echo ""
    echo "      Without Metal, the engine will still build but will use CPU only."
    echo ""
    read -p "  Continue without Metal? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    NO_METAL=1
fi

# CMake
if command -v cmake &>/dev/null; then
    ok "CMake $(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
else
    fail "CMake not found.\n  Install: brew install cmake"
fi

# Git
if command -v git &>/dev/null; then
    ok "Git $(git --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
else
    fail "Git not found."
fi

# ── Step 2: Find or install Node.js ──────────────────────────────────
step 2 "Checking Node.js..."

find_compatible_node() {
    # Priority: brew node@22 > system node (if compatible)
    local candidates=(
        "/opt/homebrew/opt/node@22/bin/node"
        "/usr/local/opt/node@22/bin/node"
    )

    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
            local ver
            ver="$("$candidate" --version 2>/dev/null | tr -d 'v')"
            local major="${ver%%.*}"
            if [ "$major" -ge 18 ] && [ "$major" -lt 24 ] 2>/dev/null; then
                echo "$candidate"
                return 0
            fi
        fi
    done

    # Check system node
    if command -v node &>/dev/null; then
        local ver
        ver="$(node --version 2>/dev/null | tr -d 'v')"
        local major="${ver%%.*}"
        if [ "$major" -ge 18 ] && [ "$major" -lt 24 ] 2>/dev/null; then
            echo "$(command -v node)"
            return 0
        fi
    fi

    return 1
}

NODE_BIN=""
if NODE_BIN=$(find_compatible_node); then
    NODE_VER="$("$NODE_BIN" --version)"
    NODE_DIR="$(dirname "$NODE_BIN")"
    ok "Node.js ${NODE_VER} at ${NODE_BIN}"
else
    # Try to install via brew
    echo "  Node.js 18-22 required (Node 24+ is not supported)."
    if command -v brew &>/dev/null; then
        echo ""
        read -p "  Install Node.js 22 via Homebrew? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "  Installing node@22..."
            brew install node@22
            NODE_BIN="/opt/homebrew/opt/node@22/bin/node"
            if [ ! -x "$NODE_BIN" ]; then
                NODE_BIN="/usr/local/opt/node@22/bin/node"
            fi
            if [ -x "$NODE_BIN" ]; then
                NODE_DIR="$(dirname "$NODE_BIN")"
                ok "Node.js $("$NODE_BIN" --version) installed"
            else
                fail "Node.js installation failed."
            fi
        else
            fail "Node.js 18-22 is required. Install manually:\n  brew install node@22"
        fi
    else
        fail "Node.js 18-22 required.\n  Install Homebrew first: https://brew.sh\n  Then: brew install node@22"
    fi
fi

# Export so npm/npx use the right Node
export PATH="${NODE_DIR}:${PATH}"

# ── Step 3: Initialize submodules ────────────────────────────────────
step 3 "Initializing git submodules..."

if [ -f "engine/ggml/CMakeLists.txt" ] && [ -f "engine/vendor/vst3sdk/CMakeLists.txt" ]; then
    ok "Submodules already initialized"
else
    git submodule update --init --recursive
    ok "Submodules initialized"
fi

# ── Step 4: Build the C++ engine ─────────────────────────────────────
step 4 "Building C++ engine (this may take a few minutes)..."

cd engine
mkdir -p build
cd build

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_BACKEND_DL=OFF
)

if [ -z "${NO_METAL}" ]; then
    CMAKE_FLAGS+=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
    echo "  GPU: Metal (Apple Silicon)"
else
    echo "  GPU: None (CPU-only build)"
fi

# Detect Ninja for faster builds
if command -v ninja &>/dev/null; then
    CMAKE_FLAGS+=(-G Ninja)
    echo "  Generator: Ninja"
else
    echo "  Generator: Make (install ninja for faster builds: brew install ninja)"
fi

cmake .. "${CMAKE_FLAGS[@]}" 2>&1 | tail -5
echo ""

NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --config Release -j "${NCPU}" 2>&1 | grep -E '^\[|Built target|Linking|error:'

echo ""
[ -f ace-server ]   && ok "ace-server"   || warn "ace-server not built"
[ -f mastering ]    && ok "mastering"    || warn "mastering not built"
[ -f mp3-codec ]    && ok "mp3-codec"    || warn "mp3-codec not built"
[ -f vst-host ]     && ok "vst-host"     || warn "vst-host not built"

cd "${ROOT_DIR}"

# ── Step 5: Install Node.js dependencies ─────────────────────────────
step 5 "Installing Node.js dependencies..."

echo "  Server..."
cd server && npm install --loglevel=warn 2>&1 | tail -3 && cd ..
ok "Server dependencies installed"

echo "  UI..."
cd ui && npm install --loglevel=warn 2>&1 | tail -3 && cd ..
ok "UI dependencies installed"

# ── Step 6: Build the UI ─────────────────────────────────────────────
step 6 "Building production UI..."

cd ui
npx vite build 2>&1 | tail -5
cd ..
ok "UI built → ui/dist/"

# ── Done! ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║         Installation complete! 🎉        ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "    1. Start the app:"
echo -e "       ${CYAN}./launch.sh${NC}"
echo ""
echo "    2. Open in your browser:"
echo -e "       ${CYAN}http://localhost:3001${NC}"
echo ""
echo "    3. Download AI models (~7 GB):"
echo "       Click Models → Get More Models in the app"
echo ""
