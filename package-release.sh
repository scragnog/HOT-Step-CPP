#!/bin/bash
# HOT-Step CPP — macOS Release Packager
#
# Builds a portable .tar.gz release that requires ZERO prerequisites.
# End users just extract and run ./launch.sh — no Xcode, no brew, no npm.
#
# What gets bundled:
#   - Pre-built C++ engine binaries (Metal GPU)
#   - Bundled Node.js 22 runtime (~40 MB)
#   - Pre-installed server dependencies (production only)
#   - Pre-built React UI
#   - Launch scripts
#
# Usage:
#   ./package-release.sh                # Build release package
#   ./package-release.sh --skip-build   # Package only (engine already built)
#
# Output:
#   HOT-Step-CPP-v{VERSION}-macOS-arm64.tar.gz

set -e

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# ── Config ───────────────────────────────────────────────────────────
VERSION="${VERSION:-1.0.0}"
ARCH="arm64"
NODE_VERSION="22.22.2"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-darwin-${ARCH}.tar.xz"
RELEASE_NAME="HOT-Step-CPP-v${VERSION}-macOS-${ARCH}"
STAGING_DIR="${ROOT_DIR}/release-staging/${RELEASE_NAME}"
OUTPUT_FILE="${ROOT_DIR}/${RELEASE_NAME}.tar.gz"
SKIP_BUILD=0

for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=1 ;;
        --version=*) VERSION="${arg#*=}"; RELEASE_NAME="HOT-Step-CPP-v${VERSION}-macOS-${ARCH}" ;;
    esac
done

# ── Colors ───────────────────────────────────────────────────────────
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

TOTAL_STEPS=7

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   HOT-Step CPP — Release Packager        ║${NC}"
echo -e "${BOLD}║      macOS ${ARCH} Portable Build       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Version:  ${VERSION}"
echo "  Output:   ${RELEASE_NAME}.tar.gz"
echo ""

# ── Find compatible Node for the build process ──────────────────────
find_node() {
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
else
    fail "Node.js 18-22 required for building. Run: brew install node@22"
fi

# ── Step 1: Build engine ─────────────────────────────────────────────
step 1 "Building C++ engine..."

if [ "$SKIP_BUILD" -eq 1 ] && [ -f "engine/build/ace-server" ]; then
    ok "Skipping build (--skip-build, binaries exist)"
else
    cd engine
    rm -rf build && mkdir build && cd build
    cmake .. \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_BACKEND_DL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tail -3
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build . --config Release -j "${NCPU}" 2>&1 | grep -E '^\[100|Built target|error:'
    cd "${ROOT_DIR}"
fi

# Verify binaries
for bin in ace-server mastering mp3-codec vst-host; do
    [ -f "engine/build/${bin}" ] || fail "engine/build/${bin} not found"
done
ok "All engine binaries built"

# ── Step 2: Install server deps (production) ─────────────────────────
step 2 "Installing server dependencies (production)..."

cd server
rm -rf node_modules
npm install --omit=dev --loglevel=warn 2>&1 | tail -3
# Rebuild native modules
npm rebuild better-sqlite3 2>&1 | tail -1
# tsx is in devDependencies but needed at runtime to run the TypeScript server
npm install tsx --save-optional --loglevel=warn 2>&1 | tail -1
ok "Server dependencies installed (production + tsx)"
cd "${ROOT_DIR}"

# ── Step 3: Build UI ─────────────────────────────────────────────────
step 3 "Building production UI..."

cd ui
[ -d "node_modules" ] || npm install --loglevel=warn
npx vite build 2>&1 | tail -3
ok "UI built"
cd "${ROOT_DIR}"

# ── Step 4: Download Node.js runtime ─────────────────────────────────
step 4 "Downloading Node.js ${NODE_VERSION} runtime..."

NODE_CACHE="${ROOT_DIR}/release-staging/node-cache"
NODE_ARCHIVE="${NODE_CACHE}/node-v${NODE_VERSION}-darwin-${ARCH}.tar.xz"
NODE_EXTRACTED="${NODE_CACHE}/node-v${NODE_VERSION}-darwin-${ARCH}"

mkdir -p "${NODE_CACHE}"

if [ -f "${NODE_EXTRACTED}/bin/node" ]; then
    ok "Using cached Node.js download"
else
    echo "  Downloading from ${NODE_URL}..."
    curl -L --progress-bar -o "${NODE_ARCHIVE}" "${NODE_URL}"
    cd "${NODE_CACHE}"
    tar xf "$(basename "${NODE_ARCHIVE}")"
    cd "${ROOT_DIR}"
    ok "Node.js ${NODE_VERSION} downloaded"
fi

# ── Step 5: Assemble release ─────────────────────────────────────────
step 5 "Assembling release package..."

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Engine binaries + dylibs
mkdir -p "${STAGING_DIR}/engine/build"
for bin in ace-server ace-lm ace-synth ace-understand mastering mp3-codec neural-codec quantize vst-host; do
    [ -f "engine/build/${bin}" ] && cp "engine/build/${bin}" "${STAGING_DIR}/engine/build/"
done
# Copy ggml dylibs
cp engine/build/libggml*.dylib "${STAGING_DIR}/engine/build/" 2>/dev/null || true

# Lua plugins — dual directory system:
#   engine/plugins/  = native/built-in plugins (solvers, schedulers, guidance)
#   plugins/         = community/user plugins (same structure, overrides native)
# ace-server scans both via engine_dir and project_dir resolution from binary path
if [ -d "engine/plugins" ]; then
    mkdir -p "${STAGING_DIR}/engine/plugins"
    cp -r engine/plugins/* "${STAGING_DIR}/engine/plugins/"
    PLUGIN_COUNT=$(find "${STAGING_DIR}/engine/plugins" -name "*.lua" | wc -l | tr -d ' ')
    ok "${PLUGIN_COUNT} native plugins copied (engine/plugins/)"
fi
if [ -d "plugins" ]; then
    cp -r plugins "${STAGING_DIR}/"
    ok "Community plugins dir copied (plugins/)"
fi

# Server (source + node_modules, no devDependencies)
mkdir -p "${STAGING_DIR}/server"
cp server/package.json server/package-lock.json "${STAGING_DIR}/server/"
cp server/tsconfig.json "${STAGING_DIR}/server/"
cp -r server/node_modules "${STAGING_DIR}/server/"
cp -r server/src "${STAGING_DIR}/server/"
[ -f server/migrate_lireek.cjs ] && cp server/migrate_lireek.cjs "${STAGING_DIR}/server/"
ok "Server copied"

# Pre-built UI
mkdir -p "${STAGING_DIR}/ui"
cp -r ui/dist "${STAGING_DIR}/ui/"
ok "UI copied"

# Node.js runtime (just the binary + npm essentials)
mkdir -p "${STAGING_DIR}/runtime"
cp "${NODE_EXTRACTED}/bin/node" "${STAGING_DIR}/runtime/node"
# Copy npm/npx for server to use tsx
mkdir -p "${STAGING_DIR}/runtime/lib"
cp -r "${NODE_EXTRACTED}/lib/node_modules" "${STAGING_DIR}/runtime/lib/"
# Create npx/npm symlinks
ln -sf "../lib/node_modules/npm/bin/npm-cli.js" "${STAGING_DIR}/runtime/npm"
ln -sf "../lib/node_modules/npm/bin/npx-cli.js" "${STAGING_DIR}/runtime/npx"
ok "Node.js runtime bundled"

# Empty directories for user content
mkdir -p "${STAGING_DIR}/models"
mkdir -p "${STAGING_DIR}/adapters"

# Scripts
cp launch.sh "${STAGING_DIR}/"
cp .env.example "${STAGING_DIR}/"
chmod +x "${STAGING_DIR}/launch.sh"

# README snippet for release
cat > "${STAGING_DIR}/README.txt" << 'HEREDOC'
HOT-Step CPP — macOS (Apple Silicon)

Quick Start:
  1. Extract this archive
  2. Run: ./launch.sh
  3. Open http://localhost:3001 in your browser
  4. Go to Models → Get More Models to download AI models (~7 GB)

If macOS blocks the app (unsigned binary):
  xattr -cr /path/to/HOT-Step-CPP-macOS-arm64/

Requirements:
  - macOS 13+ (Apple Silicon)
  - ~10 GB free disk space
  - No other software needed — Node.js is bundled

For the full README, source code, and updates:
  https://github.com/scragnog/HOT-Step-CPP
HEREDOC
ok "Release assembled"

# ── Step 6: Set permissions ──────────────────────────────────────────
step 6 "Setting permissions..."

find "${STAGING_DIR}/engine/build" -type f -perm +0111 -exec chmod +x {} \; 2>/dev/null || true
chmod +x "${STAGING_DIR}/engine/build/"* 2>/dev/null || true
chmod +x "${STAGING_DIR}/runtime/node"
# npm/npx are symlinks to .js files — make targets executable
chmod +x "${STAGING_DIR}/runtime/lib/node_modules/npm/bin/npm-cli.js" 2>/dev/null || true
chmod +x "${STAGING_DIR}/runtime/lib/node_modules/npm/bin/npx-cli.js" 2>/dev/null || true
chmod +x "${STAGING_DIR}/launch.sh"
ok "Permissions set"

# ── Step 7: Create archive ──────────────────────────────────────────
step 7 "Creating archive..."

cd release-staging
tar czf "${OUTPUT_FILE}" "${RELEASE_NAME}"
cd "${ROOT_DIR}"

# Size info
SIZE_MB=$(du -sm "${OUTPUT_FILE}" | cut -f1)
ok "${RELEASE_NAME}.tar.gz (${SIZE_MB} MB)"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║        Release package ready! 🎉         ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  File:  ${OUTPUT_FILE}"
echo "  Size:  ${SIZE_MB} MB"
echo ""
echo "  Test locally:"
echo "    cd release-staging/${RELEASE_NAME}"
echo "    ./launch.sh"
echo ""

# Clean up staging (keep the archive)
# rm -rf release-staging
