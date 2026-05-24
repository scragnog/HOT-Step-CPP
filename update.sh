#!/usr/bin/env bash
# update.sh — One-click update for HOT-Step-CPP source builders (Linux/macOS).
#
# Pulls latest code, verifies integration hooks, rebuilds everything
# incrementally. Reuses existing build infrastructure patterns.
#
# Usage:
#   ./update.sh              Incremental update (safe, default)
#   ./update.sh --force      Reset local changes before pulling (with confirmation)
#   ./update.sh --clean      Force clean engine rebuild
#   ./update.sh --skip-engine  Skip engine rebuild (UI/server changes only)
#   ./update.sh --help       Show this help
#
# For portable release users: you don't need this script.
#   Download new releases from GitHub instead.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Parse arguments ──────────────────────────────────────────────────
FORCE=0
CLEAN=0
SKIP_ENGINE=0

show_help() {
    cat <<EOF

Usage: ./update.sh [options]

Options:
  --force         Discard local changes before pulling (with confirmation)
  --clean         Force clean engine rebuild
  --skip-engine   Skip engine rebuild (UI/server changes only)
  --help, -h      Show this help

This script is for SOURCE BUILDERS who cloned the repository.
Portable release users should download new releases from GitHub.

Prerequisites:
  - Git
  - CMake 3.10+
  - Node.js LTS + npm
  - C++ compiler (gcc/clang with C++17 support)
  - NVIDIA CUDA Toolkit (optional, for GPU acceleration)
  - Vulkan SDK (optional, for Vulkan backend)

EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --help|-h) show_help ;;
        --force) FORCE=1 ;;
        --clean) CLEAN=1 ;;
        --skip-engine) SKIP_ENGINE=1 ;;
        *) echo -e "${RED}Unknown argument: $arg${NC}"; show_help ;;
    esac
done

echo ""
echo "======================================================="
echo "  HOT-Step-CPP — Smart Update"
echo "======================================================="
echo ""

# ── Phase 0: Prerequisites check ────────────────────────────────────
echo -e "${CYAN}[1/5] Checking prerequisites...${NC}"
PREREQ_OK=1

for cmd in git cmake node npm; do
    if ! command -v "$cmd" &>/dev/null; then
        # cmake is only required if we're building the engine
        if [ "$cmd" = "cmake" ] && [ "$SKIP_ENGINE" = "1" ]; then
            continue
        fi
        echo -e "  ${RED}[FAIL] $cmd not found in PATH${NC}"
        PREREQ_OK=0
    fi
done

if [ "$PREREQ_OK" = "0" ]; then
    echo ""
    echo "  Fix the above issues and try again."
    exit 1
fi
echo "  All prerequisites found."

# ── Phase 1: Pre-flight safety ──────────────────────────────────────
echo ""
echo -e "${CYAN}[2/5] Pre-flight checks...${NC}"

# Save current HEAD for changelog later
OLD_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# Check for uncommitted changes
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    if [ "$FORCE" = "1" ]; then
        echo ""
        echo -e "  ${YELLOW}WARNING: You have uncommitted changes. --force will DISCARD them.${NC}"
        echo ""
        echo "  Modified files:"
        git status --short
        echo ""
        read -p "  Discard all local changes and continue? [y/N] " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "  Aborted by user."
            exit 1
        fi
        echo "  Resetting working tree..."
        git reset --hard
        git clean -fd
    else
        echo ""
        echo -e "  ${RED}ERROR: You have uncommitted changes:${NC}"
        echo ""
        git status --short
        echo ""
        echo "  Options:"
        echo "    1. Commit or stash your changes first"
        echo "    2. Run: ./update.sh --force  (discards ALL local changes)"
        echo ""
        exit 1
    fi
else
    echo "  Working tree is clean."
fi

# Shut down running server (if any)
echo "  Checking for running server..."
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/api/status 2>/dev/null || echo "000")

if [ "$STATUS_CODE" = "200" ]; then
    echo "  Server is running — requesting graceful shutdown..."
    curl -s -X POST http://localhost:3001/api/shutdown >/dev/null 2>&1 || true

    # Wait for ace-server to exit
    retries=0
    while pgrep -x "ace-server" >/dev/null 2>&1; do
        sleep 1
        retries=$((retries + 1))
        if [ "$retries" -ge 10 ]; then
            echo "  Force-killing ace-server after 10s timeout..."
            pkill -9 -x "ace-server" 2>/dev/null || true
            sleep 2
            break
        fi
    done
    echo "  Server stopped."
else
    echo "  No running server detected."
fi

# ── Phase 2: Code sync ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}[3/5] Pulling latest code...${NC}"

if ! git pull --ff-only origin master; then
    echo ""
    echo -e "  ${RED}ERROR: git pull --ff-only failed.${NC}"
    echo "  This usually means your local branch has diverged from origin/master."
    echo "  Options:"
    echo "    1. Run: git rebase origin/master"
    echo "    2. Run: ./update.sh --force  (discards local changes)"
    echo ""
    exit 1
fi

git submodule update --init --recursive || echo "  WARNING: Submodule update had issues."

# Show what changed
NEW_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    echo ""
    echo "  Changes pulled:"
    git log --oneline "$OLD_HEAD".."$NEW_HEAD"
    echo ""
else
    echo "  Already up to date."
fi

# ── Phase 3: Hook verification ──────────────────────────────────────
echo ""
echo -e "${CYAN}[4/5] Verifying integration hooks...${NC}"

if [ -f "engine/verify-hooks.ps1" ]; then
    # On Linux we can't run PowerShell directly — do a simpler grep-based check
    HOOK_ERRORS=0

    if [ -f "engine/src/pipeline-synth-ops.cpp" ]; then
        if grep -q '#include.*"hot-step-sampler\.h"' engine/src/pipeline-synth-ops.cpp; then
            echo -e "  ${GREEN}[OK]${NC} pipeline-synth-ops.cpp -> hot-step-sampler.h"
        else
            echo -e "  ${RED}[FAIL]${NC} pipeline-synth-ops.cpp missing hot-step-sampler.h"
            HOOK_ERRORS=$((HOOK_ERRORS + 1))
        fi
    fi

    if [ -f "engine/src/model-store.h" ]; then
        if grep -q '#include.*"hot-step-params\.h"' engine/src/model-store.h; then
            echo -e "  ${GREEN}[OK]${NC} model-store.h -> hot-step-params.h"
        else
            echo -e "  ${RED}[FAIL]${NC} model-store.h missing hot-step-params.h"
            HOOK_ERRORS=$((HOOK_ERRORS + 1))
        fi
    fi

    if [ -f "engine/src/dit.h" ]; then
        if grep -q '#include.*"adapter-merge\.h"' engine/src/dit.h; then
            echo -e "  ${GREEN}[OK]${NC} dit.h -> adapter-merge.h"
        else
            echo -e "  ${RED}[FAIL]${NC} dit.h missing adapter-merge.h"
            HOOK_ERRORS=$((HOOK_ERRORS + 1))
        fi
        if grep -q '#include.*"adapter-runtime\.h"' engine/src/dit.h; then
            echo -e "  ${GREEN}[OK]${NC} dit.h -> adapter-runtime.h"
        else
            echo -e "  ${RED}[FAIL]${NC} dit.h missing adapter-runtime.h"
            HOOK_ERRORS=$((HOOK_ERRORS + 1))
        fi
    fi

    if [ "$HOOK_ERRORS" -gt 0 ]; then
        echo ""
        echo -e "  ${RED}FATAL: $HOOK_ERRORS integration hook(s) broken after pull!${NC}"
        echo "  Run the upstream-sync workflow to repair them."
        exit 1
    fi
else
    echo "  verify-hooks.ps1 not found — skipping hook check."
fi

# ── Phase 4: Build ───────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[5/5] Building...${NC}"

# --- Server dependencies ---
echo "  Installing server dependencies..."
(cd server && npm install --no-audit --no-fund 2>/dev/null) || echo "  WARNING: Server npm install had issues."
(cd server && npm rebuild better-sqlite3 2>/dev/null) || true

# --- UI dependencies ---
echo "  Installing UI dependencies..."
(cd ui && npm install --no-audit --no-fund 2>/dev/null) || echo "  WARNING: UI npm install had issues."

# --- Engine build ---
if [ "$SKIP_ENGINE" = "1" ]; then
    echo "  Skipping engine build (--skip-engine flag)."
else
    # Clean build requested?
    if [ "$CLEAN" = "1" ]; then
        echo ""
        echo -e "  ${YELLOW}WARNING: --clean flag set. Full engine rebuild.${NC}"
        read -p "  Continue with clean rebuild? [y/N] " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "  Clean rebuild skipped."
        else
            echo "  Cleaning engine build directory..."
            rm -rf engine/build
        fi
    fi

    echo "  Building engine..."

    # Auto-detect GPU backend
    CMAKE_EXTRA=""
    DETECTED=""

    if command -v nvcc &>/dev/null; then
        CMAKE_EXTRA="$CMAKE_EXTRA -DGGML_CUDA=ON"
        DETECTED="CUDA"
        echo "  CUDA toolchain: $(nvcc --version 2>/dev/null | grep release || echo 'detected')"
    fi

    if [ -n "$VULKAN_SDK" ]; then
        CMAKE_EXTRA="$CMAKE_EXTRA -DGGML_VULKAN=ON"
        if [ -n "$DETECTED" ]; then
            DETECTED="$DETECTED + Vulkan"
        else
            DETECTED="Vulkan"
        fi
    fi

    if [ -z "$DETECTED" ]; then
        DETECTED="CPU-only"
        echo "  No GPU SDK detected — building CPU-only backend."
    else
        echo "  Detected backends: $DETECTED"
    fi

    # Build engine
    mkdir -p engine/build
    cd engine/build

    if [ ! -f "CMakeCache.txt" ] || [ "$CLEAN" = "1" ]; then
        cmake .. $CMAKE_EXTRA -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON
    fi
    cmake --build . --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    cd "$SCRIPT_DIR"

    if [ $? -ne 0 ]; then
        echo ""
        echo -e "  ${RED}ERROR: Engine build failed!${NC}"
        echo "  Common fixes:"
        echo "    - Install build-essential (gcc/g++)"
        echo "    - Ensure CUDA Toolkit is in PATH (if using CUDA)"
        echo "    - Try: ./update.sh --clean"
        exit 1
    fi
    echo "  Engine build complete."
fi

# --- UI build ---
echo "  Building UI..."
(cd ui && npx vite build) || echo -e "  ${YELLOW}WARNING: UI build had issues.${NC}"
echo "  UI build complete."

# ── Phase 5: Validation & Report ─────────────────────────────────────
echo ""
echo "======================================================="

# Verify engine binary exists
ENGINE_OK=0
if [ -f "engine/build/Release/ace-server" ] || [ -f "engine/build/ace-server" ]; then
    ENGINE_OK=1
fi

UI_OK=0
if [ -f "ui/dist/index.html" ]; then
    UI_OK=1
fi

if [ "$ENGINE_OK" = "1" ]; then
    echo -e "  ${GREEN}[OK]${NC} Engine binary found"
elif [ "$SKIP_ENGINE" = "1" ]; then
    echo "  [--] Engine build skipped"
else
    echo -e "  ${RED}[!!]${NC} Engine binary NOT found — build may have failed"
fi

if [ "$UI_OK" = "1" ]; then
    echo -e "  ${GREEN}[OK]${NC} UI dist/ built"
else
    echo -e "  ${RED}[!!]${NC} UI dist/ not found — build may have failed"
fi

echo ""
if [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    echo "  Updated: ${OLD_HEAD:0:8} -> ${NEW_HEAD:0:8}"
else
    echo "  No new commits (rebuild only)."
fi
echo ""
echo "  Run ./launch.sh to start the application."
echo "======================================================="
echo ""
