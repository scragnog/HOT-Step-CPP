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
  --force         Discard changes to tracked files before pulling (with confirmation)
                  Never deletes untracked files: adapters/, models/, data/ are safe.
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

# Save current HEAD (and the ggml submodule pointer) for later
OLD_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
OLD_GGML=$(git rev-parse HEAD:engine/ggml 2>/dev/null || echo "unknown")

# Check for uncommitted changes to TRACKED files only.
#
#   --ignore-submodules=dirty: engine/ggml used to carry HOT-Step's ggml
#   changes as uncommitted edits, so an install updated from before the
#   HOT-ggml fork still has a dirty engine/ggml. Without this flag git reports
#   " m engine/ggml" there and the update refuses to run; the step after the
#   pull restores that tree.
#
#   Untracked files never count. adapters/, models/, data/ and anything else
#   the app writes are not "changes" and this script never deletes them.
if ! git diff --quiet --ignore-submodules=dirty 2>/dev/null || ! git diff --cached --quiet --ignore-submodules=dirty 2>/dev/null; then
    if [ "$FORCE" = "1" ]; then
        echo ""
        echo -e "  ${YELLOW}WARNING: You have uncommitted changes to tracked files.${NC}"
        echo "  --force will DISCARD them. Untracked files (adapters, models, data) are kept."
        echo ""
        echo "  Modified files:"
        git status --short --ignore-submodules=dirty --untracked-files=no
        echo ""
        read -p "  Discard these changes and continue? [y/N] " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "  Aborted by user."
            exit 1
        fi
        echo "  Resetting tracked files..."
        # reset --hard restores tracked files only. Never add "git clean" here:
        # it deletes untracked files, and that once wiped a user's adapters/.
        # submodule.recurse=false: engine/ggml is checked on its own below, and
        # a recursive reset would silently discard anything inside it.
        git -c submodule.recurse=false reset --hard
    else
        echo ""
        echo -e "  ${RED}ERROR: You have uncommitted changes to tracked files:${NC}"
        echo ""
        git status --short --ignore-submodules=dirty --untracked-files=no
        echo ""
        echo "  Options:"
        echo "    1. Commit or stash your changes first"
        echo "    2. Run: ./update.sh --force  (discards changes to tracked files only)"
        echo ""
        exit 1
    fi
else
    echo "  Working tree is clean."
fi

# engine/ggml must be clean before anything moves, except for one exact state:
# an install from before the HOT-ggml fork, whose engine/ggml is the old pin
# c044c6f0 with the engine/patches overlay applied (26 edited files plus eight
# untracked kernel files). That state is accepted only when the base commit is
# c044c6f0 and every one of the 34 files matches the overlay's known content
# (sha256 after stripping CR, so CRLF checkouts match). Then, and only then,
# exactly those files are restored after the pull. Anything else in engine/ggml
# (a local edit, an extra edit inside an overlay file, a partial overlay, a
# stray file, another base) stops the update here, before the pull, with every
# file left in place.
GGML_OVERLAY_BASE="c044c6f03892f9d5e98213b05f8afea1f8b0d3c9"
GGML_OVERLAY_TRACKED="include/ggml-rpc.h include/ggml.h src/ggml-alloc.c src/ggml-backend-meta.cpp src/ggml-backend.cpp src/ggml-cpu/ggml-cpu.c src/ggml-cpu/ggml-cpu.cpp src/ggml-cpu/ops.cpp src/ggml-cpu/ops.h src/ggml-cuda/CMakeLists.txt src/ggml-cuda/cpy.cu src/ggml-cuda/cpy.cuh src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/out-prod.cu src/ggml-cuda/unary.cu src/ggml-cuda/unary.cuh src/ggml-hip/CMakeLists.txt src/ggml-metal/ggml-metal-device.cpp src/ggml-metal/ggml-metal-device.h src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal.metal src/ggml-musa/CMakeLists.txt src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/vulkan-shaders/unary.comp src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp src/ggml.c"
GGML_OVERLAY_NEW="src/ggml-cuda/convrot8.cu src/ggml-cuda/convrot8.cuh src/ggml-cuda/fattn-train.cu src/ggml-cuda/fattn-train.cuh src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dkv.comp src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dq.comp src/ggml-vulkan/vulkan-shaders/fa_train_common.glsl src/ggml-vulkan/vulkan-shaders/fa_train_fwd.comp"
GGML_OVERLAY_SHA256="
54e097e2f54a0a6874ac6da34910f05738e5aa9c780a447f25f85641ff48d499 include/ggml-rpc.h
28603404395a9293da16ee552ddbd25ffd65d7db06c95315847e8dd26c859285 include/ggml.h
0a5175e1ba169c9a7b6ada3d637e9af75fac0ff02c503ace6f0ed77df1d260e9 src/ggml-alloc.c
fea0b35d80f1592041a64d14bb16697aa589879a76aa54153288b02323f2f158 src/ggml-backend-meta.cpp
4958ef48f11be667546c33c54f5028e4b4271b06a0b4adef7a10294023d871e6 src/ggml-backend.cpp
4f87e04e947eadd4bb01ee5dff4cd79a5209cb7f368f19ef135ac765953086a8 src/ggml-cpu/ggml-cpu.c
d3f614f0695bd123007d27b95f6130ac617a09fa89c6acde9220a0cc5cef7ee5 src/ggml-cpu/ggml-cpu.cpp
3e70ed8d52990b067c91dc7b769b5b3fb38b08fb02fca944db120ba29814a332 src/ggml-cpu/ops.cpp
dedb188e9f187fd89216c27c90889140ae63655ee5ca58686754f59dab2d0523 src/ggml-cpu/ops.h
d6f42a9eec46c1e828b10256110e9d21394ca3168ae9da0ae1ec604e99e15da4 src/ggml-cuda/CMakeLists.txt
697be95992b30190250e5edee00cd6612e2d6a1284a9c2da09044cce295cc3af src/ggml-cuda/convrot8.cu
d6db00cd1d973c7de46c2b8ccc83ac273c9a4f30642ec8e62cf6508db8741005 src/ggml-cuda/convrot8.cuh
89d89c749b3667aa541aeefb818436df3a4b282b209803947796915a0743bed8 src/ggml-cuda/cpy.cu
648bd317d9b980de8ade995e3f5b2323c720336de123733e5fdeedc1912ba9af src/ggml-cuda/cpy.cuh
e22cfad8df782c6fa4e42e972a11ae935abfb91f745e98bf5335b19a8f304660 src/ggml-cuda/fattn-train.cu
92ce96329ab084ef4ae8882c75a24661bd2d89175dd3548c542e33cfac432dca src/ggml-cuda/fattn-train.cuh
857a1e030c92e7aa985aac0205796e4b242876caabc3e70ebba07d11c3b2f3c1 src/ggml-cuda/ggml-cuda.cu
fba28be9a58f851ea3e7e4df7305d1ea90589537e57ef1363efc60e98aa457d0 src/ggml-cuda/out-prod.cu
0b4b5ff52b6aad57e4774023cfb8068a5e2551b98266bdb86fc09cb6f9e0cf89 src/ggml-cuda/unary.cu
f77920497798b8d6d6e338d12901f84baedaecdc33412be04a25080130355ead src/ggml-cuda/unary.cuh
32f0bedd7775c8117efd87f139a003b18fa367627ff14e772371858e35d443b4 src/ggml-hip/CMakeLists.txt
1562b93c826e1017872f9dfcb51b4b527a3f73c2a77fbe8a64402656c197fdd1 src/ggml-metal/ggml-metal-device.cpp
6a253e5b44756162b0b9d03c3bbeb7f27209088468b532c9c004fbfec8b5c483 src/ggml-metal/ggml-metal-device.h
795cdcdbbc3ada0b150345d0eaeeda7582c5d624884402dbe740d1f93c610035 src/ggml-metal/ggml-metal-ops.cpp
51b72ed95ca4e79e46d67bd80da4f6e5645f9433f88bd01f5aa7203850629cb8 src/ggml-metal/ggml-metal.metal
2adfa5007f429cdb71e78066d587d1e942f5e6d3daefde3814f859bbed2e82c3 src/ggml-musa/CMakeLists.txt
2c1ff93f645ee28a08acc643203273560cad58a13b13efd191a29e2ab5c3f97d src/ggml-vulkan/ggml-vulkan.cpp
cc659c1c46ee2964cb20dea408951ee12144850f30f9e5a7880db2b767dbca2f src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dkv.comp
576409fdc697cc58dfffc9f2ca44299f258b91f6636cb5ca6d1050516c40aa59 src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dq.comp
d6e66ba8d75bad1fa6a6c85c4e87252c7da4a1a66f46e72d8477248d6f492fc2 src/ggml-vulkan/vulkan-shaders/fa_train_common.glsl
ec25fce777e748c4da65a1e7f5e3139fdb337a928ee3a72aaaed20fb27559074 src/ggml-vulkan/vulkan-shaders/fa_train_fwd.comp
8b018353eef8fc4b34ac559f77a32a1944b1450d2f608c8fb23760de5751f944 src/ggml-vulkan/vulkan-shaders/unary.comp
4432d0084e3a053367e75a76051f4371af1143b589bc351e3229a7b140d8ce45 src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
24484864b88f541e66a62e379249a23f4c8acce9084bb9c3a32d18eee34697c0 src/ggml.c
"
ggml_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        tr -d '\r' < "$1" | sha256sum | cut -c1-64
    else
        tr -d '\r' < "$1" | shasum -a 256 | cut -c1-64
    fi
}
GGML_RESTORE_OVERLAY=0
if [ -e engine/ggml/.git ]; then
    if ! GGML_DIRTY=$(git -C engine/ggml status --porcelain --untracked-files=all); then
        echo -e "  ${RED}ERROR: cannot read engine/ggml status.${NC}"
        exit 1
    fi
    if [ -n "$GGML_DIRTY" ]; then
        GGML_PROBLEMS=""
        # 1. Only the overlay's own paths may differ: edited tracked files or
        #    its eight new files, nothing staged, deleted or elsewhere.
        while IFS= read -r line; do
            code="${line:0:2}"; path="${line:3}"
            case "$code" in
                " M") list="$GGML_OVERLAY_TRACKED" ;;
                "??") list="$GGML_OVERLAY_NEW" ;;
                *)    list="" ;;
            esac
            case " $list " in
                *" $path "*) ;;
                *) GGML_PROBLEMS="$GGML_PROBLEMS    $line"$'\n' ;;
            esac
        done <<< "$GGML_DIRTY"
        # 2. The base must be the old pin. The URL is not checked: after a manual
        #    git pull it is already the fork's, and base + content is the proof.
        if ! GGML_BASE_NOW=$(git -C engine/ggml rev-parse HEAD); then
            echo -e "  ${RED}ERROR: cannot read engine/ggml HEAD.${NC}"
            exit 1
        fi
        if [ "$GGML_BASE_NOW" != "$GGML_OVERLAY_BASE" ]; then
            GGML_PROBLEMS="$GGML_PROBLEMS    base is $GGML_BASE_NOW, not $GGML_OVERLAY_BASE"$'\n'
        fi
        # 3. Every overlay file must be present with exactly the overlay's content.
        while read -r want path; do
            [ -n "$path" ] || continue
            if [ ! -f "engine/ggml/$path" ]; then
                GGML_PROBLEMS="$GGML_PROBLEMS    missing overlay file $path"$'\n'
            elif [ "$(ggml_sha256 "engine/ggml/$path")" != "$want" ]; then
                GGML_PROBLEMS="$GGML_PROBLEMS    $path differs from the known overlay"$'\n'
            fi
        done <<< "$GGML_OVERLAY_SHA256"
        if [ -z "$GGML_PROBLEMS" ]; then
            echo "  engine/ggml carries the exact pre-fork patch overlay; it will be restored after the pull."
            GGML_RESTORE_OVERLAY=1
        else
            echo ""
            echo -e "  ${RED}ERROR: engine/ggml has local changes this updater will not touch:${NC}"
            printf '%s' "$GGML_PROBLEMS"
            echo ""
            echo "  ggml changes belong on the HOT-ggml fork (docs/dev/ggml-fork.md)."
            echo "  Move or commit them, then check with: git -C engine/ggml status"
            echo "  Nothing has been changed."
            exit 1
        fi
    fi
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

# submodule.recurse=false: with recursion on (this repo's own checkouts set
# it), the pull would check out the new engine/ggml pin itself, before the
# overlay restore and URL sync below, and fail on the pre-fork overlay or the
# stale ggml-org URL. engine/ggml is moved explicitly after the pull.
if ! git -c submodule.recurse=false pull --ff-only origin master; then
    echo ""
    echo -e "  ${RED}ERROR: git pull --ff-only failed.${NC}"
    echo "  This usually means your local branch has diverged from origin/master."
    echo "  Options:"
    echo "    1. Run: git rebase origin/master"
    echo "    2. Run: ./update.sh --force  (discards local changes)"
    echo ""
    exit 1
fi

# engine/ggml now follows HOT-ggml (docs/dev/ggml-fork.md), which carries
# HOT-Step's ggml changes as commits. A pre-fork install has them as the old
# overlay (detected in pre-flight); restore exactly those files so the checkout
# can take the fork's tracked versions. Nothing else in engine/ggml is touched.
if [ "$GGML_RESTORE_OVERLAY" = "1" ]; then
    echo "  Restoring the pre-fork overlay files in engine/ggml..."
    # shellcheck disable=SC2086
    if ! git -C engine/ggml checkout -- $GGML_OVERLAY_TRACKED 2>/dev/null; then
        # Not every listed file is modified; restore the ones that are.
        for f in $GGML_OVERLAY_TRACKED; do
            if ! git -C engine/ggml diff --quiet -- "$f"; then
                git -C engine/ggml checkout -- "$f" || { echo -e "  ${RED}ERROR: could not restore engine/ggml/$f${NC}"; exit 1; }
            fi
        done
    fi
    for f in $GGML_OVERLAY_NEW; do
        rm -f "engine/ggml/$f" || { echo -e "  ${RED}ERROR: could not remove engine/ggml/$f${NC}"; exit 1; }
    done
fi

# The submodule URL is cached in .git/config at first init, so an install from
# before the fork would keep fetching stock ggml-org and never find the pinned
# commit. sync copies the URL from .gitmodules first. Both must succeed.
if ! git submodule sync --recursive; then
    echo -e "  ${RED}ERROR: git submodule sync failed. Nothing was built.${NC}"
    exit 1
fi
if ! git submodule update --init --recursive; then
    echo -e "  ${RED}ERROR: git submodule update failed. Nothing was built.${NC}"
    echo "  Check your connection, then rerun ./update.sh"
    exit 1
fi

# Build only what this commit pins: engine/ggml at the gitlink, with no local
# changes or untracked files (ggml-cuda globs *.cu, so a stray file is compiled).
GGML_PIN=$(git rev-parse HEAD:engine/ggml) || { echo -e "  ${RED}ERROR: cannot read the engine/ggml pin.${NC}"; exit 1; }
GGML_NOW=$(git -C engine/ggml rev-parse HEAD) || { echo -e "  ${RED}ERROR: cannot read engine/ggml HEAD.${NC}"; exit 1; }
if [ "$GGML_PIN" != "$GGML_NOW" ]; then
    echo -e "  ${RED}ERROR: engine/ggml is at $GGML_NOW, but this version pins $GGML_PIN.${NC}"
    exit 1
fi
if ! GGML_DIRTY=$(git -C engine/ggml status --porcelain --untracked-files=all) || [ -n "$GGML_DIRTY" ]; then
    echo -e "  ${RED}ERROR: engine/ggml is not clean after the update:${NC}"
    printf '%s\n' "$GGML_DIRTY" | sed 's/^/    /'
    exit 1
fi
echo "  engine/ggml is at the pinned commit ${GGML_PIN:0:8}, clean."

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
