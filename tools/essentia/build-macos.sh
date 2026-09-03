#!/usr/bin/env bash
# tools/essentia/build-macos.sh — build essentia_streaming_extractor_music for
# macOS (Apple Silicon) and bundle its dylibs beside it via @loader_path.
#
# This is @beaudamion's recipe from issue #122, which is the first working
# Essentia on Apple Silicon this project has had. Adapted for CI with credit;
# the interesting decisions are theirs:
#
#   * ffmpeg@6, not ffmpeg. FFmpeg 7 removed AVCodec.sample_fmts, which Essentia
#     still reads, and FFmpeg < 5.1 lacks AVCodecContext.ch_layout, which
#     Essentia master now uses. 6 is the release where both exist.
#   * dylibs are copied next to the binary and rewritten to @loader_path, so a
#     user needs no Homebrew.
#
# Essentia master is used here (pinned) rather than the 2020 commit the Linux
# and Windows builds share, because master is what the recipe was proven on and
# that commit predates the ch_layout code ffmpeg@6 wants. The JSON the app reads
# (rhythm.bpm, tonal.key_edma) is the same across both.
#
# Usage: tools/essentia/build-macos.sh [OUT_DIR]     default ./Essentia-macos-arm64
#   ESSENTIA_COMMIT   override the pinned commit
#   WORK              scratch dir (default /tmp/essentia-build)
#   SKIP_BREW=1       do not brew install anything
set -euo pipefail

ESSENTIA_COMMIT="${ESSENTIA_COMMIT:-66a890f285d0e1988155c12d17a2068e406cdd90}"   # master, 2026-08-27
OUT="${1:-$PWD/Essentia-macos-arm64}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
WORK="${WORK:-/tmp/essentia-build}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
BIN=essentia_streaming_extractor_music
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${SKIP_BREW:-0}" != "1" ]; then
  brew install ffmpeg@6 eigen fftw libyaml taglib chromaprint libsamplerate pkg-config
fi
FF="$(brew --prefix ffmpeg@6)"
BREW="$(brew --prefix)"
export PKG_CONFIG_PATH="$FF/lib/pkgconfig:$BREW/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CXXFLAGS="-I$FF/include -I$BREW/include ${CXXFLAGS:-}"
export LDFLAGS="-L$FF/lib -L$BREW/lib ${LDFLAGS:-}"

mkdir -p "$WORK"
if [ ! -d "$WORK/essentia/.git" ]; then
  git clone --quiet https://github.com/MTG/essentia.git "$WORK/essentia"
fi
cd "$WORK/essentia"
git fetch --quiet origin "$ESSENTIA_COMMIT" 2>/dev/null || git fetch --quiet origin
git checkout --quiet "$ESSENTIA_COMMIT"
echo "essentia $(git describe --tags --always)"

# Essentia's waf imports distutils, which Python 3.12 removed; the macos-15
# runner's python3 is 3.13. A venv with setuptools puts the shim back.
PY="$WORK/venv/bin/python"
if ! "$PY" -c 'import distutils.sysconfig' 2>/dev/null; then
  python3 -m venv "$WORK/venv"
  "$PY" -m pip install -q --upgrade pip setuptools
fi
"$PY" waf distclean > /dev/null 2>&1 || true
"$PY" waf configure --mode=release --build-static --with-example=streaming_extractor_music
"$PY" waf -j"$JOBS"

# The example is named essentia_streaming_extractor_music at the 2020 commit and
# streaming_extractor_music on master; accept either.
SRC=""
for cand in essentia_streaming_extractor_music streaming_extractor_music; do
  [ -x "$WORK/essentia/build/src/examples/$cand" ] && SRC="$WORK/essentia/build/src/examples/$cand" && break
done
[ -n "$SRC" ] || { echo "build produced no extractor under $WORK/essentia/build/src/examples" >&2; ls "$WORK/essentia/build/src/examples" >&2; exit 1; }

rm -rf "${OUT:?}"/*
cp "$SRC" "$OUT/$BIN"
python3 "$HERE/bundle-dylibs.py" "$OUT" "$BIN"
cp "$WORK/essentia/COPYING.txt" "$OUT/COPYING.txt" 2>/dev/null || true
cat > "$OUT/README.txt" <<EOF
essentia_streaming_extractor_music for macOS arm64
Built from https://github.com/MTG/essentia at $ESSENTIA_COMMIT
by tools/essentia/build-macos.sh, following the recipe @beaudamion posted in
https://github.com/scragnog/HOT-Step-CPP/issues/122. Homebrew dylibs are
bundled beside the binary (@loader_path); nothing else is required.
Essentia is AGPL-3.0; see COPYING.txt.
EOF

bash "$HERE/smoke-test.sh" "$OUT/$BIN"
echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1))"
