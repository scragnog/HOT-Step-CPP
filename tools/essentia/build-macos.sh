#!/usr/bin/env bash
# tools/essentia/build-macos.sh — build essentia_streaming_extractor_music for
# macOS (Apple Silicon) and bundle its dylibs beside it via @loader_path.
#
# This is @beaudamion's recipe from issue #122, the first working Essentia on
# Apple Silicon this project has had, adapted for CI with credit. The
# interesting decisions are theirs:
#
#   * FFmpeg 6, not 7 and not 4. FFmpeg 7 removed AVCodec.sample_fmts, which
#     Essentia still reads, and FFmpeg < 5.1 lacks AVCodecContext.ch_layout,
#     which Essentia master now uses. 6 is the release where both exist.
#   * dylibs are copied next to the binary and rewritten to @loader_path, so a
#     user needs no Homebrew.
#
# Two departures. FFmpeg is built here, minimal, rather than taken from
# Homebrew: brew's ffmpeg@6 links the whole image/video codec tree, which
# bundled as 70 dylibs (libwebp, libjxl, libx265 ...) cross-referenced via
# @rpath, for a program that decodes audio. The same audio-only configure as
# the Linux recipe gives four small dylibs with no dependencies. And
# libchromaprint is left out: the music extractor never calls it, and brew's
# links ffmpeg.
#
# Essentia master is used here (pinned) rather than the commit the Linux build
# uses, because master is what the recipe was proven on and the Linux commit
# predates the ch_layout code FFmpeg 6 wants. The JSON the app reads
# (rhythm.bpm, tonal.key_edma) is the same across both.
#
# Usage: tools/essentia/build-macos.sh [OUT_DIR]     default ./Essentia-macos-arm64
#   ESSENTIA_COMMIT   override the pinned commit
#   FFMPEG_VERSION    override the FFmpeg release (must be a 6.x)
#   WORK              scratch dir (default /tmp/essentia-build)
#   SKIP_BREW=1       do not brew install anything
set -euo pipefail

ESSENTIA_COMMIT="${ESSENTIA_COMMIT:-66a890f285d0e1988155c12d17a2068e406cdd90}"   # master, 2026-08-27
FFMPEG_VERSION="${FFMPEG_VERSION:-6.1.2}"
OUT="${1:-$PWD/Essentia-macos-arm64}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
WORK="${WORK:-/tmp/essentia-build}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
BIN=essentia_streaming_extractor_music
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFPREFIX="$WORK/ffmpeg-$FFMPEG_VERSION-prefix"

if [ "${SKIP_BREW:-0}" != "1" ]; then
  brew install eigen fftw libyaml taglib libsamplerate pkg-config
fi
BREW="$(brew --prefix)"
mkdir -p "$WORK"

# ── FFmpeg, audio decoding only (same configure as build-linux.sh) ───────────
if [ ! -f "$FFPREFIX/lib/pkgconfig/libavcodec.pc" ]; then
  cd "$WORK"
  if [ ! -d "ffmpeg-$FFMPEG_VERSION" ]; then
    curl -sSL -o "ffmpeg-$FFMPEG_VERSION.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
    tar -xf "ffmpeg-$FFMPEG_VERSION.tar.xz"
  fi
  cd "ffmpeg-$FFMPEG_VERSION"
  ./configure --prefix="$FFPREFIX" \
    --enable-shared --disable-static --enable-pic \
    --disable-everything --disable-programs --disable-doc --disable-network \
    --disable-autodetect --disable-x86asm \
    --disable-avdevice --disable-avfilter --disable-postproc --disable-swscale \
    --enable-avcodec --enable-avformat --enable-swresample \
    --enable-protocol=file \
    --enable-demuxer=wav,mp3,flac,aiff,ogg,mov,matroska,aac,pcm_s16le \
    --enable-decoder=pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_f32le,pcm_f32be,pcm_f64le,pcm_u8,pcm_alaw,pcm_mulaw,flac,mp3,mp3float,vorbis,aac,alac,opus \
    --enable-parser=flac,mpegaudio,vorbis,aac,opus \
    > configure.log 2>&1 || { tail -30 configure.log >&2; exit 1; }
  make -j"$JOBS" > make.log 2>&1 || { tail -30 make.log >&2; exit 1; }
  make install > install.log 2>&1
  echo "ffmpeg $FFMPEG_VERSION (minimal) installed to $FFPREFIX"
fi
export PKG_CONFIG_PATH="$FFPREFIX/lib/pkgconfig:$BREW/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CXXFLAGS="-I$FFPREFIX/include -I$BREW/include ${CXXFLAGS:-}"
export LDFLAGS="-L$FFPREFIX/lib -L$BREW/lib ${LDFLAGS:-}"

# ── Essentia ─────────────────────────────────────────────────────────────────
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

# The example is named essentia_streaming_extractor_music at older commits and
# streaming_extractor_music on newer ones; accept either.
SRC=""
for cand in essentia_streaming_extractor_music streaming_extractor_music; do
  [ -x "$WORK/essentia/build/src/examples/$cand" ] && SRC="$WORK/essentia/build/src/examples/$cand" && break
done
[ -n "$SRC" ] || { echo "build produced no extractor under $WORK/essentia/build/src/examples" >&2; ls "$WORK/essentia/build/src/examples" >&2; exit 1; }

# ── Bundle ───────────────────────────────────────────────────────────────────
rm -rf "${OUT:?}"/*
cp "$SRC" "$OUT/$BIN"
python3 "$HERE/bundle-dylibs.py" "$OUT" "$BIN" "$FFPREFIX/lib" "$BREW/lib"
cp "$WORK/essentia/COPYING.txt" "$OUT/COPYING.txt" 2>/dev/null || true
cat > "$OUT/README.txt" <<EOF
essentia_streaming_extractor_music for macOS arm64
Built from https://github.com/MTG/essentia at $ESSENTIA_COMMIT against a
minimal FFmpeg $FFMPEG_VERSION (audio decoding only) by
tools/essentia/build-macos.sh, following the recipe @beaudamion posted in
https://github.com/scragnog/HOT-Step-CPP/issues/122. The dylibs it needs are
bundled beside the binary (@loader_path); nothing else is required.
Essentia is AGPL-3.0 (COPYING.txt); FFmpeg is LGPL-2.1+.
EOF

bash "$HERE/smoke-test.sh" "$OUT/$BIN"
echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1))"
