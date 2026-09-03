#!/usr/bin/env bash
# tools/essentia/build-linux.sh — build essentia_streaming_extractor_music for
# Linux x86_64 and bundle its shared-library dependencies beside it.
#
# WHY THIS EXISTS. The repo's Essentia/ folder holds a Windows .exe and nothing
# else, so every Linux and macOS release shipped with BPM/key detection dead
# (#104), and v1.3's Linux archives literally contained the Windows binary
# (#144). The official static Linux extractor (v2.1_beta2, 2015) is not a fix:
# it segfaults on a current kernel before writing any output.
#
# WHAT IT BUILDS. Essentia at 36ec3d92 (master, 2025-07-24): the last commit
# before the FFmpeg 5.x port (1153d559, 2025-07-29, AVChannelLayout), and well
# after libavresample was dropped for libswresample (ab42160a, 2021-12-27).
# That window is the FFmpeg 4 API. The 2020 commit the Windows .exe was built
# from (673b6a14) is NOT usable: its AudioLoader needs libavresample, which no
# current distro packages, and waf then silently builds an extractor with no
# AudioLoader that fails on every file. The JSON the app reads (rhythm.bpm,
# tonal.key_edma) is the same across these versions.
#
# FFMPEG IS BUILT HERE, MINIMAL. Linking the distro's libavcodec pulls its whole
# codec tree into the bundle — 123 libraries, 157 MB, libx265 and librsvg for
# a program that decodes audio. FFmpeg 4.4.6 is instead configured with only
# the file protocol, the demuxers and the audio decoders the app can hand it
# (essentiaClient.ts transcodes anything else to WAV first), as shared libs
# with no external dependencies. libchromaprint is left out on purpose: the
# music extractor never calls it, and the distro build of it links libavcodec.
#
# The result is NOT static. Every shared library ldd resolves is copied into
# lib/ and the ELF rpath rewritten to $ORIGIN/lib, so the folder runs on any
# x86_64 Linux with glibc >= the build host's (2.35 on Ubuntu 22.04, the
# release runner). Only glibc itself is left to the host.
#
# Usage: tools/essentia/build-linux.sh [OUT_DIR]     default ./Essentia-linux-x64
#   ESSENTIA_COMMIT   override the pinned Essentia commit
#   FFMPEG_VERSION    override the FFmpeg release (must be a 4.x)
#   WORK              scratch dir (default /tmp/essentia-build), reused if present
#   JOBS              parallel compile jobs (default nproc)
#   SKIP_APT=1        do not apt-get anything (deps already installed)
set -euo pipefail

ESSENTIA_COMMIT="${ESSENTIA_COMMIT:-36ec3d929402d491cbbb1120c54631e8c205b0ff}"
FFMPEG_VERSION="${FFMPEG_VERSION:-4.4.6}"
OUT="$(realpath -m "${1:-$PWD/Essentia-linux-x64}")"
WORK="${WORK:-/tmp/essentia-build}"
JOBS="${JOBS:-$(nproc)}"
BIN=essentia_streaming_extractor_music
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFPREFIX="$WORK/ffmpeg-$FFMPEG_VERSION-prefix"

if [ "${SKIP_APT:-0}" != "1" ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq --no-install-recommends \
    build-essential git ca-certificates curl xz-utils \
    python3 python3-distutils pkg-config patchelf \
    libeigen3-dev libyaml-dev libfftw3-dev libsamplerate0-dev libtag1-dev
fi
mkdir -p "$WORK"

# ── FFmpeg, audio decoding only ──────────────────────────────────────────────
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
# Added to the defaults, not passed as --pkg-config-path, which makes waf
# ignore the system .pc files the other libraries live in.
export PKG_CONFIG_PATH="$FFPREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# ── Essentia ─────────────────────────────────────────────────────────────────
if [ ! -d "$WORK/essentia/.git" ]; then
  git clone --quiet https://github.com/MTG/essentia.git "$WORK/essentia"
fi
cd "$WORK/essentia"
git fetch --quiet origin "$ESSENTIA_COMMIT" 2>/dev/null || git fetch --quiet origin
git checkout --quiet "$ESSENTIA_COMMIT"
echo "essentia $(git describe --tags --always)"

# --build-static links libessentia into the example, so the only shared
# libraries left to bundle are the third-party ones. Only the one example is
# built: --with-examples would compile ~40 and most need Gaia or TensorFlow.
python3 waf distclean > /dev/null 2>&1 || true
python3 waf configure --mode=release --build-static --with-example=streaming_extractor_music
python3 waf -j"$JOBS"

# The example is named essentia_streaming_extractor_music at older commits and
# streaming_extractor_music on newer ones; accept either.
SRC=""
for cand in essentia_streaming_extractor_music streaming_extractor_music; do
  [ -x "$WORK/essentia/build/src/examples/$cand" ] && SRC="$WORK/essentia/build/src/examples/$cand" && break
done
[ -n "$SRC" ] || { echo "build produced no extractor under $WORK/essentia/build/src/examples" >&2; ls "$WORK/essentia/build/src/examples" >&2; exit 1; }

# ── Bundle ───────────────────────────────────────────────────────────────────
rm -rf "$OUT"
mkdir -p "$OUT/lib"
cp "$SRC" "$OUT/$BIN"

# Everything ldd resolves is transitive already. glibc's own libraries stay
# with the host: bundling a libc that the host's dynamic loader did not come
# from does not work, and it is the one dependency every distro has.
LD_LIBRARY_PATH="$FFPREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$OUT/$BIN" \
  | awk '/=> \//{print $3}' | sort -u | while read -r lib; do
  case "$(basename "$lib")" in
    libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|ld-linux*|libresolv.so.*|libutil.so.*) continue ;;
  esac
  cp -L "$lib" "$OUT/lib/"
done
patchelf --set-rpath '$ORIGIN/lib' "$OUT/$BIN"
for l in "$OUT"/lib/*; do
  patchelf --set-rpath '$ORIGIN' "$l" 2>/dev/null || true
done
if ldd "$OUT/$BIN" | grep -q "not found"; then
  echo "unresolved libraries after bundling:" >&2; ldd "$OUT/$BIN" | grep "not found" >&2; exit 1
fi
cp "$WORK/essentia/COPYING.txt" "$OUT/COPYING.txt" 2>/dev/null || true
cat > "$OUT/README.txt" <<EOF
essentia_streaming_extractor_music for Linux x86_64
Built from https://github.com/MTG/essentia at $ESSENTIA_COMMIT against a
minimal FFmpeg $FFMPEG_VERSION (audio decoding only) by
tools/essentia/build-linux.sh. Third-party libraries are bundled in lib/
(rpath \$ORIGIN/lib); glibc >= 2.35 is required from the host.
Essentia is AGPL-3.0 (COPYING.txt); FFmpeg is LGPL-2.1+.
EOF

# Smoke test: a synthetic chord, then the two fields the app reads.
bash "$HERE/smoke-test.sh" "$OUT/$BIN"

echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1), $(ls "$OUT/lib" | wc -l) libraries)"
