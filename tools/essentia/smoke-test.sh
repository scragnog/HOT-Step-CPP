#!/usr/bin/env bash
# tools/essentia/smoke-test.sh <extractor-binary>
#
# Runs the extractor on a synthetic 6-second A-minor chord and checks the two
# fields the app actually reads (server/src/services/training/essentiaClient.ts
# and routes/analyze.ts): rhythm.bpm and tonal.key_edma. A binary that starts
# but crashes before writing JSON — which is what the 2015 static Linux build
# does — fails here rather than in a user's label job, where the client treats
# a missing output file as "no result" and says nothing.
set -euo pipefail
BIN="$1"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

python3 - "$T/chord.wav" <<'PY'
import math, struct, sys, wave
sr, secs = 44100, 6
freqs = (220.0, 261.63, 329.63)            # A3 C4 E4 — A minor
w = wave.open(sys.argv[1], "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
frames = bytearray()
for i in range(sr * secs):
    t = i / sr
    env = 1.0 if (t % 0.5) < 0.25 else 0.35     # 120 bpm pulse
    v = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs) * env * 0.6
    s = int(max(-1.0, min(1.0, v)) * 32767)
    frames += struct.pack("<hh", s, s)
w.writeframes(bytes(frames)); w.close()
PY

"$BIN" "$T/chord.wav" "$T/out.json" >"$T/log" 2>&1 || true
if [ ! -s "$T/out.json" ]; then
  echo "smoke test FAILED: extractor wrote no JSON" >&2
  tail -20 "$T/log" >&2
  exit 1
fi
python3 - "$T/out.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
bpm = d.get("rhythm", {}).get("bpm")
key = d.get("tonal", {}).get("key_edma", {})
ver = d.get("metadata", {}).get("version", {})
print(f"smoke test: bpm={bpm} key={key.get('key')} {key.get('scale')} essentia={ver.get('essentia')}")
ok = isinstance(bpm, (int, float)) and bpm > 0 and isinstance(key.get("key"), str) and key.get("key")
sys.exit(0 if ok else 1)
PY
