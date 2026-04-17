#!/bin/bash
# Roundtrip: audio -> understand -> SFT DiT -> MP3
#
# Usage: ./ace-understand.sh input.wav (or input.mp3)
#
# understand:
# input -> ace-understand.json (audio codes + metadata)
#
# ace-synth:
# ace-understand.json -> ace-understand0.mp3

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.wav|input.mp3>"
    exit 1
fi

input="$1"

../build/ace-understand \
    --src-audio "$input" \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
    -o ace-understand.json

sed -i \
    's/"audio_cover_strength": *[0-9.]*/"audio_cover_strength": 0.04/' \
    ace-understand.json

../build/ace-synth \
    --src-audio "$input" \
    --request ace-understand.json \
    --embedding ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
