#!/bin/bash

set -eu

../build/ace-lm \
    --request partial-sft.json \
    --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf

../build/ace-synth \
    --request partial-sft0.json \
    --embedding ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
