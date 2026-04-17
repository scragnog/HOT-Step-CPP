#!/bin/bash

cp ../examples/simple.json .
cp ../examples/partial.json .
cp ../examples/full.json .

../build/ace-lm --request simple.json \
  --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
  --dump-logits logits.bin --dump-tokens tokens.csv

python3 debug-lm-logits.py ../checkpoints/acestep-5Hz-lm-4B logits.bin tokens.csv

../build/ace-lm --request partial.json \
  --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
  --dump-logits logits.bin --dump-tokens tokens.csv

python3 debug-lm-logits.py ../checkpoints/acestep-5Hz-lm-4B logits.bin tokens.csv

../build/ace-lm --request full.json \
  --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
  --dump-logits logits.bin --dump-tokens tokens.csv

python3 debug-lm-logits.py ../checkpoints/acestep-5Hz-lm-4B logits.bin tokens.csv
