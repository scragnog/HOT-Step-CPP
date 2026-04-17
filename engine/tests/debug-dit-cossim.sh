#!/bin/bash

for backend in CUDA0 Vulkan0; do
    for quant in BF16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
        GGML_BACKEND=$backend ./debug-dit-cossim.py --mode all --quant $quant \
            2>&1 | tee ${backend}-${quant}.log
    done
done
