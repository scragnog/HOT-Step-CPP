@echo off

set PATH=%~dp0..\build\Release;%PATH%

ace-lm.exe ^
    --request simple.json ^
    --lm ..\models\acestep-5Hz-lm-4B-Q6_K.gguf

ace-synth.exe ^
    --request simple0.json ^
    --embedding ..\models\Qwen3-Embedding-0.6B-Q8_0.gguf ^
    --dit ..\models\acestep-v15-turbo-Q6_K.gguf ^
    --vae ..\models\vae-BF16.gguf

pause
