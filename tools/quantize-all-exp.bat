@echo off
setlocal

set Q=D:\Ace-Step-Latest\acestepcpp\acestep.cpp\build\Release\quantize-exp.exe
set M=D:\Ace-Step-Latest\acestepcpp\acestep.cpp\models
set SRC=%M%\acestep-v15-merge-base-turbo-xl-ta-0.5-BF16.gguf
set PREFIX=%M%\acestep-v15-merge-base-turbo-xl-ta-0.5

echo ============================================================
echo  Experimental Quantization Batch - XL Merge Model
echo  Source: %SRC%
echo ============================================================
echo.

REM Skip IQ4_NL (already done) and IQ4_XS (running now)

for %%T in (NVFP4 MXFP4 IQ3_S IQ3_XXS IQ2_S IQ2_XS IQ2_XXS IQ1_M IQ1_S TQ2_0 TQ1_0 Q1_0 Q3_K_S Q3_K_M Q2_K) do (
    set OUT=%PREFIX%-%%T.gguf
    if exist "!OUT!" (
        echo [SKIP] %%T already exists
    ) else (
        echo.
        echo ============================================================
        echo  Quantizing: %%T
        echo ============================================================
        "%Q%" "%SRC%" "%PREFIX%-%%T.gguf" %%T
        if errorlevel 1 (
            echo [FAIL] %%T failed with exit code %errorlevel%
        ) else (
            echo [DONE] %%T complete
        )
    )
)

echo.
echo ============================================================
echo  All quantizations complete!
echo ============================================================

REM Show file sizes
echo.
echo File sizes:
for %%T in (IQ4_NL IQ4_XS NVFP4 MXFP4 IQ3_S IQ3_XXS IQ2_S IQ2_XS IQ2_XXS IQ1_M IQ1_S TQ2_0 TQ1_0 Q1_0 Q3_K_S Q3_K_M Q2_K Q4_K_M) do (
    if exist "%PREFIX%-%%T.gguf" (
        for %%F in ("%PREFIX%-%%T.gguf") do echo   %%T: %%~zF bytes
    )
)
