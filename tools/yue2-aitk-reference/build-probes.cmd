@echo off
setlocal
REM Run from a Visual Studio x64 Developer Command Prompt at the repo root.
REM Standalone tools only: this does not rebuild or restart the HOT-Step engine.
if not exist _experiments\aitk-port\bin mkdir _experiments\aitk-port\bin
cl /nologo /std:c++17 /EHsc /W4 /fp:strict /O2 tools\yue2-aitk-reference\convrot_probe.cpp /Fe:_experiments\aitk-port\bin\convrot_probe.exe /Fo:_experiments\aitk-port\bin\convrot_probe.obj
if errorlevel 1 exit /b 1
if "%CUDA_PATH%"=="" (
    echo CUDA_PATH is required for the CUDA probe.
    exit /b 1
)
if "%AITK_CUDA_ARCH%"=="" set AITK_CUDA_ARCH=sm_120
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O2 -arch=%AITK_CUDA_ARCH% --fmad=false -Xcompiler /EHsc tools\yue2-aitk-reference\convrot_cuda.cu -lcublas -o _experiments\aitk-port\bin\convrot_cuda.exe
exit /b %errorlevel%
