@echo off
REM HOT-Step engine build (all backends: CPU variants + CUDA + Vulkan)
REM Uses vswhere to find any Visual Studio edition automatically.
REM Automatically downloads ONNX Runtime GPU SDK for SuperSep support.

REM Skip vcvars if already sourced (prevents PATH overflow on repeated runs)
if defined VSCMD_VER goto :deps

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Is Visual Studio or Build Tools installed?
    exit /b 1
)

set "VCVARS_TMP=%TEMP%\vcvars_path.txt"
"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" > "%VCVARS_TMP%" 2>nul

set "VCVARS="
for /f "usebackq tokens=*" %%i in ("%VCVARS_TMP%") do set "VCVARS=%%i"
del "%VCVARS_TMP%" 2>nul

if not defined VCVARS (
    echo ERROR: Could not find vcvars64.bat via vswhere.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)

echo Using: %VCVARS%
call "%VCVARS%"

:deps
REM ── ONNX Runtime GPU SDK (for SuperSep stem separation) ────────────
REM Auto-downloads from Microsoft's GitHub Releases if not present.
REM Users can skip this by setting ORT_ROOT or ONNXRUNTIME_ROOT env var.

set "ORT_VERSION=1.25.1"
set "ORT_DIR=%~dp0deps\onnxruntime"
set "ORT_MARKER=%ORT_DIR%\include\onnxruntime_cxx_api.h"

if defined ONNXRUNTIME_ROOT (
    echo [ORT] Using ONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT%
    goto :build
)

if exist "%ORT_MARKER%" (
    echo [ORT] Found at %ORT_DIR%
    goto :build
)

echo.
echo ══════════════════════════════════════════════════════════════════
echo  ONNX Runtime GPU SDK not found. Downloading v%ORT_VERSION%...
echo  This is needed for SuperSep stem separation (one-time download).
echo ══════════════════════════════════════════════════════════════════
echo.

set "ORT_ZIP=%TEMP%\onnxruntime-win-x64-gpu-%ORT_VERSION%.zip"
set "ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ORT_VERSION%/onnxruntime-win-x64-gpu-%ORT_VERSION%.zip"

echo [ORT] Downloading from %ORT_URL%
curl -L -o "%ORT_ZIP%" "%ORT_URL%"
if errorlevel 1 (
    echo [ORT] WARNING: Download failed. Building without SuperSep support.
    echo [ORT] You can manually download and extract to: %ORT_DIR%
    goto :build
)

echo [ORT] Extracting...
mkdir "%~dp0deps" 2>nul
powershell -NoProfile -Command "Expand-Archive -Path '%ORT_ZIP%' -DestinationPath '%~dp0deps' -Force"
if errorlevel 1 (
    echo [ORT] WARNING: Extraction failed. Building without SuperSep support.
    goto :build
)

REM Rename extracted folder (it has version in the name)
if exist "%~dp0deps\onnxruntime-win-x64-gpu-%ORT_VERSION%" (
    ren "%~dp0deps\onnxruntime-win-x64-gpu-%ORT_VERSION%" onnxruntime
)

REM Clean up zip
del "%ORT_ZIP%" 2>nul

if exist "%ORT_MARKER%" (
    echo [ORT] Successfully installed to %ORT_DIR%
) else (
    echo [ORT] WARNING: Installation may have failed. Check %ORT_DIR%
)

:build
cd /d "%~dp0"
rem rd /s /q build 2>nul
mkdir build 2>nul
cd build

REM Build with ORT auto-detection from deps directory
cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON %RELEASE_CMAKE_EXTRA%
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
