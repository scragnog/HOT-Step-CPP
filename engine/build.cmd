@echo off
REM HOT-Step engine build (CUDA, native arch only)
REM Compiles ONLY for the local GPU — fast dev builds.
REM
REM Automatically finds Visual Studio / Build Tools via vswhere.

REM --- Find vcvars64.bat dynamically ---
REM vswhere ships with VS 2017+ and VS BuildTools.
REM
REM IMPORTANT: %ProgramFiles(x86)% contains parentheses which break
REM batch for-loop parsing. We write the vswhere output to a temp file
REM and read from that instead.

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

REM Skip vcvars if already sourced (prevents PATH overflow on repeated runs)
if defined VSCMD_VER (
    echo Using cached VS environment ^(VSCMD_VER=%VSCMD_VER%^)
) else (
    echo Using: %VCVARS%
    call "%VCVARS%"
)

REM ── cuDNN 9 for CUDA stem separation ───────────────────────────────
:cudnn
REM The GPU build needs cudnn64_9.dll. We get it from
REM the nvidia-cudnn-cu12 pip package (no NVIDIA login required).
REM Only the runtime DLLs are needed — copied next to the exe.

set "CUDNN_MARKER=%~dp0build\Release\cudnn64_9.dll"

if exist "%CUDNN_MARKER%" (
    echo [cuDNN] Found cudnn64_9.dll
    goto :build
)

echo.
echo [cuDNN] cudnn64_9.dll not found. Installing via pip...
echo [cuDNN] (one-time download for CUDA-accelerated SuperSep)
echo.

python -m pip install --quiet nvidia-cudnn-cu12 2>nul
if errorlevel 1 (
    echo [cuDNN] WARNING: pip install failed. GPU stem separation may be unavailable.
    echo [cuDNN]          To fix: pip install nvidia-cudnn-cu12
    goto :build
)

REM Find the installed DLLs and copy them to build/Release
for /f "tokens=*" %%d in ('python -c "import nvidia.cudnn; import os; print(os.path.join(nvidia.cudnn.__path__[0], 'bin'))" 2^>nul') do (
    if exist "%%d\cudnn64_9.dll" (
        echo [cuDNN] Copying DLLs from %%d
        mkdir "%~dp0build\Release" 2>nul
        copy /y "%%d\cudnn*.dll" "%~dp0build\Release\" >nul 2>nul
        echo [cuDNN] Done
    ) else (
        echo [cuDNN] WARNING: Could not find cudnn64_9.dll in pip package
        echo [cuDNN]          path checked: %%d
    )
)

:build
cd /d "%~dp0"
mkdir build 2>nul
cd build

REM Only run cmake configure if not yet configured (avoids invalidating incremental builds)
if not exist "CMakeCache.txt" (
    REM HOT_STEP_CMAKE_FLAGS can be set by update.bat for auto-detected backends.
    REM When unset, defaults to CUDA-only (native dev build).
    if defined HOT_STEP_CMAKE_FLAGS (
        cmake .. %HOT_STEP_CMAKE_FLAGS% -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON
    ) else (
        cmake .. -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120a" -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON
    )
)

REM Gate the compile on the integration hooks and the ggml patch stack being
REM intact. CMake's own "already applied" test (git apply --reverse --check)
REM CANNOT be trusted: patches that touch the same lines -- flash-attn-train
REM and zz-yue2-convrot8 both add to one enum in ggml.h -- stop reversing in
REM isolation once both are applied, so CMake warns on a perfectly healthy
REM tree and stays silent in some broken ones. verify-hooks.ps1 greps for the
REM symbols themselves, which is the only answer that holds.
REM Worth the ten seconds: a `git reset --hard` (submodule.recurse=true resets
REM engine/ggml too) silently unpatches ggml, and without this gate you find
REM out twenty minutes into a CUDA compile, via hundreds of errors about an
REM undefined GGML_OP_CONVROT8 that look like a broken engine change.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify-hooks.ps1"
if errorlevel 1 (
    echo.
    echo [build] ABORTED: engine hooks or ggml patches are missing ^(see above^).
    echo [build] If you ran `git reset --hard`, that reset engine/ggml too. Recover with:
    echo [build]   del engine\ggml\src\ggml-cuda\convrot8.* engine\ggml\src\ggml-cuda\fattn-train.*
    echo [build]   git apply --ignore-whitespace engine\patches\flash-attn-train.patch
    echo [build]   git apply --ignore-whitespace engine\patches\zz-yue2-convrot8.patch
    cd ..
    exit /b 1
)

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
set BUILD_RC=%ERRORLEVEL%

cd ..
exit /b %BUILD_RC%
