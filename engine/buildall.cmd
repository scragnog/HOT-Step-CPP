@echo off
REM HOT-Step engine build (all backends: CPU variants + CUDA + Vulkan)
REM Uses vswhere to find any Visual Studio edition automatically.

REM Skip vcvars if already sourced (prevents PATH overflow on repeated runs)
if defined VSCMD_VER goto :build

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

:build
cd /d "%~dp0"
rem rd /s /q build 2>nul
mkdir build 2>nul
cd build

cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
