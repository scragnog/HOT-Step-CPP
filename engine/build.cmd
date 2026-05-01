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

cd /d "%~dp0"
mkdir build 2>nul
cd build

REM Only run cmake configure if not yet configured (avoids invalidating incremental builds)
if not exist "CMakeCache.txt" (
    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
)
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
