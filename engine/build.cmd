@echo off
REM HOT-Step engine build (CUDA, native arch only)
REM Compiles ONLY for the local GPU — fast dev builds.
REM
REM Automatically finds Visual Studio / Build Tools via vswhere.

REM --- Find vcvars64.bat dynamically ---
REM vswhere ships with VS 2017+ and VS BuildTools
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Is Visual Studio or Build Tools installed?
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat"`) do (
    set "VCVARS=%%i"
)

if not defined VCVARS (
    echo ERROR: Could not find vcvars64.bat via vswhere.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)

echo Using: %VCVARS%
call "%VCVARS%"

cd /d "%~dp0"
mkdir build 2>nul
cd build

cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
