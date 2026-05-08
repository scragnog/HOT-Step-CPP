@echo off
echo =============================================
echo   HOT-Step 9000 CPP - Installation
echo =============================================
echo.

echo [1/3] Installing server dependencies...
cd /d "%~dp0server"
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Server npm install failed
    pause
    exit /b 1
)
echo      Rebuilding native modules for your Node version...
call npm rebuild better-sqlite3
if %errorlevel% neq 0 (
    echo WARNING: better-sqlite3 rebuild failed - you may need to reinstall
)

echo.
echo [2/3] Installing UI dependencies...
cd /d "%~dp0ui"
call npm install
if %errorlevel% neq 0 (
    echo ERROR: UI npm install failed
    pause
    exit /b 1
)

echo.
echo [3/3] Building C++ engine (this may take a few minutes)...
echo      First build will also download ONNX Runtime for stem separation.
cd /d "%~dp0"
call engine\buildall.cmd
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Engine build failed. The server and UI are installed,
    echo          but you'll need to fix the build before launching.
    echo          See README for build prerequisites (Visual Studio, CUDA).
    pause
    exit /b 1
)

echo.
echo =============================================
echo   Installation complete!
echo.
echo   Run LAUNCH.bat to start, or dev.bat for development mode.
echo   No .env file needed for standard setups.
echo =============================================
pause
