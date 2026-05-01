@echo off
echo =============================================
echo   HOT-Step 9000 CPP - Installation
echo =============================================
echo.

echo [1/2] Installing server dependencies...
cd /d "%~dp0server"
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Server npm install failed
    pause
    exit /b 1
)

echo.
echo [2/2] Installing UI dependencies...
cd /d "%~dp0ui"
call npm install
if %errorlevel% neq 0 (
    echo ERROR: UI npm install failed
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
