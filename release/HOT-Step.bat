@echo off
echo =============================================
echo   HOT-Step 9000 CPP
echo   High-Performance Music Generation
echo =============================================
echo.

REM Set the distribution root for portable mode detection.
REM The server reads HOT_STEP_ROOT to resolve all paths.
set HOT_STEP_ROOT=%~dp0

REM Create models directory if it doesn't exist (first run)
if not exist "%~dp0models" mkdir "%~dp0models"

REM Open browser if no existing tab is found
start /MIN "" powershell -ExecutionPolicy Bypass -File "%~dp0open-browser-if-needed.ps1" "http://localhost:3001/"

REM ── Restart loop ──────────────────────────────────────────
REM The server writes .restart-requested when the user clicks
REM "Restart" in the UI. After node exits, we check for the
REM marker — if it exists, we delete it and relaunch.
:start
echo Starting server...
echo.
"%~dp0runtime\node.exe" "%~dp0server\server.mjs"

REM Check for restart marker
if exist "%~dp0.restart-requested" (
    del "%~dp0.restart-requested"
    echo.
    echo [HOT-Step] Restarting...
    echo.
    goto start
)

echo.
echo [HOT-Step] Server stopped. Press any key to exit.
pause >nul
