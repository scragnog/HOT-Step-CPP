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

REM Open browser after a short delay
start "" cmd /c "timeout /t 5 /nobreak > nul & start http://localhost:3001/"

REM Run server via portable Node.js
echo Starting server...
echo.
"%~dp0runtime\node.exe" "%~dp0server\server.mjs"

echo.
echo [HOT-Step] Server stopped. Press any key to exit.
pause >nul
