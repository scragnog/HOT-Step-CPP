@echo off
echo =============================================
echo   HOT-Step 9000 CPP - Production
echo =============================================
echo.

REM Build UI if dist doesn't exist
if not exist "%~dp0ui\dist" (
    echo Building UI...
    cd /d "%~dp0ui"
    call npm run build
)

REM Start server (which spawns ace-server) with restart loop
cd /d "%~dp0server"
echo Starting server...

REM Open browser after a short delay
start "" cmd /c "timeout /t 4 /nobreak > nul & start http://localhost:3001/"

:loop
call npx tsx src/index.ts
if exist "%~dp0.restart-requested" (
    del "%~dp0.restart-requested"
    echo.
    echo [HOT-Step] Restarting server...
    timeout /t 2 /nobreak > nul
    goto loop
)
