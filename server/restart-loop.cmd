@echo off
REM restart-loop.cmd — Dev-mode server wrapper with restart support.
REM Used by dev.bat to allow in-app restart without restarting Vite.

:loop
call npx tsx watch src/index.ts
if exist "%~dp0..\.restart-requested" (
    del "%~dp0..\.restart-requested"
    echo.
    echo [HOT-Step] Restarting server...
    timeout /t 2 /nobreak > nul
    goto loop
)
