@echo off
echo =============================================
echo   HOT-Step 9000 CPP - Development Mode
echo =============================================
echo.

cd /d "%~dp0"

REM Start both server and Vite dev server concurrently
REM The server spawns ace-server, Vite provides HMR
REM Server uses restart-loop.cmd for in-app restart support
start /MIN "HOT-Step Server" cmd /c "cd /d "%~dp0server" && "%~dp0server\restart-loop.cmd""
timeout /t 2 /nobreak > nul
start /MIN "HOT-Step UI" cmd /c "cd /d "%~dp0ui" && npx vite --port 3000 --host"

echo.
echo   Server: http://localhost:3001
echo   UI Dev: http://localhost:3000
echo.
echo   Open http://localhost:3000 for development
echo   Press Ctrl+C in each window to stop
