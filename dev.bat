@echo off
echo =============================================
echo   HOT-Step 9000 CPP - Development Mode
echo =============================================
echo.

cd /d "%~dp0"

REM Start both server and Vite dev server concurrently
REM The server spawns ace-server, Vite provides HMR
start "HOT-Step Server" cmd /c "cd /d "%~dp0server" && npx tsx watch src/index.ts"
timeout /t 2 /nobreak > nul
start "HOT-Step UI" cmd /c "cd /d "%~dp0ui" && npx vite --port 3000 --host"

echo.
echo   Server: http://localhost:3001
echo   UI Dev: http://localhost:3000
echo.
echo   Open http://localhost:3000 for development
echo   Press Ctrl+C in each window to stop

