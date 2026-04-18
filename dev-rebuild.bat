@echo off
REM dev-rebuild.bat — Gracefully shut down the running app, rebuild the engine, done.
REM
REM Usage: dev-rebuild.bat
REM
REM Sends a shutdown request to the Node server (port 3001), which kills
REM ace-server and itself cleanly (no orphan restarting). Then rebuilds.

echo [dev-rebuild] Requesting graceful shutdown...
curl -s -X POST http://localhost:3001/api/shutdown >nul 2>&1

REM Wait for ace-server to actually die (the shutdown route kills by port)
echo [dev-rebuild] Waiting for ace-server to exit...
set /A retries=0
:wait_loop
tasklist /FI "IMAGENAME eq ace-server.exe" 2>nul | find /I "ace-server.exe" >nul
if %errorlevel% neq 0 goto :build
timeout /t 1 /nobreak >nul
set /A retries+=1
if %retries% GEQ 10 (
    echo [dev-rebuild] Force-killing ace-server after 10s timeout...
    taskkill /F /IM ace-server.exe /T >nul 2>&1
    timeout /t 2 /nobreak >nul
)
if %retries% GEQ 15 (
    echo [dev-rebuild] FATAL: could not stop ace-server. Aborting.
    exit /b 1
)
goto :wait_loop

:build
echo [dev-rebuild] ace-server stopped. Building engine...
call "%~dp0engine\build.cmd"
echo [dev-rebuild] Done. Start the app with LAUNCH.bat to pick up changes.
