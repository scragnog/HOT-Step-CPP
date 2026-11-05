@echo off
setlocal
REM Stock Windows ships PowerShell 5.1 only; use pwsh (7+) when it is installed.
REM Resolved before any step that can replace this script, and on every re-entry.
set "HOTSTEP_PS=powershell.exe -NoProfile -ExecutionPolicy Bypass"
where pwsh >nul 2>&1 && set "HOTSTEP_PS=pwsh -NoProfile"
REM Serialize engine snapshots with rebuilds and updates in this checkout.
if /i not "%HOTSTEP_ENGINE_LOCK_ROOT%\"=="%~dp0" (
    %HOTSTEP_PS% -File "%~dp0tools\prod\engine-lock.ps1" -Script "%~f0" %*
    goto :engine_lock_result
)
goto :engine_lock_held
:engine_lock_result
exit /b %errorlevel%
:engine_lock_held
REM Existing work-channel users reserve the app and GPU for this entire command.
REM A guarded child inherits this marker; it is a recursion guard, not authentication.
if "%HOTSTEP_WORK_GUARD_ACTIVE%"=="1" (
    echo ,%HOTSTEP_WORK_GUARD_RESOURCES%,| findstr /L /C:",app-server," >nul
    if errorlevel 1 goto :missing_work_resource
    echo ,%HOTSTEP_WORK_GUARD_RESOURCES%,| findstr /L /C:",gpu," >nul
    if errorlevel 1 goto :missing_work_resource
    echo ,%HOTSTEP_WORK_GUARD_RESOURCES%,| findstr /L /C:",engine-build," >nul
    if errorlevel 1 goto :missing_work_resource
    goto :work_guarded
)
REM Work-channel reservations are opt-in; the database alone does not enable them.
if not "%HOTSTEP_WORK_CHANNEL%"=="1" goto :work_guarded
REM Repositories that have never used collaboration retain the standalone flow.
if not exist "%~dp0data\collaboration.db" goto :work_guarded
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\mcp-lyricstudio\work-run.ps1" --channel HOT-Step --name Rebuild --resource app-server --resource gpu --resource engine-build --reason "Engine rebuild" -- "%~f0"
exit /b %errorlevel%
:missing_work_resource
echo [dev-rebuild] The outer work guard must reserve app-server, gpu and engine-build.
exit /b 1
:work_guarded
REM dev-rebuild.bat — Gracefully shut down the running app, rebuild the engine, done.
REM
REM Usage: dev-rebuild.bat
REM
REM Sends a shutdown request to this checkout's configured Node server, which kills
REM ace-server and itself cleanly (no orphan restarting). Then rebuilds.

echo [dev-rebuild] Requesting graceful shutdown...
set "HOTSTEP_REPO_DIR=%~dp0"
REM Read the same active .env as the server, including a moved data folder.
set "HOTSTEP_SHUTDOWN_PORT="
pushd "%HOTSTEP_REPO_DIR%server" || exit /b 1
for /f "delims=" %%p in ('node --import tsx --input-type=module -e "import fs from 'node:fs'; import {parse} from 'dotenv'; import {locateData} from './src/dataroot/locate.ts'; const r=locateData(); if(r.pendingMove || r.pointerError) process.exit(1); const f=r.children.envFile; const e=fs.existsSync(f)?parse(fs.readFileSync(f)):{}; const p=Number(process.env.SERVER_PORT ?? e.SERVER_PORT ?? 3001); if(Number.isInteger(p)===false||p<1||p>65535) process.exit(1); console.log(p);"') do set "HOTSTEP_SHUTDOWN_PORT=%%p"
popd
if not defined HOTSTEP_SHUTDOWN_PORT (
    echo [dev-rebuild] FATAL: could not resolve this checkout's server port.
    exit /b 1
)
REM A trailing directory separator excludes sibling checkouts with a shared prefix.
if /i "%~1"=="--dry-run" (
    %HOTSTEP_PS% -File "%HOTSTEP_REPO_DIR%tools\prod\stop-checkout.ps1" -Root "%HOTSTEP_REPO_DIR%." -Port %HOTSTEP_SHUTDOWN_PORT% -DryRun
    goto :engine_lock_result
)
%HOTSTEP_PS% -File "%HOTSTEP_REPO_DIR%tools\prod\stop-checkout.ps1" -Root "%HOTSTEP_REPO_DIR%." -Port %HOTSTEP_SHUTDOWN_PORT%
if errorlevel 1 exit /b 1

:build
echo [dev-rebuild] ace-server stopped. Building engine...
call "%~dp0engine\build.cmd"
if errorlevel 1 (
    echo [dev-rebuild] BUILD FAILED - the binaries on disk are STALE.
    echo [dev-rebuild] Scroll up for the first 'error C' or 'error LNK' line.
    exit /b 1
)
echo [dev-rebuild] Done. Start the app with dev.bat to pick up changes.
