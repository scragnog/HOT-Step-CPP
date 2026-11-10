@echo off
setlocal enabledelayedexpansion
REM Stock Windows ships PowerShell 5.1 only; use pwsh (7+) when it is installed.
REM Resolved before any step that can replace this script, and on every re-entry.
set "HOTSTEP_PS=powershell.exe -NoProfile -ExecutionPolicy Bypass"
where pwsh >nul 2>&1 && set "HOTSTEP_PS=pwsh -NoProfile"
REM Serialize engine snapshots with rebuilds and updates in this checkout.
if "%HOTSTEP_UPDATE_FROM_TEMP%"=="1" goto :engine_lock_held
if /i not "%HOTSTEP_ENGINE_LOCK_ROOT%\"=="%~dp0" (
    %HOTSTEP_PS% -File "%~dp0tools\prod\engine-lock.ps1" -Script "%~f0" %*
    goto :engine_lock_result
)
goto :engine_lock_held
:engine_lock_result
exit /b %errorlevel%
:engine_lock_held

REM --- Self-copy workaround for CMD execution pointer corruption ---
REM Windows CMD reads .bat files line-by-line from disk. If git pull modifies
REM this file mid-execution, CMD loses its read offset and crashes or skips
REM to :fail. Fix: copy ourselves to %TEMP% and re-execute from there.
if not "%HOTSTEP_UPDATE_FROM_TEMP%"=="1" (
    set "HOTSTEP_UPDATE_FROM_TEMP=1"
    set "HOTSTEP_REPO_DIR=%~dp0"
    copy /y "%~f0" "%TEMP%\hotstep_update_temp.bat" >nul
    call "%TEMP%\hotstep_update_temp.bat" %*
    set "EXIT_CODE=!errorlevel!"
    del "%TEMP%\hotstep_update_temp.bat" 2>nul
    exit /b !EXIT_CODE!
)

pushd "%HOTSTEP_REPO_DIR%"

REM update.bat — One-click update for HOT-Step-CPP source builders.
REM
REM Pulls latest code, verifies integration hooks, rebuilds everything
REM incrementally. Reuses existing build infrastructure (build.cmd patterns).
REM
REM Usage:
REM   update.bat              Incremental update (safe, default)
REM   update.bat --force      Reset local changes before pulling (with confirmation)
REM   update.bat --clean      Force clean engine rebuild (warning: CUDA = 20+ min)
REM   update.bat --skip-engine  Skip engine rebuild (UI/server changes only)
REM   update.bat --help       Show this help
REM
REM For portable release users: you don't need this script.
REM   Download new releases from GitHub instead.

echo.
echo =======================================================
echo   HOT-Step-CPP — Smart Update
echo =======================================================
echo.

REM ── Parse arguments ─────────────────────────────────────────────────
set "FORCE=0"
set "CLEAN=0"
set "SKIP_ENGINE=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--force" set "FORCE=1"
if /i "%~1"=="--clean" set "CLEAN=1"
if /i "%~1"=="--skip-engine" set "SKIP_ENGINE=1"
shift
goto :parse_args

:args_done

REM ── Phase 0: Prerequisites check ───────────────────────────────────
echo [1/5] Checking prerequisites...
set "PREREQ_OK=1"

where git >nul 2>nul
if errorlevel 1 (
    echo   [FAIL] git not found in PATH
    set "PREREQ_OK=0"
)
where cmake >nul 2>nul
if errorlevel 1 (
    if "%SKIP_ENGINE%"=="0" (
        echo   [FAIL] cmake not found in PATH
        echo          Install CMake with "Add to system PATH" option.
        set "PREREQ_OK=0"
    )
)
where node >nul 2>nul
if errorlevel 1 (
    echo   [FAIL] node not found in PATH
    set "PREREQ_OK=0"
)
where npm >nul 2>nul
if errorlevel 1 (
    echo   [FAIL] npm not found in PATH
    set "PREREQ_OK=0"
)

if "%PREREQ_OK%"=="0" (
    echo.
    echo   Fix the above issues and try again.
    goto :fail
)
echo   All prerequisites found.

REM ── Phase 1: Pre-flight safety ─────────────────────────────────────
echo.
echo [2/5] Pre-flight checks...

REM Save current HEAD (and the ggml submodule pointer) for later
for /f "tokens=*" %%h in ('git rev-parse HEAD 2^>nul') do set "OLD_HEAD=%%h"
for /f "tokens=*" %%h in ('git rev-parse HEAD:engine/ggml 2^>nul') do set "OLD_GGML=%%h"

REM npm install can rewrite these generated lockfiles across npm versions.
git checkout -- server/package-lock.json ui/package-lock.json

REM Check for uncommitted changes to TRACKED files only.
REM
REM   --ignore-submodules=dirty: engine/ggml is checked on its own just below.
REM   An install from before the HOT-ggml fork carries the old patch overlay as
REM   uncommitted edits there, and without this flag git reports
REM   " m engine/ggml" and the update refuses to run before that check.
REM
REM   Untracked files never count. adapters/, models/, data/ and anything else
REM   the app writes are not "changes" and this script never deletes them.
git diff --quiet --ignore-submodules=dirty 2>nul
set "DIFF_ERR=%errorlevel%"
git diff --cached --quiet --ignore-submodules=dirty 2>nul
set "STAGED_ERR=%errorlevel%"

if "%DIFF_ERR%%STAGED_ERR%" neq "00" (
    if "%FORCE%"=="1" (
        echo.
        echo   WARNING: You have uncommitted changes to tracked files.
        echo   --force will DISCARD them. Untracked files ^(adapters, models, data^) are kept.
        echo.
        echo   Modified files:
        git status --short --ignore-submodules=dirty --untracked-files=no
        echo.
        choice /C YN /M "  Discard these changes and continue"
        if errorlevel 2 (
            echo   Aborted by user.
            goto :fail
        )
        echo   Resetting tracked files...
        REM reset --hard restores tracked files only. Never add "git clean" here:
        REM it deletes untracked files, and that once wiped a user's adapters/.
        REM submodule.recurse=false: engine/ggml is checked on its own below.
        git -c submodule.recurse=false reset --hard
    ) else (
        echo.
        echo   ERROR: You have uncommitted changes to tracked files:
        echo.
        git status --short --ignore-submodules=dirty --untracked-files=no
        echo.
        echo   Options:
        echo     1. Commit or stash your changes first
        echo     2. Run: update.bat --force  ^(discards changes to tracked files only^)
        echo.
        goto :fail
    )
) else (
    echo   Working tree is clean.
)

REM engine/ggml must be clean, or exactly the pre-fork patch overlay (base
REM c044c6f0, all 34 files at their known content). engine\ggml-pre-fork-overlay.ps1
REM decides: exit 0 clean, 2 exact overlay (restored after the pull), 1 anything
REM else, which stops here with every file left in place.
set "GGML_RESTORE_OVERLAY=0"
if exist "engine\ggml-pre-fork-overlay.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "engine\ggml-pre-fork-overlay.ps1" -Mode Check
    set "GGML_STATE=!errorlevel!"
    if "!GGML_STATE!"=="2" (
        echo   engine/ggml carries the exact pre-fork patch overlay; it will be restored after the pull.
        set "GGML_RESTORE_OVERLAY=1"
    ) else if "!GGML_STATE!" neq "0" (
        goto :fail
    )
)

REM Shut down only this checkout's server and engine.
echo   Checking for running server...
REM Read the same active .env as the server, including a moved data folder.
set "HOTSTEP_SHUTDOWN_PORT="
pushd "%HOTSTEP_REPO_DIR%server" || goto :fail
for /f "delims=" %%p in ('node --import tsx --input-type=module -e "import fs from 'node:fs'; import {parse} from 'dotenv'; import {locateData} from './src/dataroot/locate.ts'; const r=locateData(); if(r.pendingMove || r.pointerError || r.children === null) process.exit(1); const f=r.children.envFile; const e=fs.existsSync(f)?parse(fs.readFileSync(f)):{}; const p=Number(process.env.SERVER_PORT ?? e.SERVER_PORT ?? 3001); if(Number.isInteger(p)===false||p<1||p>65535) process.exit(1); console.log(p);" 2^>nul') do set "HOTSTEP_SHUTDOWN_PORT=%%p"
popd
if not defined HOTSTEP_SHUTDOWN_PORT (
    set "HOTSTEP_SHUTDOWN_PORT=3001"
    if defined SERVER_PORT set "HOTSTEP_SHUTDOWN_PORT=!SERVER_PORT!"
    echo WARNING: could not resolve this checkout's server port; using !HOTSTEP_SHUTDOWN_PORT!.
)
REM A trailing directory separator excludes sibling checkouts with a shared prefix.
%HOTSTEP_PS% -File "%HOTSTEP_REPO_DIR%tools\prod\stop-checkout.ps1" -Root "%HOTSTEP_REPO_DIR%." -Port "%HOTSTEP_SHUTDOWN_PORT%"
if errorlevel 1 goto :fail

REM ── Phase 2: Code sync ─────────────────────────────────────────────
echo.
echo [3/5] Pulling latest code...

REM submodule.recurse=false: with recursion on, the pull would check out the
REM new engine/ggml pin itself, before the overlay restore and URL sync below,
REM and fail on the pre-fork overlay or the stale ggml-org URL.
git -c submodule.recurse=false pull --ff-only origin master
if errorlevel 1 (
    echo.
    echo   ERROR: git pull --ff-only failed.
    echo   This usually means your local branch has diverged from origin/master.
    echo   Options:
    echo     1. Run: git rebase origin/master
    echo     2. Run: update.bat --force  ^(discards local changes^)
    echo.
    goto :fail
)

REM engine/ggml now follows HOT-ggml (docs/dev/ggml-fork.md), which carries
REM HOT-Step's ggml changes as commits. Restore exactly the pre-fork overlay
REM files when pre-flight recognised them; nothing else in engine/ggml is touched.
if "%GGML_RESTORE_OVERLAY%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "engine\ggml-pre-fork-overlay.ps1" -Mode Restore
    if errorlevel 1 goto :fail
)

REM The submodule URL is cached in .git/config at first init, so an install from
REM before the fork would keep fetching stock ggml-org and never find the pinned
REM commit. sync copies the URL from .gitmodules first. Both must succeed;
REM verify-hooks.ps1 (Hook 17) then requires engine/ggml clean at the pin
REM before anything is built.
git submodule sync --recursive
if errorlevel 1 (
    echo   ERROR: git submodule sync failed. Nothing was built.
    goto :fail
)
git submodule update --init --recursive
if errorlevel 1 (
    echo   ERROR: git submodule update failed. Nothing was built.
    echo   Check your connection, then rerun update.bat
    goto :fail
)

REM Show what changed
for /f "tokens=*" %%h in ('git rev-parse HEAD 2^>nul') do set "NEW_HEAD=%%h"
if "%OLD_HEAD%" neq "%NEW_HEAD%" (
    echo.
    echo   Changes pulled:
    git log --oneline %OLD_HEAD%..%NEW_HEAD%
    echo.
) else (
    echo   Already up to date.
)

REM ── Phase 3: Hook verification ─────────────────────────────────────
echo.
echo [4/5] Verifying integration hooks...

if exist "engine\verify-hooks.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "engine\verify-hooks.ps1"
    if errorlevel 1 (
        echo.
        echo   FATAL: Integration hooks are broken after pull!
        echo   The upstream files may have overwritten HOT-Step includes.
        echo   See output above for which hooks need fixing.
        echo.
        echo   Run the upstream-sync workflow to repair them.
        goto :fail
    )
) else (
    echo   ERROR: engine\verify-hooks.ps1 not found. Nothing was built.
    goto :fail
)

REM ── Phase 4: Build ──────────────────────────────────────────────────
echo.
echo [5/5] Building...

REM --- Server dependencies ---
echo   Installing server dependencies...
pushd server
call npm install --no-audit --no-fund 2>nul
if errorlevel 1 (
    echo   WARNING: Server npm install had issues.
)
REM Rebuild native modules (better-sqlite3 may need it after node update)
call npm rebuild better-sqlite3 2>nul
popd

REM --- UI dependencies ---
echo   Installing UI dependencies...
pushd ui
call npm install --no-audit --no-fund 2>nul
if errorlevel 1 (
    echo   WARNING: UI npm install had issues.
)
popd

REM --- Engine build ---
if "%SKIP_ENGINE%"=="1" (
    echo   Skipping engine build ^(--skip-engine flag^).
    goto :build_ui
)

REM Clean build requested?
if "%CLEAN%"=="1" (
    echo.
    echo   WARNING: --clean flag set. This will delete engine/build/ and
    echo   trigger a full rebuild. If you have CUDA, this takes 20+ minutes.
    echo.
    choice /C YN /M "  Continue with clean rebuild"
    if errorlevel 2 (
        echo   Clean rebuild skipped.
        goto :build_engine_incremental
    )
    echo   Cleaning engine build directory...
    if exist "engine\build" rd /s /q "engine\build"
)

:build_engine_incremental
echo   Building engine...

REM Auto-detect GPU backend — sets HOT_STEP_CMAKE_FLAGS for build.cmd
set "HOT_STEP_CMAKE_FLAGS="
set "DETECTED="

where nvcc >nul 2>nul
if not errorlevel 1 (
    set "HOT_STEP_CMAKE_FLAGS=-DGGML_CUDA=ON"
    set "DETECTED=CUDA"
    for /f "tokens=*" %%v in ('nvcc --version 2^>nul ^| findstr /C:"release"') do (
        echo   CUDA toolchain: %%v
    )
)

if defined VULKAN_SDK (
    set "HOT_STEP_CMAKE_FLAGS=!HOT_STEP_CMAKE_FLAGS! -DGGML_VULKAN=ON"
    if defined DETECTED (
        set "DETECTED=!DETECTED! + Vulkan"
    ) else (
        set "DETECTED=Vulkan"
    )
)

if not defined DETECTED (
    set "DETECTED=CPU-only"
    echo   No GPU SDK detected — building CPU-only backend.
) else (
    echo   Detected backends: !DETECTED!
)

REM Delegate to build.cmd — handles vcvars, ORT, cuDNN, and cmake build.
REM HOT_STEP_CMAKE_FLAGS env var overrides the default CUDA-only cmake flags.
REM build.cmd changes directory and does not restore it, so keep our cwd.
pushd "%HOTSTEP_REPO_DIR%"
call engine\build.cmd
set "ENGINE_RC=!errorlevel!"
popd
if not "!ENGINE_RC!"=="0" (
    echo.
    echo   ERROR: Engine build failed!
    echo   Check the output above for errors.
    echo   Common fixes:
    echo     - Install "Desktop development with C++" workload
    echo     - Ensure CUDA Toolkit is in PATH ^(if using CUDA^)
    echo     - Try: update.bat --clean  ^(forces fresh cmake config^)
    goto :fail
)
echo   Engine build complete.

REM --- UI build ---
:build_ui
echo   Building UI...
if not exist "%HOTSTEP_REPO_DIR%ui\package.json" (
    echo   ERROR: ui\package.json not found in %HOTSTEP_REPO_DIR%
    goto :fail
)
pushd "%HOTSTEP_REPO_DIR%ui" || goto :fail
if not exist "node_modules\.bin\vite.cmd" (
    echo   UI dependencies missing, running npm ci...
    call npm ci --no-audit --no-fund
    if errorlevel 1 (
        echo   npm ci failed, retrying with npm install...
        call npm install --no-audit --no-fund
        if errorlevel 1 (
            echo   ERROR: UI npm install failed. See errors above.
            popd
            goto :fail
        )
    )
)
REM Marker file: dist\index.html must end up newer than this.
set "UI_MARKER=%TEMP%\hotstep_ui_build_marker.txt"
echo marker> "%UI_MARKER%"
REM vite build from the local install only. Skips the tsc step of
REM "npm run build", which may fail on transient TypeScript errors on master.
call "node_modules\.bin\vite.cmd" build
if errorlevel 1 (
    echo   ERROR: UI build failed. See errors above.
    popd
    goto :fail
)
powershell.exe -NoProfile -Command "if ((Test-Path 'dist\index.html') -and (Get-Item 'dist\index.html').LastWriteTime -gt (Get-Item $env:UI_MARKER).LastWriteTime) { exit 0 } else { exit 1 }"
set "UI_FRESH_RC=!errorlevel!"
del "%UI_MARKER%" 2>nul
popd
if not "!UI_FRESH_RC!"=="0" (
    echo   ERROR: ui\dist\index.html was not rebuilt by the UI build.
    goto :fail
)
echo   UI build complete.

REM ── Phase 5: Validation & Report ────────────────────────────────────
echo.
echo =======================================================

REM Verify engine binary exists
set "ENGINE_OK=0"
if exist "engine\build\Release\ace-server.exe" set "ENGINE_OK=1"
if exist "engine\build\ace-server.exe" set "ENGINE_OK=1"

set "UI_OK=0"
if exist "ui\dist\index.html" set "UI_OK=1"

if "%ENGINE_OK%"=="1" (
    echo   [OK] Engine binary found
) else (
    if "%SKIP_ENGINE%"=="1" (
        echo   [--] Engine build skipped
    ) else (
        echo   [!!] Engine binary NOT found — build may have failed
    )
)

if "%UI_OK%"=="1" (
    echo   [OK] UI dist/ built
) else (
    echo   [!!] UI dist/ not found — build may have failed
)

echo.
if "%OLD_HEAD%" neq "%NEW_HEAD%" (
    echo   Updated: %OLD_HEAD:~0,8% -> %NEW_HEAD:~0,8%
) else (
    echo   No new commits ^(rebuild only^).
)
echo.
echo   Run LAUNCH.bat to start the application.
echo =======================================================
echo.

popd
endlocal
goto :eof

REM ── Help ────────────────────────────────────────────────────────────
:show_help
echo.
echo Usage: update.bat [options]
echo.
echo Options:
echo   --force         Discard changes to tracked files before pulling (with confirmation)
echo                   Never deletes untracked files: adapters/, models/, data/ are safe.
echo   --clean         Force clean engine rebuild (CUDA: 20+ min warning)
echo   --skip-engine   Skip engine rebuild (UI/server changes only)
echo   --help, -h      Show this help
echo.
echo This script is for SOURCE BUILDERS who cloned the repository.
echo Portable release users should download new releases from GitHub.
echo.
echo Prerequisites:
echo   - Git
echo   - CMake 3.10+  (in PATH)
echo   - Node.js LTS  (in PATH)
echo   - Visual Studio Build Tools ("Desktop development with C++" workload)
echo   - NVIDIA CUDA Toolkit (optional, for GPU acceleration)
echo   - Vulkan SDK (optional, for Vulkan backend)
echo.
popd
endlocal
goto :eof

REM ── Failure exit ────────────────────────────────────────────────────
:fail
echo.
echo   Update failed. See errors above.
echo.
popd
endlocal
pause
exit /b 1
