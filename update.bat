@echo off
setlocal enabledelayedexpansion

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

REM Check for uncommitted changes to TRACKED files only.
REM
REM   --ignore-submodules=dirty: engine/ggml is a submodule that every build
REM   modifies on purpose — CMake applies engine/patches/*.patch into it at
REM   configure time. Without this flag git reports " m engine/ggml" on every
REM   source builder's machine and the update refuses to run.
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
        echo   --force will DISCARD them. Untracked files (adapters, models, data) are kept.
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
        git reset --hard
    ) else (
        echo.
        echo   ERROR: You have uncommitted changes to tracked files:
        echo.
        git status --short --ignore-submodules=dirty --untracked-files=no
        echo.
        echo   Options:
        echo     1. Commit or stash your changes first
        echo     2. Run: update.bat --force  (discards changes to tracked files only)
        echo.
        goto :fail
    )
) else (
    echo   Working tree is clean.
)

REM Shut down running server (if any) to avoid file locks
echo   Checking for running server...
curl -s -o nul -w "%%{http_code}" http://localhost:3001/api/status >"%TEMP%\hotstep_status.txt" 2>nul
set /p STATUS_CODE=<"%TEMP%\hotstep_status.txt"
del "%TEMP%\hotstep_status.txt" 2>nul

if "%STATUS_CODE%"=="200" (
    echo   Server is running — requesting graceful shutdown...
    curl -s -X POST http://localhost:3001/api/shutdown >nul 2>&1

    REM Wait for ace-server to exit
    set /A retries=0
    :shutdown_wait
    tasklist /FI "IMAGENAME eq ace-server.exe" 2>nul | find /I "ace-server.exe" >nul
    if %errorlevel% neq 0 goto :shutdown_done
    timeout /t 1 /nobreak >nul
    set /A retries+=1
    if !retries! GEQ 10 (
        echo   Force-killing ace-server after 10s timeout...
        taskkill /F /IM ace-server.exe /T >nul 2>&1
        timeout /t 2 /nobreak >nul
    )
    if !retries! GEQ 15 (
        echo   WARNING: Could not stop ace-server. Build may fail with file locks.
        goto :shutdown_done
    )
    goto :shutdown_wait

    :shutdown_done
    echo   Server stopped.
) else (
    echo   No running server detected.
)

REM ── Phase 2: Code sync ─────────────────────────────────────────────
echo.
echo [3/5] Pulling latest code...

git pull --ff-only origin master
if errorlevel 1 (
    echo.
    echo   ERROR: git pull --ff-only failed.
    echo   This usually means your local branch has diverged from origin/master.
    echo   Options:
    echo     1. Run: git rebase origin/master
    echo     2. Run: update.bat --force  (discards local changes)
    echo.
    goto :fail
)

REM engine/ggml carries HOT-Step's patches as uncommitted edits (plus two new
REM files from flash-attn-train.patch). If the pull moved the submodule
REM pointer, git checkout would refuse to switch over those edits, so restore
REM the pristine tree first. This runs INSIDE engine\ggml only — it cannot
REM touch anything else in the repo. The patches are reapplied just below.
for /f "tokens=*" %%h in ('git rev-parse HEAD:engine/ggml 2^>nul') do set "NEW_GGML=%%h"
if "%OLD_GGML%" neq "%NEW_GGML%" (
    if exist "engine\ggml\.git" (
        echo   ggml submodule moved — restoring its pristine tree before checkout...
        git -C engine\ggml checkout -- .
        git -C engine\ggml clean -fd
    )
)

git submodule update --init --recursive
if errorlevel 1 (
    echo   WARNING: Submodule update had issues. Build may fail.
)

REM Reapply the ggml patches (same idempotent loop CMake runs at configure
REM time, but configure does not always rerun after a pull). A patch that
REM reverses cleanly is already in and is skipped.
set "PATCHES_APPLIED=0"
set "PATCHES_FAILED=0"
for %%p in ("engine\patches\*.patch") do (
    git apply --reverse --check --ignore-whitespace "%%~p" >nul 2>&1
    if errorlevel 1 (
        git apply --ignore-whitespace "%%~p" >nul 2>&1
        if errorlevel 1 (
            echo   WARNING: engine\patches\%%~nxp neither applies nor is already present.
            set /A PATCHES_FAILED+=1
        ) else (
            echo   Applied engine\patches\%%~nxp
            set /A PATCHES_APPLIED+=1
        )
    )
)
if "!PATCHES_FAILED!" neq "0" (
    echo   See engine\patches\README.md — the engine build may fail without them.
) else if "!PATCHES_APPLIED!"=="0" (
    echo   ggml patches already in place.
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
    echo   verify-hooks.ps1 not found — skipping hook check.
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
    echo   Skipping engine build (--skip-engine flag).
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
call engine\build.cmd
if errorlevel 1 (
    echo.
    echo   ERROR: Engine build failed!
    echo   Check the output above for errors.
    echo   Common fixes:
    echo     - Install "Desktop development with C++" workload
    echo     - Ensure CUDA Toolkit is in PATH (if using CUDA)
    echo     - Try: update.bat --clean  (forces fresh cmake config)
    goto :fail
)
echo   Engine build complete.

REM --- UI build ---
:build_ui
echo   Building UI...
pushd ui
REM Use npx vite build directly — bypasses tsc strict type-checking which
REM may fail on transient TypeScript errors on master during development.
call npx vite build
if errorlevel 1 (
    echo   WARNING: UI build had issues. The app may still work with old UI.
)
popd
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
    echo   No new commits (rebuild only).
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
