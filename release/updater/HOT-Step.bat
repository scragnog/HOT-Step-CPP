@echo off
echo =============================================
echo   HOT-Step 9000 CPP
echo   High-Performance Music Generation
echo =============================================
echo.

REM The launcher owns this install folder: it settles an interrupted update,
REM starts the version current.json names, restarts it on request and
REM installs updates. It runs on the Node runtime beside this file; each
REM version in app\ brings its own runtime for the server.
"%~dp0runtime\node.exe" "%~dp0launcher.mjs"
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    echo [HOT-Step] Stopped. Press any key to close this window.
    pause >nul
)
exit /b %CODE%
