@echo off
set "HOTSTEP_PS=powershell.exe -NoProfile -ExecutionPolicy Bypass"
where pwsh >nul 2>&1 && set "HOTSTEP_PS=pwsh -NoProfile"
%HOTSTEP_PS% -File "%~dp0run.ps1"
exit /b %errorlevel%
