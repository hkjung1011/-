@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Platform.ps1"
set "platformExit=%ERRORLEVEL%"
pause
exit /b %platformExit%
