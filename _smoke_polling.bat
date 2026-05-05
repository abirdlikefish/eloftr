@echo off
setlocal enabledelayedexpansion
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0"

set "TARGET=logs\tb_logs\roadscene_v2_debug"
set "PORT=6006"

set "TB_PID="
for /f "usebackq tokens=*" %%P in (`powershell -NoProfile -Command "(Start-Process tensorboard -ArgumentList '--logdir','%TARGET%','--port','%PORT%','--reload_interval','5' -PassThru -WindowStyle Hidden).Id"`) do set "TB_PID=%%P"
echo TB_PID=!TB_PID!

set /a TRIES=0
:loop
set /a TRIES+=1
if !TRIES! gtr 15 goto done

echo.
echo === try !TRIES! at: ===
echo %time%

netstat -ano | findstr LISTENING | findstr ":%PORT% " >nul
if !errorlevel! equ 0 ( echo   netstat: bound ) else ( echo   netstat: not yet )

powershell -NoProfile -Command "try { $null = Invoke-WebRequest -Uri http://localhost:%PORT% -UseBasicParsing -TimeoutSec 2; exit 0 } catch { exit 1 }" 2>nul
if !errorlevel! equ 0 ( echo   probe: SUCCESS && goto done ) else ( echo   probe: failed )

ping -n 2 127.0.0.1 >nul
goto loop

:done
echo.
echo === final tasklist for TB_PID=!TB_PID! children ===
wmic process where (ParentProcessId=!TB_PID!) get ProcessId,Name,CommandLine 2>nul
echo.
echo === netstat for port %PORT% ===
netstat -ano | findstr LISTENING | findstr ":%PORT% "
taskkill /PID !TB_PID! /F /T >nul 2>&1
endlocal
exit /b 0
