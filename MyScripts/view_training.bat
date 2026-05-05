@echo off
REM ============================================================
REM  view_training.bat
REM  Usage:
REM    view_training.bat X [Y]
REM      X : version number, supports sub-versions: 6_1 or 6.1 (normalized
REM          internally to 6_1). Matches logs\tb_logs\*_vX_*.
REM          Auto-detects dataset prefix: roadscene_v1..v4, m3fd_v5+, ...
REM      Y : run type (optional, default = 3)
REM            1 = debug   -> *_vX_debug
REM            2 = small   -> *_vX_small
REM            3 = final   -> *_vX_<anything except _debug / _small>
REM
REM  Behavior:
REM    - Auto-picks the first free TCP port in 6006..6020, so multiple
REM      view_training.bat instances can run side by side.
REM    - Launches tensorboard hidden in the background (PID captured).
REM    - Opens an isolated Edge / Chrome --app window (no tab bar / URL bar)
REM      via a unique --user-data-dir per port.
REM    - Closing that browser window automatically kills the matching
REM      tensorboard process and ends this script.
REM
REM    NOTE: if you force-close THIS bat window (X / Ctrl+C -> Y) before
REM    closing the browser, the cleanup step is skipped and tensorboard
REM    becomes an orphan. Prefer closing the browser window first.
REM
REM    NOTE: assumes each version number maps to a single dataset (current
REM    convention: v1..v4 = roadscene, v5+ = m3fd). If a version ever exists
REM    under multiple datasets, this picks the first one for /d returns.
REM    Sub-versions (e.g. v6_1) are treated as DISTINCT from their parent
REM    (v6) via a post-filter that rejects any *_vX_<digit>... match, so
REM    `view_training.bat 6` will never accidentally open v6_1's logs.
REM
REM  Examples:
REM    view_training.bat 1        -> open final run of v1 (roadscene_v1_contrast)
REM    view_training.bat 4        -> open final run of v4 (roadscene_v4_combined)
REM    view_training.bat 5        -> open final run of v5 (m3fd_v5_combined)
REM    view_training.bat 5 1      -> open debug run of v5 (m3fd_v5_debug)
REM    view_training.bat 5 2      -> open small run of v5 (m3fd_v5_small)
REM    view_training.bat 6        -> open final run of v6 (m3fd_v6_finetune)
REM    view_training.bat 6_1      -> open final run of v6_1 (m3fd_v6_1_finetune)
REM    view_training.bat 6.1      -> same as `view_training.bat 6_1`
REM ============================================================

setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."

set "LOGS_DIR=logs\tb_logs"

REM ---- read inputs --------------------------------------------------------
set "X=%~1"
set "Y=%~2"

REM If launched without args (e.g. double-click), prompt once for the whole
REM "X [Y]" line. We read into a buffer and split on space/comma rather than
REM letting set /p drop the entire line into X -- otherwise typing "6 0" at
REM the prompt would set X="6 0" and the *_v6 0_* glob would silently miss.
REM Y is only filled from the prompt if it wasn't already passed as %~2.
if "%X%"=="" (
    set "_input="
    set /p _input="Enter X [Y] (e.g. '6 1' / '6_1' / '6.1'; X required, Y optional 1=debug 2=small 3=final): "
    if not "!_input!"=="" (
        for /f "tokens=1,2 delims=, " %%a in ("!_input!") do (
            set "X=%%a"
            if "%Y%"=="" set "Y=%%b"
        )
    )
)
if "%X%"=="" (
    echo [ERROR] version number X is required.
    pause
    exit /b 1
)

REM Accept sub-version in either underscore (6_1) or dot (6.1) form;
REM internally we always use underscore form to match dir naming.
REM This is a no-op when X has no dot, so plain "5", "6" etc. still work.
set "X=%X:.=_%"

if "%Y%"=="" set "Y=3"

REM ---- resolve experiment directory ---------------------------------------
REM   Pattern *_vX_* matches across all dataset prefixes (roadscene, m3fd, ...).
REM   We always SCAN; we never construct the name from a hard-coded prefix.
if "%Y%"=="1" (
    set "WANT_SUFFIX=_debug"
    set "WANT_FINAL=0"
) else if "%Y%"=="2" (
    set "WANT_SUFFIX=_small"
    set "WANT_FINAL=0"
) else if "%Y%"=="3" (
    set "WANT_SUFFIX="
    set "WANT_FINAL=1"
) else (
    echo [ERROR] invalid Y=%Y%   ^(must be 1=debug, 2=small, 3=final^)
    pause
    exit /b 1
)

set "EXP="
for /d %%D in ("%LOGS_DIR%\*_v%X%_*") do (
    set "name=%%~nxD"
    REM Strip everything up to and including `_v{X}_` to inspect the kind suffix.
    REM If that suffix starts with a digit, this is a *sub-version* match
    REM (e.g. X=6 hitting m3fd_v6_1_finetune), so skip it. Without this filter
    REM, `view_training.bat 6` would non-deterministically open either
    REM m3fd_v6_finetune or m3fd_v6_1_finetune depending on FS order.
    set "tail=!name:*_v%X%_=!"
    set "first=!tail:~0,1!"
    set "is_subver=0"
    for %%C in (0 1 2 3 4 5 6 7 8 9) do if "!first!"=="%%C" set "is_subver=1"
    if "!is_subver!"=="0" (
        if "!WANT_FINAL!"=="1" (
            set "is_test=0"
            if /i "!name:~-6!"=="_debug" set "is_test=1"
            if /i "!name:~-6!"=="_small" set "is_test=1"
            if "!is_test!"=="0" if "!EXP!"=="" set "EXP=!name!"
        ) else (
            if /i "!name:~-6!"=="!WANT_SUFFIX!" if "!EXP!"=="" set "EXP=!name!"
        )
    )
)

if "%EXP%"=="" (
    echo [ERROR] no experiment found for v%X% with Y=%Y%
    echo         looked under: %LOGS_DIR%\*_v%X%_*
    pause
    exit /b 1
)

set "TARGET=%LOGS_DIR%\%EXP%"
if not exist "%TARGET%" (
    echo [ERROR] experiment directory does not exist: %TARGET%
    pause
    exit /b 1
)

REM ---- detect browser (Edge preferred, Chrome fallback) -------------------
set "BROWSER="
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    set "BROWSER=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    set "BROWSER=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    set "BROWSER=C:\Program Files\Google\Chrome\Application\chrome.exe"
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    set "BROWSER=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)
if "%BROWSER%"=="" (
    echo [ERROR] need Microsoft Edge or Google Chrome installed for app-mode auto-close.
    pause
    exit /b 1
)

REM ---- find a free TCP port in 6006..6020 ---------------------------------
set /a PORT=6006
:find_port
netstat -ano | findstr "LISTENING" | findstr ":%PORT% " >nul
if %errorlevel% equ 0 (
    set /a PORT+=1
    if !PORT! gtr 6020 (
        echo [ERROR] no free port in 6006-6020. Close some other tensorboard windows.
        pause
        exit /b 1
    )
    goto find_port
)

REM ---- header -------------------------------------------------------------
echo.
echo ============================================================
echo   Experiment : %EXP%
echo   Log dir    : %TARGET%
echo   Port       : %PORT%
echo   Browser    : %BROWSER%
echo   URL        : http://localhost:%PORT%
echo.
echo   Close the browser window to stop tensorboard ^&^& this script.
echo ============================================================
echo.

REM ---- start tensorboard hidden in background, capture its PID ------------
REM   We launch via `python -m tensorboard.main` instead of the `tensorboard`
REM   shim, because on this machine the shim's embedded shebang points to a
REM   different conda env's python (loftr_gpu), which can listen on the port
REM   but mishandles HTTP requests. Going through `python -m` forces the use
REM   of the currently-activated env's interpreter.
set "TB_PID="
for /f "usebackq tokens=*" %%P in (`powershell -NoProfile -Command "(Start-Process python -ArgumentList '-m','tensorboard.main','--logdir','%TARGET%','--port','%PORT%','--reload_interval','5' -PassThru -WindowStyle Hidden).Id"`) do set "TB_PID=%%P"

if not defined TB_PID (
    echo [ERROR] failed to start tensorboard. Is the eff_loftr conda env activated and tensorboard installed?
    pause
    exit /b 1
)
echo [INFO] tensorboard PID = %TB_PID%, waiting for it to bind port %PORT% ...

REM ---- poll TB until it's serving HTTP (cold start can take 5-10s) --------
REM   Uses flat goto labels (no multi-line if-blocks) and delayed expansion
REM   to dodge cmd's parse-time-vs-run-time variable pitfalls. Uses ping for
REM   sleep because `timeout` breaks when stdin is redirected (e.g. <nul).
REM   TimeoutSec must be >=5: even after the TCP port is bound, TB's first
REM   HTTP response is slow (it lazy-initializes plugins and scans logdir),
REM   and a 2s timeout will lose the race every time.
set /a TRIES=0
:probe_loop
set /a TRIES+=1
powershell -NoProfile -Command "try { $null = Invoke-WebRequest -Uri http://localhost:%PORT% -UseBasicParsing -TimeoutSec 5; exit 0 } catch { exit 1 }" 2>nul
if !errorlevel! equ 0 goto probe_ok
if !TRIES! geq 10 goto probe_fail
ping -n 2 127.0.0.1 >nul
goto probe_loop

:probe_fail
echo [ERROR] tensorboard not responding on http://localhost:%PORT% after 10 tries.
echo         (TB log is hidden; remove '-WindowStyle Hidden' in the script to debug.)
taskkill /PID %TB_PID% /F /T >nul 2>&1
pause
exit /b 1

:probe_ok
echo [INFO] tensorboard is up on http://localhost:%PORT% (took ~!TRIES!s).

REM ---- launch browser in app mode, BLOCK until window is closed -----------
set "USER_DATA=%TEMP%\tb_browser_%PORT%"

echo [INFO] opening browser window. Close it to stop tensorboard.

start /WAIT "" "%BROWSER%" ^
    --app="http://localhost:%PORT%" ^
    --user-data-dir="%USER_DATA%" ^
    --new-window ^
    --window-size=1400,900 ^
    --no-first-run ^
    --no-default-browser-check

REM ---- cleanup: kill tensorboard, drop temp profile -----------------------
echo.
echo [INFO] browser closed. Killing tensorboard (PID %TB_PID%) ...
taskkill /PID %TB_PID% /F /T >nul 2>&1

rmdir /s /q "%USER_DATA%" 2>nul

echo [INFO] done.
endlocal
exit /b 0
