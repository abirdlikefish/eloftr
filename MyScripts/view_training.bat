@echo off
REM ============================================================
REM  view_training.bat
REM  Usage:
REM    view_training.bat X [Y]
REM      X : version number (1, 2, 3, ...) -> matches logs\tb_logs\roadscene_vX_*
REM      Y : run type (optional, default = 3)
REM            1 = debug   -> roadscene_vX_debug
REM            2 = small   -> roadscene_vX_small
REM            3 = final   -> auto-pick the only roadscene_vX_* that is NOT _debug / _small
REM
REM  Examples:
REM    view_training.bat 1        -> open final run of v1 (roadscene_v1_contrast)
REM    view_training.bat 2 2      -> open small run of v2  (roadscene_v2_small)
REM    view_training.bat 3 1      -> open debug run of v3  (roadscene_v3_debug)
REM ============================================================

setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."

set "LOGS_DIR=logs\tb_logs"
set "PORT=6006"

REM ---- read inputs --------------------------------------------------------
set "X=%~1"
set "Y=%~2"

if "%X%"=="" (
    set /p X="Enter version number X (e.g. 1, 2, 3): "
)
if "%X%"=="" (
    echo [ERROR] version number X is required.
    pause
    exit /b 1
)

if "%Y%"=="" set "Y=3"

REM ---- resolve experiment directory ---------------------------------------
set "EXP="
if "%Y%"=="1" (
    set "EXP=roadscene_v%X%_debug"
) else if "%Y%"=="2" (
    set "EXP=roadscene_v%X%_small"
) else if "%Y%"=="3" (
    REM scan logs\tb_logs\roadscene_vX_* and pick the first that is NOT _debug / _small
    for /d %%D in ("%LOGS_DIR%\roadscene_v%X%_*") do (
        set "name=%%~nxD"
        set "is_test=0"
        if /i "!name:~-6!"=="_debug" set "is_test=1"
        if /i "!name:~-6!"=="_small" set "is_test=1"
        if "!is_test!"=="0" (
            if "!EXP!"=="" set "EXP=!name!"
        )
    )
) else (
    echo [ERROR] invalid Y=%Y%   ^(must be 1=debug, 2=small, 3=final^)
    pause
    exit /b 1
)

if "%EXP%"=="" (
    echo [ERROR] no experiment found for v%X% with Y=%Y%
    echo         looked under: %LOGS_DIR%\roadscene_v%X%_*
    pause
    exit /b 1
)

set "TARGET=%LOGS_DIR%\%EXP%"
if not exist "%TARGET%" (
    echo [ERROR] experiment directory does not exist: %TARGET%
    pause
    exit /b 1
)

REM ---- launch tensorboard -------------------------------------------------
echo.
echo ============================================================
echo   Experiment : %EXP%
echo   Log dir    : %TARGET%
echo   URL        : http://localhost:%PORT%
echo   Press Ctrl+C in this window to stop TensorBoard.
echo ============================================================
echo.

start "" http://localhost:%PORT%
tensorboard --logdir "%TARGET%" --port %PORT%

endlocal
