@echo off
REM ============================================================
REM  eval_roadscene_finetuned.bat
REM  Usage:
REM    eval_roadscene_finetuned.bat X [Y] [Z]
REM
REM  Inputs (all support -1 as "use default"):
REM    X : version number (1, 2, 3, 4, ...)
REM          -> resolves to logs\tb_logs\roadscene_vX_<final>
REM          -> auto-skips _debug / _small experiments
REM          -> required (no real default; -1 is treated as missing)
REM    Y : lightning version_Y under that experiment (optional)
REM          -> default = highest-numbered version_N
REM             (numbering may be non-contiguous; we pick the max)
REM    Z : ckpt rank by precision@3px (optional)
REM          1..5 = pick the 1st..5th best ckpt (sorted desc by p@3px)
REM          6    = pick last.ckpt
REM          -> default = 1 (best p@3px)
REM
REM  Note: positional args naturally enforce "Z requires Y";
REM        if you want default Y but custom Z, pass Y=-1.
REM
REM  cfg picked: configs\loftr\eloftr_full_<vX_suffix>.py
REM              (suffix derived from the experiment dir name)
REM
REM  Examples:
REM    eval_roadscene_finetuned.bat 3              -> v3, latest version, best p@3px
REM    eval_roadscene_finetuned.bat 2 1            -> v2, version_1, best p@3px
REM    eval_roadscene_finetuned.bat 4 -1 6         -> v4, latest version, last.ckpt
REM    eval_roadscene_finetuned.bat 3 2 3          -> v3, version_2, 3rd best p@3px
REM ============================================================

setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

set "LOGS_DIR=logs\tb_logs"

REM ---- read inputs (cli or interactive) -----------------------------------
set "X=%~1"
set "Y=%~2"
set "Z=%~3"

REM If launched with no args (e.g. double-click), prompt once for the whole
REM "X [Y] [Z]" line so set /p doesn't swallow spaces into X. Tokens may be
REM separated by spaces or commas; missing tokens stay at their cmdline /
REM default values.
if "%X%"=="" (
    set "_input="
    set /p _input="Enter X [Y] [Z] (e.g. '2 1' or '3 -1 6'; -1 = use default): "
    if not "!_input!"=="" (
        for /f "tokens=1,2,3 delims=, " %%a in ("!_input!") do (
            set "X=%%a"
            set "Y=%%b"
            set "Z=%%c"
        )
    )
)

REM Treat -1 as "use default" (applies to both cmdline args and interactive input)
if "%X%"=="-1" set "X="
if "%Y%"=="-1" set "Y="
if "%Z%"=="-1" set "Z="

if "%X%"=="" (
    echo [ERROR] version number X is required.
    pause
    exit /b 1
)

REM Default Z = 1 (best p@3px)
if "%Z%"=="" set "Z=1"

REM Validate Z in 1..6
set "VALID_Z=0"
for %%v in (1 2 3 4 5 6) do if "!Z!"=="%%v" set "VALID_Z=1"
if "!VALID_Z!"=="0" (
    echo [ERROR] Z must be 1..6 ^(got Z=!Z!^)
    pause
    exit /b 1
)

REM ---- resolve experiment directory (skip _debug / _small) ----------------
set "EXP="
for /d %%D in ("%LOGS_DIR%\roadscene_v%X%_*") do (
    set "name=%%~nxD"
    set "is_test=0"
    if /i "!name:~-6!"=="_debug" set "is_test=1"
    if /i "!name:~-6!"=="_small" set "is_test=1"
    if "!is_test!"=="0" (
        if "!EXP!"=="" set "EXP=!name!"
    )
)

if "%EXP%"=="" (
    echo [ERROR] no final experiment found for v%X% under %LOGS_DIR%\roadscene_v%X%_*
    echo         ^(excluding _debug / _small^)
    pause
    exit /b 1
)

set "EXP_DIR=%LOGS_DIR%\!EXP!"

REM ---- resolve version_Y (default = numeric max) --------------------------
if "%Y%"=="" (
    set "MAX_V="
    for /d %%V in ("!EXP_DIR!\version_*") do (
        set "vname=%%~nxV"
        set "vnum=!vname:version_=!"
        REM skip non-numeric / empty names
        set "is_num=1"
        if "!vnum!"=="" set "is_num=0"
        for /f "delims=0123456789" %%C in ("!vnum!") do set "is_num=0"
        if "!is_num!"=="1" (
            if "!MAX_V!"=="" set "MAX_V=!vnum!"
            if !vnum! GTR !MAX_V! set "MAX_V=!vnum!"
        )
    )
    if "!MAX_V!"=="" (
        echo [ERROR] no version_N found under !EXP_DIR!
        pause
        exit /b 1
    )
    set "Y=!MAX_V!"
)

set "VERSION_DIR=!EXP_DIR!\version_!Y!"
if not exist "!VERSION_DIR!" (
    echo [ERROR] version directory does not exist: !VERSION_DIR!
    echo         existing versions:
    for /d %%V in ("!EXP_DIR!\version_*") do echo           %%~nxV
    pause
    exit /b 1
)

set "CKPT_DIR=!VERSION_DIR!\checkpoints"
if not exist "!CKPT_DIR!" (
    echo [ERROR] no checkpoints directory: !CKPT_DIR!
    pause
    exit /b 1
)

REM ---- pick ckpt by Z -----------------------------------------------------
REM Z=6 -> last.ckpt; Z=1..5 -> Z-th best by precision@3px (desc).
if "!Z!"=="6" (
    set "FT_CKPT=!CKPT_DIR!\last.ckpt"
    set "Z_TAG=last"
) else (
    set "Z_TAG=top!Z!"
    set "TMP_RES=%TEMP%\eloftr_ckpt_choice_%RANDOM%.txt"
    if exist "!TMP_RES!" del "!TMP_RES!"
    powershell -NoProfile -Command "Get-ChildItem -Path '!CKPT_DIR!' -Filter 'epoch=*.ckpt' -ErrorAction SilentlyContinue | Sort-Object { if ($_.Name -match 'precision@3px=([\d.]+)') { [double]$matches[1] } else { -1 } } -Descending | Select-Object -Skip (!Z!-1) -First 1 -ExpandProperty FullName | Out-File -FilePath '!TMP_RES!' -Encoding oem"
    set "FT_CKPT="
    if exist "!TMP_RES!" (
        for /f "usebackq delims=" %%F in ("!TMP_RES!") do if "!FT_CKPT!"=="" set "FT_CKPT=%%F"
        del "!TMP_RES!" 2>nul
    )
    if "!FT_CKPT!"=="" (
        echo [ERROR] no ckpt found at rank Z=!Z! under !CKPT_DIR!
        echo         available epoch=*.ckpt files:
        dir /b "!CKPT_DIR!\epoch=*.ckpt" 2>nul
        pause
        exit /b 1
    )
)

if not exist "!FT_CKPT!" (
    echo [ERROR] cannot find ckpt: !FT_CKPT!
    pause
    exit /b 1
)

REM ---- cfg ----------------------------------------------------------------
set "EXP_TAIL=!EXP:roadscene_=!"
set "FT_CFG=configs\loftr\eloftr_full_!EXP_TAIL!.py"

if not exist "!FT_CFG!" (
    echo [ERROR] cannot find LoFTR config: !FT_CFG!
    pause
    exit /b 1
)

set "OUT_DIR=dump\roadscene_eval_!EXP_TAIL!_version!Y!_!Z_TAG!"

echo.
echo ============================================================
echo   Experiment : !EXP!
echo   Version    : version_!Y!
echo   Z (rank)   : !Z!  (!Z_TAG!)
echo   Ckpt       : !FT_CKPT!
echo   Cfg        : !FT_CFG!
echo   Out dir    : !OUT_DIR!
echo ============================================================
echo.

python MyScripts\eval_roadscene.py ^
  --ckpt "!FT_CKPT!" ^
  --main_cfg "!FT_CFG!" ^
  --list_path data\RoadScene\index\test_pairs.txt ^
  --out_dir "!OUT_DIR!" ^
  --thr 0.1

endlocal
pause
