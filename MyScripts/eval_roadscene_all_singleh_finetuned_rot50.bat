@echo off
REM ============================================================
REM  eval_roadscene_all_singleh_finetuned_rot50.bat
REM
REM  Same X/Y/Z resolution logic as eval_roadscene_finetuned.bat,
REM  but evaluates on the FULL RoadScene index (train+val+test=221
REM  pairs) under v11/v12-style AGGRESSIVE single-side Homography
REM  augmentation with DOUBLED rotation (rot_deg=50):
REM
REM    warp VIS only (image1), IR (image0) untouched
REM    rot_deg=50 / scale=[0.75,1.25] / trans=0.12 / persp=0.08
REM    seed=123 for reproducibility
REM
REM  Visualisation: only the first 10 pairs get a PNG; overall.txt
REM  and summary.csv still aggregate across ALL 221 pairs.
REM
REM  cfg picked: configs\loftr\eloftr_eval_aggressive_singleh_rot50.py
REM    (1-channel baseline architecture + aggressive H aug kwargs,
REM     rot_deg=50; scale / trans / persp identical to the rot25 cfg).
REM    NOTE: this is forced regardless of which v_X cfg the model was
REM    trained with -- the script assumes the ckpt is a 1-channel
REM    model (USE_EDGE_INPUT=False, BACKBONE_IN_CHANNELS=1). For
REM    2-channel ckpts (v7/v10/v11/v12 with PC input) use the
REM    original eval_roadscene_finetuned.bat instead, since loading
REM    them under this 1-channel cfg produces missing_keys /
REM    unexpected_keys warnings on the first conv.
REM
REM  Usage:
REM    eval_roadscene_all_singleh_finetuned_rot50.bat X [Y] [Z]
REM
REM  Inputs (all support -1 as "use default"):
REM    X : version number, supports sub-versions: 6_1 or 6.1
REM    Y : lightning version_Y under that experiment (optional)
REM          -> default = highest-numbered version_N
REM    Z : ckpt rank by auto-detected RANK_KEY (optional)
REM          RANK_KEY is read from ckpt filename:
REM            - if 'auc@10=...' present       -> sort by auc@10 desc
REM              (v13+ pose-supervised: monitor='auc@10')
REM            - elif 'precision@3px=...' present -> sort by precision@3px desc
REM              (v0-v12 H-supervised: monitor='precision@3px')
REM          1..5 = pick the 1st..5th best ckpt under that key
REM          6    = pick last.ckpt
REM          -> default = 1 (best ckpt under the auto-detected key)
REM
REM  Prerequisite: data\RoadScene\index\all_pairs.txt must exist.
REM    type data\RoadScene\index\train_pairs.txt ^
REM         data\RoadScene\index\val_pairs.txt ^
REM         data\RoadScene\index\test_pairs.txt ^
REM      ^> data\RoadScene\index\all_pairs.txt
REM
REM  Output dir: dump\roadscene_eval_v<X>_version<Y>_<topZ|last>_all_singleh_aggressive_rot50
REM
REM  Examples:
REM    eval_roadscene_all_singleh_finetuned_rot50.bat 3
REM    eval_roadscene_all_singleh_finetuned_rot50.bat 6_1 -1 6
REM ============================================================

setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

set "LOGS_DIR=logs\tb_logs"
set "EVAL_CFG=configs\loftr\eloftr_eval_aggressive_singleh_rot50.py"
set "ALL_LIST=data\RoadScene\index\all_pairs.txt"

if not exist "!ALL_LIST!" (
    echo [ERROR] !ALL_LIST! does not exist.
    echo         Generate it once with:
    echo           type data\RoadScene\index\train_pairs.txt data\RoadScene\index\val_pairs.txt data\RoadScene\index\test_pairs.txt ^> !ALL_LIST!
    pause
    exit /b 1
)

if not exist "!EVAL_CFG!" (
    echo [ERROR] eval cfg not found: !EVAL_CFG!
    pause
    exit /b 1
)

REM ---- read inputs (cli or interactive) -----------------------------------
set "X=%~1"
set "Y=%~2"
set "Z=%~3"

if "%X%"=="" (
    set "_input="
    set /p _input="Enter X [Y] [Z] (e.g. '2 1', '6_1 -1 6' or '3 -1 6'; -1 = use default): "
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

REM Accept sub-version in either underscore (6_1) or dot (6.1) form;
REM internally we always use underscore form to match dir naming.
set "X=%X:.=_%"

REM Default Z = 1 (best ckpt under the auto-detected RANK_KEY; see L33).
if "%Z%"=="" set "Z=1"

REM Validate Z in 1..6
set "VALID_Z=0"
for %%v in (1 2 3 4 5 6) do if "!Z!"=="%%v" set "VALID_Z=1"
if "!VALID_Z!"=="0" (
    echo [ERROR] Z must be 1..6 ^(got Z=!Z!^)
    pause
    exit /b 1
)

REM ---- resolve experiment directory (skip _debug / _small + sub-versions) -
set "EXP="
for /d %%D in ("%LOGS_DIR%\*_v%X%_*") do (
    set "name=%%~nxD"
    set "tail=!name:*_v%X%_=!"
    set "first=!tail:~0,1!"
    set "is_subver=0"
    for %%C in (0 1 2 3 4 5 6 7 8 9) do if "!first!"=="%%C" set "is_subver=1"
    if "!is_subver!"=="0" (
        set "is_test=0"
        if /i "!name:~-6!"=="_debug" set "is_test=1"
        if /i "!name:~-6!"=="_small" set "is_test=1"
        if "!is_test!"=="0" if "!EXP!"=="" set "EXP=!name!"
    )
)

if "%EXP%"=="" (
    echo [ERROR] no final experiment found for v%X% under %LOGS_DIR%\*_v%X%_*
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

REM ---- pick ckpt by Z (rank by auc@10 -> precision@3px desc) -------------
REM Z=6 -> last.ckpt. Z=1..5 -> Z-th best ckpt under the auto-detected
REM RANK_KEY:
REM   v0-v12 H-supervised  ckpt filename has 'precision@3px=...'
REM                        (monitor='precision@3px' at train time).
REM   v13+   pose-supervised ckpt filename has 'auc@10=...'
REM                        (monitor='auc@10' at train time, megadepth-style).
REM The powershell sort key uses an if/elseif so both ckpt families are
REM ranked correctly in one bat invocation -- mixing the two in the same
REM checkpoints/ dir is not expected, but if it happens auc@10 wins
REM (v13+ takes precedence over v0-v12).
if "!Z!"=="6" (
    set "FT_CKPT=!CKPT_DIR!\last.ckpt"
    set "Z_TAG=last"
) else (
    set "Z_TAG=top!Z!"
    set "TMP_RES=%TEMP%\eloftr_ckpt_choice_!RANDOM!!RANDOM!.txt"
    if exist "!TMP_RES!" del "!TMP_RES!"
    powershell -NoProfile -Command "Get-ChildItem -Path '!CKPT_DIR!' -Filter 'epoch=*.ckpt' -ErrorAction SilentlyContinue | Sort-Object { if ($_.Name -match 'auc@10=([\d.]+)') { [double]$matches[1] } elseif ($_.Name -match 'precision@3px=([\d.]+)') { [double]$matches[1] } else { -1 } } -Descending | Select-Object -Skip (!Z!-1) -First 1 -ExpandProperty FullName | Out-File -FilePath '!TMP_RES!' -Encoding oem"
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

REM ---- output dir ---------------------------------------------------------
set "OUT_DIR=dump\roadscene_eval_v!X!_version!Y!_!Z_TAG!_all_singleh_aggressive_rot50"

echo.
echo ============================================================
echo   Experiment : !EXP!
echo   Version    : version_!Y!
echo   Z (rank)   : !Z!  (!Z_TAG!)
echo   Ckpt       : !FT_CKPT!
echo   Cfg        : !EVAL_CFG!
echo   Index      : !ALL_LIST! (221 pairs)
echo   Aug        : single-side aggressive (rot 50 / scale 0.75-1.25 / trans 0.12 / persp 0.08)
echo   Out dir    : !OUT_DIR!
echo ============================================================
echo.
echo   NOTE: this script forces a 1-channel eval cfg
echo         (eloftr_eval_aggressive_singleh_rot50.py). If !FT_CKPT! is a
echo         2-channel model (v7 / v10 / v11 / v12 with
echo         USE_EDGE_INPUT=True or BACKBONE_IN_CHANNELS=2),
echo         load_state_dict will report missing_keys / unexpected_keys
echo         for the first conv and PC fields. For 2-channel ckpts
echo         use the original eval_roadscene_finetuned.bat instead.
echo.

python MyScripts\eval_roadscene.py ^
  --ckpt "!FT_CKPT!" ^
  --main_cfg "!EVAL_CFG!" ^
  --data_cfg configs\data\roadscene_trainval.py ^
  --list_path "!ALL_LIST!" ^
  --out_dir "!OUT_DIR!" ^
  --thr 0.1 ^
  --apply_homography ^
  --no_homography_dual ^
  --seed 123 ^
  --max_save_figures 10

endlocal
pause
