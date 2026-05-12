@echo off
REM ============================================================
REM  eval_metu_vistir_finetuned.bat
REM  Sister of MyScripts\eval_m3fd_finetuned.bat: same X / Y / Z
REM  semantics, but evaluates v10/v11 (or future msyn-trained)
REM  ckpts on METU_VISTIR (real-captured drone VIS+LWIR pairs)
REM  using the project-native test.py pose-based pipeline:
REM
REM    test.py -> trainer.test -> PL_LoFTR.test_step
REM      -> compute_symmetrical_epipolar_errors
REM      -> compute_pose_errors (RANSAC / LO-RANSAC)
REM      -> aggregate_metrics  ->  auc@5/10/20 + prec@5e-4
REM
REM  Three-subset reporting (one cfg per subset, run sequentially):
REM    all              : 10 npz / 2590 pair  (full set)
REM    cloudy_cloudy    :  6 npz / 1382 pair  (same illumination)
REM    cloudy_sunny     :  4 npz / 1208 pair  (cross illumination,
REM                                            hardest robustness slice)
REM
REM  Usage:
REM    eval_metu_vistir_finetuned.bat X [Y] [Z]
REM
REM  Inputs (all support -1 as "use default"):
REM    X : version number (10, 11, ...).
REM          -> resolves to logs\tb_logs\msyn_vX_* (so X=10 picks
REM             msyn_v10_ddp / msyn_v10_singlecard, X=11 picks
REM             msyn_v11_dualh_aggressive_ddp, etc.)
REM          -> auto-skips _debug / _small experiments
REM          -> auto-skips sub-version siblings (X=10 will not pick v10_1)
REM          -> required (no real default; -1 is treated as missing)
REM    Y : lightning version_Y under that experiment (optional)
REM          -> default = highest-numbered version_N
REM    Z : ckpt rank by RANK_KEY=precision@3px (optional)
REM          1..5 = pick the 1st..5th best ckpt
REM          6    = pick last.ckpt
REM          -> default = 1 (best p@3px)
REM
REM  cfg picked: configs\loftr\eloftr_full_vX*msyn*.py
REM              (precise glob to msyn-series cfg only; baseline
REM               eloftr_full.py and m3fd-only configs ignored)
REM
REM  Output dirs: dump\metu_eval_v<X>_version<Y>_<topZ|last>\{all,
REM               cloudy_cloudy,cloudy_sunny}\overall.txt
REM
REM  Examples:
REM    eval_metu_vistir_finetuned.bat 10           -> v10 best p@3px
REM    eval_metu_vistir_finetuned.bat 10 -1 6      -> v10 last.ckpt
REM    eval_metu_vistir_finetuned.bat 11           -> v11 best p@3px
REM    eval_metu_vistir_finetuned.bat 10 0 1       -> v10 version_0 top1
REM ============================================================

setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

set "LOGS_DIR=logs\tb_logs"
REM ckpt sort key. Today's v10/v11 ModelCheckpoint monitors precision@3px
REM (RoadScene/M3FD/Megadepth_Syn pipeline). v12+ that monitors auc-style
REM values can flip this single line, e.g. set "RANK_KEY=auc@10".
set "RANK_KEY=precision@3px"

REM ---- read inputs (cli or interactive) -----------------------------------
set "X=%~1"
set "Y=%~2"
set "Z=%~3"

if "%X%"=="" (
    set "_input="
    set /p _input="Enter X [Y] [Z] (e.g. '10', '11 -1 6' or '10 0 1'; -1 = use default): "
    if not "!_input!"=="" (
        for /f "tokens=1,2,3 delims=, " %%a in ("!_input!") do (
            set "X=%%a"
            set "Y=%%b"
            set "Z=%%c"
        )
    )
)

if "%X%"=="-1" set "X="
if "%Y%"=="-1" set "Y="
if "%Z%"=="-1" set "Z="

if "%X%"=="" (
    echo [ERROR] version number X is required.
    pause
    exit /b 1
)

REM Sub-version normalisation: 6_1 / 6.1 / 10_1 / 10.1 -> internal underscore form.
set "X=%X:.=_%"

if "%Z%"=="" set "Z=1"

set "VALID_Z=0"
for %%v in (1 2 3 4 5 6) do if "!Z!"=="%%v" set "VALID_Z=1"
if "!VALID_Z!"=="0" (
    echo [ERROR] Z must be 1..6 ^(got Z=!Z!^)
    pause
    exit /b 1
)

REM ---- resolve experiment directory ---------------------------------------
REM EXP glob is msyn_v<X>_* because pose-based eval is for v10/v11+ which
REM all live under logs\tb_logs\msyn_v<X>_<suffix>. Sub-version skip: when
REM X=10, msyn_v10_* greedy-matches msyn_v10_1 (hypothetical future) whose
REM tail (after _v10_) starts with `1`. Reject those so X=10 never picks
REM v10_1's logs and vice versa.
set "EXP="
for /d %%D in ("%LOGS_DIR%\msyn_v%X%_*") do (
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
    echo [ERROR] no msyn experiment found for v%X% under %LOGS_DIR%\msyn_v%X%_*
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

REM ---- pick ckpt by Z (rank by RANK_KEY desc) ----------------------------
if "!Z!"=="6" (
    set "FT_CKPT=!CKPT_DIR!\last.ckpt"
    set "Z_TAG=last"
) else (
    set "Z_TAG=top!Z!"
    set "TMP_RES=%TEMP%\eloftr_metu_ckpt_choice_!RANDOM!!RANDOM!.txt"
    if exist "!TMP_RES!" del "!TMP_RES!"
    powershell -NoProfile -Command "Get-ChildItem -Path '!CKPT_DIR!' -Filter 'epoch=*.ckpt' -ErrorAction SilentlyContinue | Sort-Object { if ($_.Name -match '!RANK_KEY!=([\d.]+)') { [double]$matches[1] } else { -1 } } -Descending | Select-Object -Skip (!Z!-1) -First 1 -ExpandProperty FullName | Out-File -FilePath '!TMP_RES!' -Encoding oem"
    set "FT_CKPT="
    if exist "!TMP_RES!" (
        for /f "usebackq delims=" %%F in ("!TMP_RES!") do if "!FT_CKPT!"=="" set "FT_CKPT=%%F"
        del "!TMP_RES!" 2>nul
    )
    if "!FT_CKPT!"=="" (
        echo [ERROR] no ckpt found at rank Z=!Z! under !CKPT_DIR!  ^(RANK_KEY=!RANK_KEY!^)
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
REM Glob configs\loftr\eloftr_full_v<X>*msyn*.py to pin the cfg to the
REM msyn-series (v10_msyn_ddp / v10_msyn_singlecard / v11_dualh_aggressive_msyn_ddp).
REM Sub-version skip is unnecessary here because the _msyn_ infix already
REM rules out baseline / m3fd / roadscene cfgs.
set "FT_CFG="
set "N_CFG=0"
for %%F in ("configs\loftr\eloftr_full_v%X%*msyn*.py") do (
    set "cname=%%~nF"
    set "ctail=!cname:*eloftr_full_v%X%_=!"
    set "cfirst=!ctail:~0,1!"
    set "cis_subver=0"
    for %%C in (0 1 2 3 4 5 6 7 8 9) do if "!cfirst!"=="%%C" set "cis_subver=1"
    if "!cis_subver!"=="0" (
        set /a "N_CFG+=1"
        if "!FT_CFG!"=="" set "FT_CFG=%%~F"
    )
)
if "!FT_CFG!"=="" (
    echo [ERROR] no cfg matching configs\loftr\eloftr_full_v%X%*msyn*.py
    pause
    exit /b 1
)
if !N_CFG! GTR 1 (
    echo [WARN] multiple cfgs match eloftr_full_v%X%*msyn*.py:
    for %%F in ("configs\loftr\eloftr_full_v%X%*msyn*.py") do echo           %%~nxF
    echo        using !FT_CFG!
)

REM ---- METU data layout sanity --------------------------------------------
set "METU_NPZ_PROBE=data\METU_VISTIR\index\scene_info_test\cloudy_cloudy_scene_1.npz"
if not exist "!METU_NPZ_PROBE!" (
    echo [ERROR] METU dataset not found: !METU_NPZ_PROBE!
    echo         expected layout:
    echo           data\METU_VISTIR\{cloudy,sunny}\scene_*\{visible,thermal}\images\IM_*.jpg
    echo           data\METU_VISTIR\index\scene_info_test\cloudy_*_scene_*.npz
    pause
    exit /b 1
)
set "METU_LIST_DIR=assets\metu_vistir_test_lists"
for %%S in (all cloudy_cloudy cloudy_sunny) do (
    if not exist "!METU_LIST_DIR!\%%S.txt" (
        echo [ERROR] missing METU subset list: !METU_LIST_DIR!\%%S.txt
        pause
        exit /b 1
    )
)

REM ---- output dir ---------------------------------------------------------
set "OUT_DIR=dump\metu_eval_v!X!_version!Y!_!Z_TAG!"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

echo.
echo ============================================================
echo   Eval set   : METU_VISTIR  (3 subsets)
echo   Experiment : !EXP!
echo   Version    : version_!Y!
echo   Z (rank)   : !Z!  (!Z_TAG!,  RANK_KEY=!RANK_KEY!)
echo   Ckpt       : !FT_CKPT!
echo   Cfg (LoFTR): !FT_CFG!
echo   Out dir    : !OUT_DIR!
echo ============================================================
echo.

REM ---- run all 3 subsets sequentially -------------------------------------
REM Each call drives MyScripts/visualize_metu_vistir.py:
REM   - progress prints DIRECTLY to this cmd window (no `> file` redirect),
REM     so you see one line per pair in real time
REM   - figures saved to <SUB_OUT>\figures\ (capped at --max_figs)
REM   - overall.txt + summary.csv written by the python script itself
REM
REM Default backend = OpenCV RANSAC at thr=2.0. Empirically (30-pair smoke
REM test on v10 ckpt) LO-RANSAC produces statistically identical auc but
REM is ~30% slower; switch to LO-RANSAC by editing RANSAC_FLAG below
REM (requires `pip install poselib`).
set "RANSAC_FLAG=--ransac RANSAC --ransac_thr 2.0 --ransac_times 5"

REM Cap on saved figures per subset (set 0 to disable figures entirely).
set "MAX_FIGS=10"

for %%S in (all cloudy_cloudy cloudy_sunny) do (
    set "SUB=%%S"
    set "SUB_OUT=!OUT_DIR!\!SUB!"
    if not exist "!SUB_OUT!" mkdir "!SUB_OUT!"
    set "DATA_CFG=configs\data\metu_vistir_test_!SUB!.py"
    if not exist "!DATA_CFG!" (
        echo [ERROR] missing data cfg: !DATA_CFG!
        pause
        exit /b 1
    )
    echo.
    echo === METU subset: !SUB!  ^(figures+csv+overall.txt at !SUB_OUT!^) ===

    python MyScripts\visualize_metu_vistir.py ^
      --ckpt "!FT_CKPT!" ^
      --main_cfg "!FT_CFG!" ^
      --data_cfg "!DATA_CFG!" ^
      --out_dir "!SUB_OUT!" ^
      !RANSAC_FLAG! ^
      --max_figs !MAX_FIGS! ^
      --thr 0.1 ^
      --num_workers 8

    if errorlevel 1 (
        echo [WARN] subset !SUB! exited non-zero. See progress lines above.
    ) else (
        echo [OK]   subset !SUB! done.
    )
)

echo.
echo ============================================================
echo  All METU subsets finished. Summary:
for %%S in (all cloudy_cloudy cloudy_sunny) do (
    echo   !OUT_DIR!\%%S\overall.txt
)
echo ============================================================
endlocal
pause
