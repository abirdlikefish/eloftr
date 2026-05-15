@echo off
REM ============================================================
REM  eval_metu_vistir_finetuned.bat
REM  Sister of eval_metu_vistir_official.bat. Evaluates ANY finetuned
REM  ckpt (v0..v9 RoadScene/M3FD-trained or v10/v11/v12+ Megadepth_Syn-
REM  trained) on METU_VISTIR (real-captured drone VIS+LWIR pairs) under
REM  the same protocol as the ELoFTR baseline run by
REM  eval_metu_vistir_official.bat, so finetuned numbers and the
REM  outdoor.ckpt baseline (paper Table 3 ELoFTR row 2.88 / 7.88 / 17.72)
REM  are directly comparable.
REM
REM  Protocol: MINIMA / XoFTR test_relative_pose_infrared.py (CVPR 2025)
REM    - ransac_thr=1.5, ransac_times=1 (single estimate_pose call)
REM    - long edge = 640 (paper Sec 5.1: "long dimension equal to 640")
REM    - LOFTR.MATCH_COARSE.THR = 0.2 (MINIMA load_loftr default;
REM      eloftr_full.py cfg: "recommend 0.2 for full model")
REM    - NPE off (test long edge 640 < train 832; no extrapolation needed)
REM    - side0=thermal / side1=vis (cfg default for v10+ -- modemb_ir
REM      bound to image0 needs thermal in image0; NOT overridden here).
REM      Mirror cross-version comparison caveat: ELoFTR baseline
REM      (official.bat) overrides to side0=vis to match MINIMA; the two
REM      protocols therefore differ in image0 modality. See
REM      align-metu-eval-minima_*.plan.md for the asymmetry rationale.
REM    - undistort via cv2.getOptimalNewCameraMatrix alpha=0 -> new_K
REM      (dataset side, src/datasets/metu_vistir.py)
REM    - pad_to_square=False (dataset default; bs=1 forward at native shape)
REM    - AUC: per-npz error_auc -> split('_scene')[0] class mean ->
REM      2-class mean (all_class_mean compares with paper Table 3)
REM
REM  History: prior protocol used ransac_thr=2.0 / 5-restart / cfg default
REM  megasize=832 and pooled all 2590 pair errors into a single AUC
REM  (project-native test.py path). That protocol drastically
REM  under-reports METU AUC and is incompatible with paper baseline.
REM  results/eval_summary.md sec 6 v0..v12 METU rows produced under the
REM  old protocol are NO LONGER directly comparable to new runs and need
REM  a full rerun once this script is committed.
REM
REM  Usage:
REM    eval_metu_vistir_finetuned.bat X [Y] [Z]
REM
REM  Inputs (all support -1 as "use default"):
REM    X : version number (0, 1, ..., 10, 11, 12, ...). Resolves to
REM        logs\tb_logs\*_vX_* (any dataset prefix: roadscene_, m3fd_,
REM        msyn_, ...). Auto-skips _debug / _small / sub-version siblings
REM        (X=6 will not pick v6_1; X=10 will not pick v10_1). Required.
REM    Y : lightning version_Y under that experiment (optional; default
REM        = numeric max).
REM    Z : ckpt rank by RANK_KEY (optional; default = 1).
REM        RANK_KEY auto-detects from ckpt filename:
REM          - if 'auc@10=...' is present  -> sort by auc@10 desc
REM            (v13+ pose-supervised pipeline; train-time monitor='auc@10').
REM          - elif 'precision@3px=...' is present -> sort by precision@3px desc
REM            (v0-v12 H-supervised pipeline; train-time monitor='precision@3px').
REM          - else -> ckpt gets score -1 and falls back to filename order.
REM        1..5 = pick the 1st..5th best ckpt; 6 = last.ckpt.
REM
REM  cfg picked: configs\loftr\eloftr_full_vX_*.py
REM              (auto-glob with sub-version skip filter; e.g. X=0 picks
REM               eloftr_full_v0_baseline.py, X=10 picks v10_msyn_*.py
REM               and the multi-match warning surfaces both ddp/singlecard).
REM
REM  Output: dump\metu_eval_v<X>_version<Y>_<topZ|last>\all\overall.txt
REM          (per-class breakdown appears inside the same overall.txt;
REM          single full-set run, no need for 3-subset loop).
REM
REM  Examples:
REM    eval_metu_vistir_finetuned.bat 0            -> v0 (RoadScene-trained)
REM    eval_metu_vistir_finetuned.bat 6_1          -> v6_1 (M3FD-trained)
REM    eval_metu_vistir_finetuned.bat 10           -> v10 best p@3px
REM    eval_metu_vistir_finetuned.bat 10 -1 6      -> v10 last.ckpt
REM    eval_metu_vistir_finetuned.bat 11           -> v11 best p@3px
REM    eval_metu_vistir_finetuned.bat 10 0 1       -> v10 version_0 top1
REM
REM  Note on side0/side1:
REM    cfg default is METU_SIDE0='thermal' / METU_SIDE1='vis' (set in
REM    configs\data\metu_vistir_test_all.py). This matches every finetuned
REM    ckpt's training-time convention -- v0..v9 (RoadScene / M3FD via
REM    src/datasets/roadscene.py L376-378) AND v10+ (Megadepth_Syn) all
REM    put IR in image0 at training. modemb_ir, when used (v2+), is bound
REM    to image0 so thermal-as-image0 stays correct. Do NOT override here.
REM    The only side0=vis case is eval_metu_vistir_official.bat which
REM    runs outdoor.ckpt (RGB-RGB MegaDepth, never saw IR).
REM ============================================================

setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

set "LOGS_DIR=logs\tb_logs"
REM ckpt sort key. ModelCheckpoint monitor differs per training pipeline:
REM   v0-v12  (H-supervised RoadScene/M3FD/Megadepth_Syn) -> monitor='precision@3px'
REM           -> ckpt filename: epoch=*-precision@1px=*-precision@3px=*-precision@5px=*.ckpt
REM   v13+    (pose-supervised Megadepth_Syn LoFTR-style)  -> monitor='auc@10'
REM           -> ckpt filename: epoch=*-auc@5=*-auc@10=*-auc@20=*.ckpt
REM The powershell sort below tries 'auc@10=' first (v13+) and falls back to
REM 'precision@3px=' (v0-v12) so the same bat works for both ckpt families
REM without manual switching. RANK_KEY here is a display-only label used in
REM echo + error messages; the actual sort logic is hardcoded in the
REM powershell line.
set "RANK_KEY=auc@10 -> precision@3px (auto)"

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
REM EXP glob *_v%X%_* matches any dataset prefix (roadscene_, m3fd_, msyn_, ...)
REM so the same script can eval v0..v9 (RoadScene/M3FD-trained) and
REM v10/v11/v12+ (Megadepth_Syn-trained) ckpts on METU under one protocol.
REM Sub-version skip: when X=10, *_v10_* greedy-matches msyn_v10_1
REM (hypothetical future) whose tail (after _v10_) starts with `1`. Reject
REM those so X=10 never picks v10_1's logs and vice versa.
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
    echo [ERROR] no experiment found for v%X% under %LOGS_DIR%\*_v%X%_*
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
REM RANK_KEY (auc@10 for v13+ pose ckpts, precision@3px for v0-v12 H ckpts;
REM see L86-95 above for the auto-detect rationale). The powershell sort
REM key uses an if/elseif so both ckpt families are ranked correctly in
REM one bat invocation -- mixing the two in the same checkpoints/ dir is
REM not expected, but if it happens auc@10 wins (v13+ takes precedence).
if "!Z!"=="6" (
    set "FT_CKPT=!CKPT_DIR!\last.ckpt"
    set "Z_TAG=last"
) else (
    set "Z_TAG=top!Z!"
    set "TMP_RES=%TEMP%\eloftr_metu_ckpt_choice_!RANDOM!!RANDOM!.txt"
    if exist "!TMP_RES!" del "!TMP_RES!"
    powershell -NoProfile -Command "Get-ChildItem -Path '!CKPT_DIR!' -Filter 'epoch=*.ckpt' -ErrorAction SilentlyContinue | Sort-Object { if ($_.Name -match 'auc@10=([\d.]+)') { [double]$matches[1] } elseif ($_.Name -match 'precision@3px=([\d.]+)') { [double]$matches[1] } else { -1 } } -Descending | Select-Object -Skip (!Z!-1) -First 1 -ExpandProperty FullName | Out-File -FilePath '!TMP_RES!' -Encoding oem"
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
REM Glob configs\loftr\eloftr_full_v<X>_*.py to pick the matching cfg
REM regardless of suffix (baseline / finetune / msyn_ddp / dualh_aggressive_
REM msyn_ddp etc). Sub-version skip: when X=6, eloftr_full_v6_*.py matches
REM both v6_finetune.py and v6_1_finetune.py; reject the latter so X=6
REM picks only v6_finetune.py (and X=6_1 picks only v6_1_finetune.py).
REM Same convention as eval_roadscene_finetuned.bat L218-244.
set "FT_CFG="
set "N_CFG=0"
for %%F in ("configs\loftr\eloftr_full_v%X%_*.py") do (
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
    echo [ERROR] no cfg matching configs\loftr\eloftr_full_v%X%_*.py
    echo         ^(sub-version siblings, if any, are skipped on purpose^)
    pause
    exit /b 1
)
if !N_CFG! GTR 1 (
    echo [WARN] multiple cfgs match eloftr_full_v%X%_*.py:
    for %%F in ("configs\loftr\eloftr_full_v%X%_*.py") do echo           %%~nxF
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
echo   Eval set   : METU_VISTIR  (msyn-trained ckpt)
echo   Experiment : !EXP!
echo   Version    : version_!Y!
echo   Z (rank)   : !Z!  (!Z_TAG!,  RANK_KEY=!RANK_KEY!)
echo   Ckpt       : !FT_CKPT!
echo   Cfg (LoFTR): !FT_CFG!
echo   Out dir    : !OUT_DIR!
echo   Protocol   : MINIMA / XoFTR (CVPR 2025)
echo                ransac_thr=1.5  times=1  thr=0.2  megasize=640  npe=off
echo                side0=thermal  side1=vis  pad_to_square=False
echo                (sister of eval_metu_vistir_official.bat; same protocol
echo                 except side0/1 = thermal/vis vs vis/thermal -- see header)
echo ============================================================
echo.

REM Each call drives MyScripts/visualize_metu_vistir.py: progress prints to
REM this cmd window in real time; figures saved to <SUB_OUT>\figures\;
REM overall.txt + summary.csv written by the python script.

REM RANSAC: MINIMA / XoFTR protocol = thr 1.5, single shot.
set "RANSAC_FLAG=--ransac RANSAC --ransac_thr 1.5 --ransac_times 1"
set "MAX_FIGS=10"

REM FT_PROTOCOL (MINIMA / XoFTR test_relative_pose_infrared.py):
REM   --thr 0.2         MATCH_COARSE.THR matches MINIMA load_loftr default
REM                     + eloftr_full.py cfg ("recommend 0.2 for full model").
REM                     NOTE: train-side ModelCheckpoint monitor uses THR=0.1
REM                     to pick best.ckpt by prec@1/3/5px; eval-side THR=0.2
REM                     is the MINIMA-aligned cross-version-comparable setting.
REM   --megasize 640    paper Sec 5.1 "long dimension equal to 640"
REM                     (no --npe: test long edge 640 < train 832, no need).
REM   side0 / side1     NOT overridden here. cfg default (METU_SIDE0=thermal,
REM                     METU_SIDE1=vis) is the v10+ training-time convention
REM                     (modemb_ir is bound to image0). The ELoFTR baseline
REM                     bat overrides to side0=vis because outdoor.ckpt has
REM                     no modemb and MINIMA's reference image_paths[id0][0]
REM                     is visible. Mind this asymmetry when comparing.
set "FT_PROTOCOL=--thr 0.2 --megasize 640"

REM ====================================================================
REM  SUBSETS: under MINIMA protocol the visualize_metu_vistir.py python
REM  end already groups per-npz AUC into cloudy_cloudy / cloudy_sunny /
REM  all_class_mean in a single full-set run. The 3 subsets are now
REM  redundant; keep SUBSETS=all by default.
REM ====================================================================
set "SUBSETS=all"

REM ====================================================================
REM  STRIDE: 步长抽样, 加速 smoke / sanity test
REM     1   : 完整跑 (默认, 生产/正式 eval 用)
REM     10  : 每 10 对取 1 对 (~259/2590, ~10x 加速, 覆盖所有 scene)
REM     50  : 每 50 对取 1 对 (~52 pair, 极速 sanity check)
REM  注: STRIDE > 1 时 auc@5/10/20 数字仅供方向性预览, 跨版本对比仍需 STRIDE=1
REM ====================================================================
set "STRIDE=1"

for %%S in (%SUBSETS%) do (
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
      !FT_PROTOCOL! ^
      --stride !STRIDE! ^
      --num_workers 8

    if errorlevel 1 (
        echo [WARN] subset !SUB! exited non-zero. See progress lines above.
    ) else (
        echo [OK]   subset !SUB! done.
    )
)

echo.
echo ============================================================
echo  All METU subsets finished (finetuned ckpt). Summary:
for %%S in (%SUBSETS%) do (
    echo   !OUT_DIR!\%%S\overall.txt
)
echo ============================================================
endlocal
pause
