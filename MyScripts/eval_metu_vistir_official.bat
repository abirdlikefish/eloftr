@echo off
REM ============================================================
REM  eval_metu_vistir_official.bat
REM  Evaluate the official outdoor.ckpt (real-multiview RGB-RGB MegaDepth
REM  training) on METU_VISTIR (real-multiview RGB-LWIR test). Serves as
REM  a baseline for v10/v11 cross-modal training -- if v10 ckpt does not
REM  beat outdoor.ckpt on METU, the cross-modal training did not transfer
REM  to real LWIR.
REM
REM  Three subsets: all / cloudy_cloudy / cloudy_sunny  (same protocol
REM  as eval_metu_vistir_finetuned.bat).
REM
REM  Output: dump\metu_eval_official\{all,cloudy_cloudy,cloudy_sunny}\overall.txt
REM ============================================================
setlocal enabledelayedexpansion

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr
cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

set "OFFICIAL_CKPT=weights\eloftr_outdoor.ckpt"
set "FT_CFG=configs\loftr\eloftr_full.py"
set "OUT_DIR=dump\metu_eval_official"

if not exist "!OFFICIAL_CKPT!" (
    echo [ERROR] official ckpt missing: !OFFICIAL_CKPT!
    echo         download from EfficientLoFTR release page first.
    pause
    exit /b 1
)
if not exist "!FT_CFG!" (
    echo [ERROR] baseline cfg missing: !FT_CFG!
    pause
    exit /b 1
)

REM METU data sanity (mirrors the finetuned bat).
set "METU_NPZ_PROBE=data\METU_VISTIR\index\scene_info_test\cloudy_cloudy_scene_1.npz"
if not exist "!METU_NPZ_PROBE!" (
    echo [ERROR] METU dataset not found: !METU_NPZ_PROBE!
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

if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

echo.
echo ============================================================
echo   Eval set   : METU_VISTIR  (3 subsets, official outdoor.ckpt baseline)
echo   Ckpt       : !OFFICIAL_CKPT!
echo   Cfg (LoFTR): !FT_CFG!
echo   Out dir    : !OUT_DIR!
echo ============================================================
echo.

REM Each call drives MyScripts/visualize_metu_vistir.py: progress prints to
REM this cmd window in real time; figures saved to <SUB_OUT>\figures\;
REM overall.txt + summary.csv written by the python script.
REM
REM Default = OpenCV RANSAC at thr=2.0 (zero extra dependency); LO-RANSAC
REM needs `pip install poselib`.
set "RANSAC_FLAG=--ransac RANSAC --ransac_thr 2.0 --ransac_times 5"
set "MAX_FIGS=10"

for %%S in (all cloudy_cloudy cloudy_sunny) do (
    set "SUB=%%S"
    set "SUB_OUT=!OUT_DIR!\!SUB!"
    if not exist "!SUB_OUT!" mkdir "!SUB_OUT!"
    set "DATA_CFG=configs\data\metu_vistir_test_!SUB!.py"
    echo.
    echo === METU subset: !SUB!  ^(figures+csv+overall.txt at !SUB_OUT!^) ===
    python MyScripts\visualize_metu_vistir.py ^
      --ckpt "!OFFICIAL_CKPT!" ^
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
echo  All METU subsets finished (official baseline). Summary:
for %%S in (all cloudy_cloudy cloudy_sunny) do (
    echo   !OUT_DIR!\%%S\overall.txt
)
echo ============================================================
endlocal
pause
