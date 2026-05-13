@echo off
REM ============================================================
REM  eval_metu_vistir_official.bat
REM  Evaluate the official outdoor.ckpt (RGB-RGB MegaDepth training) on
REM  METU_VISTIR (real-captured drone VIS+LWIR pairs). Serves as the
REM  ELoFTR baseline cited by MINIMA paper Table 3 (2.88 / 7.88 / 17.72).
REM
REM  Protocol: MINIMA / XoFTR test_relative_pose_infrared.py (CVPR 2025)
REM    - ransac_thr=1.5, ransac_times=1 (single estimate_pose call)
REM    - long edge = 640 (paper Sec 5.1: "long dimension equal to 640")
REM    - LOFTR.MATCH_COARSE.THR = 0.2 (MINIMA load_loftr default)
REM    - NPE off (test long edge 640 < train 832; no extrapolation needed)
REM    - side0=vis / side1=thermal (mirrors MINIMA load_vis_tir_pairs_npz)
REM    - undistort via cv2.getOptimalNewCameraMatrix alpha=0 -> new_K
REM      (dataset side, see src/datasets/metu_vistir.py L254-273)
REM    - pad_to_square=False (dataset default; bs=1 forward at native shape)
REM    - AUC: per-npz error_auc -> split('_scene')[0] class mean ->
REM      2-class mean (all_class_mean = paper Table 3 row)
REM
REM  History: prior protocol mirrored ELoFTR outdoor_full_auc.sh
REM  (RGB-RGB MegaDepth-1500: ransac_thr=0.5 / 5-restart / megasize=1152
REM  + NPE / pool 2590 pair into single error_auc). That underestimated
REM  METU AUC by ~10-20x and is no longer used. See plan
REM  align-metu-eval-minima_*.plan.md for protocol audit details.
REM
REM  Output: dump\metu_eval_official\all\overall.txt (single full set run;
REM  per-class breakdown is computed inside python from per-scene AUCs).
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
echo   Eval set   : METU_VISTIR  (official outdoor.ckpt baseline)
echo   Ckpt       : !OFFICIAL_CKPT!
echo   Cfg (LoFTR): !FT_CFG!
echo   Out dir    : !OUT_DIR!
echo   Protocol   : MINIMA / XoFTR (CVPR 2025)
echo                ransac_thr=1.5  times=1  thr=0.2  megasize=640  npe=off
echo                side0=vis  side1=thermal  pad_to_square=False
echo                target: paper Table 3 ELoFTR = 2.88 / 7.88 / 17.72
echo ============================================================
echo.

REM Each call drives MyScripts/visualize_metu_vistir.py: progress prints to
REM this cmd window in real time; figures saved to <SUB_OUT>\figures\;
REM overall.txt + summary.csv written by the python script.

REM RANSAC: MINIMA / XoFTR protocol = thr 1.5, single shot.
set "RANSAC_FLAG=--ransac RANSAC --ransac_thr 1.5 --ransac_times 1"
set "MAX_FIGS=10"

REM OFFICIAL_PROTOCOL (MINIMA / XoFTR test_relative_pose_infrared.py):
REM   --thr 0.2         MATCH_COARSE.THR matches MINIMA load_loftr default
REM                     + eloftr_full.py cfg ("recommend 0.2 for full model")
REM   --megasize 640    paper Sec 5.1 "long dimension equal to 640"
REM                     (no --npe: test long edge 640 < train 832, no need)
REM   --metu_side0 vis  image0 = visible side (mirrors MINIMA load_vis_tir_pairs_npz
REM   --metu_side1 thermal   image_paths[id0][0]=visible, [id1][1]=thermal)
REM
REM Note: prior protocol used --thr 0.1 --megasize 1152 --npe which is
REM ELoFTR's RGB-RGB MegaDepth-1500 reproduce config (outdoor_full_auc.sh);
REM that drastically under-reports METU AUC. See plan
REM align-metu-eval-minima_*.plan.md.
set "OFFICIAL_PROTOCOL=--thr 0.2 --megasize 640 --metu_side0 vis --metu_side1 thermal"

REM ====================================================================
REM  SUBSETS: under MINIMA protocol the visualize_metu_vistir.py python
REM  end already groups per-npz AUC into cloudy_cloudy / cloudy_sunny /
REM  all_class_mean in a single full-set run. The 3 subsets (all /
REM  cloudy_cloudy / cloudy_sunny) on the data-cfg level are now redundant
REM  (cloudy_cloudy.txt and cloudy_sunny.txt rows will appear inside the
REM  'all' run's overall.txt [per-class] section). Keep SUBSETS=all by
REM  default; the other two are left as a smoke-test convenience.
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
    echo.
    echo === METU subset: !SUB!  ^(figures+csv+overall.txt at !SUB_OUT!^) ===
    python MyScripts\visualize_metu_vistir.py ^
      --ckpt "!OFFICIAL_CKPT!" ^
      --main_cfg "!FT_CFG!" ^
      --data_cfg "!DATA_CFG!" ^
      --out_dir "!SUB_OUT!" ^
      !RANSAC_FLAG! ^
      --max_figs !MAX_FIGS! ^
      !OFFICIAL_PROTOCOL! ^
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
echo  All METU subsets finished (official baseline). Summary:
for %%S in (%SUBSETS%) do (
    echo   !OUT_DIR!\%%S\overall.txt
)
echo ============================================================
endlocal
pause
