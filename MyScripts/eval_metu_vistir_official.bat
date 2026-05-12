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
echo   Eval set   : METU_VISTIR  (official outdoor.ckpt baseline)
echo   Ckpt       : !OFFICIAL_CKPT!
echo   Cfg (LoFTR): !FT_CFG!
echo   Out dir    : !OUT_DIR!
echo   Protocol   : ransac_thr=0.5  thr=0.1  megasize=1152  npe=on
echo                side0=vis  side1=thermal  (mirrors outdoor_full_auc.sh)
echo ============================================================
echo.

REM Each call drives MyScripts/visualize_metu_vistir.py: progress prints to
REM this cmd window in real time; figures saved to <SUB_OUT>\figures\;
REM overall.txt + summary.csv written by the python script.
REM
REM ====================================================================
REM  RANSAC_FLAG: 官方 outdoor_full_auc.sh 用 ransac_thr=0.5 (像素阈值,
REM  作用在归一化坐标系 K^-1 后, 经验上 0.5 是 MegaDepth/METU 这类外景
REM  的标准 AUC 复现值). 我们之前用的 2.0 = 4x 偏松 -> RANSAC 把 outlier
REM  当 inlier 喂给 essential matrix -> R/t 估计漂移 -> AUC 大幅低估.
REM  ransac_times=5 沿用官方 EVAL_TIMES=5 (5 次随机种子 RANSAC 取最好).
REM ====================================================================
set "RANSAC_FLAG=--ransac RANSAC --ransac_thr 0.5 --ransac_times 5"
set "MAX_FIGS=10"

REM ====================================================================
REM  OFFICIAL_PROTOCOL: 复现 outdoor.ckpt 的 4 个关键参数 (与 finetuned
REM  ckpts 故意不一致, finetuned 走 metu_vistir_test_*.py cfg 默认值)
REM
REM    --thr 0.1            ^ MATCH_COARSE.THR; 官方 outdoor_full_auc.sh
REM                           显式 --thr 0.1, NOT cfg 注释里的 0.2.
REM    --megasize 1152      ^ 长边 resize=1152 (cfg 默认 832); METU 4K 图
REM                           保留更多纹理, 跨模态匹配关键.
REM    --npe                ^ 启用 NPE = [832,832,1152,1152]; 告诉 RoPE
REM                           "训练在 832, 测试在 1152", 频率正确外推.
REM                           必须配 --megasize 1152 一起用.
REM    --metu_side0 vis     ^ image0=vis, image1=thermal; outdoor.ckpt 是
REM    --metu_side1 thermal   纯 RGB 训练 (无 modemb), 经验上 vis-as-image0
REM                           比 thermal-as-image0 (cfg 默认) 高 ~4x AUC.
REM ====================================================================
set "OFFICIAL_PROTOCOL=--thr 0.1 --megasize 1152 --npe --metu_side0 vis --metu_side1 thermal"

REM ====================================================================
REM  SUBSETS: 想跑哪几个子集就留哪几个; 直接改这一行
REM     all           : 2590 pair (10 npz)
REM     cloudy_cloudy : 1382 pair (6 npz, 同光照, 较易)
REM     cloudy_sunny  : 1208 pair (4 npz, 跨光照, 最难)
REM
REM  例:
REM     set "SUBSETS=all cloudy_cloudy cloudy_sunny"   REM 全部 (默认, 完整 eval)
REM     set "SUBSETS=cloudy_cloudy"                    REM 只同光照
REM     set "SUBSETS=cloudy_sunny"                     REM 只跨光照
REM     set "SUBSETS=all"                              REM 只跑 all
REM ====================================================================
set "SUBSETS=cloudy_cloudy"

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
