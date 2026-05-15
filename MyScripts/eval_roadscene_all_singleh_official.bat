@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================
REM  eval_roadscene_all_singleh_official.bat
REM
REM  Evaluate the official outdoor checkpoint on the FULL RoadScene
REM  index (train + val + test = 221 pairs) under v11/v12-style
REM  AGGRESSIVE single-side Homography augmentation:
REM
REM    warp VIS only (image1), IR (image0) untouched
REM    rot_deg=25 / scale=[0.75,1.25] / trans=0.12 / persp=0.08
REM    seed=123 for reproducibility
REM
REM  Visualisation: only the first 10 pairs get a PNG; overall.txt
REM  and summary.csv still aggregate across ALL 221 pairs.
REM
REM  Prerequisite: data\RoadScene\index\all_pairs.txt must exist.
REM  Generate it once (cmd) with:
REM    type data\RoadScene\index\train_pairs.txt ^
REM         data\RoadScene\index\val_pairs.txt ^
REM         data\RoadScene\index\test_pairs.txt ^
REM      ^> data\RoadScene\index\all_pairs.txt
REM  (Then verify 221 lines via: find /v /c "" data\RoadScene\index\all_pairs.txt)
REM
REM  cfg picked: configs\loftr\eloftr_eval_aggressive_singleh.py
REM    (1-channel baseline architecture + aggressive H aug kwargs)
REM
REM  Output dir: dump\roadscene_eval_official_all_singleh_aggressive
REM ============================================================

if not exist "data\RoadScene\index\all_pairs.txt" (
    echo [ERROR] data\RoadScene\index\all_pairs.txt does not exist.
    echo         Generate it once with:
    echo           type data\RoadScene\index\train_pairs.txt data\RoadScene\index\val_pairs.txt data\RoadScene\index\test_pairs.txt ^> data\RoadScene\index\all_pairs.txt
    pause
    exit /b 1
)

python MyScripts\eval_roadscene.py ^
  --ckpt weights\eloftr_outdoor.ckpt ^
  --main_cfg configs\loftr\eloftr_eval_aggressive_singleh.py ^
  --data_cfg configs\data\roadscene_trainval.py ^
  --list_path data\RoadScene\index\all_pairs.txt ^
  --out_dir dump\roadscene_eval_official_all_singleh_aggressive ^
  --thr 0.1 ^
  --apply_homography ^
  --no_homography_dual ^
  --seed 123 ^
  --max_save_figures 10

endlocal
pause
