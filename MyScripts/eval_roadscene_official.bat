@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Evaluate the official outdoor checkpoint on the RoadScene test split
REM under v11-style dual-side Homography aug (warp both IR and VIS, fixed
REM seed=123 for reproducibility). Outputs go to
REM dump\roadscene_eval_official_dualh\. To restore the legacy no-aug eval,
REM remove the trailing --apply_homography --homography_dual --seed 123
REM flags from the python invocation below and drop the _dualh suffix.
python MyScripts\eval_roadscene.py ^
  --ckpt weights\eloftr_outdoor.ckpt ^
  --list_path data\RoadScene\index\test_pairs.txt ^
  --out_dir dump\roadscene_eval_official_dualh ^
  --thr 0.1 ^
  --apply_homography ^
  --homography_dual ^
  --seed 123

pause
