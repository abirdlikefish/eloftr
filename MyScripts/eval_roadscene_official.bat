@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Evaluate the official outdoor checkpoint on the RoadScene test split.
REM Outputs go to dump\roadscene_eval_official\.
python MyScripts\eval_roadscene.py ^
  --ckpt weights\eloftr_outdoor.ckpt ^
  --list_path data\RoadScene\index\test_pairs.txt ^
  --out_dir dump\roadscene_eval_official ^
  --thr 0.1

pause
