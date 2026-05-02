@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Edit FT_CKPT below to point at the finetuned checkpoint you want to score.
REM Example: logs\tb_logs\roadscene_finetune\version_0\checkpoints\last.ckpt
set FT_CKPT=logs\tb_logs\roadscene_finetune\version_0\checkpoints\last.ckpt

if not exist "%FT_CKPT%" (
    echo Cannot find finetuned ckpt: %FT_CKPT%
    echo Please edit MyScripts\eval_roadscene_finetuned.bat and update FT_CKPT.
    pause
    exit /b 1
)

python MyScripts\eval_roadscene.py ^
  --ckpt "%FT_CKPT%" ^
  --list_path data\RoadScene\index\test_pairs.txt ^
  --out_dir dump\roadscene_eval_finetuned ^
  --thr 0.1

pause
