@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Edit FT_CKPT below to point at the finetuned checkpoint you want to score.
REM Edit FT_CFG to the LoFTR yacs config the ckpt was trained with so the
REM matcher class matches (otherwise modemb / contrastive params in v2/v3
REM ckpts will be reported as unexpected_keys and silently dropped).
REM
REM Example pairings:
REM   v0 baseline    -> configs\loftr\eloftr_full.py
REM   v1 contrast    -> configs\loftr\eloftr_full_v1_contrast.py
REM   v2 modemb      -> configs\loftr\eloftr_full_v2_modemb.py
REM   v3 combined    -> configs\loftr\eloftr_full_v3_combined.py
set FT_CKPT=logs\tb_logs\roadscene_v3_combined\version_1\checkpoints\last.ckpt
set FT_CFG=configs\loftr\eloftr_full_v3_combined.py

if not exist "%FT_CKPT%" (
    echo Cannot find finetuned ckpt: %FT_CKPT%
    echo Please edit MyScripts\eval_roadscene_finetuned.bat and update FT_CKPT.
    pause
    exit /b 1
)
if not exist "%FT_CFG%" (
    echo Cannot find LoFTR config: %FT_CFG%
    echo Please edit MyScripts\eval_roadscene_finetuned.bat and update FT_CFG.
    pause
    exit /b 1
)

python MyScripts\eval_roadscene.py ^
  --ckpt "%FT_CKPT%" ^
  --main_cfg "%FT_CFG%" ^
  --list_path data\RoadScene\index\test_pairs.txt ^
  --out_dir dump\roadscene_eval_finetuned ^
  --thr 0.1

pause
