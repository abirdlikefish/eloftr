@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v1 (Step 1) FULL FINETUNE: same hyperparameters as run_roadscene_finetune.bat
REM (the baseline) except for two things:
REM   1. main_cfg_path  -> eloftr_full_v1_contrast.py (USE_CONTRASTIVE = True)
REM   2. exp_name       -> roadscene_v1_contrast
REM Everything else (lr / epochs / batch_size / data split / seed) MUST match
REM the baseline run for a fair v1-vs-baseline comparison in Step 3.
REM
REM batch_size=2 is the minimum InfoNCE needs to gain cross-scene negatives.
REM Increase batch_size if GPU memory allows -- larger batch makes the
REM contrastive signal stronger.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v1_contrast.py ^
  --exp_name=roadscene_v1_contrast ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=4 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=50 ^
  --limit_train_batches=1.0 ^
  --limit_val_batches=1.0 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=20 ^
  --disable_mp ^
  --thr 0.1

pause
