@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v4 SMALL: 100 train batches, 20 val batches, 3 epochs (smoke test).
REM
REM Key things to compare against v2_small / v3_small in TensorBoard:
REM   - val precision@3px curve: with backbone unfrozen + moderate LR (1.25e-4),
REM     v4 should NOT crash early like v3 (TRUE_LR=5e-4) and should ramp faster
REM     than v2 (effective TRUE_LR ~1.25e-5 in the WARMUP_STEP=1875 trap).
REM   - mod_emb_*_norm growth: with backbone fully trainable, modality
REM     separation can flow through conv weights instead of through modemb;
REM     watch whether modemb norm saturates (good) or keeps growing like v2/v3.
REM   - loss_contrast / loss_i2v / loss_v2i should all still appear
REM     (confirming v4 still inherits v1 InfoNCE on top of v2 modemb).
REM   - LR should ramp 0 -> 1.25e-4 over ~32 steps (WARMUP_STEP=2 auto-scaled
REM     to 32 at bs=4) and stay flat through 3 epochs (MSLR fires at 10/15/20,
REM     not visible in this 3-epoch smoke test -- only in v4_combined.bat).
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v4_combined.py ^
  --exp_name=roadscene_v4_small ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=2 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=20 ^
  --limit_train_batches=100 ^
  --limit_val_batches=20 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=3 ^
  --disable_mp ^
  --thr 0.1

pause
