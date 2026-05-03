@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v3 SMALL: 100 train batches, 20 val batches, 3 epochs (smoke test).
REM
REM Key things to compare against v2_small in TensorBoard:
REM   - val precision@3px curve should NO LONGER drop after epoch 4. Either
REM     EarlyStopping fires (patience=4) once it plateaus, or the curve
REM     plateaus instead of degrading (frozen backbone caps overfit capacity).
REM   - mod_emb_*_norm growth should be slower/saturating compared to v2,
REM     because the focal-loss "shortcut" through scale amplification is
REM     weaker once backbone is frozen.
REM   - loss_contrast / loss_i2v / loss_v2i should all still appear
REM     (confirming v3 inherits v1's InfoNCE on top of v2's modemb).
REM   - LR should jump to ~5e-4 within the first step (WARMUP_STEP=1) and
REM     drop to ~2.5e-4 after epoch 5 (MSLR milestone) -- only visible in
REM     run_roadscene_v3_combined.bat which trains long enough.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v3_combined.py ^
  --exp_name=roadscene_v3_small ^
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
