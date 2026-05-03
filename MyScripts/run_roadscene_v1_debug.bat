@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v1 (Step 1) DEBUG: very fast smoke test of the symmetric InfoNCE.
REM batch_size=2 is the minimum needed for cross-batch negatives. With bs=1
REM the InfoNCE negative pool collapses to in-image only and the test would
REM not actually exercise the cross-scene path.
REM
REM Expect to see four NEW scalars in TensorBoard within the first step:
REM   - train/loss_contrast    (~ log(B*L) approx 8-10 initially)
REM   - train/loss_i2v
REM   - train/loss_v2i
REM   - train/n_pos            (number of positive matches per batch)
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v1_contrast.py ^
  --exp_name=roadscene_v1_debug ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=0 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=1 ^
  --limit_train_batches=10 ^
  --limit_val_batches=2 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=1 ^
  --disable_ckpt ^
  --disable_mp ^
  --thr 0.1

pause
