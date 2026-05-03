@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v1 (Step 1) SMALL: 100 train batches, 20 val batches, 3 epochs.
REM Sanity check that symmetric InfoNCE actually helps before launching the
REM full finetune. batch_size>=2 is required (see run_roadscene_v1_debug.bat).
REM
REM Watch for in TensorBoard:
REM   - loss_contrast trends downward (initial ~8-10 -> ~3-4 after 3 epochs)
REM   - loss_c stays in baseline range (0.01 * loss_contrast remains < 10% of loss_c)
REM   - precision@3px on val matches or beats baseline 0.471
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v1_contrast.py ^
  --exp_name=roadscene_v1_small ^
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
