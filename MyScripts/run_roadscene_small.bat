@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM RoadScene small: 100 train batches, 20 val batches, 3 epochs. Used to
REM sanity-check the full training loop (loss curve, plot, ckpt) before the
REM long finetune. Drop --num_workers to 0 if Windows DataLoader fails.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full.py ^
  --exp_name=roadscene_small ^
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
