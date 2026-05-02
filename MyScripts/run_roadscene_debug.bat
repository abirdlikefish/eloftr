@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM RoadScene debug: 10 train batches per epoch, 2 val batches, 1 epoch.
REM --ckpt_path only loads the official weights as initialisation; it does NOT
REM restore optimizer/scheduler/epoch (use --resume_from_checkpoint for that).
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full.py ^
  --exp_name=roadscene_debug ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=1 ^
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
