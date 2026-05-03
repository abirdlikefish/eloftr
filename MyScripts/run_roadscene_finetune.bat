@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM RoadScene finetune: full train/val split per epoch starting from the
REM official outdoor weights. ModelCheckpoint monitors precision@3px.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full.py ^
  --exp_name=roadscene_finetune ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=4 ^
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
