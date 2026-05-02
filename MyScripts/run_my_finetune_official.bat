@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

python train.py ^
  data\MyTrainData\config\my_train_debug.py ^
  configs\loftr\eloftr_full.py ^
  --exp_name=my_train_finetune_official ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=4 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=2 ^
  --log_every_n_steps=500 ^
  --limit_val_batches=0.25 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=30 ^
  --disable_mp ^
  --thr 0.1

pause
