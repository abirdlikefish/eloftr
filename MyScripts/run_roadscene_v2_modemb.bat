@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v2 (Step 2) FULL FINETUNE: same hyperparameters as the baseline /
REM v1 finetune so that all three runs (baseline / v1 / v2) are
REM apples-to-apples in Step 3. Only main_cfg_path / exp_name differ.
REM
REM v2 = v1 (symmetric InfoNCE) + learnable modality embedding (zeros init).
REM
REM batch_size=4 to match run_roadscene_v1_contrast.bat. With batch_size>=2
REM the InfoNCE negative pool spans cross-scene tokens; modality emb works
REM with any batch_size but is logged regardless.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v2_modemb.py ^
  --exp_name=roadscene_v2_modemb ^
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
