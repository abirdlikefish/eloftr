@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v2 (Step 2) SMALL: 100 train batches, 20 val batches, 3 epochs.
REM Same recipe as run_roadscene_v1_small.bat, only main_cfg_path / exp_name
REM differ. Lets us compare v2 vs v1 with apples-to-apples hyperparameters.
REM
REM Watch in TensorBoard:
REM   - train/mod_emb_ir_norm  : should rise from 0 to roughly 0.5 - 5 by
REM                              the end of 3 epochs. If it stays < 0.01,
REM                              the embedding is "dead" -- consider switching
REM                              MODALITY_EMB_INIT to 'normal_0.02' next time.
REM   - train/mod_emb_vis_norm : same expectation.
REM   - val precision@3px      : should match or beat v1 (and the baseline).
REM   - val loss_c             : stays in baseline range (modemb does NOT
REM                              destabilise the focal loss at step 0 thanks
REM                              to zero init).
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v2_modemb.py ^
  --exp_name=roadscene_v2_small ^
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
