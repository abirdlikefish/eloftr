@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v3 FULL FINETUNE (a.k.a. v3.1): same hyperparameters as v2
REM (run_roadscene_v2_modemb.bat) EXCEPT for these intentional changes
REM (all coming from v3 config or this bat). See the v3 config docstring
REM for the rationale behind each value.
REM   - FREEZE_BACKBONE: false -> true   (16M -> ~5.7M trainable)
REM   - FREEZE_BN:       false -> true   (lock pretrained running stats)
REM   - EARLY_STOPPING:  false -> true   (patience=4 on val precision@3px)
REM   - CANONICAL_LR:    8e-3  -> 8e-3   (kept; TRUE_LR = 5e-4 at bs=4)
REM   - WARMUP_STEP:     1875  -> 1      (effectively disabled; pretrained init)
REM   - MSLR_MILESTONES: [8,12,16,20,24] -> [5, 7]  (decay before EarlyStop fires)
REM   - max_epochs:      20    -> 20     (kept; EarlyStopping cuts it short)
REM
REM Step1 (InfoNCE) and Step2 (modemb) remain enabled via config inheritance,
REM so v3 is a strict superset of v2 in terms of model features.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v3_combined.py ^
  --exp_name=roadscene_v3_combined ^
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
