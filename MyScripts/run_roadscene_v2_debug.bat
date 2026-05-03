@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v2 (Step 2) DEBUG: very fast smoke test of modality embedding + InfoNCE.
REM batch_size=2 is the minimum needed for cross-batch negatives.
REM
REM ===== Step 2 acceptance checklist (look at the console + TensorBoard) =====
REM (a) Startup log MUST contain exactly these two missing keys:
REM       missing_keys (kept at init value): ['modality_emb_ir', 'modality_emb_vis']
REM     and NO unexpected_keys. If either condition fails, abort and check
REM     that the official ckpt path is correct.
REM (b) TensorBoard scalars from Step 1 are still present:
REM       train/loss_contrast, train/loss_i2v, train/loss_v2i, train/n_pos
REM (c) NEW Step 2 scalars appear from step 0:
REM       train/mod_emb_ir_norm   -> starts at 0 (zeros init), should grow
REM       train/mod_emb_vis_norm  -> starts at 0, should grow
REM (d) Step 0 emb_norm == 0 confirms the math: feat + 0 = feat, so step 0
REM     is byte-identical to v1.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v2_modemb.py ^
  --exp_name=roadscene_v2_debug ^
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
