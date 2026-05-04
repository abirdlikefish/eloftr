@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v4 DEBUG: smoke test of "v3 minus FREEZE_BACKBONE plus softer LR / ES".
REM
REM ===== v4 acceptance checklist (look for these in the startup log) =====
REM (a) NO 'Froze backbone: ...' line                  <-- v4 unfreezes backbone
REM (b) Froze all BatchNorm2d layers (eval-mode + no-grad)
REM (c) Trainable params: ~16.0M / Total: ~16.0M       <-- KEY signal vs v3 (~5.7M)
REM (d) missing_keys (kept at init value): ['modality_emb_ir', 'modality_emb_vis']
REM (e) EarlyStopping enabled (monitor=precision@3px, mode=max, patience=8)
REM
REM In TensorBoard, after a few steps you should see ALL of:
REM   - train/loss_c, train/loss_f, train/loss_l   (baseline)
REM   - train/loss_contrast, train/loss_i2v, train/loss_v2i, train/n_pos   (step1 from v1)
REM   - train/mod_emb_ir_norm, train/mod_emb_vis_norm   (step2 from v2)
REM Confirming v4 still inherits v1 + v2 cumulatively, only the v3 freeze + LR
REM knobs are different.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v4_combined.py ^
  --exp_name=roadscene_v4_debug ^
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
