@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v5 DEBUG: smoke test of "v4 REVISION 2 architecture + M3FD data".
REM
REM ===== v5 acceptance checklist (look for these in the startup log) =====
REM (a) [rank 0]: building RoadSceneDataset (dataset_name=M3FD) from
REM     data/M3FD_Detection/index/train_pairs.txt
REM (b) Same line for val_pairs.txt
REM (c) NO 'Froze backbone: ...' line                         (FREEZE_BACKBONE=False)
REM (d) NO 'Froze all BatchNorm2d layers' line                (FREEZE_BN=False)
REM (e) Froze backbone BatchNorm2d layers (eval-mode + no-grad);
REM     fine_preprocess BN remains trainable                  (FREEZE_BACKBONE_BN=True)
REM (f) Trainable params: ~15.99M / Total: ~16.0M             (KEY signal vs v3 ~5.7M)
REM (g) missing_keys (kept at init value): ['modality_emb_ir', 'modality_emb_vis']
REM (h) EarlyStopping enabled (monitor=precision@3px, mode=max, patience=3)
REM
REM In TensorBoard, after a few steps you should see ALL of:
REM   - train/loss_c, train/loss_f, train/loss_l   (baseline)
REM   - train/loss_contrast, train/loss_i2v, train/loss_v2i, train/n_pos   (step1 from v1)
REM   - train/mod_emb_ir_norm, train/mod_emb_vis_norm   (step2 from v2)
REM Confirming v5 still inherits v1+v2+v3+v4 cumulatively, only the data
REM source + schedule knobs are different.
REM
REM ===== M3FD-specific sanity (optional follow-up) =====
REM M3FD has 6.5% non-1024x768 images (smallest 400x280, coarse 35x60).
REM In a 10-batch debug at bs=2 the chance of hitting at least one small
REM image is ~23%. If this debug passes BUT subsequent _small / _combined
REM training crashes with mask-related errors like:
REM     Calculated padded input size per channel: (... x 0)
REM     Sizes of tensors must match except in dimension 1/2
REM bump --limit_train_batches from 10 to 50-100 here and rerun debug to
REM force coverage of the small-size samples and confirm transformer.py
REM agg_size guard (line 159-163) handles 35x60 coarse mask correctly.
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v5_m3fd.py ^
  --exp_name=m3fd_v5_debug ^
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
