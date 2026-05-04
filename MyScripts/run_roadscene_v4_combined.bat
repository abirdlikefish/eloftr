@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v4 FULL FINETUNE: a softer v3.
REM
REM ============================================================================
REM REVISION 2 (BN-only fix): roll back LR-only revision (CANONICAL_LR back to
REM   2e-3, max_epochs back to 30); switch FREEZE_BN True->False AND set new
REM   FREEZE_BACKBONE_BN=True so only backbone BN is pinned, fine_preprocess BN
REM   (2 layers) is now trainable + train-mode.
REM ----------------------------------------------------------------------------
REM Previous LR-only revision (TRUE_LR ~2.4e-5, 80 epochs) was refuted: result
REM dropped to ~0.684 (worse than original v4's 0.711). Hypothesis updated:
REM the gap to v2 (~0.80) is dominated by frozen fine_preprocess BN, not LR.
REM This revision keeps original v4's optimization regime intact and only
REM toggles the BN freeze granularity. See v4 config docstring REVISION 2.
REM ============================================================================
REM
REM Diff vs v3 (run_roadscene_v3_combined.bat). All knobs come from the v4
REM config file or this bat. See the v4 config docstring for full rationale.
REM   - FREEZE_BACKBONE:    true  -> false  (16M trainable, restoring v2's capacity)
REM   - FREEZE_BN:          true  -> false  (REVISION 2: release fine BN; was true in REVISION 1)
REM   - FREEZE_BACKBONE_BN: --    -> true   (REVISION 2 NEW: pin backbone BN only)
REM   - EARLY_STOPPING:     true  -> true   (kept)
REM   - ES_PATIENCE:        4     -> 8      (mid-LR runs need more room to recover)
REM   - CANONICAL_LR:       8e-3  -> 2e-3   (REVISION 2: rolled back to original v4)
REM   - WARMUP_STEP:        1     -> 2      (auto-scaled to ~32 steps = ~1 epoch)
REM   - MSLR_MILESTONES:    [5,7] -> [10,15,20]  (three decays after full LR)
REM   - max_epochs:         20    -> 30     (REVISION 2: rolled back to original v4)
REM
REM Effective TRUE LR trajectory at bs=4 (~40 train batches/epoch), post-REVISION 2:
REM   step    0       LR ramps from 0  (WARMUP_STEP=2 -> 32 steps after _scaling)
REM   step  ~32       LR = 1.25e-4 (full)
REM   epoch 10 end    LR = 6.25e-5 (MSLR step #1, gamma=0.5)
REM   epoch 15 end    LR = 3.13e-5 (MSLR step #2)
REM   epoch 20 end    LR = 1.56e-5 (MSLR step #3)
REM   epoch 20-30     LR stays 1.56e-5
REM   EarlyStopping (patience=8) most often fires before epoch 30.
REM
REM Step1 (InfoNCE) and Step2 (modemb) remain enabled via v3 -> v2 -> v1
REM inheritance, so v4 is a strict superset of v2 in features. The only
REM differences from v2 are FREEZE_BACKBONE_BN + EarlyStopping + the new LR schedule.
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v4_combined.py ^
  --exp_name=roadscene_v4_combined ^
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
  --max_epochs=30 ^
  --disable_mp ^
  --thr 0.1

pause
