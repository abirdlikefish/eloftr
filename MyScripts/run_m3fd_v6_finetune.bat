@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v6 M3FD SLOW-LR FINETUNE: continue from v5 ep9 ckpt to lift in-domain p@1px.
REM
REM ============================================================================
REM Why v6 = v5 ckpt resume (not retrain from pretrained)
REM
REM   v5 实测 (eloftr-cross-modal-experiments §5.6):
REM     In-domain (M3FD test 210):  p@1=0.4352  p@3=0.8072  p@5=0.8777  mpe=2.07
REM     OOD (RoadScene test 22):    p@1=0.1858  p@3=0.5989  p@5=0.7825  mpe=3.39
REM     -> v5 OOD beats v2 OOD on EVERY metric (v5 learned generalizable rep)
REM     -> v5's only gap is in-domain p@1 (v2's 0.7567 is RoadScene memorisation)
REM
REM   Root cause of v5 in-domain p@1 gap: v5 effective LR in epoch 0-3 is
REM   ~6x v2's (~1.25e-4 vs ~2.08e-5). Fine sub-pixel L2 offset oscillates
REM   under high LR and never settles to <1px precision.
REM
REM   v6 fix: load v5 ep9 ckpt (BN converged + general backbone preserved),
REM   then continue training with v2-style slow LR (TRUE_LR=2.5e-5) for
REM   50 epoch. Additive-not-subtractive: keep v5's OOD wins, only refine
REM   in-domain p@1.
REM
REM ============================================================================
REM Diff vs v5 (run_m3fd_v5_combined.bat)
REM
REM   - main_cfg:           v5_m3fd.py       -> v6_finetune.py       (slow LR override)
REM   - exp_name:           m3fd_v5_combined -> m3fd_v6_finetune
REM   - --ckpt_path:        weights\eloftr_outdoor.ckpt
REM                         -> logs\tb_logs\m3fd_v5_combined\version_0\checkpoints\
REM                            epoch=9-precision@1px=0.418-precision@3px=0.789-precision@5px=0.864.ckpt
REM       This repo's --ckpt_path (train.py:72 + lightning_loftr.py:64-67) only
REM       loads model weights via load_state_dict(strict=False). Optimizer state,
REM       LR scheduler state, epoch counter, global_step are ALL fresh (NOT a
REM       PL --resume_from_checkpoint). v6 starts at epoch 0 with v5 weights +
REM       v6's own LR schedule, which is exactly what we want (no inherited
REM       1.56e-5 from v5's MSLR decay).
REM   - CANONICAL_LR (in v6 cfg):       2e-3       -> 4e-4   (TRUE_LR=2.5e-5)
REM   - WARMUP_STEP (in v6 cfg):        20         -> 50     (800 abs step ~0.85 epoch)
REM   - MSLR_MILESTONES (in v6 cfg):    [3,5,7]    -> [15,25,35]
REM   - ES_PATIENCE (in v6 cfg):        3          -> 12
REM   - max_epochs (here):              10         -> 50     (~47k abs step total)
REM   - Everything else (sampler, freeze, contrastive, modemb) inherited via
REM     v5 config from v4 REVISION 2. Do NOT override sampler in v6 config
REM     -- yacs would re-merge and might confuse readers; the inheritance
REM     chain is enough.
REM
REM ============================================================================
REM Effective TRUE LR trajectory at bs=4 (~945 train batches/epoch)
REM
REM   step      0       LR ramps from 0  (WARMUP=50 -> 800 abs step after _scaling)
REM   step    800       LR = 2.5e-5 (full; ~0.85 epoch in)
REM   epoch  15 end     LR = 1.25e-5  (MSLR step #1, ~14k step in)
REM   epoch  25 end     LR = 6.25e-6  (MSLR step #2, ~23k step)
REM   epoch  35 end     LR = 3.13e-6  (MSLR step #3, ~33k step)
REM   epoch  35-50      LR stays 3.13e-6 (deep refinement, ~47k step total)
REM   EarlyStopping (patience=12) likely fires between epoch 25-45.
REM
REM   Total training time estimate: ~20h on RTX 4060 (v5 was 3.9h for 9.45k step,
REM   v6 is 5x longer at ~47k step, same per-step cost).
REM
REM ============================================================================
REM Validation gates -- read launch log carefully
REM
REM   1. "Load 'logs\tb_logs\m3fd_v5_combined\...epoch=9-...ckpt' as pretrained
REM      checkpoint" MUST appear. If missing, --ckpt_path was not picked up.
REM
REM   2. "missing_keys" / "unexpected_keys" both should be EMPTY after the
REM      load_state_dict(strict=False) call (v5 ckpt has the full v1+v2+v3+v4+v5
REM      graph, identical to v6's architecture). If "modality_emb_ir /
REM      modality_emb_vis" missing, you accidentally pointed at a pre-v2 ckpt.
REM
REM   3. Epoch 0 val end: p@1px MUST be >= 0.4181 (v5 terminal).
REM      Reasoning: epoch 0 trains under 0 -> 2.5e-5 warmup ramp, very gentle,
REM      so val p@1 should be within +/- 0.005 of v5's 0.4181. If MUCH lower
REM      (e.g. 0.30), graph mismatch silently re-init some weights -- check
REM      missing_keys list closely.
REM
REM   4. PL progress bar Epoch 0: must show "... 945/1155" (sampler override
REM      inherited from v5 via `from v5_m3fd import cfg`). If "50/260", the
REM      inheritance broke -- check v6 config does NOT re-import from v4.
REM
REM ============================================================================
REM Expected outcomes
REM
REM   Strong success (M3FD val p@1 >= 0.55 + RoadScene OOD p@1 >= 0.25):
REM     Slow-LR refinement closes the gap. Ship v6 as final cross-modal
REM     finetune. Optional: v7 MSBN to push p@1 toward 0.65+.
REM
REM   Weak success (M3FD val p@1 in 0.45-0.55, OOD didn't move):
REM     LR helped but fine path's modality-blindness is the structural
REM     ceiling. Move to v7 MSBN per cross-modal-experiments §6 path E1.
REM
REM   Failure (p@1 stuck at 0.42 from epoch 0 onward):
REM     Either ckpt didn't load (gate #1 failed) or weights loaded but the
REM     full v6 schedule has zero learning effect (LR too low). Try v6.5
REM     with CANONICAL_LR raised to 6e-4 ~ 8e-4.
REM
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v6_finetune.py ^
  --exp_name=m3fd_v6_finetune ^
  --ckpt_path=logs\tb_logs\m3fd_v5_combined\version_0\checkpoints\epoch=9-precision@1px=0.418-precision@3px=0.789-precision@5px=0.864.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=4 ^
  --num_workers=6 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=50 ^
  --limit_train_batches=1.0 ^
  --limit_val_batches=1.0 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=50 ^
  --disable_mp ^
  --thr 0.1

pause
