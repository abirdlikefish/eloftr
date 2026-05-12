"""v13 single-card cfg, used for the ~5-min sanity smoke before 4-GPU DDP ship.

Inherits eloftr_full_v13_pose_msyn_ddp.py and only adjusts LR / WARMUP so
single-card bs=4 reaches the SAME TRUE_LR (1.25e-4) and 1-ep warmup count
as the 4-GPU run.

Math:
  _scaling_singlecard = (4 * 1) / 64 = 0.0625
  TRUE_LR  = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4  (matches DDP)
  actual WARMUP step = WARMUP_STEP / _scaling = 1800 / 0.0625 = 28800 step

The huge nominal WARMUP_STEP (1800 cfg -> 28800 actual) is only ever
"used" in the 1-ep smoke because PL's --max_epochs=1 --limit_train_batches=10
runs only 10 steps total; the actual learning rate during smoke stays
linear-warmup near zero. The gate is "no NaN + b_ids >= 50", not
loss-convergence.
"""
from configs.loftr.eloftr_full_v13_pose_msyn_ddp import cfg

# Single-card reverse-scaling: keep TRUE_LR == DDP TRUE_LR
cfg.TRAINER.CANONICAL_LR = 2e-3
cfg.TRAINER.WARMUP_STEP = 1800
