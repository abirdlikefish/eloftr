"""v19 single-card cfg, used for the ~5-min sanity smoke before 4-GPU DDP ship.

Inherits eloftr_full_v19_pose_msyn_rot25_only_ddp.py (which itself inherits
v18 ddp + 1 rot_deg knob change) and only adjusts LR / WARMUP so single-card
bs=4 reaches the SAME TRUE_LR (1.25e-4) and actual-WARMUP-step count as the
4-GPU run. Math identical to v14/v18 singlecard:

  _scaling_singlecard = (4 * 1) / 64 = 0.0625
  TRUE_LR  = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4 (matches DDP)
  actual WARMUP step = WARMUP_STEP / _scaling = 1800 / 0.0625 = 28800 step

The huge nominal WARMUP_STEP is harmless in smoke because PL's --max_epochs=1
--limit_train_batches=10 only runs 10 steps, all in linear-warmup phase. The
gate is "no NaN + b_ids >= 30 + H_vis valid + no IndexError on backward path",
not loss convergence.

Note: H aug (cfg.DATASET.MGDPT_HOMOGRAPHY_AUG=True with rot_deg=25) is
inherited from the ddp parent; data.py train-only gate keeps val/test
unaugmented even in the single-card smoke.
"""
from configs.loftr.eloftr_full_v19_pose_msyn_rot25_only_ddp import cfg

# Single-card reverse-scaling: keep TRUE_LR == DDP TRUE_LR
cfg.TRAINER.CANONICAL_LR = 2e-3
cfg.TRAINER.WARMUP_STEP = 1800
