"""v15 single-card cfg, used for the ~5-min sanity smoke before 4-GPU DDP ship.

Inherits eloftr_full_v15_pose_msyn_ddp.py and only adjusts LR / WARMUP so
single-card bs=4 reaches the SAME TRUE_LR (1.25e-4) and actual-WARMUP-step
count as the 4-GPU run.

Math:
  _scaling_singlecard = (4 * 1) / 64 = 0.0625
  TRUE_LR  = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4 (matches DDP)
  actual WARMUP step = WARMUP_STEP / _scaling = 1800 / 0.0625 = 28800 step

The huge nominal WARMUP_STEP (1800 cfg -> 28800 actual) is only ever
"used" in the 1-ep smoke because PL's --max_epochs=1 --limit_train_batches=10
runs only 10 steps total; the actual learning rate during smoke stays in
linear-warmup phase near zero. The gate is "no NaN + b_ids >= 50 + n_pos >= 30",
not loss-convergence.

v15 specific smoke gate (vs v14 smoke):
  - 期望 ep0 step 1 TB 出现新 scalar: loss_contrast / loss_i2v / loss_v2i / n_pos
  - loss_contrast 初始 ~ log(B*L1) = log(4*6400) ~ 10.1 (InfoNCE 等概率猜的上限)
    随训练快速下降到 [3, 8]; smoke 10 step 看不出趋势但应该 < 10.5
  - n_pos > 30 / batch -> 有足够 contrastive 信号
  - 若 n_pos = 0 几乎所有 step -> contrastive 退化, 检查 spv_b_ids 是否
    在 MegaDepth pose path 下被正确填 (sanity check 5 应该早就 PASS 过)
"""
from configs.loftr.eloftr_full_v15_pose_msyn_ddp import cfg

# Single-card reverse-scaling: keep TRUE_LR == DDP TRUE_LR
cfg.TRAINER.CANONICAL_LR = 2e-3
cfg.TRAINER.WARMUP_STEP = 1800
