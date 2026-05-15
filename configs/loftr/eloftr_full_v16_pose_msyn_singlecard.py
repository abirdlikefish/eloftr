"""v16 single-card cfg, used for the ~5-min sanity smoke before 4-GPU DDP ship.

Inherits eloftr_full_v16_pose_msyn_ddp.py and only adjusts LR / WARMUP so
single-card bs=4 reaches the SAME TRUE_LR (1.25e-4) and actual-WARMUP-step
count as the 4-GPU run.

Math:
  _scaling_singlecard = (4 * 1) / 64 = 0.0625
  TRUE_LR  = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4 (matches DDP)
  actual WARMUP step = WARMUP_STEP / _scaling = 1800 / 0.0625 = 28800 step

The huge nominal WARMUP_STEP (1800 cfg -> 28800 actual) is only ever
"used" in the 1-ep smoke because PL's --max_epochs=1 --limit_train_batches=10
runs only 10 steps total; the actual learning rate during smoke stays in
linear-warmup phase near zero. The gate is "no NaN + b_ids >= 50 + mod_emb
正确初始化为 0", not loss-convergence.

v16 specific smoke gate (vs v14 smoke):
  - 期望 ep0 step 1 TB 出现新 scalar: train/mod_emb_ir_norm / vis_norm
  - mod_emb_ir/vis_norm 在 step 0 = 0.0 exactly (zeros init 数学保证)
  - smoke 10 step 后 norm 微微 > 0 (gradient 推动开始, 健康信号)
  - 若 mod_emb_ir/vis_norm 在 smoke 10 step 仍 = 0.0 -> 检查 USE_MODALITY_EMB
    cfg merge 是否真的传到 matcher (loftr.py:47-62 应该见 init_mode='zeros')
  - train_loss 跟 v14 ep0 step 1 同水平 (zeros init 让 step 0 byte-identical)
"""
from configs.loftr.eloftr_full_v16_pose_msyn_ddp import cfg

# Single-card reverse-scaling: keep TRUE_LR == DDP TRUE_LR
cfg.TRAINER.CANONICAL_LR = 2e-3
cfg.TRAINER.WARMUP_STEP = 1800
