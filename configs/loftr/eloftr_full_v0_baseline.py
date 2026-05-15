"""v0 (baseline): pure finetune from eloftr_outdoor.ckpt on RoadScene.

零模型修改, 所有 v1-v15 引入的 toggle 保持 default False:
  - USE_CONTRASTIVE      = False  (v1 引入)
  - USE_MODALITY_EMB     = False  (v2 引入)
  - USE_EDGE_INPUT       = False  (v7 引入)
  - USE_CLAHE_IR         = False  (v7 引入)
  - USE_MSBN             = False  (v8 引入)
  - BACKBONE_IN_CHANNELS = 1      (v7 改 2)

模型架构跟 eloftr_full.py + outdoor.ckpt 完全一致, ckpt -> model load 是
byte-clean strict load (无 inflate / 无 missing_keys / 无 unexpected_keys).

数据增强侧启用 v12 同款 aggressive 强度的单侧 H aug:
  - ROAD_HOMOGRAPHY_DUAL = False   单侧 (跟 v12 dual=True 形成单变量 ablation)
  - ROAD_HOMOGRAPHY_PROB = 1.0     每 sample 必触发 (单侧无共视塌陷)
  - ROAD_HOMOGRAPHY_KWARGS aggressive 4 参数 (跟 v12 完全一致)

代码路径 src/datasets/roadscene.py:558-578: H_ir = I (恒等),
H_vis = _random_homography(aggressive), 只对 image1 (VIS) 做 cv2.warpPerspective.
v0 vs v12 是干净的 dual=False vs dual=True 单变量 ablation;
v0 vs v1 是干净的 USE_CONTRASTIVE 单变量 ablation (但 v1 当前用 weak preset,
新对齐 ablation 需要 ship v1.1 = v1 + aggressive H aug, 见 results/eval_summary).

Schedule 修正 (NOT 沿用 eloftr_full.py 默认):
  eloftr_full.py 的 CANONICAL_LR=8e-3 + WARMUP_STEP=1875 是为 MegaDepth bs=64
  设计的; 在 RoadScene bs=2 单卡反向缩放后 WARMUP_actual = 1875 / (2/64) =
  60000 step 远超 max_epochs=30 总 step ~3000, 训练全程都卡在 warmup 内
  LR 几乎为 0. 跟 v3 cfg docstring 反复指出的 "原 1875 不适合 RoadScene"
  结论一致 (configs/loftr/eloftr_full_v3_combined.py 第 21-22 行).

  v0 沿用 v4 REVISION 2 实测验证的 RoadScene 单卡 finetune schedule:
    CANONICAL_LR = 4e-3       -> TRUE_LR = 4e-3 * 2/64 = 1.25e-4
                                 (跟 v4/v9/v10/v13 实测验证的 finetune LR 一致)
    WARMUP_STEP  = 2          -> actual = 2 / 0.03125 = 64 step (~0.6 ep warmup)
    MSLR         = [10,15,20] -> 第一次 decay 在 ep 10 让 LR 在 plateau 前衰减
    EARLY_STOPPING = True / PATIENCE = 8
                                 (RoadScene val=22 pair 单点波动 ±0.02-0.03,
                                  需要高 patience 容忍 val 噪声)
    N_SAMPLES_PER_SUBSET 沿用 default=200 (100 step/ep 配 bs=2)

bat 配套: --max_epochs=30 --batch_size=2 --gpus=1 --check_val_every_n_epoch=1

LR schedule timeline (假设跑满 max_epochs=30, 实际可能被 ES 提前停):
  ep 0      step 0-64    warmup linear 0 -> 1.25e-4
  ep 0.6-10 step 64-1000 full LR 1.25e-4
  ep 10     step 1000    MSLR #1, LR x= 0.5 -> 6.25e-5
  ep 15     step 1500    MSLR #2, LR x= 0.5 -> 3.13e-5
  ep 20     step 2000    MSLR #3, LR x= 0.5 -> 1.56e-5
  ep 20-30  step 2000-3000 低 LR 微调
"""
from configs.loftr.eloftr_full import cfg

# === 数据增强: 单侧 aggressive H aug (跟 v12 共用 KWARGS, DUAL=False 单变量区分) ===
cfg.DATASET.ROAD_HOMOGRAPHY_DUAL = False
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08

# === Schedule: 沿用 v4 REVISION 2 RoadScene 实测验证的 finetune schedule ===
cfg.TRAINER.CANONICAL_LR = 4e-3            # TRUE_LR = 1.25e-4 at bs=2 (反向缩放 2/64)
cfg.TRAINER.WARMUP_STEP = 2                # actual ~64 step (~0.6 ep)
cfg.TRAINER.MSLR_MILESTONES = [10, 15, 20]
cfg.TRAINER.EARLY_STOPPING = True
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 8
