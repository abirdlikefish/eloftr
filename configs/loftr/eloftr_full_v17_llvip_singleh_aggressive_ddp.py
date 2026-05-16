"""v17 = v14 best ckpt finetune + LLVIP + aggressive single-side H aug + DDP.

------------------------------------------------------------------------------
设计目标
------------------------------------------------------------------------------
在 v14 已经做过跨模态训练 (Megadepth_Syn pose 监督, ep12 best, auc@10=0.273)
的基础上, finetune 到 LLVIP 像素对齐 IR-VIS 数据集, 用 v11/v12 同款
aggressive single-side H aug (rot=25 / scale=0.75-1.25 / trans=0.12 / persp=0.08)
让模型学会 eval-time aug 场景 (与 eval_roadscene_all_singleh_finetuned.bat 协议
完全一致).

v17 不是普通 finetune, 是"多重 distribution shift transfer learning":
  - dataset:   Megadepth_Syn (155K合成 IR-VIS, 跨视角) -> LLVIP (12K真实 IR-VIS, 像素对齐)
  - 监督:      pose (auc@10) -> H 监督 (precision@3px, IR-VIS path)
  - aug:       v14 无 H aug (cross-view 自然有 viewpoint 变化)
               -> v17 aggressive single-side H (3x v0-v10 强度, image1=VIS only)
  - geometry:  cross-view (有 parallax) -> monocular pixel-aligned (无 parallax)
  - BN stats:  Megadepth 综合分布 -> LLVIP 街景 IR-VIS 分布

------------------------------------------------------------------------------
继承链 + override 数 (3 组 = 10 项 cfg override)
------------------------------------------------------------------------------
parent: eloftr_full_v14_pose_msyn_ddp.py (v14 ddp cfg)
                                          |
                                  + 4 项 H aug (ROAD_HOMOGRAPHY_KWARGS)
                                  + 2 项 sampler (DDP image-level pre-shard 配套)
                                  + 4 项 finetune schedule V2 (中庸版)
                                          |
                                v17 cfg (this file)

v14 自动继承的 (NOT override here, 顺带列出避免审查时混淆):
- USE_CONTRASTIVE / USE_MODALITY_EMB / USE_EDGE_INPUT / USE_CLAHE_IR /
  USE_MSBN / BACKBONE_IN_CHANNELS=2: 全部 default False (v14 是 B-pure stack
  短路继承自 eloftr_full.py, 跟 v9_e2e -> v10 链解耦)
- IMG_RESIZE: 由 data cfg llvip_trainval.py 设 ROAD_IMG_RESIZE=640 (跟 v14 一致,
  finetune 不破坏起点)
- bs=4, --gpus=4 -> effective_bs=16 -> _scaling=0.25 (TRUE_LR=CANONICAL_LR*0.25,
  actual WARMUP step=WARMUP_STEP/0.25)
- ENABLE_PLOTTING=False (TB 文件 ~220MB, 关 train figure)
- EVAL_TIMES=1 (v14 显式 reset 5->1; v17 走 IR-VIS path _compute_roadscene_metrics
  根本不调 RANSAC, 这个字段对 v17 train val 不生效, 仅防御性继承)
- NPE: 不显式设, train.py:129-130 fallback [832,832,832,832] (v14 NPE bug
  post-mortem, v17 同样不踩坑)

------------------------------------------------------------------------------
Schedule V2 推导 (取 V1 保守 / V2 中庸 / V3 激进 中庸)
------------------------------------------------------------------------------
v14 cold start 对照: CANONICAL_LR=5e-4 / WARMUP=450 (actual 1800 step) /
  MSLR=[6,10,14] / ES patience=5 / max_ep=18.

V2 (本 cfg) 设计:
- CANONICAL_LR=4e-4 (TRUE_LR=1e-4 = v14 cold start 的 0.8x):
    * 不取 v14 同 LR (5e-4 / TRUE 1.25e-4): 保护 v14 cross-modal feature, 避免
      aggressive aug 把已学好的 IR-VIS alignment 冲坏.
    * 不取 V1 保守 LR (2.5e-4 / TRUE 6.25e-5): max_ep=12 窗口内, 6.25e-5 适应
      新 H aug task 偏慢, model 可能跑不到最优.
- WARMUP_STEP=75 (actual 300 step ~0.4 ep):
    * 比 V1 100 (400 step) 短: finetune 不需要长 warmup (model 已 well-trained),
      300 step 足够吸收 aggressive aug 初期震荡 + BN running stats 在新数据集
      上稳定.
- MSLR_MILESTONES=[2,4,6] 配 ES patience=5:
    * best_ep 估 ep 2-3 (v14 已 cross-modal align, LLVIP 像素对齐很简单, raw
      load 就接近最优, 主要 epoch 用来适应 H aug).
    * ES 在 best_ep + 5 = ep 7-8 触发. milestones [2,4,6] 让三段 LR drop 都早
      于 ES 窗口:
        MSLR1 (ep 2): 跟 best_ep 同时或紧随, drop LR 看是否再涨 (二次 fine-tune)
        MSLR2 (ep 4): 三次 fine-tune
        MSLR3 (ep 6): 跟 ES 触发前留 1 ep 缓冲
    * 比 V1 [3,5,7] 更早 drop, cover best_ep=ep 1 的极端情况 (v14 raw load
      可能直接最优).
- EARLY_STOPPING_PATIENCE=5 (沿用 v14): finetune setting 标准值, 给 LR drop 后
  reactivation 充足空间.

------------------------------------------------------------------------------
Sampler 配置 (DDP image-level pre-shard 配套, 关键 bug 防御)
------------------------------------------------------------------------------
LLVIP 走 IR-VIS short-circuit (data.py:289-297) 在 mode='train' 时把 dataset
切到 per-rank, 每 rank 的 dataset.__len__() = 12025/4 = 3006 (n_subset=1
的 ConcatDataset).

sampler.py:53-61 在 SB_SUBSET_SAMPLE_REPLACEMENT=False 时:
  if len_subset >= n_samples_per_subset: rand_tensor[:n_samples_per_subset]
  else: padding with replacement (3006 unique + 9019 duplicate = 12025)

所以 N_SAMPLES_PER_SUBSET 必须 = 3006 (= 全集/world_size), 跟 v10/v11/v12 ddp
cfg 设 28750 (= 115000/4) 同样的 DDP 修正方式. **设 12025 会 4x 重复 sample,
浪费 wall-clock (10h 而不是 2.8h).**

单卡场景 (本地 sanity-only) 时 dataset.__len__() = 12025 (no pre-shard);
3006 会让 sampler 只取 25% 数据, 但 sanity-only 只验启动日志 + shape +
loss 不爆, 25% 也够. 如果要单卡跑完整 12025, 单独建 v17 singlecard cfg
override N_SAMPLES_PER_SUBSET=12025.

------------------------------------------------------------------------------
Wall-clock 估算 (服务器 4×3090 DDP, V2 schedule)
------------------------------------------------------------------------------
v14 实测黄金基准 (msyn_v14_pose_ddp 11.892 h / 41000 step):
  联立解 train step time t + val step time ~1.2t (megadepth val 含 RANSAC):
    t ≈ 1.0 s/step (含 sync_bn + ckpt save + DataLoader prefetch)

v17 batch 数:
  - train: 12025/4 (DDP image-level pre-shard) /bs=4 = 752 step/rank/ep
  - val: 3463/4 (DistributedSampler) /bs=4 * --limit_val_batches=0.5 = 108 step/rank/ep
  - per ep: 752 + 108 = 860 step

v17 时间:
  - train per ep: 752 * 1.0 = 752 s = 12.5 min
  - val per ep: 108 * 0.6 = 65 s = 1.1 min (IR-VIS path 不调 RANSAC, 比 megadepth
    val 轻 40%)
  - ckpt save + epoch boundary: ~25 s = 0.4 min
  - per ep total: ~14 min
  - max_ep=12 跑满: ~2.8 h (上限, 罕见)
  - ES @ ep 7-8 大概率: ~1.6-1.9 h, center ~1.7 h

------------------------------------------------------------------------------
Acceptance gate (val precision@3px)
------------------------------------------------------------------------------
v14 raw load 在 LLVIP 上的 val precision@3px 数字未知 (需 sanity-only 跑 1 ep
ep0 数字校准), 用 RoadScene 同 cfg 类比 baseline ~0.85-0.90:

  Strong : val precision@3px >= 0.95
  Medium : val precision@3px in [0.90, 0.95)
  Flat   : val precision@3px in [0.85, 0.90), 跟 v14 raw load 持平
  Fail   : val precision@3px < 0.85, 或 ep0 step 100 内 NaN/loss 爆

------------------------------------------------------------------------------
风险与回退方案
------------------------------------------------------------------------------
1. aggressive aug + V2 LR=1e-4 冲坏 v14 特征 (ep0 step 内 NaN / loss > 5.0):
   一级备选: WARMUP_STEP 75 -> 150 (actual 600 step ~0.8 ep) 给更长 ramp
   二级备选 (回退到 V1 保守 schedule):
     CANONICAL_LR 4e-4 -> 2.5e-4 (TRUE 1e-4 -> 6.25e-5) +
     WARMUP_STEP 75 -> 100 + MSLR=[3,5,7]
   终极: H 强度回退到 v0-v10 weak preset

2. finetune over-shoot v14 在 LLVIP 之外 (V2 风险高于 V1):
   v17 在 LLVIP 涨, 但 OOD (M3FD/RoadScene/METU) 反而退
   缓解: ep0/ep4/ep8 ckpt 都跑 OOD eval, 选 OOD 持平 + LLVIP 最强的折中点
   如果 OOD 退化超过 5% (M3FD precision@3px), 重训用 V1 schedule

3. best_ep < ep 2 (raw load 已最优, MSLR1 在 ep 2 太晚):
   V2 已经 cover, MSLR1 紧随 best_ep + ES patience=5 给 4 ep 二次精调机会.
   极端 best_ep=ep 0: 直接交付 v14 raw load 也 OK, v17 跑 sanity-only 即可结论.

4. v14 best ckpt auc@10=0.273 偏低:
   不影响 v17 计划 (v17 用 v14 当 pretrained, 即使 v14 不强 v17 还是能学)
"""
from configs.loftr.eloftr_full_v14_pose_msyn_ddp import cfg


# === Aggressive single-side H aug (eval_roadscene_all_singleh_finetuned.bat 同款) ===
# v0-v10 默认是 weak preset (rot=10 / scale=0.9-1.1 / trans=0.05 / persp=0.03);
# v17 升到 v11/v12 aggressive (3x 强度) 让 model 学会 eval-time aug 场景.
# DUAL 保持 default False -> 单侧 warp VIS only (image1), 与 eval bat 一致.
# PROB 由 data cfg llvip_trainval.py 设 1.0 (100% trigger), 这里不重设.
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08

# === LLVIP sampler override (DDP image-level pre-shard 后的 per-rank 全集) ===
# IR-VIS short-circuit (data.py:289-297) 在 mode='train' 时把 dataset 切到 per-rank,
# 每 rank 的 dataset.__len__() = 12025 / 4 = 3006 (n_subset=1 的 ConcatDataset).
# sampler.py:53-61 在 SB_SUBSET_SAMPLE_REPLACEMENT=False 时:
#   if len_subset >= n_samples_per_subset: rand_tensor[:n_samples_per_subset]
#   else: padding with replacement (3006 unique + 9019 duplicate = 12025)
# 所以 N_SAMPLES_PER_SUBSET 必须 = 3006 (= 全集/world_size), 跟 v10/v11/v12 ddp cfg
# 设 28750 (= 115000/4) 同样的 DDP 修正方式. 设 12025 会 4x 重复 sample, 浪费 wall-clock.
# 单卡场景 (本地 sanity-only) 时 dataset.__len__() = 12025 (no pre-shard),
# 3006 让 sampler 只取 25%, 但 sanity 验证启动日志 + shape + loss 不爆够用.
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 3006   # = 12025 / world_size=4, image-level pre-shard 后 per-rank size
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False   # randperm 全覆盖

# === Finetune schedule V2 (中庸: 短 WARMUP + 略高 LR + 早 MSLR, 适应 H aug task) ===
# 设计动机详见 docstring 顶部 "Schedule V2 推导" 段.
cfg.TRAINER.CANONICAL_LR = 4e-4         # actual = 4e-4 * 0.25 = 1e-4 (= v14 cold start 0.8x)
cfg.TRAINER.WARMUP_STEP = 75            # actual = 75 / 0.25 = 300 step ~0.4 ep
cfg.TRAINER.MSLR_MILESTONES = [2, 4, 6]  # 三段 LR drop 都早于 ES @ ep 7-8 窗口
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 5  # finetune from v14 + monitor 切换, 留更长 patience 避免误 ES

# bat / sh:
#   --max_epochs=12, --batch_size=4, --gpus=4, --num_workers=12, --pin_memory=true,
#   --check_val_every_n_epoch=1, --log_every_n_steps=500,
#   --limit_train_batches=1.0, --limit_val_batches=0.5,
#   --num_sanity_val_steps=0, --disable_mp, --thr 0.1,
#   --ckpt_path=logs/tb_logs/msyn_v14_pose_ddp/version_0/checkpoints/
#               epoch=12-auc@5=0.151-auc@10=0.273-auc@20=0.428.ckpt
