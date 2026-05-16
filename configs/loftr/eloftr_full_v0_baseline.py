"""v0 (baseline): pure cold-start finetune from eloftr_outdoor.ckpt on **LLVIP** + 4-GPU DDP.

------------------------------------------------------------------------------
设计目标
------------------------------------------------------------------------------
v0 = cross-modal IR-VIS finetune 起点, 用 LLVIP (12025 train / 3463 test 像素
对齐 IR-VIS) + aggressive single-side H aug (rot=25 / scale=0.75-1.25 /
trans=0.12 / persp=0.08, image1=VIS only) cold-start 训练 outdoor.ckpt.

zero 模型修改, 所有 v1-v15 引入的 toggle 保持 default False:
  - USE_CONTRASTIVE      = False  (v1 引入)
  - USE_MODALITY_EMB     = False  (v2 引入)
  - USE_EDGE_INPUT       = False  (v7 引入)
  - USE_CLAHE_IR         = False  (v7 引入)
  - USE_MSBN             = False  (v8 引入)
  - BACKBONE_IN_CHANNELS = 1      (v7 改 2)

模型架构跟 eloftr_full.py + outdoor.ckpt 完全一致, ckpt -> model load 是
byte-clean strict load (无 inflate / 无 missing_keys / 无 unexpected_keys).

------------------------------------------------------------------------------
v0 历史变迁 (RoadScene -> LLVIP, 重要警告)
------------------------------------------------------------------------------
**旧 v0** (本文件 git history): RoadScene 178-pair single-card cold-start +
aggressive single-side H aug, schedule {CANONICAL_LR=4e-3, WARMUP_STEP=2,
MSLR=[10,15,20], ES PATIENCE=8}.

**新 v0** (本文件当前): LLVIP 12025-pair 4-GPU DDP cold-start + aggressive
single-side H aug, schedule {CANONICAL_LR=5e-4, WARMUP_STEP=450, MSLR=[4,8,12],
ES PATIENCE=5}.

**ablation 影响**: v1-v15 全部基于旧 v0 RoadScene 单卡 baseline 做 ablation
(USE_CONTRASTIVE / USE_MODALITY_EMB / etc), 覆写后 v1-v15 跟 v0 不再形成单变量
对照 (v0 数据集/硬件/schedule 都换了). 历史 v1-v15 vs v0 RoadScene 对比数字
(results/eval_summary.md) 仍然有效, 但今后新建 v* 应基于新 v0 LLVIP DDP 起点.

如需复跑老 v0 RoadScene 单卡实验, 从 git log 拿历史版本即可.

------------------------------------------------------------------------------
继承链 + override (3 组 = 11 项 cfg override)
------------------------------------------------------------------------------
parent: eloftr_full.py (baseline, 短路继承 NOT 经过 v9_e2e/v10/v14 链, 跟旧
v0 RoadScene 同款继承结构, 让 USE_* 全 default False)
                                          |
                                  + 5 项 H aug (DUAL=False, PROB=1.0, 4 参数)
                                  + 2 项 sampler (DDP image-level pre-shard 配套)
                                  + 4 项 cold-start schedule (LR/WARMUP/MSLR/ES)
                                          |
                                v0 cfg (this file)

eloftr_full.py 自动继承的关键字段 (NOT override here, 顺带列出避免审查混淆):
- WARMUP_TYPE='linear' / WARMUP_RATIO=0.1 (warmup 起点 = 0.1 * TRUE_LR = 1.25e-5)
- MSLR_GAMMA=0.5 (每次 LR drop 减半)
- OPTIMIZER='adamw' / ADAMW_DECAY=0.1
- RANSAC_PIXEL_THR=0.5 (IR-VIS path 不调 RANSAC, 不生效但保留)
- LOFTR.EVAL_TIMES=5 (IR-VIS path 不调 RANSAC, 不生效)
- LOFTR.COARSE.NPE: 不显式设 (train.py:129-130 fallback [832,832,832,832] 与
  outdoor.ckpt 兼容; 显式设 NPE 会触发 v14 NPE bug post-mortem)
- LOFTR.BACKBONE_IN_CHANNELS=1 (默认; outdoor.ckpt 第一层 conv 也是 1 通道,
  strict load 兼容, 不需要 inflate hook)
- IMG_RESIZE / DF / PAD_SIZE: 由 data cfg llvip_trainval.py 设 (640 / 32 / 640)
- bs=4, --gpus=4 -> effective_bs=16 -> _scaling=0.25 (TRUE_LR=CANONICAL_LR*0.25,
  actual WARMUP step=WARMUP_STEP/0.25)
- ENABLE_PLOTTING (eloftr_full.py 默认 True; 但 v0 不显式 reset 因为 LLVIP 训练
  规模 18 ep * 752 step/ep ~ 13k step, log_every_n_steps=500 -> ~27 train figure,
  events.tfevents 控制在 ~500MB 可接受)
- FREEZE_BACKBONE / FREEZE_BN / FREEZE_BACKBONE_BN: 全 False (默认; v0 finetune
  全模型, 不冻结)

------------------------------------------------------------------------------
Cold-start outdoor.ckpt 跨模态实测黄金标准 (本 cfg schedule 推导基础)
------------------------------------------------------------------------------
v10 / v13 / v14 都是 outdoor.ckpt cold-start 跨模态 DDP, 实测同一组 schedule:

| 项 | v10 (Megadepth_Syn H) | v13 (Megadepth_Syn pose 832) | v14 (Megadepth_Syn pose 640) | **v0-LLVIP** | v17 (warm-start) |
|---|---|---|---|---|---|
| 起点 ckpt | outdoor.ckpt | outdoor.ckpt | outdoor.ckpt | **outdoor.ckpt** | v14 best ep12 |
| 监督 | H | pose | pose | **H** | H |
| 数据集 train pairs | 115000 | 153 scene 8.86M | 同 v13 | **12025** | 12025 |
| bs / gpus | 4 / 4 | 2 / 4 | 4 / 4 | **4 / 4** | 4 / 4 |
| TRUE_LR | 1.25e-4 | 1.25e-4 | 1.25e-4 | **1.25e-4** | 1e-4 (0.8x) |
| actual warmup step | 1800 | 1800 | 1800 | **1800** | 300 (短 finetune) |
| MSLR (epoch) | [3,5,7] | [3,5,7] | [6,10,14] | **[4,8,12]** | [2,4,6] |
| ES PATIENCE | 3 | 3 | 5 | **5** | 5 |
| max_ep | 12 | 12 | 18 | **18** | 12 |
| best_ep 实测 | ep 11 saturate | ep 6 | TODO | **est ep 6-9** | est ep 2-3 |

v0-LLVIP schedule 设计:
- TRUE_LR=1.25e-4 (CANONICAL_LR=5e-4 * 0.25): 跟 v10/v13/v14 cold-start 完全一致.
  cold-start 对 outdoor.ckpt 跨模态需要稳定 LR, 比 v17 finetune 的 1e-4 略高.
- actual warmup=1800 step (WARMUP_STEP=450 / 0.25): 跟 v10/v14 字节对齐.
  cold-start 需要 long warmup 让 outdoor.ckpt RGB-RGB 特征向 IR-VIS task 平滑过渡;
  warmup 起点 LR=0.1 * 1.25e-4 = 1.25e-5 (WARMUP_RATIO=0.1 来自 eloftr_full.py).
- MSLR=[4,8,12]: LLVIP H 监督任务比 Megadepth_Syn pose 监督简单 (像素对齐 + monocular,
  无 parallax 计算), best_ep 估早 6-9, MSLR1 必须前移到 ep 4 才能 cover best_ep
  早段; 跟 v10 [3,5,7] 比略晚 1-2 ep (12025 train pairs vs v10 115000, single
  ep step count 一致 ~752 step/ep 但 best_ep 类似的"一半数据集见过"启发式).
- ES PATIENCE=5 (沿用 v14): cold-start 噪声 + LR drop 后 reactivation 需要 buffer.
- max_ep=18 (沿用 v14): 给 ES @ best_ep+5 充分余地 (worst case best_ep=12 -> ES
  @ ep 17 < max_ep=18).

WARMUP=True 显式保留 (旧 v0 第 66 行已设, 关键防御性保留):
- default.py:374 _CN.TRAINER.EARLY_STOPPING = False, eloftr_full.py 不显式设
- v13/v14/v17 cfg 都只设 PATIENCE 不设 EARLY_STOPPING (继承 default False),
  所以这些实验实际 ES 从未启用, max_ep 跑满或人工停
- v0 必须显式 cfg.TRAINER.EARLY_STOPPING = True 才能让 ES 生效

------------------------------------------------------------------------------
H aug 参数 (跟旧 v0 + v11/v12/v17 完全一致)
------------------------------------------------------------------------------
ROAD_HOMOGRAPHY_DUAL = False (单侧 warp image1=VIS only)
ROAD_HOMOGRAPHY_PROB = 1.0   (每 sample 必触发, 单侧无共视塌陷)
ROAD_HOMOGRAPHY_KWARGS:
  rot_deg     = 25.0   (3x v0-v10 weak preset 10)
  scale_range = [0.75, 1.25]  (3x weak [0.9, 1.1])
  trans_ratio = 0.12   (2.4x weak 0.05)
  persp_ratio = 0.08   (2.7x weak 0.03)

代码路径 src/datasets/roadscene.py:558-578: H_ir = I (恒等),
H_vis = _random_homography(aggressive), 只对 image1 (VIS) 做 cv2.warpPerspective.

eval_roadscene_all_singleh_finetuned.bat / eval_metu_vistir_finetuned.bat 都是
同款 aggressive single-side H aug 协议, train/eval distribution match.

------------------------------------------------------------------------------
Sampler 配置 (DDP image-level pre-shard 配套, 关键 bug 防御)
------------------------------------------------------------------------------
LLVIP 走 IR-VIS short-circuit (data.py:289-297) 在 mode='train' 时把 dataset
切到 per-rank, 每 rank 的 dataset.__len__() = 12025 / 4 = 3006 (n_subset=1
的 ConcatDataset).

sampler.py:53-61 在 SB_SUBSET_SAMPLE_REPLACEMENT=False 时:
  if len_subset >= n_samples_per_subset: rand_tensor[:n_samples_per_subset]
  else: padding with replacement (3006 unique + 9019 duplicate = 12025)

所以 N_SAMPLES_PER_SUBSET 必须 = 3006 (= 全集/world_size). 设 12025 会让每 rank
重复 sample 4x, 浪费 wall-clock (10h+ 而不是 2.8h, 实验周期翻 4 倍).

跟 v10/v11/v12/v17 ddp cfg 设 28750 (= 115000/4) / 3006 (= 12025/4) 同款 DDP
修正模式.

------------------------------------------------------------------------------
LR schedule timeline (max_ep=18, 跑满情况)
------------------------------------------------------------------------------
ep 0     step    0-1800   warmup linear 1.25e-5 -> 1.25e-4 (WARMUP_RATIO=0.1)
ep 0-4   step 1800-3008   full LR 1.25e-4 (ep0 完成 752 step / ep, warmup 占 ep0~2.4)
ep 4     step 3008        MSLR #1, LR x= 0.5 -> 6.25e-5
ep 4-8   step 3008-6016   ramp down 期 1
ep 8     step 6016        MSLR #2, LR x= 0.5 -> 3.13e-5
ep 8-12  step 6016-9024   ramp down 期 2
ep 12    step 9024        MSLR #3, LR x= 0.5 -> 1.56e-5
ep 12-18 step 9024-13536  低 LR 微调
(实际 ES @ best_ep + 5 大概率提前停, best_ep 估 6-9 -> ES @ ep 11-14)

------------------------------------------------------------------------------
Wall-clock 估算 (服务器 4×3090 DDP, v14 实测 step time 1.0 s 推算)
------------------------------------------------------------------------------
v14 实测黄金基准: msyn_v14_pose_ddp 11.892 h / 41000 step -> t ≈ 1.0 s/step
(含 sync_bn + ckpt save + DataLoader prefetch)

LLVIP 跟 v14 都走 832/640 IMG_RESIZE 同分辨率 + bs=4 + sync_bn + 4-GPU NCCL,
step time 假设 0.83 s/step (略快, IR-VIS dataset I/O 比 megadepth pose path
轻; 实际首次 sanity-only 跑后校准).

v0-LLVIP batch 数:
  - train: 12025/4 (DDP image-level pre-shard) /bs=4 = 752 step/rank/ep
  - val: 3463/4 (DistributedSampler) /bs=4 * --limit_val_batches=0.5 = 108 step/rank/ep
  - per ep: 752 + 108 = 860 step

v0-LLVIP 时间:
  - train per ep: 752 * 0.83 = 624 s = 10.4 min
  - val per ep: 108 * 0.6 = 65 s = 1.1 min (IR-VIS path 不调 RANSAC)
  - ckpt save + epoch boundary: ~25 s = 0.4 min
  - per ep total: ~12 min
  - max_ep=18 跑满: ~3.6 h
  - ES @ best_ep+5 大概率 (best_ep est 6-9): ~2.2-2.8 h, center ~2.5 h

------------------------------------------------------------------------------
Acceptance gate (val precision@3px, sanity-only mode 1 ep 后校准)
------------------------------------------------------------------------------
LLVIP val mode 在 data.py:310 自动 homography_aug=False (不论 cfg 设啥), 所以
val 是 IR-VIS pixel-aligned + 无 aug 的"简单题". outdoor.ckpt 在该简单题上的
raw precision@3px 数字未实测, sanity-only 1-ep 跑后校准:

  ep0 末 val precision@3px: estimate wide range 0.30-0.70 (待校准)
    LLVIP IR-VIS 像素严格对齐 + 街景 LWIR/RGB 结构边缘高度相似 + val 无 H aug,
    outdoor.ckpt 虽未见过 IR 但通用结构特征匹配可能拿到 0.50+.

  全 ship max_ep=18 跑完 best ep val:
    Strong : >= 0.95
    Medium : in [0.90, 0.95)
    Flat   : in [0.85, 0.90), 跟旧 v0 RoadScene baseline 类似量级
    Fail   : < 0.85, 或 ep0 step 100 内 NaN/loss > 5.0

启动日志硬性 dispatch gate (3 条必须出现):
  [rank N]: building RoadSceneDataset (dataset_name=LLVIP) from data/LLVIP/index/{train,test}_pairs.txt
  [rank N]: aligned IR-VIS train pre-shard: 3006/12025 samples (disjoint across 4 ranks; seed=...)
  EarlyStopping enabled (monitor=precision@3px, mode=max, patience=5)

------------------------------------------------------------------------------
风险与回退方案
------------------------------------------------------------------------------
1. cold-start aggressive aug + outdoor.ckpt 冲坏 (ep0 step 100 内 NaN / loss > 5.0):
   一级备选: WARMUP_STEP 450 -> 750 (actual 3000 step ~4 ep) 给更长 ramp
   二级备选: H 强度回退到 v0-v10 weak preset (rot=10 / scale=0.9-1.1 / trans=0.05 / persp=0.03)
   终极: ROAD_HOMOGRAPHY_PROB 1.0 -> 0.5 (一半 sample 不 aug)

2. best_ep < ep 4 (outdoor.ckpt raw load 已最优, MSLR1 在 ep 4 太晚):
   极端 best_ep=ep 1: ES PATIENCE=5 在 ep 6 触发, MSLR1 ep 4 只 cover 2 ep 二次精调
   缓解: max_ep / ES PATIENCE 已经给 cushion, 实测后若发现 best_ep 太早, retry
       MSLR=[2,5,8]; 若 raw load 已最优, 直接交付 outdoor.ckpt 也 OK

3. best_ep > ep 13 (LR drop 后还能涨, MSLR3 ep 12 太早):
   ES @ best_ep+5 = ep 18+ 可能撞 max_ep 边界, retry max_ep=24 + MSLR=[6,12,18]

4. step time 大于估算 0.83 s/step:
   sanity-only 1-ep 实测后, 如 step time > 1.2 s/step (>50% 超估), 检查
   GPU 散热 (v14 SKILL §6 GPU 2 87C thermal throttle) 或 dataloader 卡顿
   (--num_workers 12 是否被其他用户占满)
"""
from configs.loftr.eloftr_full import cfg

# === 数据增强: 单侧 aggressive H aug (跟 v11/v12/v17 共用 KWARGS, DUAL=False 单变量区分) ===
cfg.DATASET.ROAD_HOMOGRAPHY_DUAL = False
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08

# === LLVIP sampler override (DDP image-level pre-shard 后的 per-rank 全集) ===
# IR-VIS short-circuit (data.py:289-297) 在 mode='train' 时把 dataset 切到 per-rank,
# 每 rank 的 dataset.__len__() = 12025 / 4 = 3006 (n_subset=1 的 ConcatDataset).
# N_SAMPLES_PER_SUBSET 必须 = 3006 (设 12025 会 4x 重复 sample 浪费 wall-clock).
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 3006
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False

# === Cold-start outdoor.ckpt 跨模态 schedule (沿用 v10/v14 实测黄金标准) ===
cfg.TRAINER.CANONICAL_LR = 5e-4              # TRUE_LR = 5e-4 * 0.25 = 1.25e-4
cfg.TRAINER.WARMUP_STEP = 450                # actual = 450 / 0.25 = 1800 step
cfg.TRAINER.MSLR_MILESTONES = [4, 8, 12]
cfg.TRAINER.EARLY_STOPPING = True            # 关键: default.py:374 默认 False, 必须显式 True
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 5      # 跟 v14 一致

# sh:
#   --max_epochs=18, --batch_size=4, --gpus=4, --num_workers=12, --pin_memory=true,
#   --check_val_every_n_epoch=1, --log_every_n_steps=500,
#   --limit_train_batches=1.0, --limit_val_batches=0.5,
#   --num_sanity_val_steps=0, --disable_mp, --thr 0.1,
#   --ckpt_path=weights/eloftr_outdoor.ckpt
