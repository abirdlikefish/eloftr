EfficientLoFTR v17 = finetune from v14 best ckpt + LLVIP pixel-aligned IR-VIS
+ aggressive single-side H aug (image1=VIS only, eval_roadscene_all_singleh_
finetuned.bat 同款) + 4-GPU DDP. 第一个引入 LLVIP 数据集 (15488 pair, 12025
train + 3463 test, 1280x1024 .jpg, 街景低光照 + 热红外, BUPT-AI-CZ 官方发布)
的 v* 实验, 也是第一个把"已经做过跨模态训练的 v14 ckpt"作为 finetune 起点
的实验 (v0-v16 都从 outdoor.ckpt cold start).

v17 不是普通 finetune, 是"多重 distribution shift transfer learning":
  dataset (Megadepth_Syn 155K合成 IR-VIS 跨视角 -> LLVIP 12K真实 IR-VIS 像素对齐)
  + 监督 (pose/auc@10 -> H/precision@3px IR-VIS path)
  + aug (无 H aug -> aggressive single-side H 3x v0-v10 强度)
  + geometry (cross-view 有 parallax -> monocular pixel-aligned 无 parallax)
  + BN stats (Megadepth 综合 -> LLVIP 街景 IR-VIS).
v17 = path K (LLVIP scale-down + aug-up finetune route), 测试"v14 cross-modal
feature 能否泛化到真 LLVIP IR-VIS + H aug task".

继承链: eloftr_full.py (v0 baseline) -> v13 (pose 监督) -> v14 (640 + 工程加速) ->
v17 (THIS skill, finetune + LLVIP + aggressive H). v17 在 v15/v16 之外的另一
平行分支 ablation, 但 v17 用了 finetune 而 v15/v16 用 cold start, 不是同一
对照组 (v17 是 dataset+aug ablation, v15/v16 是 loss/embedding ablation).

------------------------------------------------------------------------------
文件清单 (1 行白名单修改 + 4 个新文件 + 1 个可选 .bat + 服务器子目录软链)
------------------------------------------------------------------------------
1. src/utils/data_source.py: ALIGNED_IRVIS_SOURCES 加 "llvip" (1 行)
2. MyScripts/make_llvip_splits.py: 扫 data/LLVIP/{infrared,visible}/{train,test}/,
   输出 train_pairs.txt (12025 行 'train/<id>.jpg') + test_pairs.txt (3463 行
   'test/<id>.jpg'). **行格式带子目录前缀**, 让 cfg ROAD_*_SUBDIR='infrared'
   (NOT 'infrared/train') 单值同时支持 train (TRAIN_LIST_PATH) 与 val
   (VAL_LIST_PATH) 两种 split.
3. configs/data/llvip_trainval.py: 复制 m3fd cfg 改 5 字段 (DATA_SOURCE='LLVIP'
   / TRAIN_LIST_PATH=train_pairs.txt / VAL_LIST_PATH=test_pairs.txt 用 LLVIP
   官方 test 当 val pool / ROAD_IMG_RESIZE=640 / ROAD_HOMOGRAPHY_AUG=True).
4. configs/loftr/eloftr_full_v17_llvip_singleh_aggressive_ddp.py: 继承
   eloftr_full_v14_pose_msyn_ddp.py + 10 项 override 分 3 组:
     4 项 H aug:        ROAD_HOMOGRAPHY_KWARGS.{rot_deg=25, scale=[0.75,1.25],
                        trans=0.12, persp=0.08} (eval bat 同款)
     2 项 sampler:      N_SAMPLES_PER_SUBSET=3006 (= 12025/4, DDP image-level
                        pre-shard 配套, **绝不是 12025**!) +
                        SB_SUBSET_SAMPLE_REPLACEMENT=False
     4 项 schedule V2:  CANONICAL_LR=4e-4 (TRUE 1e-4) / WARMUP_STEP=75 (actual
                        300 step) / MSLR=[2,4,6] / patience=5
5. MyScripts/run_llvip_v17_singleh_aggressive_ddp.sh: 基于 v14 ddp.sh 改 5 处
   (data_cfg / main_cfg / exp_name / +--ckpt_path指向v14 / max_epochs 18->12
   / limit_val_batches 0.2->0.5) + prereq 检查换成 LLVIP + v14 ckpt. 含
   --sanity-only 1-ep smoke 模式.
6. MyScripts/run_llvip_v17_singleh_aggressive_singlecard.bat: 可选, 本地 Windows
   sanity 用. 单卡 setting (gpus=1 / no sync_bn / num_workers=6 /
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True / limit_train_batches=0.05).

------------------------------------------------------------------------------
关键设计决策 (重要!)
------------------------------------------------------------------------------
A. **N_SAMPLES_PER_SUBSET=3006 而不是 12025** (DDP 关键 bug 防御):
   IR-VIS short-circuit (data.py:289-297) 在 mode='train' 时把 dataset 切到
   per-rank, 每 rank 的 dataset.__len__() = 12025/4 = 3006 (n_subset=1 的
   ConcatDataset, 因为 LLVIP 是单 subset 不是 ScanNet/MegaDepth 多 npz).
   sampler.py:53-61 在 SB_SUBSET_SAMPLE_REPLACEMENT=False 时:
     if len_subset >= n_samples_per_subset: rand_tensor[:n_samples_per_subset]
     else: padding with replacement (3006 unique + 9019 duplicate = 12025)
   设 N_SAMPLES_PER_SUBSET=12025 会让 sampler 走 padding 分支让每 rank 重复
   4x sample, **wall-clock 直接 4x (~10h 而不是 ~2.8h)**. 这是 v10/v11/v12
   ddp cfg 已经踩过的坑 (设 28750 = 115000/4 而不是 115000), v17 同样修正.

B. **ROAD_*_SUBDIR='infrared' 不带 train/test 前缀** (单 cfg 兼容 train+val):
   LLVIP 有官方 train/test 子分(infrared/{train,test}/, visible/{train,test}/),
   不像 RoadScene/M3FD 那样 flat. 如果 cfg 写 ROAD_IR_SUBDIR='infrared/train',
   train 能跑但 val 失败 (val_list 指向 test/ 但 cfg 指向 train/), 要建两个
   data cfg. 解决方案: cfg ROAD_IR_SUBDIR='infrared' (parent 目录) +
   splits 文件每行带子目录前缀 (如 'train/010001.jpg' 或 'test/190001.jpg'),
   RoadSceneDataset 的 osp.join(ir_dir, name) 自动拼出最终路径. 一份 cfg
   驱动两种 split.

C. **--ckpt_path vs --resume_from_checkpoint** (v17 第一次用 finetune):
   --ckpt_path: 仅 torch.load + load_state_dict, optimizer/scheduler/monitor
                全新建. v9 finetune from outdoor.ckpt 用的是这个.
   --resume_from_checkpoint: PL Trainer 原生 flag, 恢复完整训练状态 (含
                             monitor=auc@10 历史 best). v17 切到 precision@3px
                             监控会冲突, **不能用**.
   v17 用 --ckpt_path.

D. **val pool = LLVIP test 全 3463 pair (--limit_val_batches=0.5 实跑 1731)**:
   v17 不切 LLVIP train 出 val (留 12025 全集训练最大化), 用 LLVIP test 当
   val pool. limit 0.5 留 50% 给未来 holdout 灵活性, 实际 val 1731 pair
   跟 LoFTR megadepth_val_1500 业界量级一致. v17 不在 LLVIP 跑独立 test
   (OOD eval 在 M3FD/RoadScene/METU).

------------------------------------------------------------------------------
Schedule V2 推导 (V1 保守 / V2 中庸 / V3 激进, 取中庸)
------------------------------------------------------------------------------
v14 cold start 对照: CANONICAL_LR=5e-4 / WARMUP=450 (actual 1800 step) /
  MSLR=[6,10,14] / ES patience=5 / max_ep=18.

V2 (本 cfg) 设计:
- CANONICAL_LR=4e-4 (TRUE 1e-4 = v14 cold start 0.8x):
    不取 v14 同 LR (1.25e-4): 保护 v14 cross-modal feature 不被 aggressive aug 冲坏.
    不取 V1 保守 LR (6.25e-5): max_ep=12 窗口内适应新 H aug task 偏慢.
- WARMUP_STEP=75 (actual 300 step ~0.4 ep):
    比 V1 100 (400 step) 短: finetune 不需要长 warmup, 300 step 足够吸收
    aggressive aug 初期震荡 + BN running stats 在新数据集稳定.
- MSLR=[2,4,6] 配 ES patience=5:
    best_ep 估 ep 2-3 (v14 已 cross-modal align, LLVIP 像素对齐很简单, raw
    load 接近最优, 主要 epoch 用来适应 H aug). ES 在 best_ep+5=ep 7-8 触发.
    [2,4,6] 让三段 LR drop 都早于 ES 窗口:
      MSLR1 (ep 2): 跟 best_ep 紧随, drop LR 看是否再涨 (二次 fine-tune)
      MSLR2 (ep 4): 三次 fine-tune
      MSLR3 (ep 6): 跟 ES 触发前留 1 ep 缓冲
    比 V1 [3,5,7] 更早 drop, cover best_ep=ep 1 的极端 (raw load 直接最优).
- patience=5 (沿用 v14): 给 LR drop 后 reactivation 充足空间.

V1 (保守备选, 二级回退): CANONICAL_LR=2.5e-4 (TRUE 6.25e-5) / WARMUP=100 /
  MSLR=[3,5,7]. 优点: 最大化保护 v14 feature, OOD 退化少. 缺点: 适应 H aug
  task 慢, max_ep 可能不够.
V3 (激进备选, 实际不用): CANONICAL_LR=5e-4 (TRUE 1.25e-4 = v14 同 LR) /
  WARMUP=50 / MSLR=[2,4,6] / patience=3. 优点: 快收敛. 缺点: aggressive aug +
  高 LR 可能严重偏离 v14 -> OOD 大幅退化.

------------------------------------------------------------------------------
Wall-clock 估算 (服务器 4×3090 DDP, v14 实测 step time 1.0 s 推算)
------------------------------------------------------------------------------
v14 实测黄金基准: msyn_v14_pose_ddp wall_clock 11.892 h / total_global_steps
41000 / 18 ep -> 联立解 train step time t + val step time ~1.2t (megadepth val
含 EVAL_TIMES=1 RANSAC): 41000*t + 18*56*1.2*t = 11.892*3600 -> t ≈ 1.0 s/step
(含 sync_bn + ckpt save + DataLoader prefetch overhead).

v17 batch 数:
  train: 12025/4 (DDP image-level pre-shard) /bs=4 = 752 step/rank/ep
  val:   3463/4 (DistributedSampler) /bs=4 * --limit_val_batches=0.5 = 108 step/rank/ep
  per ep: 860 step

v17 时间 (per ep):
  train: 752 * 1.0 = 752 s = 12.5 min
  val:   108 * 0.6 = 65 s  = 1.1 min  (IR-VIS path 走 _compute_roadscene_metrics
                                       不调 RANSAC, 比 megadepth val 轻 40%)
  ckpt save + epoch boundary: ~25 s = 0.4 min
  per ep total: ~14 min

v17 wall-clock:
  max_ep=12 跑满: ~2.8 h (上限, 罕见)
  ES @ ep 9 (best_ep 偏晚): ~2.1 h
  ES @ ep 7-8 大概率 (best_ep ep 2-3): ~1.6-1.9 h, center ~1.7 h
  ES @ ep 6 (best_ep ep 1, raw load 已最优): ~1.4 h
  ES @ ep 5 (罕见早): ~1.2 h
  预估范围 1.2-2.8 h, 最可能 ~1.6-1.9 h, center ~1.7 h

------------------------------------------------------------------------------
Acceptance gate (val precision@3px, sanity-only mode 1 ep ep0 数字校准)
------------------------------------------------------------------------------
v14 raw load 在 LLVIP 上的 val precision@3px 数字未知 (待 sanity 后填),
用 RoadScene 同 cfg 类比 baseline ~0.85-0.90:
  Strong : val precision@3px >= 0.95
  Medium : val precision@3px in [0.90, 0.95)
  Flat   : val precision@3px in [0.85, 0.90), 跟 v14 raw load 持平
  Fail   : val precision@3px < 0.85, 或 ep0 step 100 内 NaN/loss 爆

------------------------------------------------------------------------------
风险与回退方案
------------------------------------------------------------------------------
1. aggressive aug + V2 LR=1e-4 冲坏 v14 特征 (NaN / loss > 5.0):
   一级备选: WARMUP_STEP 75 -> 150 (actual 600 step ~0.8 ep) 给更长 ramp
   二级备选 (回退到 V1 保守): CANONICAL_LR 4e-4 -> 2.5e-4 / WARMUP 75 -> 100 /
                              MSLR=[3,5,7]
   终极: H 强度回退到 v0-v10 weak (rot=10/scale=0.9-1.1/trans=0.05/persp=0.03),
         但失去与 eval bat 协议一致性

2. finetune over-shoot v14 在 LLVIP 之外 (V2 风险高于 V1):
   v17 在 LLVIP 涨, 但 OOD (M3FD/RoadScene/METU) 反而退
   缓解: ep0/ep4/ep8 ckpt 都跑 OOD eval, 选 OOD 持平 + LLVIP 最强折中点
   如果 OOD 退化超过 5% (M3FD precision@3px), 重训用 V1 schedule

3. best_ep < ep 2 (raw load 已最优):
   V2 已 cover, MSLR1 紧随 best_ep + ES patience=5 给 4 ep 二次精调.
   极端 best_ep=ep 0: 直接交付 v14 raw load 也 OK, v17 sanity-only 即可结论.

4. v14 best ckpt auc@10=0.273 偏低 (vs v13 0.4096 -33%):
   不影响 v17 计划 (v17 用 v14 当 pretrained, 即使 v14 不强 v17 还是能学)

------------------------------------------------------------------------------
Ship 状态 (2026-05-15)
------------------------------------------------------------------------------
NOT YET shipped, 实现完成但未跑 ship.
- 6 个文件已创建 + 1 行 src 修改: ✓
- splits 已本地生成: train 12025 + test 3463 ✓
- git push: 用户手动
- 服务器软链: 用户手动 (subdirectory softlink: data/LLVIP/{infrared,visible}/
  -> /data/xyjiang/Datasets/Infrared_image_datasets/LLVIP/{infrared,visible}/,
  data/LLVIP/index/ 仓库内实体可写)
- 服务器 sanity-only smoke (1 ep): 待跑
- 服务器 ship (12 ep, ~1.6-1.9 h): 待跑
- METU/M3FD/RoadScene OOD eval: 待跑
- 实测 best_ep / wall-clock / val precision@3px / OOD 数字: 待 backfill

Use when running/debugging/extending v17 / 解释 v17 vs v14 ablation /
finetune from v14 vs cold start / N_SAMPLES_PER_SUBSET DDP 修正 (3006 vs
12025) / aggressive H aug 在 finetune 上的协议 / Schedule V2 vs V1/V3
中庸推导 / val pool 用 LLVIP test 全 + limit 0.5 协议 / ROAD_*_SUBDIR=
'infrared' 单 cfg 兼容 train+val / IR-VIS path 不调 RANSAC val 加速 /
LLVIP 数据集接入 / 写论文 v17 finetune 实验章节 / OOD over-shoot trade-off /
回退方案 V2 -> V1.

Triggers: v17 / v17_llvip / v17_singleh_aggressive / llvip_v17_singleh_
aggressive_ddp / LLVIP / 12025 + 3463 LLVIP 官方 split / N_SAMPLES_PER_SUBSET
3006 / 12025 sampler bug / DDP image-level pre-shard / 4x 重复 sample 浪费
wall-clock / ROAD_HOMOGRAPHY_KWARGS aggressive (rot 25 scale 0.75-1.25 trans
0.12 persp 0.08) / image1 VIS warped only / DUAL=False 单侧 / Schedule V2
中庸 / V1 保守 / V3 激进 / TRUE_LR=1e-4 v14 cold start 0.8x / WARMUP 75
actual 300 step 0.4 ep / MSLR [2,4,6] / patience 5 / best_ep 估 ep 2-3 /
ES @ ep 7-8 1.6-1.9h / per ep 14 min 752 train + 108 val + 25s ckpt /
finetune from v14 (NOT cold start outdoor.ckpt) / --ckpt_path NOT
--resume_from_checkpoint / monitor 切换 auc@10 -> precision@3px / val pool
LLVIP test 全 3463 + limit 0.5 实跑 1731 / ROAD_IR_SUBDIR='infrared' 不带
train/test 前缀 splits 行带前缀 / 1 cfg 驱动 train+val / 多重 distribution
shift transfer learning / dataset+监督+aug+geometry+BN 都变 / 子目录软链
infrared visible 而不是整目录 / data/LLVIP/index 仓库内实体可写 / Annotations
可选软链 / OOD over-shoot 风险 V2 高于 V1 / 回退路径 V2 -> V1 / 5% OOD 退化
threshold M3FD precision@3px / IR-VIS path val 不调 RANSAC _compute_roadscene_
metrics 只 H 投影 + linalg.norm,
English 'finetune from v14 not cold start', 'multi-distribution shift transfer
learning', 'aggressive single-side H aug eval bat protocol', 'N_SAMPLES_PER_SUBSET
3006 not 12025 DDP bug', 'Schedule V2 balanced LR=1e-4', 'val pool LLVIP test
with limit 0.5', 'subdirectory prefix in splits files for single cfg train+val',
'IR-VIS path val no RANSAC saves 40%', 'ROAD_*_SUBDIR parent only no train test
prefix', 'best_ep estimated ep 2-3 patience 5 ES ep 7-8'.

v17 = path K (LLVIP scale-down + aug-up finetune route). Inheritance:
  eloftr_full.py (v0 baseline)
  -> v13 (B-pure pose 监督, eloftr-v13-pose)
  -> v14 (640 + 工程加速, eloftr-v14-resolution-640)
  -> v17 (THIS skill, finetune + LLVIP + aggressive H, 区别于 v15/v16 平行
          ablation 因为用了 finetune 而不是 cold start)

Companion:
  eloftr-v14-resolution-640 (parent / v17 ckpt 起点 epoch=12-auc@5=0.151-
    auc@10=0.273-auc@20=0.428.ckpt)
  eloftr-v11-dualh (aggressive H aug 同款 KWARGS preset 来源)
  eloftr-v12-dualh-baseline (aggressive H aug 同款 KWARGS preset)
  eloftr-v10-msyn (DDP image-level pre-shard 设计 + N_SAMPLES_PER_SUBSET=
    per-rank size 修正先例 28750 = 115000/4)
  eloftr-cross-modal-experiments (chain 路线总览; v17 是 path K)
  eloftr-server-multigpu (DDP runtime + 5 traps + step time 实测)
  eloftr-eval-pipeline (eval_roadscene_all_singleh_finetuned.bat 协议 /
    OOD eval M3FD/RoadScene/METU)
  eloftr-results (跨版本数字总表; v17 数字待 backfill into results/eval_summary.md)
  eloftr-tb-summary (训练侧 KPI 总表; v17 wall-clock / best_ep / val precision@3px
    待 backfill into results/tb_summary.md)
  eloftr-yurupeng-workspace (服务器子目录软链协议: data/LLVIP/{infrared,visible}/
    -> 只读源, data/LLVIP/index/ 仓库内实体)
