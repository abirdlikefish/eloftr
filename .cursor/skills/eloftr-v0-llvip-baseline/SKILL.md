EfficientLoFTR v0 = pure cold-start finetune from eloftr_outdoor.ckpt + LLVIP
pixel-aligned IR-VIS + aggressive single-side H aug (image1=VIS only,
eval_roadscene_all_singleh_finetuned.bat 同款) + 4-GPU DDP. v0 是 cross-modal
IR-VIS 实验的 cold-start baseline 起点 (跟 v17 = v14 warm-start finetune 是
sister ablation, 同数据集 LLVIP / 同 H aug / 不同 ckpt 起点).

------------------------------------------------------------------------------
v0 历史变迁: RoadScene 单卡 (旧) -> LLVIP DDP (新)
------------------------------------------------------------------------------
**旧 v0** (git history before 2026-05-16):
  数据: RoadScene 178-pair single-card cold-start
  schedule: CANONICAL_LR=4e-3, WARMUP_STEP=2 (actual 64 step), MSLR=[10,15,20],
            ES PATIENCE=8
  入口: MyScripts/run_roadscene_v0_baseline.bat (Windows 单卡)
  作用: v1-v15 ablation 链的 RoadScene 同硬件 baseline 对照

**新 v0** (2026-05-16 覆写, 本 skill 描述):
  数据: LLVIP 12025 train / 3463 test 像素对齐 IR-VIS, 4-GPU DDP
  schedule: CANONICAL_LR=5e-4 (TRUE 1.25e-4), WARMUP_STEP=450 (actual 1800 step),
            MSLR=[4,8,12], ES PATIENCE=5, max_ep=18
  入口: MyScripts/run_llvip_v0_baseline_ddp.sh (Linux DDP)
  作用: cross-modal IR-VIS finetune cold-start 起点 (LLVIP scale + DDP infra)
  ablation 影响: v1-v15 跟新 v0 不再形成单变量对照 (数据集 / 硬件 / schedule
            全部换了); 历史 v1-v15 vs 旧 v0 RoadScene 对比数字
            (results/eval_summary.md) 仍然有效, 但今后新建 v* 应基于新 v0
            LLVIP DDP 起点

如需复跑老 v0 RoadScene 单卡实验, 从 git log -- configs/loftr/eloftr_full_v0_baseline.py
+ git log -- MyScripts/run_roadscene_v0_baseline.bat 拿历史版本即可.

------------------------------------------------------------------------------
v0 vs v17 sister ablation matrix (同数据集 LLVIP, cold-start vs warm-start)
------------------------------------------------------------------------------

| 维度 | v0 (cold-start, **本 skill**) | v17 (warm-start, eloftr-v17-llvip) |
|---|---|---|
| 起点 ckpt | weights/eloftr_outdoor.ckpt (RGB-RGB MegaDepth, 未见 IR) | logs/tb_logs/msyn_v14_pose_ddp/.../ep12 (cross-modal aligned) |
| 数据集 train | LLVIP 12025 | 同 |
| 数据集 val | LLVIP test 3463 (limit 0.5 = 1731 effective) | 同 |
| H aug | aggressive single-side (rot=25/scale=[0.75,1.25]/trans=0.12/persp=0.08) | 同 |
| BN running stats | 全部从 outdoor.ckpt MegaDepth 综合分布出发 | 已经从 v14 Megadepth_Syn 跨模态稳定 |
| TRUE_LR | 1.25e-4 (cold-start golden) | 1e-4 (= cold-start * 0.8x, 保护 v14 feature) |
| actual warmup | 1800 step (~2.4 ep) 长 ramp 让 RGB->IR 平滑过渡 | 300 step (~0.4 ep) 短 ramp finetune |
| MSLR | [4,8,12] cover best_ep 6-9 | [2,4,6] cover best_ep 2-3 |
| ES PATIENCE | 5 | 5 |
| max_ep | 18 (cold-start 需更多 ep) | 12 (finetune 短 schedule) |
| best_ep est | ep 6-9 | ep 2-3 (raw load 已接近最优) |
| wall-clock est | ES 大概率 ~2.5h, 跑满 ~3.6h | ES 大概率 ~1.7h, 跑满 ~2.8h |
| acceptance Strong | val P@3 >= 0.95 | val P@3 >= 0.95 |
| acceptance Flat   | val P@3 in [0.85, 0.90) (对照 outdoor.ckpt raw + 学到 H aug) | val P@3 in [0.85, 0.90) (对照 v14 raw load) |

v0 跟 v17 的核心 ablation 问题: cross-modal feature alignment 是要从头学
(v0 outdoor.ckpt 起点), 还是先 Megadepth_Syn 合成 IR pretrain 再 finetune (v17 路径)
更好? OOD 评测 (M3FD/RoadScene/METU) 数字会回答这个问题.

------------------------------------------------------------------------------
文件清单 (覆写 1 个 cfg + 删除 1 个 .bat + 新建 1 个 .sh, LLVIP infra 复用 v17)
------------------------------------------------------------------------------
1. configs/loftr/eloftr_full_v0_baseline.py (覆写, 旧 RoadScene -> 新 LLVIP DDP):
   短路继承 eloftr_full.py (NOT 经过 v9_e2e/v10/v14 链), USE_* 全 default False
   (跟旧 v0 同款继承结构), 3 组 11 项 override:
     5 项 H aug:        DUAL=False, PROB=1.0, KWARGS.{rot=25, scale=[0.75,1.25],
                        trans=0.12, persp=0.08}
     2 项 sampler:      N_SAMPLES_PER_SUBSET=3006 (= 12025/4, DDP image-level
                        pre-shard 配套, **不是 12025**!), SB_SUBSET_SAMPLE_REPLACEMENT=False
     4 项 schedule:     CANONICAL_LR=5e-4 (TRUE 1.25e-4), WARMUP_STEP=450
                        (actual 1800), MSLR=[4,8,12], ES PATIENCE=5
     + 1 项关键防御:    cfg.TRAINER.EARLY_STOPPING = True (default False, 必须
                        显式设, 否则 ES 失效)

2. MyScripts/run_roadscene_v0_baseline.bat (删除):
   旧 RoadScene 单卡 entry, 覆写后失效 (cfg 已不配 RoadScene). 历史版本从
   git log 拿.

3. MyScripts/run_llvip_v0_baseline_ddp.sh (新建, 拷贝 v17 sh 改 5 处):
   - main_cfg : v17 cfg -> v0 cfg
   - --exp_name : llvip_v17_singleh_aggressive_ddp -> llvip_v0_baseline_ddp
   - --ckpt_path : v14 best ckpt -> weights/eloftr_outdoor.ckpt
   - --max_epochs : 12 -> 18
   - prereq 列表 : v14 ckpt 路径 -> weights/eloftr_outdoor.ckpt
   含 --sanity-only 1-ep smoke 模式 (v17 sh 同款).

LLVIP infrastructure 全部复用 v17 已落地的:
- src/utils/data_source.py 的 ALIGNED_IRVIS_SOURCES 包含 "llvip"
- MyScripts/make_llvip_splits.py 生成 train_pairs.txt / test_pairs.txt
- configs/data/llvip_trainval.py (DATA_SOURCE='LLVIP', ROAD_*_SUBDIR='infrared'/'visible',
  ROAD_IMG_RESIZE=640, train list / val list 都用 LLVIP test 当 val pool)

详见 [eloftr-v17-llvip SKILL §文件清单](../eloftr-v17-llvip/SKILL.md).

------------------------------------------------------------------------------
关键设计决策
------------------------------------------------------------------------------
A. **EARLY_STOPPING = True 必须显式设** (default 是 False, 关键防御性保留):
   default.py:374 _CN.TRAINER.EARLY_STOPPING = False; eloftr_full.py 不显式设;
   实地核查 v13/v14/v17 cfg 都**只设 PATIENCE 不设 EARLY_STOPPING** (继承
   default False), 所以这些实验实际 ES 从未启用 (max_ep 跑满或人工停).
   v13 SKILL 文档"9 ep ES 即将触发"是误描述.
   v0 cfg 必须 cfg.TRAINER.EARLY_STOPPING = True 才能让 ES 生效, 否则 wall-clock
   从大概率 ~2.5h 退化成跑满 ~3.6h. 这是覆写时**最容易丢的一行**, 已在 cfg
   docstring "WARMUP=True 显式保留" 段警告.

B. **Cold-start outdoor.ckpt 跨模态 schedule 黄金标准** (沿用 v10/v14 实测验证):
   v10 (Megadepth_Syn H 监督) / v13 (Megadepth_Syn pose 832) / v14 (Megadepth_Syn
   pose 640) 都用 outdoor.ckpt cold-start 跨模态训练, 实测同一组 schedule 数字:
     TRUE_LR = 1.25e-4 (= CANONICAL_LR * _scaling, 4-GPU bs=4 -> _scaling=0.25)
     actual warmup step = 1800 (= WARMUP_STEP / _scaling, 让 RGB->IR 平滑过渡)
     warmup 起点 LR = 0.1 * TRUE_LR = 1.25e-5 (WARMUP_RATIO=0.1, eloftr_full.py 默认)
   v0-LLVIP 沿用同一组. 不取 v17 的 0.8x LR (1e-4 / 300 step 短 warmup), 因为 v17
   是 finetune 保护已学的 v14 feature; v0 是 cold-start 需要稳定长 ramp.

C. **MSLR=[4,8,12] 为何比 v10 [3,5,7] 晚 1-2 ep**:
   v10 在 Megadepth_Syn 115K train pair 上 ep 11 saturate, MSLR=[3,5,7] 让三段
   LR drop 都早于 saturate. v0-LLVIP 在 LLVIP 12025 train pair (10x 小), 但
   single-ep step count 一样 (~752 step/ep, 因为 DDP image-level pre-shard 后
   per-rank size = 12025/4 / bs=4 = 752), best_ep 估 6-9 (LLVIP H 监督任务比
   Megadepth_Syn pose 监督简单, IR-VIS 像素对齐 + monocular 无 parallax),
   MSLR1 必须前移到 ep 4 才能 cover best_ep 早段. ES PATIENCE=5 + max_ep=18
   配合给 cushion (worst case best_ep=12 -> ES @ ep 17 < max_ep=18).

D. **N_SAMPLES_PER_SUBSET=3006 而不是 12025** (跟 v17 同款 DDP bug 防御):
   IR-VIS short-circuit (data.py:289-297) 在 mode='train' + world_size>1 时把
   dataset 切到 per-rank, 每 rank 的 dataset.__len__() = 12025/4 = 3006
   (n_subset=1 的 ConcatDataset). sampler.py:53-61 在 SB_SUBSET_SAMPLE_REPLACEMENT=False
   时:
     if len_subset >= n_samples_per_subset: rand_tensor[:n_samples_per_subset]
     else: padding with replacement (3006 unique + 9019 duplicate = 12025)
   设 12025 会让 sampler 走 padding 分支让每 rank 重复 4x sample, **wall-clock
   直接 4x (~10h 而不是 ~2.5h)**. 跟 v10 (28750=115000/4) / v17 (3006=12025/4)
   同款 DDP 修正模式.

E. **零 src/ 修改** (LLVIP 接入工作 v17 已完成):
   v0-LLVIP 是 v17 之后第二个用 LLVIP 数据集的实验, 不需要再碰
   src/utils/data_source.py / make_llvip_splits.py / configs/data/llvip_trainval.py;
   只覆写 v0 cfg + 新建 sh + 删除老 .bat. 后续 v18+ 用 LLVIP 也是同套零 src
   修改流程.

------------------------------------------------------------------------------
LR schedule timeline (max_ep=18 跑满情况, 实际 ES 大概率提前停)
------------------------------------------------------------------------------
ep 0     step    0-1800   warmup linear 1.25e-5 -> 1.25e-4 (WARMUP_RATIO=0.1)
ep 0-4   step 1800-3008   full LR 1.25e-4 (ep0 完成 752 step, warmup 占 ep 0-2.4)
ep 4     step 3008        MSLR #1, LR x= 0.5 -> 6.25e-5
ep 4-8   step 3008-6016   ramp down 期 1
ep 8     step 6016        MSLR #2, LR x= 0.5 -> 3.13e-5
ep 8-12  step 6016-9024   ramp down 期 2
ep 12    step 9024        MSLR #3, LR x= 0.5 -> 1.56e-5
ep 12-18 step 9024-13536  低 LR 微调
(ES @ best_ep + 5 大概率提前停; best_ep 估 6-9 -> ES @ ep 11-14)

------------------------------------------------------------------------------
Wall-clock 估算 (服务器 4x 3090 DDP, v14 实测 step time 1.0 s 推算 0.83s)
------------------------------------------------------------------------------
v14 实测黄金基准: msyn_v14_pose_ddp 11.892h / 41000 step -> t ~= 1.0 s/step
(含 sync_bn + ckpt save + DataLoader prefetch).
LLVIP 跟 v14 同 IMG_RESIZE=640 + bs=4 + sync_bn + 4-GPU NCCL, step time 假设
0.83 s/step (略快, IR-VIS dataset I/O 比 megadepth pose path 轻; 实际 sanity
1-ep 校准).

v0-LLVIP batch 数:
  train: 12025/4 (DDP image-level pre-shard) /bs=4 = 752 step/rank/ep
  val:   3463/4 (DistributedSampler) /bs=4 * --limit_val_batches=0.5 = 108 step/rank/ep
  per ep: 860 step

v0-LLVIP 时间 (per ep):
  train: 752 * 0.83 = 624 s = 10.4 min
  val:   108 * 0.6 = 65 s   = 1.1 min (IR-VIS path 走 _compute_roadscene_metrics 不调 RANSAC)
  ckpt save + epoch boundary: ~25 s = 0.4 min
  per ep total: ~12 min

v0-LLVIP wall-clock:
  max_ep=18 跑满: ~3.6 h (上限)
  ES @ best_ep+5 大概率 (best_ep est 6-9): ~2.2-2.8 h, center ~2.5 h
  ES @ ep 11 (best_ep ep 6): ~2.2 h
  ES @ ep 14 (best_ep ep 9): ~2.8 h

------------------------------------------------------------------------------
Acceptance gate (val precision@3px, sanity-only mode 1 ep 后校准)
------------------------------------------------------------------------------
LLVIP val mode 在 data.py:310 自动 homography_aug=False (不论 cfg 设啥),
所以 val 是 IR-VIS pixel-aligned + 无 aug 的"简单题". outdoor.ckpt 在该简单
题上的 raw precision@3px 数字未实测, sanity-only 1-ep 跑后校准:

ep0 末 val precision@3px: estimate wide range 0.30-0.70 (待校准)
  LLVIP IR-VIS 像素严格对齐 + 街景 LWIR/RGB 结构边缘高度相似 + val 无 H aug,
  outdoor.ckpt 虽未见过 IR 但通用结构特征匹配可能拿到 0.50+. 这跟 v17 ep0
  raw load 数字 (待实测) 比, 大概率低 0.10-0.30 (v17 起点 v14 已 cross-modal
  align).

全 ship max_ep=18 跑完 best ep val:
  Strong : val P@3 >= 0.95 -> v0 cold-start 可达 v17 finetune 同水平
  Medium : val P@3 in [0.90, 0.95) -> finetune 路径 v17 略胜
  Flat   : val P@3 in [0.85, 0.90) -> 跟旧 v0 RoadScene baseline 类似量级
  Fail   : val P@3 < 0.85, 或 ep0 step 100 内 NaN/loss > 5.0

启动日志硬性 dispatch gate (3 条必须出现):
  [rank N]: building RoadSceneDataset (dataset_name=LLVIP) from data/LLVIP/index/{train,test}_pairs.txt
  [rank N]: aligned IR-VIS train pre-shard: 3006/12025 samples (disjoint across 4 ranks; seed=...)
  EarlyStopping enabled (monitor=precision@3px, mode=max, patience=5)

------------------------------------------------------------------------------
风险与回退方案
------------------------------------------------------------------------------
1. cold-start aggressive aug + outdoor.ckpt 冲坏 (ep0 step 100 内 NaN / loss > 5.0):
   一级备选: WARMUP_STEP 450 -> 750 (actual 3000 step ~4 ep) 给更长 ramp
   二级备选: H 强度回退到 v0-v10 weak preset (rot=10/scale=0.9-1.1/trans=0.05/persp=0.03),
             但失去与 eval bat 协议一致性
   终极: ROAD_HOMOGRAPHY_PROB 1.0 -> 0.5 (一半 sample 不 aug)

2. best_ep < ep 4 (outdoor.ckpt raw load 已最优, MSLR1 在 ep 4 太晚):
   极端 best_ep=ep 1: ES PATIENCE=5 在 ep 6 触发, MSLR1 ep 4 只 cover 2 ep 二次精调
   缓解: max_ep / ES PATIENCE 已经给 cushion, 实测后若发现 best_ep 太早, retry
       MSLR=[2,5,8]; 若 raw load 已最优, 直接交付 outdoor.ckpt + sanity 1-ep 即可结论

3. best_ep > ep 13 (LR drop 后还能涨, MSLR3 ep 12 太早):
   ES @ best_ep+5 = ep 18+ 可能撞 max_ep=18 边界, retry max_ep=24 + MSLR=[6,12,18]

4. step time 显著大于估算 0.83 s/step:
   sanity-only 1-ep 实测后, 如 step time > 1.2 s/step (>50% 超估), 检查
   GPU 散热 (v14 SKILL §6 GPU 2 87C thermal throttle) 或 dataloader 卡顿
   (--num_workers 12 是否被其他用户占满)

5. v0 ckpt OOD eval 反不如 v17 / v14 ckpt:
   v17 优势 = 已经跨模态 align, OOD 通常更稳; v0 cold-start 直接学 LLVIP 街景,
   OOD 风险更高. 评测时若 v0 在 M3FD/RoadScene/METU 退化超过 v17 ckpt 5%,
   v0 定位为 "LLVIP in-domain baseline only", 不参与 OOD SOTA 排名

------------------------------------------------------------------------------
Ship 状态 (2026-05-16)
------------------------------------------------------------------------------
NOT YET shipped, 实现完成但未跑 ship.
- cfg 覆写 + .bat 删除 + .sh 新建: ✓
- ReadLints: ✓ 无 lint
- LLVIP splits / 子目录软链 / outdoor.ckpt 准备: 复用 v17 (用户手动)
- 服务器 sanity-only smoke (1 ep): 待跑
- 服务器 ship (18 ep, ~2.5 h): 待跑
- METU/M3FD/RoadScene OOD eval: 待跑
- 实测 best_ep / wall-clock / val precision@3px / OOD 数字: 待 backfill

------------------------------------------------------------------------------
Use when running/debugging/extending v0 (LLVIP DDP) / 解释 v0 vs v17 cold-start
vs warm-start ablation / 解释新旧 v0 (RoadScene 单卡 vs LLVIP DDP) 转换 /
EARLY_STOPPING=True 显式保留必要性 / cold-start outdoor.ckpt 跨模态 schedule
黄金标准 / N_SAMPLES_PER_SUBSET=3006 DDP 修正 / sanity-only smoke + 3 条
启动日志硬 gate / wall-clock 2.5h 估算 / 写论文 v0 LLVIP baseline 实验章节 /
v0 ckpt OOD eval / v0 vs 旧 v1-v15 RoadScene ablation 链断裂的影响.

Triggers: v0 / v0_baseline / v0_llvip / v0_baseline_ddp / llvip_v0_baseline_ddp /
LLVIP cold-start / outdoor.ckpt LLVIP DDP / cross-modal cold-start baseline /
新 v0 LLVIP / 旧 v0 RoadScene / v0 覆写 / v0 历史变迁 / EARLY_STOPPING=True
关键防御 / default False 隐藏陷阱 / cold-start golden schedule TRUE_LR 1.25e-4
WARMUP 1800 / MSLR [4,8,12] cover best_ep 6-9 / ES PATIENCE 5 / max_ep 18 /
N_SAMPLES_PER_SUBSET 3006 / SB_SUBSET_SAMPLE_REPLACEMENT False / aggressive
single-side H aug 沿用 v11/v12/v17 / 零 src 修改复用 v17 LLVIP infra /
启动日志 3 条 dispatch gate / wall-clock 2.5h 估算 / sanity-only 1-ep smoke /
v0 vs v17 sister ablation cold-start vs warm-start / OOD eval v0 ckpt 风险 /
v1-v15 ablation 链断裂,
English 'v0 cold-start LLVIP DDP', 'v0 vs v17 cold-start vs warm-start sister
ablation', 'v0 RoadScene to LLVIP DDP migration', 'EARLY_STOPPING True must
be explicit', 'cold-start outdoor.ckpt cross-modal golden schedule', 'reuse
v17 LLVIP infrastructure zero src modification', 'aggressive single-side H
aug shared with v11 v12 v17', '3 dispatch gates in startup log', 'wall-clock
estimate 2.5h ES likely', 'v1 to v15 ablation chain broken by v0 overwrite'.

v0 = path L (LLVIP cold-start baseline route, 2026-05-16 新增). Inheritance:
  eloftr_full.py (MegaDepth baseline, USE_* 全 default False)
  -> v0 (THIS skill, + LLVIP + aggressive H aug + DDP + cold-start schedule)
v0 vs v17 是 sister ablation (同数据集 LLVIP / 同 H aug / 不同 ckpt 起点);
v0 vs 旧 v0 RoadScene 是历史变迁 (数据集 + 硬件 + schedule 全换).

Companion:
  eloftr-v17-llvip (sister warm-start finetune from v14 on same LLVIP dataset;
    v17 是 v0 的"训过的 ckpt 起点"对照组; LLVIP infra 由 v17 落地, v0 复用)
  eloftr-v14-resolution-640 (v17 的 ckpt 起点; 不直接影响 v0 但 schedule
    黄金标准来自 v14 实测)
  eloftr-v10-msyn (DDP image-level pre-shard 设计 + N_SAMPLES_PER_SUBSET=
    per-rank size 修正先例 28750 = 115000/4)
  eloftr-v11-dualh / eloftr-v12-dualh-baseline (aggressive H aug 同款 KWARGS
    preset 来源)
  eloftr-cross-modal-experiments (chain 路线总览; 旧 v0 RoadScene 在 §0 链
    根, 新 v0 LLVIP DDP 是 §4 path L 新增节点, 待回填)
  eloftr-server-multigpu (DDP runtime + 5 traps + step time 实测)
  eloftr-eval-pipeline (eval_roadscene_all_singleh_finetuned.bat 协议 /
    OOD eval M3FD/RoadScene/METU)
  eloftr-results (跨版本数字总表; v0-LLVIP 数字待 backfill into
    results/eval_summary.md, **必须明确标注新 v0 vs 旧 v0 区分**)
  eloftr-tb-summary (训练侧 KPI 总表; v0 wall-clock / best_ep / val P@3
    待 backfill into results/tb_summary.md)
  eloftr-yurupeng-workspace (服务器子目录软链协议: data/LLVIP/{infrared,visible}/
    -> 只读源, data/LLVIP/index/ 仓库内实体)
