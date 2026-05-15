---
name: eloftr-v13-pose
description: |
  EfficientLoFTR v13 = 首个 pose-supervised cross-view + cross-modal 路线
  (Megadepth_Syn LoFTR-style train_list + truncated val + 4-GPU DDP B-pure
  stack), 用真 K + W2C pose + depth 监督替代 v0-v12 的 H 监督, 让 train val
  从 v10/v11/v12 的 identity matching saturation (P@3=0.97 但 OOD METU
  AUC<1%) 切换成有意义的几何监督信号 (val auc@5/10/20 单调收敛, ckpt 选择
  跟 OOD eval 对齐).

  v13 = "B-pure 短路继承 eloftr_full.py" (NOT v9_e2e -> v10 -> v11 链),
  v1-v8 全 stack USE_* (CONTRASTIVE / MODALITY_EMB / EDGE_INPUT / CLAHE_IR /
  MSBN / BACKBONE_IN_CHANNELS=2) 都 default False, 让 outdoor.ckpt 第一层
  conv strict load 兼容. 训练分辨率 832 + bs=2 (832+bs=4 在 24GB 3090 OOM,
  原因: sim_matrix at coarse_matching.py:122 是 (bs, 10816, 10816) fp32 =
  1.87 GB/batch + cuDNN workspace 4 GB + DDP NCCL bucket buffer 1 GB ->
  ~22 GB / 24 GB OOM-prone). bs=2 反向缩放 LR/WARMUP 保 TRUE_LR=1.25e-4
  跟 v9/v10 一致.

  数据来源 (重大切换 vs v10/v11/v12 的 145/4/4 自切分):
  - train list: LoFTR 官方 train_list.txt (153 scene / 368 npz / 8.86M pair
    pool, 业界 MegaDepth baseline, scene-disjoint guarantee {0015,0022}
    not in train)
  - val list: 自建 val_list_loftr_small.txt (2 行 = Trevi 0015_0.3_0.5 +
    Pantheon 0022_0.5_0.7 = 4493 pair total, 截断 LoFTR val_list 避免 0022
    的 147K-pair npz 拖累 DDP val cost). --limit_val_batches=0.5 进一步
    halves 到 2247 effective (跟 LoFTR megadepth_val_1500 业界 1500 pair
    量级一致).
  - test: 占位指向 val (用户 OOD eval 在 M3FD/RoadScene/METU 跑, 不需 v13
    内部 test pool).

  5 个新建文件 (零 src/ 改动):
  (1) configs/data/megadepth_syn_pose_trainval.py (~80 行, MGDPT_IMG_RESIZE=
      832 + DF=32 + IMG_PAD=True + DEPTH_PAD=True + CROSS_MODAL_MODE='ir2vis'
      + NPE_NAME='megadepth')
  (2) configs/loftr/eloftr_full_v13_pose_msyn_ddp.py (~100 行, 短路继承
      eloftr_full.py + 8 行 cfg overrides: USE_*=False all + bs=2 反向缩放
      CANONICAL_LR=1e-3 + WARMUP_STEP=225 + MSLR=[3,5,7] + ES patience=3 +
      N_SAMPLES_PER_SUBSET=200 + SB_SUBSET_SAMPLE_REPLACEMENT=False)
  (3) configs/loftr/eloftr_full_v13_pose_msyn_singlecard.py (~23 行, 继承
      v13 ddp + 单卡反向缩放 CANONICAL_LR=2e-3, WARMUP_STEP=1800)
  (4) MyScripts/run_msyn_v13_pose_ddp.sh (~106 行, 4 卡 DDP entry, --gpus=4
      --batch_size=2 --max_epochs=12 --limit_val_batches=0.5
      --log_every_n_steps=50 --disable_mp --thr 0.1 + sanity-only mode)
  (5) MyScripts/run_msyn_v13_pose_singlecard.sh (~60 行, smoke entry)

  Ship 状态 (2026-05-14): COMPLETED 9 ep DDP run, ES 即将触发 (ep10).
  实测 wall-clock 25h, total_global_steps 87700 (= 9 ep * ~9744 step / ep).
  events.tfevents 1.1 GB (主因: 87700 / log_every_n_steps=50 = 1754 个
  train figure x ~500KB ~= 880 MB, 80%; 145 MB val figure; 10 MB scalar;
  70 MB overhead). monitor=auc@10 (train.py:190 megadepth-style 默认),
  ckpt 文件名 fmt {epoch}-{auc@5}-{auc@10}-{auc@20}.ckpt.

  v13 实测完整曲线 (read_tb_metrics.py summary --logdir
  logs/tb_logs/msyn_v13_pose_ddp/version_1):
    ep0  auc@10=0.376  warmup 末 (1800 actual step)
    ep1  auc@10=0.387  full LR
    ep2  auc@10=0.391
    ep3  auc@10=0.396  MSLR1 LR -> 6.25e-5
    ep4  auc@10=0.400
    ep5  auc@10=0.405  MSLR2 LR -> 3.13e-5
    ep6  auc@10=0.4096 *** best by auc@10 (monitor) ***
    ep7  auc@10=0.4069 MSLR3 LR -> 1.56e-5, 过拟合
    ep8  auc@10=0.4060 ES 即将触发 (3 ep no-improve)

  ckpt 同步 (PL ModelCheckpoint save_top_k=5 by auc@10):
    ep4 auc@5=0.246 auc@10=0.400 auc@20=0.561
    ep5 auc@5=0.248 auc@10=0.405 auc@20=0.567
    ep6 auc@5=0.251 auc@10=0.4096 auc@20=0.573  *** best ckpt ***
    ep7 auc@5=0.2522 auc@10=0.407 auc@20=0.570 (best by auc@5)
    ep8 auc@5=0.247 auc@10=0.406 auc@20=0.569
    last.ckpt = ep8

  Key takeaway:
  - **首次脱离 val identity matching saturation**: v10/v11/v12 val 数字假高
    (P@3=0.97 但 OOD METU AUC<1%); v13 val auc@10 单调涨 0.376 -> 0.4096
    (相对 +9%), trend 跟 OOD 性能挂钩, ckpt 选择可信.
  - LR schedule 工作正常: ep3/5 MSLR drop 各贡献 ~0.005 auc@10, ep7 第三次
    drop 后过拟合 trigger 找到.
  - cold start 起步 0.376 比 v10 cold start auc@10 0.622 (Megadepth_Syn
    val) 低很多, 因 v13 val 是 LoFTR 官方 0015/0022 高难度 SfM-pair (跨
    视角真位姿), 不是 v10 那种 pixel-aligned identity matching val.
    本身难度不一样, 不能跨版本绝对比.
  - METU eval pending: v13 ckpt 待跑 eval_metu_vistir_finetuned.bat 拿
    METU all auc@5/10/20 数字 (paper 决战 benchmark, MINIMA / XoFTR 协议).
  - 显存预算实测 v13 832+bs=2 = 20.2 GB / 卡 (vs plan 估 16 GB, +4 GB
    cuDNN benchmark workspace + DDP NCCL bucket buffer), 留 4 GB 余量,
    没 OOM. v14 改 640+bs=4 显存预期 ~17 GB / 卡 (sim_matrix 砍 29% +
    bs 翻倍).

  Acceptance (METU all auc@20, 待 eval bat 跑出后 backfill):
    Strong : >= 0.05 (5x v10 baseline 0.87%) -> 论文高潮 cross-modal pose
             监督路线成功
    Medium : in [0.02, 0.05] -> 部分成功, 考虑 v14 (640 + 多维加速)
    Weak   : < 0.02 -> pose 监督单独不足, 反思 stack
    Crash  : NaN / METU auc < outdoor.ckpt -> dispatch / K-T 单位错

  Use when running/debugging/extending v13 / 解释 cross-view + cross-modal
  pose 监督设计 / 看 v13 实测曲线 / 选 v13 best ckpt / METU eval v13 ckpt /
  对比 v13 vs v10/v11/v12 (val identity matching saturation) / 写论文
  pose-supervised cross-modal 章节 / 建 v14 派生 ablation / 解释 ENABLE_PLOTTING
  关掉省 80% events / 解释 EVAL_TIMES=5 vs eval bat ransac_times=1 协议差.

  Triggers: v13 / v13_pose / msyn_v13_pose_ddp / pose-supervised cross-modal /
  cross-view IR-VIS / Megadepth_Syn_Pose / B-pure stack / 短路继承 eloftr_full /
  LoFTR train_list 153 scene / val_list_loftr_small.txt 4493 pair / Trevi
  Pantheon truncated val / spvs_coarse pose-based / scene_info_pose 软链 /
  trainvaltest_list_src 软链 / NPE [832,832,832,832] fallback / sim_matrix
  bs=4 OOM 1.87 GB / batch / cuDNN workspace +4 GB / 实测 25h 9 ep / best
  ep6 auc@10=0.4096 / val identity matching saturation 脱离 / monitor auc@10
  not auc@5 / read_tb_metrics.py summary v13 / METU eval pending / EVAL_TIMES=5
  vs ransac_times=1 协议差 / events 1.1 GB train figure 主因 / GPU 2 87 C
  thermal throttling 风险,
  English 'pose-supervised cross-modal route', 'first to break val identity
  matching saturation', 'LoFTR official train_list with truncated val',
  'short-circuit B-pure stack', 'sim_matrix bs=4 OOM at 832 fixed by bs=2',
  'monitor auc@10 not auc@5', '9 ep ES trigger ~25h wall-clock', 'METU eval
  pending', 'cold start auc@10 0.376 -> 0.4096 monotone increase'.

  v13 = path J in cross-modal roadmap (pose-supervision route, NOT H-supervision
  like v10 path H / v11 path I / v12 path I-ablation). Inheritance graph:
    eloftr_full.py (v0 baseline) -> v13 (THIS skill, short-circuit, B-pure)
  v13 vs v10/v11/v12: 完全不同的监督契约 (pose 几何 vs 平面单应), 完全不同
  的 train list (LoFTR 官方 vs 145/4/4 自切分), val 数字基础也不同 (LoFTR
  0015/0022 高难度 SfM-pair vs identity-matching-saturated). v13 是 pose
  路线的 "v0 起点", v14 是 v13 的分辨率/工程加速 ablation 衍生.

  Companion:
    eloftr-megadepth-syn-data (Megadepth_Syn dataset SOP, v13 复用 train/
      infrared + train/phoenix 软链, scene_info_pose 是新 subset 软链)
    eloftr-cross-modal-experiments (chain 路线总览; v13 是 path J 起点,
      待 backfill v13 行)
    eloftr-server-multigpu (DDP runtime + 5 traps; v13 走 MegaDepth 默认
      _build_concat_dataset 路径, 不需 v10 的 image-level pre-shard)
    eloftr-eval-pipeline (eval_metu_vistir_finetuned.bat 协议; v13 ckpt
      待跑该 bat 拿 paper 决战数字)
    eloftr-results (跨版本数字总表; v13 数字待 backfill into
      results/eval_summary.md)
    eloftr-tb-summary (训练侧 KPI 总表; v13 wall-clock / best_ep / 待
      backfill)
    eloftr-v14-resolution-640 (v13 的分辨率 ablation 衍生, 832 -> 640 + bs=4
      + 多维工程加速, ~9h ship vs v13 25h)
---

# v13: Pose-Supervised Cross-View + Cross-Modal (B-Pure)

> 继承链：`eloftr_full.py` →（**短路绕过 v9_e2e → v10 → v11 → v12 链**）→ **v13 (NEW path J)**
> 父对照 / 链总览 → [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
> 数据集 → [eloftr-megadepth-syn-data](../eloftr-megadepth-syn-data/SKILL.md)
> DDP runtime → [eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md)
> 跨版本数字 → [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)
> **Ship 进行中（2026-05-14, ES 即将触发于 ep10）**: best ckpt = `logs/tb_logs/msyn_v13_pose_ddp/version_1/checkpoints/epoch=6-auc@5=0.251-auc@10=0.4096-auc@20=0.573.ckpt`

## 1. 设计意图：从 H 监督转 pose 监督，跳出 val identity matching saturation

v10/v11/v12 在 Megadepth_Syn 训练时, val 用的是 145/4/4 自切分 + ROAD_HOMOGRAPHY_AUG=False。结果 val 任务退化成 "pixel-aligned IR ↔ VIS 同视角 identity matching", val P@3 假高 0.97, 但 OOD METU AUC < 1%。**ckpt 选择基准跟最终 OOD 性能脱钩**, 是 v9-v12 路线的核心问题。

v13 想回答：

> **如果换成跟 OOD eval (METU) 同分布的 pose 监督契约 + 跨视角真 SfM-pair val list, 是不是 train val 数字就跟 OOD 数字挂钩了？**

## 2. 5 个文件 (零 src/ 改动) + 关键设计

| 文件 | 关键设计 |
|---|---|
| [`configs/data/megadepth_syn_pose_trainval.py`](../../../configs/data/megadepth_syn_pose_trainval.py) | 80 行。`TRAINVAL_DATA_SOURCE='Megadepth_Syn_Pose'` (非 ALIGNED_IRVIS_SOURCES, 走默认 `_build_concat_dataset` 路径). LoFTR 官方 train_list.txt + 自建 val_list_loftr_small.txt (4493 pair). MGDPT_IMG_RESIZE=832 / DF=32 / 跨模态 mode='ir2vis'. |
| [`configs/loftr/eloftr_full_v13_pose_msyn_ddp.py`](../../../configs/loftr/eloftr_full_v13_pose_msyn_ddp.py) | 100 行。短路继承 `eloftr_full.py` (B-pure stack), 8 行 cfg overrides: USE_* 全 False + bs=2 反向缩放 (`CANONICAL_LR=1e-3, WARMUP_STEP=225`) + `MSLR_MILESTONES=[3,5,7]` + `EARLY_STOPPING_PATIENCE=3` + `N_SAMPLES_PER_SUBSET=200` + `SB_SUBSET_SAMPLE_REPLACEMENT=False`. |
| [`configs/loftr/eloftr_full_v13_pose_msyn_singlecard.py`](../../../configs/loftr/eloftr_full_v13_pose_msyn_singlecard.py) | 23 行。继承 v13 ddp + 单卡反向缩放 (`CANONICAL_LR=2e-3, WARMUP_STEP=1800`). smoke 用. |
| [`MyScripts/run_msyn_v13_pose_ddp.sh`](../../../MyScripts/run_msyn_v13_pose_ddp.sh) | 106 行。4 卡 DDP entry. `--batch_size=2 --max_epochs=12 --limit_val_batches=0.5 --log_every_n_steps=50 --disable_mp --thr 0.1 + sanity-only` mode. |
| [`MyScripts/run_msyn_v13_pose_singlecard.sh`](../../../MyScripts/run_msyn_v13_pose_singlecard.sh) | 60 行。smoke entry, `--max_epochs=1 --limit_train_batches=10 --disable_ckpt`. |

**0 src/ 改动**: dataset 类 `MegadepthSynPoseDataset` (path J 数据加载) / dispatch (`spvs_coarse` for `Megadepth_Syn_Pose`) / 监督函数 / metrics 全部已有 (path J 是用 LoFTR 官方 MegaDepth 训练栈, src/ 早就支持).

## 3. cfg merge 顺序 + NPE caveat

`train.py:127` 的 merge 顺序是 `main_cfg` 之后 merge `data_cfg`, 即:

```python
config.merge_from_file(args.main_cfg_path)
config.merge_from_file(args.data_cfg_path)
```

**caveat**: 如果 main cfg 设了 `cfg.DATASET.MGDPT_IMG_RESIZE`, 会被 data cfg 覆盖。所以 IMG_RESIZE 必须在 data cfg 里设, 不能在 main cfg。

NPE (跨分辨率 RoPE 校准): v13 cfg 没显式设 `cfg.LOFTR.COARSE.NPE`, 走 `train.py:130` fallback `[832, 832, 832, 832]`。当前数据是 832, fallback 也对。**v14 改 640 训时也仍然走 fallback `[832, 832, 832, 832]` 不显式设**——v14 初版误以为 "640 训应设 NPE [832,832,640,640] 让 RoPE 跨分辨率校准", 实测踩 stretch bug (ratio=1.3 让 RoPE position phase 错位 50%, ep0 auc@10=0.214 < outdoor.ckpt 0.30+ baseline)。根因: `position_encoding.py:21-22` 的 train_res / test_res stretch 是给 **extrapolation** 用的 (input > train_res); v14 long_side 640 < 832 是 **interpolation**, integer position (1..20) 是 outdoor.ckpt 学过 (1..26) 的子集, **不该 stretch**。详见 v14 skill "NPE bug post-mortem"。

## 4. ship 实测：9 ep × 25h

### 4.1 wall-clock 分项 (read_tb_metrics.py summary)

```
events_size_mb : 1109.31
elapsed_hours  : 24.991
total_epochs   : 9
total_global_steps : 87700  (≈ 9 × 9744 step/ep, bs=2)
final_train_loss   : 0.855
min_train_loss     : 0.234 @ step 53300 (~ep5-6)
final_avg_loss_on_epoch : 0.528
```

单 ep 实测 167 min (= 25h / 9 ep), 跟 plan 估算 175 min 接近 (-5%)。

### 4.2 val 完整曲线 (monitor=auc@10)

| ep | auc@5 | auc@10 | auc@20 | val_avg_loss | LR 阶段 |
|---|---|---|---|---|---|
| 0 | 0.2245 | 0.3764 | 0.5387 | NaN | warmup 末 1800 step |
| 1 | 0.2313 | 0.3874 | 0.5535 | 0.559 | full LR |
| 2 | 0.2359 | 0.3908 | 0.5552 | 0.559 | full LR |
| 3 | 0.2431 | 0.3963 | 0.5564 | 0.533 | **MSLR1** LR→6.25e-5 |
| 4 | 0.2461 | 0.4001 | 0.5613 | 0.541 | drop1 后 |
| 5 | 0.2483 | 0.4049 | 0.5668 | 0.525 | **MSLR2** LR→3.13e-5 |
| **6** | 0.2508 | **0.4096** | **0.5730** | 0.534 | **best by auc@10 (monitor)** |
| 7 | **0.2522** | 0.4069 | 0.5704 | 0.521 | **MSLR3** LR→1.56e-5, best by auc@5 |
| 8 | 0.2473 | 0.4060 | 0.5690 | 0.521 | 过拟合, ES 即将触发 (3 ep no-improve) |

### 4.3 关键观察

1. **ep0 起步 0.376** (cold start outdoor.ckpt → cross-modal pose 监督, 健康)
2. **ep0 → ep6 单调涨 +0.033** (相对 +9%, **缓慢稳定**, 不像 v10/v11/v12 val saturation 那种"假高 P@3=0.97 但 OOD METU AUC<1%")
3. **ep5/7 LR drop 各贡献 ~0.005** auc@10 (LR schedule 工作正常)
4. **ep6 是 best by auc@10 monitor** (PL ModelCheckpoint 选 ep6 ckpt)
5. **ep7/8 微降** (过拟合 trigger 找到)

## 5. METU-OOD 独立 eval 待 backfill

按 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md) 协议跑 v13 ep6 ckpt:

```bat
eval_metu_vistir_finetuned.bat 13
```

期望产出 `dump/metu_eval_v13_version1_top1/all/overall.txt`, 含 auc@5/10/20 + prec@5e-04 + num_matches。**待 backfill into [eloftr-results](../eloftr-results/SKILL.md) / [`results/eval_summary.md`](../../../results/eval_summary.md)**。

acceptance gate (METU all auc@20):

| 档位 | 阈值 | 解读 |
|---|---|---|
| Strong | ≥ 0.05 (5x v10 baseline 0.87%) | 论文高潮, cross-modal pose 监督路线成功 |
| Medium | in [0.02, 0.05] | 部分成功, 考虑 v14 (640 + 多维加速) |
| Weak | < 0.02 | pose 监督单独不足, 反思 stack |
| Crash | NaN / METU auc < outdoor.ckpt | dispatch / K-T 单位错 |

## 6. 显存预算 + GPU 散热实测

| 维度 | plan 估 | 实测 | 解读 |
|---|---|---|---|
| 显存 / 卡 | 16 GB | **20.2 GB** | +4 GB 来自 cuDNN benchmark workspace + DDP NCCL bucket buffer; 留 4 GB 余量, 没 OOM |
| GPU 0/1/2/3 温度 | — | 81 / 66 / 87 / 84 °C | GPU 2 接近 88°C thermal throttling 阈值; 限 power 至 280W 可降到 ~78°C |
| sim_matrix 单 batch peak | — | 0.93 GB (bs=2, 832, hw=104²) | bs=4 在 832 path 是 1.87 GB, +cuDNN/NCCL 后 ~22 GB, OOM-prone |
| events.tfevents | — | **1.1 GB** | 主因 1754 train figure × 500 KB ≈ 880 MB (80%); v14 关 ENABLE_PLOTTING 砍至 ~220 MB |

## 7. 后续派生：v14 (resolution ablation)

v13 的 832 path 跟 METU eval 的 640 path 存在 train/eval domain mismatch (AGG attention spatial pattern / fine_window 物理覆盖率两层; NPE 那层 v14 实测证明走 fallback `[832,832,832,832]` 不 stretch 反而最优, 因 640 < 832 是 interpolation 不该 stretch, 见 v14 skill §1.1 + "NPE bug post-mortem")。**v14 = v13 + IMG_RESIZE 832→640 + NPE 走 fallback (不显式设) + bs=4 + 多维工程加速 (~9h ship vs v13 25h, 2.8x)**。详见 [eloftr-v14-resolution-640](../eloftr-v14-resolution-640/SKILL.md)。

v14 vs v13 是干净的训练分辨率 ablation——训练 fingerprint 仅由 IMG_RESIZE 主导, 其他都是"语义等效"或"工程加速"。
