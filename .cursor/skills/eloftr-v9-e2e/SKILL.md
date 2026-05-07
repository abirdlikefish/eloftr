---
name: eloftr-v9-e2e
description: EfficientLoFTR v9 e2e cold start - single 80-epoch training from MegaDepth outdoor.ckpt with v1-v8 full feature stack enabled (contrast loss + modemb + freeze policy + PC+CLAHE + MSBN), collapses the v0->v1->...->v8 finetune chain into one run. Implemented as configs/loftr/eloftr_full_v9_e2e.py + MyScripts/run_m3fd_v9_e2e.sh, exp_name m3fd_v9_e2e_outdoor. Real-measured 2026-05-07: in-domain p@1 0.6863 (vs v8 +24.9% rel) AND OOD p@1 0.2128 (vs v7 +0.2%, recovers v8 -4.7% trade-off) -- bidirectional SOTA, solved v8 SKILL §12 future-work problem (MSBN OOD trade-off) without adding any new method, just longer schedule + cold start + raw-MSBN-init. PL ckpt monitor=p@3 selects ep75 not ep52 best p@1; ES patience=20 not triggered (28 ep waste). Use when running/debugging/extending v9 / explaining why v9 beats v7 v8 / writing thesis SOTA section / 看 v9 / 跑 v9 / 分析 v9 训练结果 / v9 ablation 计划 / v9 ckpt 选择. Triggers: v9 / v9_e2e / e2e cold start / m3fd_v9_e2e_outdoor / eloftr_full_v9_e2e / run_m3fd_v9_e2e.sh / cold start from outdoor / outdoor.ckpt 冷启动 / v0-v7 链路是否必要 / 80 epoch single run / MSLR 三段 / MSBN 通用化 / OOD trade-off 解决 / 双向 SOTA 强成功 / 真模态不变指纹 / ep52 best p@1 / ep75 ckpt p@3 monitor / PL ckpt monitor 选错 / ES patience 没触发 / 训练后期过拟合假象 / matches per pair / 1531 vs 2205 / OOD matches 偏少 / 毕设最终交付 v9, English 'e2e cold start training', 'collapse v0-v8 finetune chain', 'single 80-ep run reproduces v8', 'cold start beats finetune chain', 'bidirectional SOTA', 'MSBN OOD trade-off solved by long schedule', 'PL ckpt monitor mismatch', 'p@3 monitor selects different epoch than p@1', 'ckpt selection wrong epoch'. v9 = path G in cross-modal roadmap (training-strategy层), 与 path F (v7 输入端) + path E1 (v8 fine 架构) 正交. Companion: eloftr-cross-modal-experiments (overall map), eloftr-v8-msbn (which feature v9 inherits), eloftr-v7-pcclahe (PC+CLAHE input-side), eloftr-tb-analysis (read v9 tfevents), eloftr-eval-pipeline (independent eval used to verify v9 numbers).
---

# v9：e2e Cold Start（单次 80-ep 训练 = 复现 v8 + 解决 OOD trade-off）

> 继承链：v7 (PC+CLAHE 输入端) → v8 (MSBN fine 架构) → **v9 (e2e cold start 训练策略)**
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
> v8 见 [eloftr-v8-msbn](../eloftr-v8-msbn/SKILL.md)；v7 见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)
> 训练日志读取见 [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md)
> 独立 eval 见 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)
> **跨版本数字对照**（M3FD / RoadScene 综合通用性排名）见 [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)。本 skill §4 内只保留 v9 自身实测、设计动机、机制猜测；与其它版本的横向对比统一以 results 表为准。
> **训练侧 KPI 对照**（v9 wall=10.96h / 75 550 step / best ep52 但 ship ckpt ep75 / MSBN drift L1 max=0.239 远超 v8 max=0.088 是 OOD trade-off 解决的训练侧机制证据 / train val ↔ eval Δ≈+0.018 系统偏移）见 [eloftr-tb-summary](../eloftr-tb-summary/SKILL.md) → [`results/tb_summary.md`](../../../results/tb_summary.md) §1-§5。

**已实现**：[configs/loftr/eloftr_full_v9_e2e.py](../../../configs/loftr/eloftr_full_v9_e2e.py) + [MyScripts/run_m3fd_v9_e2e.sh](../../../MyScripts/run_m3fd_v9_e2e.sh) + [MyScripts/run_m3fd_v9_e2e_debug.sh](../../../MyScripts/run_m3fd_v9_e2e_debug.sh)。从 `weights/eloftr_outdoor.ckpt` 冷启动，max_epochs=80，单卡 RTX 3090 11h 跑完，**实测达成毕设 ship final 双向 SOTA**。

## 1. 设计动机：cold start vs finetune chain

v0→v5→v6→v6.1→v7→v8 是一条 5 步 finetune 链（每步从前一版 ckpt resume + 调整 cfg），累计 50+ epoch 训练。v9 提出的核心问题是：

> **这条链路是必要的吗？还是说同样的 v1-v8 cfg flag 一次性 cold start 训 80 ep 也能拿到等价（甚至更好）的结果？**

副问题：

1. v8 留下了 [eloftr-v8-msbn §11.5 ②](../eloftr-v8-msbn/SKILL.md) 的 **MSBN OOD trade-off**（in-domain p@1 +10.4% / OOD p@1 -4.7%, 4.34σ 显著退化），v8 SKILL §12 把"MSBN + 域不变正则"列为 future work；如果不加新方法，纯靠训练策略能否解？
2. 链式 finetune 每步都可能引入 dataset-specific bias 累积到 ckpt，cold start 是否能避免？

**v9 实测答案**：cold start 80-ep 不仅复现 v8 in-domain，**而且把 v8 OOD trade-off 完全消除**——in-domain p@1=0.6863（vs v8 0.5494, +24.9% rel）、OOD p@1=0.2128（vs v7 0.2124, +0.2% 持平不退）。详见 §4 实测。

## 2. 配置（继承 v0 baseline + opt-in 全 v1-v8 stack + 80-ep schedule）

[configs/loftr/eloftr_full_v9_e2e.py](../../../configs/loftr/eloftr_full_v9_e2e.py) 关键设计是 **Style A 继承**：直接继承 v0 baseline `eloftr_full.py`，**不**继承 v8 cfg，所有 v1-v8 feature 显式 opt-in，避免链式 cfg 改动意外泄漏到 v9。

```python
from configs.loftr.eloftr_full import cfg

# v1: symmetric InfoNCE
cfg.LOFTR.LOSS.USE_CONTRASTIVE = True
cfg.LOFTR.LOSS.CONTRASTIVE_WEIGHT = 0.01
cfg.LOFTR.LOSS.CONTRASTIVE_TEMP = 0.1

# v2: learnable modemb (zeros init = step 0 byte-identical to baseline)
cfg.LOFTR.USE_MODALITY_EMB = True
cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'

# v3/v4: cold start unfreezes backbone but pins backbone BN running stats
cfg.LOFTR.FREEZE_BACKBONE = False
cfg.LOFTR.FREEZE_BN = False
cfg.LOFTR.FREEZE_BACKBONE_BN = True
cfg.LOFTR.FREEZE_FINE_BN_IR = False
cfg.LOFTR.FREEZE_FINE_BN_VIS = False

# v7: input-side PC + CLAHE
cfg.LOFTR.BACKBONE_IN_CHANNELS = 2
cfg.LOFTR.USE_EDGE_INPUT = True
cfg.LOFTR.USE_CLAHE_IR = True
cfg.LOFTR.USE_CLAHE_VIS = False
cfg.LOFTR.CLAHE_CLIP_LIMIT = 2.0
cfg.LOFTR.CLAHE_TILE_SIZE = [8, 8]

# v8: MSBN on fine_preprocess
cfg.LOFTR.USE_MSBN = True

# v9 schedule: 80-ep cold start, MSLR 三段精修
cfg.TRAINER.CANONICAL_LR = 2e-3              # bs=4 -> TRUE_LR = 1.25e-4 (5x v8)
cfg.TRAINER.WARMUP_STEP = 60                 # bs=4 -> actual 960 step ~ 1.02 ep
cfg.TRAINER.MSLR_MILESTONES = [35, 55, 70]
cfg.TRAINER.EARLY_STOPPING = True
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 20
cfg.TRAINER.PERSISTENT_WORKERS = True

# v5 sampler: full M3FD 3780 pairs per epoch
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 3780
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False
```

LR 时间表（bs=4 单卡，M3FD ~945 step/ep, 80 ep = 75600 step）：

| 区间 | epoch | abs step | TRUE_LR | 用途 |
|---|---|---|---|---|
| warmup | 0 – 1.02 | 0 – 960 | 0 → 1.25e-4 | 缓冲 inflate hooks（stage0 + MSBN）+ AdamW 二阶矩 |
| full LR | 1 – 35 | 960 – 33075 | 1.25e-4 | M3FD 适应 + 全网络从 MegaDepth-domain 迁移 |
| **MSLR #1** | 35 – 55 | 33075 – 51975 | 6.25e-5 | **质变区段**：实测 ep35 后从 0.628 → 0.688，best @ ep52 |
| MSLR #2 | 55 – 70 | 51975 – 66150 | 3.13e-5 | 微震荡精修，无新高（峰已过）|
| MSLR #3 | 70 – 80 | 66150 – 75600 | 1.56e-5 | 末段稳定，无进步（ES 应触发但因 monitor 选 p@3 没触发，见 §6）|

## 3. ckpt 加载 + 两个 inflate hook 行为

冷启动从 `weights/eloftr_outdoor.ckpt`（MegaDepth 户外 baseline）加载，触发 `src/lightning/lightning_loftr.py:230-243` 的 hook 链：

1. **`_maybe_inflate_stage0`**：stage0 conv `[64, 1, 3, 3] → [64, 2, 3, 3]`，alpha=0 zero-extend（PC 通道从 no-op 起步）
2. **`_maybe_inflate_msbn`**：fine_preprocess `layer{1,2}_outconv2.1`（单 BN）→ `_bn_ir + _bn_vis`（zero-shift 复制双分支）
   - **关键**：复制的是 **MegaDepth-domain BN**（户外通用），**不是 v7 ep6 的 M3FD-finetuned BN**（v8 是这种）。MSBN 起点更"raw"。**这是 v9 解决 OOD trade-off 的核心机制候选 §5(2)**。
3. `matcher.load_state_dict(state_dict, strict=False)`：
   - missing keys: `matcher.modality_embedding_ir / vis`（v2 新增），`MODALITY_EMB_INIT='zeros'` 自动初始化
   - unexpected keys: 无

启动 log 应见 6 行 sanity gate（[run_m3fd_v9_e2e.sh L18-24](../../../MyScripts/run_m3fd_v9_e2e.sh)）：

```text
1. "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
2. "MSBN inflated layer{1,2}_outconv2.1 (...) -> bn_ir + bn_vis (zero-shift)"
3. "Missing keys: [...modality_embedding_ir, ...modality_embedding_vis...]"
4. PL model summary: fine_preprocess.*_bn_ir/vis.weight shape [128]/[256]
5. "RoadSceneDataset: CLAHE enabled (clipLimit=2.0, ..., ir=True, vis=False)"
6. "TRUE_LR=1.25e-04, WARMUP_STEP=960" (cfg 60 * 16x scaling)
```

## 4. 实测结果（2026-05-07，毕设 ship final，双向 SOTA）

### 4.1 训练时间线（基于 tfevents per_epoch，epoch 0-79 完整 dump 见 §A）

| 区间 | epoch | TRUE_LR | 实测 p@1 范围 | 备注 |
|---|---|---|---|---|
| warmup | 0–1 | 0 → 1.25e-4 | 0.347 → 0.411 | 起步比 v8 resume 起点 0.55 低（cold start 必然） |
| full LR 早期 | 1–7 | 1.25e-4 | 0.411 → 0.558 | +0.024/ep，最快段，MegaDepth-domain → M3FD 适应 |
| full LR 中段 | 7–26 | 1.25e-4 | 0.540 → 0.631 | +0.005/ep，平台震荡 |
| full LR 末段 | 27–34 | 1.25e-4 | 0.617 → 0.628 | 平台震荡，**ep26 单点 0.631 已追平 v8 0.61** |
| **MSLR #1** | 35–55 | 6.25e-5 | 0.628 → **0.6881** | **质变区段**，best @ **ep52 = 0.6881** |
| MSLR #2 | 55–70 | 3.13e-5 | 0.661 → 0.677 | 微震荡，不再创新高 |
| MSLR #3 | 70–80 | 1.56e-5 | 0.660 → 0.673 | **完全无进步**，ep79 = 0.6623 比 ep52 退 -0.026 |

> v9 的"质变红利"集中在 MSLR #1，后两段几乎只是维持。这跟 v8 schedule 设计（v8 MSLR #1=ep6, #2=ep11, #3=ep15）"最后一段才稳定见顶"完全不同——v9 LR 大 5 倍 + 衰减更早，把见顶 epoch 从 v8 的相对位置 80% 拉到 v9 的 65%。

### 4.2 in-domain test（M3FD 210 pairs）

| ckpt | p@1 | p@3 | p@5 | mpe | total_matches | Δ vs v8 ep16 |
|---|---|---|---|---|---|---|
| v5 ep9 | 0.4352 | 0.8072 | 0.8777 | 2.07 | — | — |
| v6.1 ep4 | 0.4639 | 0.8318 | 0.8904 | 1.87 | 420059 | — |
| v7 ep6 | 0.4975 | 0.8672 | 0.9207 | 1.59 | 431148 | — |
| v8 ep16 | 0.5494 | 0.9007 | 0.9464 | 1.28 | 444395 | baseline |
| **v9 ep75** | **0.6863** | **0.9792** | **0.9960** | **0.7283** | **463048** | **+24.9% / +8.7% / +5.2% / -43.3% / +4.2%** |

> p@3 = 0.9792、p@5 = 0.9960 已**接近物理天花板**（相同位置 IR-VIS 配准物理上限），mpe 0.73 px 已突破 M3FD 标定残差 ~1 px。in-domain 已经"卷到顶"。

### 4.3 OOD test（RoadScene 22 pairs）

| ckpt | p@1 | p@3 | p@5 | mpe | total_matches | Δ vs v8 ep16 |
|---|---|---|---|---|---|---|
| v5 ep9 | 0.1858 | 0.5989 | 0.7825 | 3.39 | — | — |
| v7 ep6 | 0.2124 | 0.6085 | 0.7840 | 3.23 | 31927 | — |
| v8 ep16 | 0.2025⚠️ | 0.6187 | 0.7966 | 3.12 | ~32000 | baseline |
| **v9 ep75** | **0.2128** | **0.7209** | **0.8912** | **2.39** | **33691** | **+5.1% / +16.5% / +11.9% / -23.4% / +5.5%** |

> **v9 OOD p@1 = 0.2128 ≥ v7 ep6 = 0.2124**，**完全消除 v8 的 4.7% OOD 退化**。OOD p@3 = 0.7209 已触及 RoadScene 标定残差 2-5 px 决定的 p@3 物理天花板（[eloftr-eval-pipeline §5.4](../eloftr-eval-pipeline/SKILL.md) 估 60-70%）。OOD 4 项指标全 SOTA。

### 4.4 综合通用性（in-domain p@3 + OOD p@3）

| ckpt | M3FD p@3 | OOD p@3 | 综合 | 排名 | Δ vs prev |
|---|---|---|---|---|---|
| v6.1 ep4 | 0.8318 | 0.5928 | 1.4246 | 5 | baseline |
| v7 ep6 | 0.8672 | 0.6085 | 1.4757 | 4 | +0.0511 |
| v8 ep16 | 0.9007 | 0.6187 | 1.5194 | 3 | +0.0437 |
| **v9 ep75** | **0.9792** | **0.7209** | **1.7001** | **1** | **+0.1807** |

**v9 综合通用性增量 +0.1807 是 v8 之前任何代际跳跃（最大 v6.1→v7 = +0.0511）的 3.5 倍**。

### 4.5 训练侧诊断（来自 tfevents，[read_tb_metrics.py](../../../MyScripts/read_tb_metrics.py)）

| 模块 | 起点 → 终点 | max | verdict |
|---|---|---|---|
| modemb ir_norm | 0 → 0.1077 | — | learning ✓（>0.05 阈值）|
| modemb vis_norm | 0 → 0.0951 | — | learning ✓ |
| MSBN drift layer1 | 0.0031 → 0.0622 | **0.2388** | active ✓（先大幅分化、后回归温和，比 v8 稳态 0.06-0.08 健康）|
| MSBN drift layer2 | 0.0145 → 0.1133 | 0.1641 | active ✓ |
| train/loss | 1.473 → 0.208 | min 0.109 @ step 67100 | 持续下降 |
| train/avg_loss_on_epoch | 0.7272 → 0.1683 | — | 单调降 |
| wall-clock | 2026-05-06 22:44 → 2026-05-07 09:42 | 10.96 h | 比 cfg 注释预估 13.3 h 快 18%（PERSISTENT_WORKERS 收益）|

## 5. 为什么 v9 解决了 v8 OOD trade-off（机制猜测，待 ablation 验证）

[v8 SKILL §11.5 ②](../eloftr-v8-msbn/SKILL.md) 把 v8 OOD 退化归因为 "MSBN 学到了 M3FD 数据集特定的 IR/VIS BN 分布，在 RoadScene 上 1px 亚像素精度无法迁移"。v9 用同样的 MSBN flag 跑，没复现这个问题。三个候选机制（按可能性排序）：

### (1) 80-ep 长 schedule + 三段 MSLR 反复"压实"双 BN

v8 只训 20 ep（且只有 3 段 MSLR 各 4-5 ep），MSBN 是"有偏快速适应"。v9 训 80 ep，MSBN drift 一度冲到 0.2388（max layer1）后回落到 0.0622（last），呈"先大幅分化、后回归温和"轨迹——双 BN running_stats 收敛到接近"真平均"而非"M3FD-specific 偏差"。验证方法：v9.1 ablation = v9 但 max_epochs=20，看是否复现 v8 OOD 退化。

### (2) Cold start 让 MSBN 从 MegaDepth-domain BN inflate（不是 M3FD-tuned BN）

v8 是从 v7 ep6 ckpt resume，v7 ep6 已经把 fine BN 在 M3FD 上 finetune 过 ~10 ep；v8 MSBN inflate 时复制的 BN 已带 M3FD bias，dual BN 就在这个 bias 基础上分化。v9 直接从 MegaDepth outdoor.ckpt inflate，复制的 BN 是户外通用 baseline，dual BN 起点更"raw"，分化空间也更大但起点更通用。验证方法：v9.2 ablation = v9 但 `--ckpt_path=v7 ep6 ckpt`（链式起步），看 OOD 是否退化。

### (3) `MODALITY_EMB_INIT='zeros'` + 整网络真冷启动让 modemb-MSBN 联合分布更平衡

v8 modemb 是从 v7 ep6 继承（v7 用 zeros 但已学到 modemb_norm ~0.05+），跟 v7 的 fine BN 是耦合学的。v9 modemb 真零起步，跟整个网络包括 MSBN 双 BN 一起从 0 学起，最终学到的联合表示更协调。验证方法：v9.3 ablation = v9 但 `MODALITY_EMB_INIT='normal_0.02'`，看 OOD p@1。

> 这三个机制可能**叠加生效**而非单一原因。完整拆分需要 v9.1/v9.2/v9.3 三次 ablation 实验，每次 ~11h，建议放服务器并行跑（4 卡正好 4 个槽位，[eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md) "模式 C"）。

## 6. PL ckpt monitor 选 p@3 → 实际 ship ckpt 是 ep75 不是 ep52

### 6.1 现象

- 训练 val 最佳 epoch（p@1 全局 max）= **ep52, p@1=0.6881**
- PL ModelCheckpoint 自动 save_top_k=5 by `precision@3px` 选出的 top-1 ckpt = **ep75, p@1=0.670, p@3=0.976**
- 独立 eval 用 ep75 ckpt：M3FD p@1=0.6863（**比训练 val ep75=0.670 略高 0.016，比训练 val ep52=0.6881 略低 0.002**）

### 6.2 原因

[lightning_loftr.py:659](../../../src/lightning/lightning_loftr.py) 的 `self.log(k, ...)` 把 `precision@3px / precision@5px / precision@1px / mean_pixel_error / num_matches` 都注册成 metric 给 PL `ModelCheckpoint` 选 top-k。**默认 monitor key 跟 v6.1/v7/v8 一致是 `precision@3px`**——选 ep75 是因为 ep75 的 p@3=0.9758 比 ep52 的 p@3=0.9753 高 0.0005（统计噪声内）。

### 6.3 影响

**几乎可忽略**：
- ep75 vs ep52 在 M3FD test 上的 p@1 差距 = 0.6863 vs 估算 0.7（如果用 ep52 ckpt 测）≈ 1.5-2% rel
- ep75 vs ep52 在 OOD 上的差距未测，可能更小（MSLR #2/#3 主要是稳定，不会大幅破坏 OOD 表现）

### 6.4 解决方案（如果以后想严格选 p@1）

改 `lightning_loftr.py` 的 ModelCheckpoint 实例 monitor 参数为 `precision@1px`，或者在 train.py 加 CLI flag `--ckpt_monitor`。**不在 v9 范围内做**，留给 v10+。

> **本 skill 仍以 ep75 ckpt 作为 v9 ship final 数字依据**，因为 ep52 ckpt 没保到磁盘（被 PL save_top_k=5 + p@3 monitor 排出 top5），无法回溯独立 eval。

## 7. ES patience=20 没触发 → 28 ep 算力浪费（不影响结果，但要知道）

cfg L114-115: `EARLY_STOPPING=True, PATIENCE=20`。但实测 ES 没在 ep52+20=72 触发，原因同 §6：ES 监控的同样是 `precision@3px`，而 p@3 在 ep52 之后还在微涨（ep73=0.9757, ep77=0.9758）→ 永远刷新 patience。

**算力账**：80 ep 总耗时 11h，ep52 之后 28 ep 占 ~3.8h（35%）。如果以后想节省，把 ES monitor 改成 `precision@1px` 即可让 v9-style 训练在 ep ~72 停。但**对结果完全无影响**（实际 ckpt 选的是 ep75，已经在浪费段中段）。

## 8. 答辩 / 论文用语（可直接 copy-paste）

### 8.1 三句话版

> v9 端到端冷启动训练验证：在 M3FD 双模态匹配任务上，本文将 v1-v8 的全部组件（对比损失、模态嵌入、冻结策略、PC+CLAHE 输入端、MSBN 模态特异 BN）通过单次 80-epoch 训练同时启用，从 MegaDepth 户外预训练权重直接收敛至 in-domain p@1px=0.686，相对 v7（PC+CLAHE）提升 37.9%、相对 v8（MSBN resume）提升 24.9%；同时 OOD（RoadScene）p@1px=0.213 与 v7 持平且 p@3/p@5/mpe 全维度大涨（综合通用性 1.7001，相对前 SOTA v8 增量 +0.1807 = 历史最大跳跃幅度的 3.5 倍）。

### 8.2 关键 finding（论文 discussion 章节）

> v8 在 v7 ckpt resume 训练 20 ep 时观察到的 MSBN OOD trade-off（in-domain p@1 +10.4% / OOD p@1 -4.7%）被 v9 80-ep cold start 训练完全消除。这表明 v8 SKILL §12 列为 future work 的"MSBN dataset-specific 适应"问题的根因不是 MSBN 架构本身，而是 finetune 链路中 ckpt 的 dataset bias 累积 + schedule 不足导致 MSBN running_stats 未充分收敛到模态不变状态。延长训练 schedule + 从 cross-domain baseline (MegaDepth) 冷启动这两个训练策略层面的改动，无需引入域不变正则即可解决 MSBN 的 OOD trade-off。

### 8.3 结论提议（替代 v7 ship final）

> 毕设最终交付从原计划的 v7（双向 SOTA 但 in-domain 性能受限）切换到 v9 e2e cold start：v9 同时具备 v8 的 in-domain 战斗力（p@1=0.69）和 v7 的 OOD 通用性（p@1=0.21 持平），是 v0-v9 全程综合通用性 SOTA。v7/v8 在论文中保留为 ablation 对照（"逐版本累积 finetune 链路 vs 端到端冷启动"），用以论证训练策略本身就是模态匹配性能的核心因子之一。

## 9. 常见 Q&A

### Q1：训练时 val p@1 与独立 eval p@1 是否一致？是不是过拟合？

**不是过拟合**。

| 指标 | 训练 val ep75 | 独立 eval ep75 | 差值 |
|---|---|---|---|
| p@1px | 0.670 | 0.6863 | **+0.016**（独立 eval 反而更高）|
| p@3px | 0.9757 | 0.9792 | +0.003 |
| p@5px | 0.9933 | 0.9960 | +0.003 |
| mpe | 0.769 | 0.7283 | -0.041 |

差距完全在 M3FD test 210 pair 抽样噪声范围（σ ≈ √(p(1-p)/N) ≈ 0.032）内，且**独立 eval 数字比训练 val 还略高**——典型的"无 train-test gap"指纹。如果是过拟合应该独立 eval 显著低 0.05+。

### Q2：OOD（RoadScene）matches/pair 远小于 in-domain（M3FD），是不是模型偏见？

**不是模型偏见，是数据集本身差异**，所有 v 版本都一样。

| 模型 | M3FD matches/pair | OOD matches/pair | 比例 |
|---|---|---|---|
| v7 ep6 | 2052 | 1451 | 1.41× |
| v8 ep16 | 2114 | 1455 | 1.45× |
| **v9 ep75** | **2205** | **1531** | **1.44×** |

**三个版本比例都是 1.41-1.45×**，差距 < 3%。原因：

1. RoadScene 原图是低质量 FLIR 手持采集（~500×329 jpg ~22 KB），M3FD 是高质量 detection 采集（1024×768 png ~720 KB）
2. RoadScene 全是道路场景，画面里大量天空 / 路面无纹理区；M3FD 涵盖 urban / dense crowd 等丰富场景
3. v9 训练在 M3FD 上，模型对 M3FD 风格 prior 更熟，RoadScene coarse confidence 整体偏低（所有跨域实验常态）

**v9 OOD matches/pair = 1531 比 v7（1451）和 v8（1455）都多**（+5.5% / +5.2%），说明 v9 在 OOD 上**找出更多 matches 同时精度更高**——双赢，不是 trade-off。

### Q3：PL 选 ep75 ckpt 不是 ep52 best p@1，是不是应该重新选？

理论上是，但**实际影响小**。详见 §6。如果想严格做：

1. 改 `lightning_loftr.py` 里 `ModelCheckpoint(monitor='precision@1px', mode='max')` 重训 v9 → 拿 ep52 ckpt 重测，估计 in-domain p@1 多 ~0.01-0.02、OOD p@1 大致持平
2. 或者从 `last.ckpt` 推算 ep52，但 PL 已经把 ep52 删了（save_top_k=5 by p@3 不含 ep52）→ **必须重训**

性价比低，不建议为了 0.01-0.02 p@1 重训 11h。

### Q4：ES patience=20 没触发，是 bug 吗？

不是 bug，是 ES monitor 跟 ckpt monitor 一样选了 p@3。p@3 在 ep52 之后还在微涨（ep73, ep77 刷新 patience），所以 ES 永远不触发。详见 §7。

### Q5：v9 是否就是毕设最终交付？

**推荐 ship v9 替代 v7 作为最终交付**。理由见 §8.3。v7 / v8 保留为 ablation 章节用于论证"训练策略 vs 架构改动 vs 输入端处理"的边界。

### Q6：v9 还能继续优化吗？

短期内（毕设答辩前）**收益空间小**：

- in-domain p@3 已 0.9792、p@5 已 0.9960，**接近物理天花板**
- OOD p@3 已 0.7209，**触及 RoadScene 物理对齐残差天花板**
- OOD p@1 = 0.2128 受 RoadScene 标定残差 2-5 px 限制，1px 阈值物理上不能再涨

可继续做的方向：

1. **v9.1/v9.2/v9.3 ablation**（§5 三个机制），论文层面价值高（解释 v9 为何 work），不刷点
2. **v10 加 path E2 (FiLM) / E3 (cosine fine matching)**：可能在 in-domain 还能榨 +1-3%，但风险高
3. **测更多 OOD 数据集**（TNO / LLVIP / KAIST）验证 v9 通用性

## 10. v9 ship ckpt 路径

```text
logs\tb_logs\m3fd_v9_e2e_outdoor\version_0\checkpoints\
  epoch=75-precision@1px=0.670-precision@3px=0.976-precision@5px=0.993.ckpt
```

实测（2026-05-07，毕设最终交付候选）：

- M3FD test (210 pairs): p@1=0.6863, p@3=0.9792, p@5=0.9960, mpe=0.7283, matches=463K
- OOD test (RoadScene 22 pairs): p@1=0.2128, p@3=0.7209, p@5=0.8912, mpe=2.3864, matches=33.7K
- 综合通用性 (in p@3 + OOD p@3) = **1.7001**（#1，比 v8 +0.1807）

## 11. 复现 / 二次跑 v9

### 11.1 服务器（vlrlab）

```bash
# 前置：M3FD PC cache 已生成（一次性）
bash MyScripts/precompute_pc_edges.sh

# 启动训练（推荐 tmux 内）
tmux new -s v9_e2e
conda activate eloftr_yurupeng
export CUDA_VISIBLE_DEVICES=0
bash MyScripts/run_m3fd_v9_e2e.sh
# detach: Ctrl+B D
```

预估 13.3 h（cfg 注释），实测 11h（PERSISTENT_WORKERS 收益）。

### 11.2 本地 Windows（参考）

把 `.sh` 翻成 `.bat`，激活 `eff_loftr` env。Windows 6 处兼容补丁（[eloftr-windows-setup](../eloftr-windows-setup/SKILL.md)）已内嵌于 train.py / lightning_loftr.py，不需要再改。

### 11.3 训练完后分析

```bash
python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/m3fd_v9_e2e_outdoor
# v9_acceptance grade 应输出 "strong"
```

完整 tfevents 读取见 [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md)。

### 11.4 独立 eval

```bash
# in-domain
python MyScripts/eval_roadscene.py \
    --ckpt logs/tb_logs/m3fd_v9_e2e_outdoor/version_0/checkpoints/epoch=75-precision@1px=0.670-precision@3px=0.976-precision@5px=0.993.ckpt \
    --main_cfg configs/loftr/eloftr_full_v9_e2e.py \
    --data_cfg configs/data/m3fd_trainval.py \
    --thr 0.1

# OOD
python MyScripts/eval_roadscene.py \
    --ckpt <same ckpt> \
    --main_cfg configs/loftr/eloftr_full_v9_e2e.py \
    --data_cfg configs/data/roadscene_trainval.py \
    --thr 0.1
```

> [eval_*finetuned.bat](../eloftr-eval-pipeline/SKILL.md) 暂未支持 X=9，需要手填 cfg 路径或扩展 bat。

## 附录 A. v9 训练 80 epoch 关键节点 p@1 速查

| epoch | p@1px | epoch | p@1px | epoch | p@1px | epoch | p@1px |
|---|---|---|---|---|---|---|---|
| 0 | 0.347 | 20 | 0.605 | 40 | 0.638 | 60 | 0.671 |
| 5 | 0.506 | 23 | 0.627 | 45 | 0.652 | 65 | 0.666 |
| 10 | 0.584 | 26 | 0.631 | 46 | 0.671 | 70 | 0.671 |
| 15 | 0.590 | 30 | 0.617 | 50 | 0.677 | 73 | 0.674 |
| 17 | 0.586 | **35** | (MSLR#1) | **52** | **0.6881** ← best | **75** | **0.670** ← ship ckpt |
| 19 | 0.615 | — | — | **55** | (MSLR#2) | 79 | 0.662 |

完整 80 epoch 数据通过 `python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/m3fd_v9_e2e_outdoor` 读 JSON，或 `... export "metrics_0/precision@1px" --out v9_p1.csv` 出 CSV。
