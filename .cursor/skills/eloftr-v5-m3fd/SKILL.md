---
name: eloftr-v5-m3fd
description: EfficientLoFTR v5 cross-modal experiment - scale up from RoadScene (177 train) to M3FD (3780 train, 21x step density), inheriting v4 REVISION 2's FREEZE_BACKBONE_BN strategy. Use when user mentions v5, v5_m3fd, eloftr_full_v5_m3fd, m3fd_v5_combined, run_m3fd_v5_*.bat, "M3FD epoch only 50 step instead of 945", N_SAMPLES_PER_SUBSET, SB_SUBSET_SAMPLE_REPLACEMENT, "v5 OOD beats v2 OOD", "v5 vs v2 OOD matrix", "v2 RoadScene 22 张过拟合", or wants to debug v5 / understand why v5 beats v2 OOD. Triggers: v5 / v5_m3fd / m3fd 训练 / 数据扩展 / 21 倍 step / OOD 矩阵 / 跨数据集对比 / 跑 v5 / 看 v5 / 为什么 v5 OOD 更强 / 路径 B 数据扩展, English 'scale up training data', 'why v5 beats v2 OOD', 'M3FD baseline', 'data scaling fixes BN convergence', 'v2 overfits 22 RoadScene pairs', 'progress bar 945 vs 50', 'sampler override mandatory'. v5 inherits v4 REVISION 2 + v2 modemb + v1 contrast architecture; data integration details see eloftr-m3fd-data.
---

# v5：M3FD 数据扩展（路径 B 实现）

> 继承链：v4 REVISION 2 → **v5 (M3FD)** → v6 / v6.1 → v7
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
> v4 见 [eloftr-v3-v4-freeze](../eloftr-v3-v4-freeze/SKILL.md)；v6 见 [eloftr-v6-finetune](../eloftr-v6-finetune/SKILL.md)
> M3FD 数据接入见 [eloftr-m3fd-data](../eloftr-m3fd-data/SKILL.md)

## 1. 配置（继承 v4 REVISION 2 全部架构）

[configs/loftr/eloftr_full_v5_m3fd.py](../../../configs/loftr/eloftr_full_v5_m3fd.py)：

```python
from configs.loftr.eloftr_full_v4_combined import cfg

# 必须 opt-in：单数据源 IR-VIS 强制走全集
cfg.TRAINER.N_SAMPLES_PER_SUBSET         = 3780   # default=200, override 到 M3FD 全集
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False  # default=True, 走 randperm 让每对各见 1 次/epoch

# Schedule 适配 M3FD 21x step 密度
cfg.TRAINER.WARMUP_STEP             = 20          # 320 abs step ≈ 3.4% total training
cfg.TRAINER.MSLR_MILESTONES         = [3, 5, 7]   # M3FD epoch ~21x longer
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3           # M3FD val=210 less noisy than RoadScene val=22
```

> 前两个 override 是 v5 能成立的**前提**。漏掉任意一个会让 v5 退化成"5% 数据 + 50 step/epoch"，schedule 全部失准。原理见下节"sampler 默认值陷阱"。

## 2. sampler 默认值陷阱（v5 强制 opt-in 的根因）

LoFTR 的训练 sampler ([src/datasets/sampler.py](../../../src/datasets/sampler.py)) 是为 ScanNet/MegaDepth 的"多场景"结构设计的——每个 .npz = 一个 scene = sampler 眼里的一个 subset，每 epoch 从**每个 subset 抽 200 个样本**（with replacement）。`N_SAMPLES_PER_SUBSET=200` 是 **per-scene quota**，不是 per-dataset budget。

RoadScene/M3FD 每个数据集只有一个扁平图像列表，[src/lightning/data.py:248](../../../src/lightning/data.py) 把它包装成 `ConcatDataset([ds])` → `n_subset == 1`，结果 200 这个默认值退化为"per-dataset 上限 200 样本/epoch"。

| 数据集 | 全集 | 默认每 epoch 实际样本 | 行为 |
|--------|------|---------------------|------|
| RoadScene train | 177 | 200 (with replacement) | 巧合：177 < 200，期望覆盖 ~63%/epoch，50 epoch 后基本全见过；**v0-v4 的所有结果都是在这个默认值下产生的**，但因为数据集本身 < 200，没有被欠采样 |
| M3FD train | 3780 | 200 (with replacement) | **严重欠采样**：每 epoch 只见 5.3% 的数据；10 epoch 总共 ≤2000 个独特样本（约 47% 数据从未见过）；"21× step 密度"假设彻底破产 |

### 2.1 必须 override 的两个开关（M3FD/LLVIP/KAIST 等大数据集）

```python
cfg.TRAINER.N_SAMPLES_PER_SUBSET         = 3780     # 改成数据集全集大小
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False    # 走 randperm 而不是 randint
```

**两个都必须改**：
- 只改前者 → `sampler.py:50-52` 仍走 `torch.randint(0, 3780, (3780,))`，with-replacement，每 epoch 期望唯一样本仅 ≈ 3780 × (1 − 1/e) ≈ **2389 个**
- 同时改后者 → 走 `sampler.py:53-61` 的 `torch.randperm(3780)[:3780]` 分支，**每 epoch 完整 permutation，3780 全部各见 1 次**，跨 epoch 因为 PL 每 epoch 重建 sampler（[data.py:367](../../../src/lightning/data.py)）所以 permutation 重新随机

### 2.2 `default.py` 不改 N_SAMPLES_PER_SUBSET 的原因

200 这个默认值对 ScanNet/MegaDepth 是正确的——94/196 个 scene × 200 = 18800/39200 样本/epoch，正好匹配 LoFTR 原训练 step 预算。改 default 会破坏 ScanNet/MegaDepth 训练。

**单数据源 IR-VIS 必须由各自的 LoFTR config 显式 opt-in**（v5_m3fd 已加，未来 v5_llvip / v5_kaist 也必须各自加）。

### 2.3 进度条诊断

PL 训练时第 0 个 epoch 的进度条 `Epoch 0: ... A/B [train+val]`：
- ✅ 正确（v5 sampler override 已生效）：`B = 3780/4 + 210/1 = 1155`，train 占 945
- ❌ 错误（默认值生效）：`B = 200/4 + 210/1 = 260`，train 占 50
- 看到 `50/260` **立即停训**：sampler override 没生效

## 3. Bat 入口

- [run_m3fd_v5_debug.bat](../../../MyScripts/run_m3fd_v5_debug.bat) / [_small.bat](../../../MyScripts/run_m3fd_v5_small.bat) / [_combined.bat](../../../MyScripts/run_m3fd_v5_combined.bat)
- `--exp_name` 前缀从 `roadscene_v*` 改为 `m3fd_v5_*`（区分历史 RoadScene 实验）
- `--max_epochs=10` 在 .bat 里设（M3FD epoch 长 21×，10 个就够 9450 step）

## 4. v5 实测：训练曲线指纹

v5 跑完 10 epoch（max_epochs 满，未触发 EarlyStopping），训练时长 3h54m，events 文件 93MB（间接验证 945 step/epoch 的 sampler override 生效）。

Top-5 ckpt（其余被 ModelCheckpoint 挤出）：

```text
epoch 3:  p@1=0.393  p@3=0.784  p@5=0.868
epoch 6:  p@1=0.399  p@3=0.784  p@5=0.865
epoch 7:  p@1=0.417  p@3=0.784  p@5=0.859
epoch 8:  p@1=0.418  p@3=0.786  p@5=0.860
epoch 9:  p@1=0.418  p@3=0.789  p@5=0.864   ← best
```

Δ 形状：**p@5 微降 / p@3 微升 / p@1 大涨**——正好是 v4 REVISION 2 的 "p@5 反超 + p@3 平 + p@1 暴跌" 的**镜像反转**。这印证 [eloftr-v3-v4-freeze §6.4](../eloftr-v3-v4-freeze/SKILL.md) 的 BN 收敛假设：v5 ~9450 step 远超 ~3000 step BN 收敛阈值，fine BN running stats 已基本收敛，p@1 不再塌。

## 5. In-domain (M3FD val) vs RoadScene val 对比

| 指标 | v4 REV2 (RS, ep9) | v5 (M3FD, ep9) | 解读 |
|---|---|---|---|
| p@1px | 0.329（崩） | **0.418** ↑ | BN 收敛后 fine 不再乱推 |
| p@3px | 0.716 | **0.789** ↑ | 真实涨点，但 v2 RoadScene val 噪声大不可直接比 |
| p@5px | 0.814 | 0.864 | 同上 |

## 6. 关键 OOD 评估：v2 vs v5 在 RoadScene/M3FD 双 test 矩阵

**这才是判断 v5 是否成功的真正证据**——同 val set 的 apples-to-apples 对比。

| 训练 → 测试 | RoadScene test (22 对) | M3FD test (210 对) |
|---|---|---|
| **v2 v1 ep62** (RS-train) | p@1=**0.7567** p@3=**0.8086** p@5=0.8151 mpe=**1.97** | p@1=0.1192 p@3=0.4971 p@5=0.6948 mpe=4.66 |
| **v5 v0 ep9** (M3FD-train) | p@1=0.1858 p@3=0.5989 p@5=0.7825 mpe=3.39 | p@1=0.4352 p@3=0.8072 p@5=0.8777 mpe=2.07 |

**对角线（in-domain）**：v5 in-domain p@3 (0.8072) ≈ v2 in-domain p@3 (0.8086)，**几乎平手**；v5 p@5 (0.8777) 反超 v2 (0.8151)。**v5 唯一短板是 in-domain p@1**（0.4352 vs v2 0.7567）。

**反对角线（OOD）**：

| OOD 指标对称对比 | v2 → M3FD | v5 → RoadScene | v5 优势 |
|---|---|---|---|
| OOD p@1px | 0.1192 | 0.1858 | **+56%** |
| OOD p@3px | 0.4971 | 0.5989 | **+20%** |
| OOD p@5px | 0.6948 | 0.7825 | **+13%** |
| OOD mpe | 4.66 | 3.39 | **-27%（更小更好）** |

**v5 在每一个 OOD 指标上都显著优于 v2 OOD**。综合通用性（in-domain + OOD 的 p@3px）：v5 = 1.4061，v2 = 1.3057，v5 领先 8%。

## 7. v2 高 p@1 的真相：**RoadScene 22 张 val/test 严重过拟合**

v2 RoadScene-train 的 in-domain p@1=0.7567 跨到 M3FD 直接崩到 0.1192——**绝对损失 0.64，相对衰减 84%**。这种崩坏不是"换 val 集难度差异"能解释的（如果是难度差异，p@3 也应该等比例崩，但 p@3 只衰减 38%）。

机制：
- RoadScene v2 训练 60+ epoch × ~88 step/epoch = ~5000 step
- 训练集 177 张 with-replacement 每张被见 ~50 次
- v2 effective LR 极小（~2e-5）给了模型**精细记忆**每张训练图局部模式的时间
- 同 split 流程产出的 22 张 test 与 22 张 val 高度同分布 → test 也被"间接记忆"
- 跨到 M3FD（训练时从未见过的图）→ v2 的精细记忆完全失效

> **v2 的 p@1=0.7567 其实是负面信号，不是优点**——它说明 v2 的 fine refinement 学的是"22 张图的记忆"，不是"通用 IR-VIS sub-pixel 对齐能力"。

## 8. v5 = 路径 B 胜利的最终判定

按 [configs/loftr/eloftr_full_v5_m3fd.py](../../../configs/loftr/eloftr_full_v5_m3fd.py) docstring 的判定标准：
- ✅ **假设确认**：v5 in-domain p@3=0.807 ≥ 0.80，且 p@1=0.435 远超 v4 REV2 的 0.329 → BN 收敛 + 数据扩展双重假设成立
- ✅ **路径 B 核心收益**：v5 OOD 全面碾压 v2 OOD → **学到了真正的通用 IR-VIS 表示，不是数据集特化的记忆**
- ⚠️ **唯一短板**：in-domain p@1 还有 ~0.10-0.15 提升空间（vs 理论 ~0.55-0.65 的"通用 + 精修双修"上限）

短板归因：
1. **v5 effective LR 太高**：v5 epoch 0-3 一直跑 1.25e-4，是 v2 同期 ~2e-5 的 6 倍。fine sub-pixel offset 在高 LR 下永久震荡，无法精细收敛 → 由 [eloftr-v6-finetune](../eloftr-v6-finetune/SKILL.md) 慢 LR resume 解决。
2. **结构性瓶颈**：modemb 在 fine 路径完全失效（[eloftr-v2-modemb §modemb 信号路径](../eloftr-v2-modemb/SKILL.md)），fine_matching 对模态盲 → v7 走输入端 PC+CLAHE 间接缓解（[eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)），**v8 走 fine 架构 MSBN 直接破解**（[eloftr-v8-msbn](../eloftr-v8-msbn/SKILL.md)）。

## 9. v6+ 联合训练候选（未实现）

如果想做 RoadScene + M3FD + LLVIP 联合训练（cross-dataset），工程改动：
1. 写 `LLVIPDataset` 类（参考 [src/datasets/roadscene.py](../../../src/datasets/roadscene.py) 模板，复用 A3 padding + Homography aug）
2. 用 `ConcatDataset(roadscene_train, m3fd_train, llvip_train)` + 域均衡 sampler
3. **修复**：[eloftr-m3fd-data §8](../eloftr-m3fd-data/SKILL.md) 的 "单 batch 多源 assert" 必须先去掉
4. **val/test 仍只用 RoadScene 22 + 22 对** 或独立 test 集，保证可比性
5. bs 维持 4（16GB VRAM 限制）；max_epochs 减小

预期：解决 BN 收敛 + 过拟合 + InfoNCE 负样本池 + modemb 步数四个瓶颈，代价是 2-3 天工程。
