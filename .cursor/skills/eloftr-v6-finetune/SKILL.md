---
name: eloftr-v6-finetune
description: EfficientLoFTR v6 / v6.1 cross-modal experiments - resume from v5 ckpt with v2-style slow LR to lift fine sub-pixel p@1px, plus v6.1 spillover hotfix for PyTorch allocator fragmentation. Use when running, debugging, or extending v6 / v6.1. Triggers: v6 / v6.1 / v6_finetune / v6_1_finetune / 慢 LR finetune / 从 v5 ckpt 继续训 / resume / 续训 / fine 精修 / --ckpt_path / spillover / 显存碎片 / expandable_segments / PYTORCH_CUDA_ALLOC_CONF / 跑 v6 / 看 v6.1 / 选 v6.1 ep4 / PERSISTENT_WORKERS / N_VAL_PAIRS_TO_PLOT, English 'resume from ckpt', 'slow LR fine refinement', 'PyTorch allocator fragmentation', 'CUDA Virtual Memory API', 'GPU-Util 100% but power low', 'VRAM 16GB shared GPU memory', 'weights-only resume', 'gate 7-9 ckpt resume sanity', '"PCIe spillover"'. v6 is path D in cross-modal roadmap; v6.1 is v6's spillover hotfix sharing the same schedule.
---

# v6 / v6.1：从 v5 ckpt 慢 LR 精修 + Spillover 修复

> 继承链：v5 (M3FD) → **v6 (resume + 慢 LR)** → **v6.1 (spillover hotfix)** → v7 (PC + CLAHE)
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
> v5 见 [eloftr-v5-m3fd](../eloftr-v5-m3fd/SKILL.md)；v7 见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)
> **跨版本数字对照**（v6 vs v6.1 ckpt 选择、v6/v6.1 共用 v6_1_finetune.py cfg 的脚注）见 [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)。
> **训练侧 KPI 对照**（v6 spillover 中断 wall=2.73h / v6.1 hotfix wall=1.52h / 同 cfg val P@1 +4.1% 的实测）见 [eloftr-tb-summary](../eloftr-tb-summary/SKILL.md) → [`results/tb_summary.md`](../../../results/tb_summary.md) §1 注 ¹ + §5 观察 3。

> **v6 / v6.1 共一个 skill**：v6.1 是 v6 的 spillover hotfix，零 cfg override，schedule 完全继承——两者一起读才能正确归因 expandable_segments。

## 1. 设计动机：为什么不重训而是 resume

[eloftr-v5-m3fd](../eloftr-v5-m3fd/SKILL.md) 的 OOD 矩阵证明 v5 已经学到通用 IR-VIS 表示，唯一短板是 **in-domain p@1=0.4352 / OOD p@1=0.1858**——根本原因是 v5 早期 LR 太高让 fine sub-pixel 永久震荡。v6 = 在 v5 ckpt 上**用 v2-style 极慢 LR 继续训练**，目标是把 p@1 从 0.435 推向 0.55-0.65 的"通用 + 精修双修"上限。

| 选项 | 优势 | 劣势 |
|---|---|---|
| 重新从 pretrained 训（v5 配方 + 慢 LR） | 干净 | ~9000 step 把 BN 重新养到收敛，浪费 v5 已有成果 |
| **v5 ckpt resume + 慢 LR**（v6 选） | 直接接管 v5 已收敛的 BN running stats + 通用 backbone，前 0 step 就在"v5 终态"上 | 需要 `--ckpt_path` 加载 + 不能动 sampler/数据/freeze 配置（破坏 ckpt 兼容） |
| 切到 RoadScene 慢 LR 精修 | 可能复刻 v2 in-domain 0.74 | **几乎肯定破坏 v5 OOD 优势**——RoadScene 22 张训练集太小，会把 v5 通用 BN 拉回到记忆模式 |

v6 选 resume 路径是为了 **"加法不减法"**：保住 v5 已经赢的 OOD 通用性，只补 in-domain p@1 这块短板。

## 2. v6 关键 schedule 重设（vs v5）

[configs/loftr/eloftr_full_v6_finetune.py](../../../configs/loftr/eloftr_full_v6_finetune.py)：

```python
from configs.loftr.eloftr_full_v5_m3fd import cfg

cfg.TRAINER.CANONICAL_LR            = 4e-4   # v5=2e-3, /5 → TRUE_LR=2.5e-5 ≈ v2 effective peak
cfg.TRAINER.WARMUP_STEP             = 50     # v5=20; resume 不需要长 warmup, 50 → 800 abs step
cfg.TRAINER.MSLR_MILESTONES         = [15, 25, 35]  # v5=[3,5,7]; 让大部分时间停在 full LR
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 12    # v5=3; 慢 LR 收敛慢 + p@1 优化曲线噪声大需更高 patience

# Performance overrides (v6-only opt-in via cfg)
cfg.TRAINER.PERSISTENT_WORKERS      = True   # default False; keep workers alive across 50 epoch, saves ~25min
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT     = 1      # default 32; 50 epoch x 35 figs would bloat TB by ~70%
```

继承自 v5 不变的字段（**不要在 v6 config 里再覆写一遍**）：`N_SAMPLES_PER_SUBSET = 3780` / `SB_SUBSET_SAMPLE_REPLACEMENT = False` / `FREEZE_BACKBONE_BN = True` / `USE_CONTRASTIVE = True` / `USE_MODALITY_EMB = True`。

### LR 数学

`train.py` auto-scaling：`TRUE_LR = CANONICAL_LR × bs/64 = 4e-4 × 4/64 = 2.5e-5`，正好与 v2 effective peak（~2.67e-5）同量级。

| 区间 | epoch | abs step | TRUE_LR |
|---|---|---|---|
| warmup | 0 - 0.85 | 0 - 800 | 0 → 2.5e-5 |
| full | 0.85 - 15 | 800 - 14175 | 2.5e-5 |
| MSLR #1 | 15 - 25 | 14175 - 23625 | 1.25e-5 |
| MSLR #2 | 25 - 35 | 23625 - 33075 | 6.25e-6 |
| MSLR #3 | 35 - 50 | 33075 - 47250 | 3.13e-6 |

总训练 step ≈ 47k（约 v5 9.45k 的 5 倍），fine BN 有充分时间在低 LR 下精细收敛。预计 ~20h（v5 3.9h × 5）。

## 3. `--ckpt_path` 在本仓的精确语义（**和 PL 原生 --resume_from_checkpoint 不同**）

[train.py:72](../../../train.py) + [src/lightning/lightning_loftr.py:64-67](../../../src/lightning/lightning_loftr.py)：

```python
if pretrained_ckpt:
    state_dict = torch.load(pretrained_ckpt, ...)['state_dict']
    self.matcher.load_state_dict(state_dict, strict=False)
```

**只 load model weights**，不恢复以下任何状态：
- AdamW second moment（重新初始化为 0）
- LR scheduler step count / last LR（按 v6 config 从 0 走）
- epoch counter / global_step（从 0 开始）
- PL trainer state、EarlyStopping counter（freshly init）

后果：
- v6 的 `max_epochs=50` 是 **literally 跑 50 个新 epoch**（而不是 50−9=41）
- v6 的 `WARMUP_STEP=50 → 800 abs step` 在 v6 epoch 0 真实生效
- v6 的 `MSLR=[15,25,35]` 也按 v6 epoch 计数生效
- v5 ep9 的 1.56e-5 LR + 已衰减的 MSLR 状态都被丢弃，正是想要的

> 想要严格断点续训（恢复 epoch + scheduler + optimizer），需换 PL 原生 `--resume_from_checkpoint`，并解决 PL 1.x 的 `weights_only` 兼容（[eloftr-windows-setup](../eloftr-windows-setup/SKILL.md) 表 6 已 patch）。本项目目前**不需要**此用法。

## 4. v6 性能优化（v6-only opt-in via cfg）

v5 跑了 3h54m，v6 50 epoch 预计 ~20h。原 plan 提了 4 个加速方案，v6 只采纳 2 个零风险项，且通过 cfg opt-in 严格隔离。

### 4.1 决策矩阵

| 改动 | 在 v6 上的取舍 | 原因 |
|---|---|---|
| persistent_workers | **保留 (opt-in)** | `cfg.TRAINER.PERSISTENT_WORKERS` 默认 False + v6 主动 opt-in；v0-v5 重训保字节级一致 |
| 去掉 --disable_mp | **不做** | finetune 已收敛模型对 fp16 数值噪声比从头训更敏感，保留 v0-v5 兜底实践 |
| N_VAL_PAIRS_TO_PLOT=1 | **保留 (opt-in)** | 不影响指标，仅减 figure 数；v0-v5 行为不变 |
| --limit_val_batches=50 | **不做** | M3FD val `shuffle=False`，PL 取前 50 张 = systematic bias，v6 ckpt 选择会偏 |

### 4.2 persistent_workers 与 numpy RNG

[src/datasets/roadscene.py:249](../../../src/datasets/roadscene.py) 用 `np.random` 做 augmentation；项目里没有 `worker_init_fn`，所以：

| 场景 | numpy state |
|---|---|
| 不开 persistent_workers（v0-v5 默认） | 每 epoch worker 重 spawn，numpy state 从主进程 fork 时 global state 重新开始 |
| 开 persistent_workers（v6 opt-in） | worker 跨 epoch 复用，numpy state 在 epoch 间持续累积 |

两条路径**统计意义完全等价**（augmentation 分布相同），但**具体随机数序列不字节级一致**。v0-v5 走默认 False 严格复现历史；v6 走 True 单 epoch 内一致但跨 epoch 不可严格复现。

### 4.3 N_VAL_PAIRS_TO_PLOT vs ENABLE_PLOTTING

两者完全正交：

| Flag | 控制位置 | v6 状态 |
|---|---|---|
| `cfg.TRAINER.ENABLE_PLOTTING` | [lightning_loftr.py:285](../../../src/lightning/lightning_loftr.py) **训练阶段**画图 | False（[m3fd_trainval.py:73](../../../configs/data/m3fd_trainval.py) 强制） |
| `cfg.TRAINER.N_VAL_PAIRS_TO_PLOT` | [lightning_loftr.py:311](../../../src/lightning/lightning_loftr.py) **验证阶段**画图 | 1（v6 config 显式 opt-in） |

## 5. v6 实测：spillover 中断

| epoch | p@1 | p@3 | p@5 | wall time | 状态 |
|---|---|---|---|---|---|
| 0 (val end) | 0.400 | 0.788 | 0.865 | ~25 min | ckpt load + warmup ramp |
| 1 | 0.409 | 0.773 | 0.849 | 06:56 | fast (~7 min/epoch baseline) |
| 2 | 0.423 | 0.790 | 0.863 | 06:47 | |
| 3 | 0.417 | 0.787 | 0.861 | 06:46 | |
| 4 | 0.408 | 0.794 | 0.869 | 07:01 | |
| 5 | 0.421 | 0.796 | 0.866 | 06:46 | |
| **6** | **0.425** | **0.801** | **0.873** | **46:39** | **spillover starts (16.6 GB VRAM)** |
| 7 | — | — | — | aborted | >80 min projected at 4.36 s/step → user 中断 |

v6 ep6 = v6 全程最佳 + 已经在每个 in-domain 指标上反超 v5 ep9（v5 = 0.418 / 0.789 / 0.864）。

## 6. v6.1：v6 spillover 修复版（PyTorch allocator fragmentation cure）

**已实现**：[configs/loftr/eloftr_full_v6_1_finetune.py](../../../configs/loftr/eloftr_full_v6_1_finetune.py) + [MyScripts/run_m3fd_v6_1_finetune.bat](../../../MyScripts/run_m3fd_v6_1_finetune.bat)。**v6 schedule 全继承，零 override**——v6.1 跟 v6 的全部差别集中在 .bat 一行环境变量 + ckpt 起点上移。

### 6.1 诊断：PyTorch caching allocator 碎片化

`nvidia-smi` 在 v6 ep7 中段：

```text
VRAM:  15876 / 16303 MiB  (97.4%, only 427 MiB free)
Power: 85W / 300W         (28%; GPU underutilised but kernels at 100%)
```

"GPU-Util 100% + 功耗远低于 TDP" 是 PCIe-bottlenecked GPU 的指纹——VRAM 满了，PyTorch 用 shared GPU memory（system RAM via PCIe 4.0 ~32 GB/s vs VRAM ~700 GB/s = **22× 慢**）。

根因：默认 PyTorch caching allocator 是 slab-based（固定尺寸块）。v6 跑 6 epoch 后混合分配模式（train batch + val batch + matplotlib figure + ckpt save）让 free pool 碎片化，即使总 free > 1 GB，也找不到单个连续大块给新 gradient tensor 用 → fallback 到 shared memory。

加重因素：
- Cursor / Edge WebView2 / Epic Games / Steam / NVIDIA App / asus_framework 等 21 个非训练 GPU 进程共享 16 GB，悄悄占了 2-4 GB
- v6 的 `PERSISTENT_WORKERS=True` 让 6 worker + pinned memory buffer 跨 50 epoch 持续存活

### 6.2 Cure：expandable_segments

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

机制：把 PyTorch allocator 从 slab-based 切换为 CUDA Virtual Memory API。物理页可以非连续，但通过 GPU MMU 映射成虚拟连续段。**虚拟地址永远连续 → 碎片不可能发生**。代价：每个 segment 第一次分配多 ~2-3 ms，后续零成本。

### 6.3 v6.1 vs v6 diff（共 4 项，全部在 .bat）

1. **新加** `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（必须在 `python` 之前、`conda activate` 之后）
2. `main_cfg`: `eloftr_full_v6_finetune.py` → `eloftr_full_v6_1_finetune.py`（零 override 纯继承，仅日志可追溯）
3. `--exp_name`: `m3fd_v6_finetune` → `m3fd_v6_1_finetune`（独立 TB 目录）
4. `--ckpt_path`: v5 ep9 → **v6 ep6**（`epoch=6-precision@1px=0.425-precision@3px=0.801-precision@5px=0.873.ckpt`）

其他全部继承 v6（schedule / sampler / freeze / contrastive / modemb / persistent_workers / N_VAL_PAIRS_TO_PLOT），**故意一变量改动**，便于将来归因 expandable_segments 是否真治根。

### 6.4 操作 SOP（启动 v6.1 前必看）

1. 用任务管理器关掉非训练 GPU 应用（Cursor / Edge WebView2 / Epic Games / Steam / NVIDIA App / TranslucentTB / Notepad / 浏览器 WebGL 标签页）
2. **从独立 WindowsTerminal 双击 .bat 启动**，不要从 Cursor 集成终端启动（关 Cursor 会杀训练）
3. `nvidia-smi` 确认非训练进程 VRAM 占用 < 1 GB 才启动
4. 头 30 行启动日志通过 v6 的 gate 1-6 + v6.1 的 gate 7-9（见各 config docstring）

### 6.5 v6.1 新增 validation gate（追加到 v6 的 6 个 gate 之后）

| Gate | 时机 | 通过条件 | 失败处理 |
|---|---|---|---|
| 7 (allocator sanity) | ep1 后 nvidia-smi | python.exe VRAM ~12-13 GB 且 ep2-10 稳定，不超 14 GB | `set` 命令位置错了，必须在 `python` 之前 |
| 8 (no second spillover) | ep1-50 全程 | per-epoch ~7 min 全程稳定 | 通常是外部 GPU app 启动，关掉即可 |
| 9 (ckpt resume sanity) | ep0 val end | p@1 ≥ 0.420（v6 ep6 是 0.425，AdamW 重置 ±0.005 OK） | 检查 missing_keys；ckpt path 拼写 |

## 7. v6.1 实测训练曲线

| epoch | p@1 | p@3 | p@5 | wall time | 状态 |
|---|---|---|---|---|---|
| 0 (val end) | 0.421 | 0.804 | 0.873 | 13:40 | gate 9 通过 (≥0.420) |
| 1 | 0.419 | 0.787 | 0.858 | 12:40 | |
| 2 | 0.424 | 0.796 | 0.866 | 10:32 | |
| 3 | 0.415 | 0.780 | 0.851 | 09:21 | |
| **4** | **0.443** | **0.814** | **0.879** | 10:18 | **NEW SOTA on all in-domain metrics** |
| 5 | 0.432 | 0.808 | 0.876 | 10:06 | val 见顶后波动 |
| 6 | 0.423 | 0.805 | 0.872 | 09:57 | |
| 7 | 0.426 | 0.798 | 0.866 | 10:25 | train loss 仍降 (0.398→0.348) |
| 8 | — | — | — | aborted at 39% | 原因不明 |

v6.1 ep4 全面反超 v6 ep6（p@1 +4.2% / p@3 +1.6% / p@5 +0.7%）。Spillover 修复成功：per-epoch ~10 min 全程稳定。

## 8. v5 / v6 / v6.1 综合对比

### 8.1 in-domain (M3FD test 210)

| ckpt | p@1 | p@3 | p@5 | mpe | Δ vs v5 ep9 |
|---|---|---|---|---|---|
| v5 ep9 | 0.4352 | 0.8072 | 0.8777 | 2.0747 | baseline |
| v6 ep6 | 0.4426 | 0.8208 | 0.8863 | 1.9630 | p@1 +1.7% / p@3 +1.7% / p@5 +1.0% / mpe -5.4% |
| v6.1 ep4 | **0.4639** | **0.8318** | **0.8904** | **1.8685** | p@1 +6.6% / p@3 +3.0% / p@5 +1.4% / mpe -9.9% (在 v6 / v6.1 链路上是 in-domain test SOTA，但被 v7 ep6 全面反超 — 见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)) |

> v6.1 ep4 in-domain test 实测时间：2026-05-06；total_matches=420,059。
> v6.1 ep4 val→test gap：p@1 +0.021 / p@3 +0.018 / p@5 +0.011（与之前按 v5/v6 推测的 +0.017/+0.018 几乎完美吻合）。

### 8.2 OOD (RoadScene test 22)

| ckpt | p@1 | p@3 | p@5 | mpe | Δ vs v5 ep9 |
|---|---|---|---|---|---|
| v5 ep9 | **0.1858** | 0.5989 | 0.7825 | 3.3856 | baseline |
| v6 ep6 | 0.1767 | **0.6010** | 0.7818 | 3.3908 | p@1 -4.9% / p@3 +0.4% / p@5 -0.1% / mpe +0.2% |
| v6.1 ep4 | 0.1762 | 0.5928 | 0.7818 | **3.3478** | p@1 -5.2% / p@3 -1.0% / p@5 -0.1% / **mpe -1.1%** |

v6 ep4-6 三个 ckpt OOD p@1 在 [0.173, 0.179] 区间窄幅浮动，**没有恶化趋势**——证明 v6 慢 LR 路径的 OOD trade-off 是稳定 trade-off 而不是 progressive collapse。v6.1 ep4 (0.176) 落在同一区间 → **在 v6/v6.1 schedule 下** trade-off 已饱和。

> **后续打破**：v7 (PC + CLAHE 输入端优化) 实测把 OOD p@1 从 v6.1 的 0.176 拉到 0.212（+20.5% rel），打破"trade-off 饱和"的判定 — 见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md) §11。这说明饱和不是模型能力上限，而是 v6/v6.1 同 schedule（数据 + 慢 LR resume）的局部最优；输入端引入模态不变特征即可突破。

### 8.3 关键判定（5 条）

1. **v6.1 ep4 = M3FD in-domain 全面 SOTA**（val p@1 0.443 反超 v6 ep6 的 0.425，+4.2% rel；val p@3 0.814 反超 0.801，+1.6% rel）
2. **OOD trade-off 在可接受区间稳定**（v5→v6→v6.1 OOD p@1 = 0.186→0.177→0.176，**v6→v6.1 几乎不动**，差 0.0005 << 22-pair 测试的 1-pair 量子 4.5%）
3. **OOD mpe 反而改善**（v6.1 vs v5: -1.1%）：fine refinement 微 M3FD-overfit 把 <1px 桶推到 1-3px 桶（→ p@1 跌），同时把长尾大误差收紧（→ mpe 改善），不是单方面退化
4. **expandable_segments 完全治根**（v6.1 全程 ~10 min/epoch 无 spillover）
5. **v6/v6.1 schedule 下达局部最优**：M3FD val p@1 = 0.443 接近 docstring "弱成功" 阈值 0.45，距离 "强成功" 0.55 还有 24%；patience=12 内剩余 9 epoch 突破有限。**进一步突破必须改输入或架构**——v7 实测路径 F（PC + CLAHE 输入端）已突破 v6.1 天花板：M3FD test p@1 0.4639 → **0.4975** (+7.2% rel)，OOD p@1 0.1762 → **0.2124** (+20.5% rel)，**双向 SOTA**。详见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md) §11。

## 9. 推荐交付：v6.1 ep4 ckpt

```text
logs\tb_logs\m3fd_v6_1_finetune\version_0\checkpoints\
  epoch=4-precision@1px=0.443-precision@3px=0.814-precision@5px=0.879.ckpt
```

理由：
- in-domain (val) 三指标全面 SOTA
- OOD trade-off 稳定不恶化（同 v6 ep6 同一窄区间内）
- OOD mpe 反而改善
- 综合通用性 (proxy) 持平 v5 baseline，与 v6 ep6 几乎平手

## 10. Bat 入口

- v6: [run_m3fd_v6_finetune.bat](../../../MyScripts/run_m3fd_v6_finetune.bat)
- v6.1: [run_m3fd_v6_1_finetune.bat](../../../MyScripts/run_m3fd_v6_1_finetune.bat)
- v6 启动样例：
  ```bat
  python train.py ^
    configs\data\m3fd_trainval.py ^
    configs\loftr\eloftr_full_v6_finetune.py ^
    --exp_name=m3fd_v6_finetune ^
    --ckpt_path=logs\tb_logs\m3fd_v5_combined\version_0\checkpoints\epoch=9-precision@1px=0.418-precision@3px=0.789-precision@5px=0.864.ckpt ^
    --max_epochs=50 ^
    --disable_mp ...
  ```
