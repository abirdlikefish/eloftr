---
name: eloftr-server-multigpu
description: 'EfficientLoFTR on Linux + 4x RTX 3090 server (vlrlab @ 222.20.94.235:8708) via SSH + Cursor Remote-SSH. Covers SSH key setup, conda env, why eloftr-windows-setup 6 patches all REMAIN (DDP-compatible), train.py PYTORCH_CUDA_ALLOC_CONF order-of-eval bug, plus 5 DDP traps: LR/WARMUP auto-scaling 4x-multiplies effective LR breaking v8 MSBN zero-shift; sync_batchnorm rewrites MSBN dual BN; RandomConcatSampler NOT auto-sharded for aligned IR-VIS; DDP non-determinism breaks strict v8 repro; large batch generalization gap. Includes v9 recipe (reproduce v8 on server: mode A single-card / B DDP / C parallel ablation). Use when running/debugging/planning training/eval on the Linux multi-GPU server. Triggers: 服务器 / SSH / vlrlab / 远程开发 / Cursor Remote-SSH / 私钥 / 4 卡 3090 / DDP / 多卡 / sync_batchnorm / NCCL / RandomConcatSampler 不分片 / LR 自动缩放 4 倍 / WARMUP 反向放大 / v9 复现 / mode A B C / 并行 ablation, English ''remote SSH'', ''4-card 3090'', ''multi-GPU DDP traps'', ''reproduce v8 on server''.'
---

# Linux + 多卡 DDP 速查（与 eloftr-windows-setup 对偶）

> 本仓库官方代码就是为 Linux + 多卡 + NCCL 设计的，所以 [eloftr-windows-setup](../eloftr-windows-setup/SKILL.md) 6 处补丁**不需要回滚**（它们对 Linux 多卡也兼容）。
> 真正的隐藏地雷是 **train.py 第 124-130 行的 LR / WARMUP / MSLR 自动缩放**：v0..v8 的 cfg 全部按 bs=4 单卡推算，4 卡时被静默放大 4-8 倍，是这条链路最危险的多卡陷阱。
> §1 服务器接入；§2 Linux 环境与 Windows 补丁的关系；§3 4 卡的三种用法；§4 DDP 五个隐藏地雷；§5 v9 = 复现 v8 的精确改动清单；§6 v0..v8 全链路多卡注意点速查。

## 1. 服务器接入 + 远程开发

### 1.1 服务器信息（导师下发）

| 项 | 值 |
|---|---|
| Host alias | 任取（推荐 `vlrlab`） |
| HostName | 主 `222.20.94.235`，备 `222.20.99.26` |
| Port | `8708` |
| User | `xyjiang` |
| 登录方式 | 私钥 (`vlrlab`)，**不是密码** |
| 资源 | 4 张 RTX 3090 24GB（共享，需用 `nvidia-smi` 看占用） |

### 1.2 私钥获取与权限设置

私钥文件本身在导师机器上（路径 `C:\Users\rog\Desktop\ssh-keys\vlrlab`），**找导师私聊获取**（微信/QQ/邮件均可，别群里发）。文件无后缀，是文本，开头为 `-----BEGIN OPENSSH PRIVATE KEY-----`。`.pub` 结尾的是公钥，对登录无用。

放到 `C:\Users\<本机用户名>\.ssh\vlrlab` 后必须立刻设权限（否则 SSH 报 `UNPROTECTED PRIVATE KEY FILE!` 拒绝使用）：

```powershell
icacls C:\Users\<本机用户名>\.ssh\vlrlab /inheritance:r
icacls C:\Users\<本机用户名>\.ssh\vlrlab /remove "Everyone" "Users" "Authenticated Users" "BUILTIN\Users" 2>$null
icacls C:\Users\<本机用户名>\.ssh\vlrlab /grant:r "${env:USERNAME}:(R)"
```

### 1.3 SSH config

`C:\Users\<本机用户名>\.ssh\config`：

```ssh-config
Host vlrlab
    HostName 222.20.94.235
    Port 8708
    User xyjiang
    IdentityFile C:\Users\<本机用户名>\.ssh\vlrlab
    ServerAliveInterval 60

Host vlrlab-bak
    HostName 222.20.99.26
    Port 8708
    User xyjiang
    IdentityFile C:\Users\<本机用户名>\.ssh\vlrlab
    ServerAliveInterval 60
```

`ssh vlrlab` 测连通；连不上换 `ssh vlrlab-bak`；都不通先 `Test-NetConnection 222.20.94.235 -Port 8708` 检查网络层（很多实验室服务器只对校园网开放，要挂校园 VPN）。

### 1.4 远程开发（推荐 Cursor Remote-SSH）

Cursor 装 `Remote - SSH` 扩展 → 左下角 `><` → `Connect to Host > vlrlab` → `Open Folder > ~/efficient_loftr`。本地只跑 UI，所有读写、终端、AI 助手都在远端运行，体验 ≈ 本地。**不要用 SSHFS / rsync 同步 + 本地编辑**，索引慢且 AI 上下文易跑飞。

## 2. Linux 基础环境

### 2.1 [eloftr-windows-setup](../eloftr-windows-setup/SKILL.md) 6 处补丁是否回滚

**全部保留，不回滚**。逐项确认：

| # | Windows 症状 | Linux 多卡命中分支 | 是否影响 Linux |
|---|---|---|---|
| 1 (`np.Inf = np.inf`) | NumPy 2.0+ 删 `np.Inf` | 服务器 NumPy 任何版本，留着无害 | 兼容 |
| 2 (`if WORLD_SIZE>1 plugins.insert DDPPlugin`) | Win 单卡无 NCCL | **多卡时正好走这个分支**（这就是设计意图） | 兼容（多卡必须） |
| 3 ([data.py](../../../src/lightning/data.py) `dist.is_initialized()` 守卫) | 单卡 dist 未初始化 | **多卡时 `dist.is_initialized()=True`**，走标准 DDP 分支 | 兼容 |
| 4 ([plotting.py](../../../src/utils/plotting.py) `.detach()`) | autograd 图泄漏 | 跨平台一致 | 兼容 |
| 5 ([lightning_loftr.py](../../../src/lightning/lightning_loftr.py) `weights_only=False`) | PyTorch 2.6 默认变化 | 服务器 PyTorch 2.6 时仍需 | 兼容 |
| 6 ([train.py](../../../train.py) `torch.load` monkey-patch) | PL 1.3.5 内部不传 weights_only | PL 仍是 1.3.5 硬约束 | 兼容 |

**结论**：现状代码可直接拿到 Linux 4 卡跑，无需回滚任何一处。

### 2.2 conda 环境

```bash
conda create -n eloftr python=3.10 -y
conda activate eloftr

# PyTorch 版本最好与本地保持一致 (2.6)，因为 patch 5/6 是为 2.6 写的
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# requirements.txt: 注意 pytorch-lightning==1.3.5 是硬约束，不能升
# opencv_python==4.4.0.46 在 Linux 装不上，放宽到 >=4.5,<5
pip install -r requirements.txt
pip install phasepack       # PC 边缘预计算，v7+ 必装
```

### 2.3 ★ PYTORCH_CUDA_ALLOC_CONF 的 order-of-eval 隐藏 bug — 已修 (2026-05-06)

[train.py:51-56](../../../train.py) 当前版本：

```python
import os
# Use setdefault so the .sh / .bat caller can pre-set PYTORCH_CUDA_ALLOC_CONF
# (e.g. expandable_segments:True for the v6.1 spillover cure) before invoking
# train.py. A hardcoded assignment would silently overwrite the caller's value
# at every run -- see .cursor/skills/eloftr-server-multigpu/SKILL.md SS2.3.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024")
```

**历史问题**：旧版本是硬赋值 `os.environ[...] = "max_split_size_mb:1024"`，会强制覆盖 .bat / .sh 里通过 `set` / `export` 设置的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。换言之 v6.1 / v7 / v8 写在 .bat 里的 spillover cure（[eloftr-v6-finetune §6](../eloftr-v6-finetune/SKILL.md)）**实际上没有生效**——bat 设值 → train.py 启动 → train.py 立刻覆盖。Windows 上"看起来生效"是因为 PyTorch 在第一次 CUDA op 之前才读这个变量，巧合下 .bat 设的值偶尔保留。

**修复后**：`setdefault` 只在 caller 没设时才赋默认值。.sh 里的 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 现在能真正传递到 PyTorch。

服务器 24GB 比本地 16GB 富裕 spillover 风险大幅降低，但仍建议 .sh 启动前显式 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 作为防御性兜底（[MyScripts/precompute_pc_edges.sh](../../../MyScripts/precompute_pc_edges.sh) 已采用此模式）。

## 3. 多卡 ≠ 显存合并：DDP 是 4 个独立 24GB

### 3.1 显存不共通

LoFTR 16M 参数在 PyTorch / PL 默认的 DDP 数据并行下：

- **每张卡装一份完整模型**（4 份独立副本）
- 每张卡处理不同 mini-batch（sampler 自动分片，**aligned IR-VIS 例外，见 §4.3**）
- 每 step 通过 NCCL all-reduce 平均梯度（PCIe 3090 间 ~25GB/s，16M × 4B = 64MB → ~2.5ms/step，< 1% 单 step 时间）

**单卡 batch_size 仍受单卡 24GB 限制**。LoFTR 在 ROAD_IMG_RESIZE=480 大约 bs=8 安全（本地 16GB 是 bs=4）。"4 卡变 96GB"是误区。模型并行 / pipeline 并行需要切模型，LoFTR 这种 16M 规模**永远走 DDP 数据并行**。

### 3.2 4 卡的三种用法

| 模式 | 配置 | 用途 | 推荐度 |
|---|---|---|---|
| **A. 单卡复现** | `CUDA_VISIBLE_DEVICES=0`, `--gpus=1`, bs=4 | 字节级复现 v0..v8 历史数字 | ★★★★★ 最稳 |
| **B. 4 卡 DDP 单实验** | `--gpus=4`, bs=4（effective_bs=16），**必须改 cfg**（§4.1） | 加速同一实验 wall-clock 4 倍 | ★★★ 慎用 |
| **C. 4 卡并行 ablation** | 4 个独立单卡进程，各占 1 张卡（`CUDA_VISIBLE_DEVICES=0/1/2/3`） | 一次跑 v9 / v9.1 / v9.2 / v9.3 矩阵 | ★★★★★ 最划算 |
| ~~D. 模型并行~~ | — | LoFTR 16M 不需要 | 用不到 |

**关键洞察**：多卡的真正价值不是"显存合并大 batch"，而是模式 C 的**任务并行**。同样 1 小时 wall clock，模式 C 跑 4 个 ablation，模式 B 只跑 1 个。

## 4. DDP 多卡的 5 个隐藏地雷

### 4.1 ★ LR / WARMUP / MSLR 自动缩放放大灾

[train.py:124-130](../../../train.py)：

```python
config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes              # 4
config.TRAINER.TRUE_BATCH_SIZE = WORLD_SIZE * batch_size          # 4×4=16
_scaling = TRUE_BATCH_SIZE / CANONICAL_BS                          # 16/64 = 0.25
config.TRAINER.TRUE_LR = CANONICAL_LR * _scaling                  # LR 看似变小，但...
config.TRAINER.WARMUP_STEP = math.floor(WARMUP_STEP / _scaling)   # WARMUP 反向放大!
```

注意 `_scaling` 是 `TRUE_BATCH_SIZE / CANONICAL_BS`：单卡 bs=4 时 _scaling=0.0625；4 卡 bs=4 时 _scaling=0.25 → **TRUE_LR 增大 4 倍**（同时 WARMUP 步数也 ÷ 4 即放大 4 倍）。以 v8 cfg（CANONICAL_LR=4e-4, WARMUP_STEP=30, MSLR=[6,11,15], M3FD ~945 step/epoch）为例：

| 配置 | _scaling | TRUE_LR | WARMUP | 单 epoch step | warmup 占 epoch | LR 偏移 vs v8 |
|---|---|---|---|---|---|---|
| 单卡 v8（本地） | 0.0625 | 2.5e-5 | 30 | 945 | 0.032 | baseline |
| 4 卡 bs=4 | 0.25 | **1e-4 (4×)** | **120 (4×)** | 237 | 0.51 | **LR 4 倍** |
| 4 卡 bs=8 | 0.50 | **2e-4 (8×)** | **240 (8×)** | 118 | **2.0（warmup>1ep!）** | **LR 8 倍** |

**直接后果**：

1. v8 MSBN 的 zero-shift 复制要求 ep0 forward 字节级 = v7 ep6。**第一次梯度更新 LR 4 倍**会立刻把 `bn_vis` 推离 zero-shift → [eloftr-v8-msbn §7](../eloftr-v8-msbn/SKILL.md) Gate 15（"ep0 val end p@1 ≥ 0.470"）极可能失败。
2. `MSLR_MILESTONES=[6,11,15]` 是 epoch 数，但每 epoch 步数变 1/4，LR decay 时机的**总 step 进度**与单卡完全不同。
3. WARMUP=240 步 > 单 epoch 118 步时，整训练前 2 ep 都在线性 warmup，等价于"全程低 LR + 不到 milestone"。

**多卡 cfg 修复模板**（想让 effective LR / schedule ≈ 单卡 v8）：

```python
# 在 v9 多卡 cfg 里手动反向缩放
cfg.TRAINER.CANONICAL_LR = 4e-4 / 4         # 1e-4，让 4 卡 _scaling=0.25 时 TRUE_LR=2.5e-5
cfg.TRAINER.WARMUP_STEP = math.ceil(30 * 0.25)  # 7 → 自动缩放后 floor(7/0.25)=28 ≈ 30
# MSLR_MILESTONES 不动（epoch 数不变，但每 epoch step 数变了，总 step 进度仍不一致）
```

[eloftr-cross-modal-experiments §2](../eloftr-cross-modal-experiments/SKILL.md) 列了这条坑的早期单卡版本（v1/v2 的 WARMUP=1875 灾），但没展开多卡场景；多卡场景就在本节。

### 4.2 sync_batchnorm × MSBN 相互作用（v8/v9 独家）

[train.py:220](../../../train.py)：

```python
sync_batchnorm=config.TRAINER.WORLD_SIZE > 1
```

多卡时 PL 会把所有 `nn.BatchNorm2d` 替换为 `nn.SyncBatchNorm`，**包括 v8 新增的 MSBN 4 个 BN**（`fine_preprocess.layer{1,2}_outconv2_bn_ir/vis`）。

| 配置 | bn_ir 看到的样本 | bn_vis 看到的样本 | 数学等价 v8? |
|---|---|---|---|
| 单卡 bs=4 | batch 内 IR 样本 (~4) | batch 内 VIS 样本 (~4) | ✓ |
| 4 卡 DDP bs=4 + sync_bn | **全卡 IR 样本 (~16)** | **全卡 VIS 样本 (~16)** | ✗ |

这是**好事**（4 倍样本量让 running_mean/var 更稳，[eloftr-v8-msbn](../eloftr-v8-msbn/SKILL.md) Gate 16 `bn_drift_ratio` 也会受影响）但**不再字节级等价 v8**。

严格复现的两条路：
1. 在 [train.py:220](../../../train.py) 强制 `sync_batchnorm=False`（但要确认 backbone BN 也不需要 sync）
2. 接受多卡 ≠ 单卡，跑出 0.515 ± 0.01 而不是严格 0.520，论文里声明

### 4.3 ★ RandomConcatSampler 在 DDP 下不分片（aligned IR-VIS 灾区）

> **★ 2026-05-08 已通过 image-level pre-shard 真修复**（v10 落地，详见本节末"修复"段）。下面的"问题描述"段落保留作为历史背景 + 排查时的诊断指南；**v10+ 4 卡训练默认享受修复，不需要手动 cfg N_SAMPLES_PER_SUBSET 缩到 1/world_size**。

#### 问题描述（v0-v9 + v10 修复前）

[src/datasets/sampler.py:16-17](../../../src/datasets/sampler.py)：

```python
NOTE: This sampler behaves differently with DistributedSampler.
      It assume the dataset is splitted across ranks instead of replicated.
```

[src/lightning/data.py train_dataloader](../../../src/lightning/data.py) **不**用 `DistributedSampler` 包装 `RandomConcatSampler`，配合 [train.py:221](../../../train.py) 的 `replace_sampler_ddp=False`，意味着：

- ScanNet/MegaDepth 路径：[data.py 的 `get_local_split(npz_names, world_size, rank, seed)`](../../../src/lightning/data.py) 把 npz 文件按 rank 分片，每张卡只构建自己那份 ConcatDataset（数据已分片，sampler 对全集采样无重复）→ DDP 正确。
- **Aligned IR-VIS（M3FD / RoadScene / Megadepth_Syn）路径（v10 修复前）**：短路分支**直接把全集 ConcatDataset 给所有 rank**，**不调用 `get_local_split`**。配合不分片的 RandomConcatSampler + 全局 `torch.manual_seed(cfg.TRAINER.SEED)`，4 张卡抽到完全相同的 indices → 4× 重复 step + 实际 effective_bs 只有 4（不是 16）。

#### 修复（v10 落地，aligned IR-VIS 短路分支增加 image-level pre-shard）

v10 mode B 4 卡 DDP 是这条 design gap 首次实战。修复在 [src/lightning/data.py L248-273](../../../src/lightning/data.py)（mirrors ScanNet/MegaDepth 的 scene-level `get_local_split` 模式）：

```python
if is_aligned_irvis(data_source):
    # DDP image-level pre-sharding (mirrors the scene-level
    # get_local_split call below for ScanNet/MegaDepth).
    with open(scene_list_path, 'r', encoding='utf-8') as f:
        all_names = [line.strip() for line in f.readlines() if line.strip()]
    if mode == 'train' and self.world_size > 1:
        local_names = list(get_local_split(
            all_names, self.world_size, self.rank, self.seed))
        logger.info(
            f'[rank {self.rank}]: aligned IR-VIS train pre-shard: '
            f'{len(local_names)}/{len(all_names)} samples '
            f'(disjoint across {self.world_size} ranks; seed={self.seed})')
    else:
        local_names = all_names
    # ...
    ds = RoadSceneDataset(..., names=local_names, ...)
```

`RoadSceneDataset.__init__` 接受新加的 `names: Optional[List[str]] = None`（None = 走原 `_read_index_file(list_path)` 路径，向后兼容 v0-v9 单卡）。Pre-shard **只在 train mode** 触发；val / test 保持 full + PL 默认 `DistributedSampler` 切片。

修复后 v10 mode B 实测 log:
```text
[rank 0]: aligned IR-VIS train pre-shard: 26521/106083 samples (disjoint across 4 ranks; seed=66)
[rank 1]: aligned IR-VIS train pre-shard: 26521/106083 samples ...
[rank 2/3]: ...
```
26521 × 4 ≈ 106083 全集 ✓ 4 rank 严格不重叠 1 epoch 见 train 全集。

详见 [eloftr-v10-msyn §4.1](../eloftr-v10-msyn/SKILL.md)。

#### sanity check（任何 4 卡 aligned IR-VIS 训练首次启动必看）

1. 启动 log 应有 4 条 `[rank N]: aligned IR-VIS train pre-shard: <local>/<total> samples ...` → pre-shard 触发
2. `<local>` × `world_size` ≈ `<total>` （±1 round-up 误差）→ disjoint 切片正确
3. PL 进度条 `Epoch 0: ... <X>/<Y>`，`X` = 单 rank `N_SAMPLES_PER_SUBSET / batch_size`（不是 `<total> / batch_size`）

如果 log **没出现 pre-shard 行**：要么 mode 不是 train（val/test 不分片是预期）, 要么 `world_size <= 1`（单卡退化到 full set, 是预期）, 要么本仓库未升到 v10 修复版本（git pull）。

#### v10 cfg 里 N_SAMPLES_PER_SUBSET 的语义变化

修复前: 因为 sampler 不分片, 单 rank 跑 N_SAMPLES_PER_SUBSET 个 sample, **4 rank 抽相同的 N_SAMPLES**, 实际只用了 N_SAMPLES 个 unique 样本。
修复后: 单 rank 拿到 `local_shard_size = total / world_size`, RandomConcatSampler 按 `subset_replacement=False` 先 randperm(local_shard) 然后取 N_SAMPLES_PER_SUBSET（如果 N_SAMPLES > local_shard 则 with-replacement 补差）。**4 rank * N_SAMPLES = 单 epoch 总 sample passes**, 设 N_SAMPLES ≈ local_shard 即可让 1 epoch 覆盖 train 全集。

v10 mode B cfg: `N_SAMPLES_PER_SUBSET = 28750` ≈ `local_shard 26521`, 4 * 28750 = 115K ≈ train 全集 106083, 完美覆盖。

### 4.4 复现性退化（DDP 非确定性）

DDP 引入的非确定性来源：
- NCCL all-reduce 浮点累加顺序（4 卡求和 ≠ 1 卡求和的 FP32 表示）
- 各 worker 的 dataloader 种子初始化（PL 1.3.5 对 DDP 不完美）
- sync_bn 统计聚合顺序

后果：多卡跑 v9 复现 v8，**±0.005 是合理范围**，不要期望严格 0.5494。严格复现就用模式 A。

### 4.5 大 batch generalization gap（Keskar 2017）

effective_batch 4 → 16/32 后，模型容易陷入 sharp minima，val precision 可能不升反降。文献观察阈值 batch>8K，effective_bs=16 仍在安全区，但 32 已有迹象。

**v8 见顶 ep16 p@1=0.520 是 bs=4 stochastic 噪声搜到的好点**，大 batch 可能直接错过。模式 B 跑出来的 v9 可能见顶提前到 ep10-12 但峰值仅 0.510，**不一定是 bug 而是 Keskar 效应**。

## 5. v9 实操：在 4 卡 3090 服务器跑 v8 全套架构

### 5.0 决策记录 (2026-05-06): yurupeng 选 e2e cold start 路线 (B3), 不严格复现 v8 ep16

v9 服务器实操有 3 条候选路线 (从严格复现到完全新实验):

| 路线 | 起点 ckpt | cfg / schedule | 期望结果 |
|---|---|---|---|
| B1. 严格 v8 复现 | v7 ep6 ckpt (rsync from windows ~192MB) | v8 cfg + max_epochs=20, schedule 不动 | p@1 ∈ [0.515, 0.525] (§5.5 范围) |
| B2. v8 cfg + outdoor 起点 (不推荐) | `weights/eloftr_outdoor.ckpt` | v8 cfg 不改 | 不可预测, 远差于 v8 (schedule 严重不匹配 cold start) |
| **B3. e2e cold start (yurupeng 选择)** | `weights/eloftr_outdoor.ckpt` | 新建 `configs/loftr/eloftr_full_v9_e2e.py`: max_epochs=80, WARMUP=300, CANONICAL_LR=2e-3, MSLR=[40,60,75], ES patience=20 | 新实验数据点, 评估"v0-v7 链路是否必须"问题 |

§5.1-§5.5 保留为 B1 (严格复现) 路线模板, 仍可作为 B3 的实操对照. **B3 路线的差异**:

- §5.2 必改清单的 `--ckpt_path` 改为 `weights/eloftr_outdoor.ckpt` (省去 v7 ep6 同步的 rsync)
- §5.2 `--exp_name` 改为 `m3fd_v9_e2e_outdoor`
- §5.3 第 4 项 "v7 ep6 ckpt 必需" 取消; 第 1-3 项 (代码/M3FD/PC 缓存) 仍需要
- §5.4 启动脚本改 `bash MyScripts/run_m3fd_v9_e2e.sh` (新建; 同样在 tmux 内, ~13.3h max_ep=80)
- §5.5 验收范围**不再对标 v8 0.5494**, 重新设计 (强成功 ≥0.51 / 中成功 [0.45, 0.51] / 弱成功 [0.40, 0.45] / 失败 <0.40, OOD 不退过 0.21)

PC 缓存预计算入口 [MyScripts/precompute_pc_edges.sh](../../../MyScripts/precompute_pc_edges.sh) (Linux 版本, 默认 M3FD + RoadScene 一起算 ~35-50 min) 已在 2026-05-06 落地. 无需 rsync windows 本地的 PC cache, 服务器直接预计算更快 (80 核 CPU). 预计算时如果 phasepack 报 `pyfftw could not be imported` warning, 是 fallback 到 scipy fftpack, 速度 ~30% 损失但不影响正确性. 详见 [eloftr-yurupeng-workspace §5.1](../eloftr-yurupeng-workspace/SKILL.md).

### 5.1 决策树

```mermaid
flowchart TD
    goal{"v9 想要的最重要的东西"}
    goal -->|"严格复现 v8 ep16 数字<br/>(p@1=0.5494, p@3=0.9007)"| modeA["模式 A 单卡复现<br/>CUDA_VISIBLE_DEVICES=0<br/>bs=4, schedule 完全继承 v8<br/>★ v9 必做的第一步"]
    goal -->|"一次跑完<br/>v9 + v9.1/.2/.3 ablation"| modeC["模式 C 4 卡并行 ablation<br/>每张卡独立单卡 bs=4<br/>wall clock ~1h vs 串行 ~13h"]
    goal -->|"必须 1 小时拿到 1 个 v8 复现"| modeB["模式 B 4 卡 DDP<br/>必须改 cfg (§4.1 修复模板)<br/>接受 ±0.005 抖动 + sync_bn × MSBN 差异<br/>慎用"]
    modeA --> nextstep{"v9 复现成功后"}
    nextstep -->|"做 v8.x ablation"| modeC
    nextstep -->|"提高分辨率/扩 epoch"| moreexp["在模式 A 基础上微调 cfg"]
```

### 5.2 .bat → .sh 必改清单（模式 A 单卡，改动最小）

| 项 | 本地 .bat | 服务器 .sh | 备注 |
|---|---|---|---|
| 激活 conda | `call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr` | `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` | env 实际在 `/data/xyjiang/envs/eloftr_yurupeng` 但 conda 通过 envs_dirs 识别, source 仍 work |
| 路径分隔符 | `configs\loftr\...` | `configs/loftr/...` | 续行 `^` → `\` |
| 环境变量 | `set X=Y` | `export X=Y` | |
| 末尾 | `pause` | 删掉 | |
| `--ckpt_path` (B1) | `logs\tb_logs\m3fd_v7_pcclahe\...epoch=6-...ckpt` | 同名相对路径，**等号需引号**：`--ckpt_path='logs/.../epoch=6-....ckpt'` | v7 ep6 ckpt 必须传到服务器同位置 |
| `--ckpt_path` (B3) | — | `--ckpt_path=weights/eloftr_outdoor.ckpt` | e2e cold start, 不需要 rsync v7 ep6 |
| `--gpus` | `1` | `1`（模式 A） | 模式 B/C 见 §3.2 |
| `--num_workers` | `6` | `8` | 服务器 CPU 核多 |
| `--pin_memory` | `false`（本地内存紧） | `true` | 服务器内存充足，DataLoader→GPU 拷贝更快 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`（被 train.py 覆盖，§2.3） | 同；[train.py:52](../../../train.py) 已修为 `setdefault` (2026-05-06) | |
| `--exp_name` | `m3fd_v8_msbn` | B1: `m3fd_v9_msbn_repro` / B3: `m3fd_v9_e2e_outdoor` | 不要混进 v8 的 tb_logs |

### 5.3 数据 / ckpt 同步

服务器需要的文件：

1. **代码**：`git push` 本地 → 服务器 `git clone`，**不要 scp 整个仓库**（有 .git 历史 + logs + ckpt 上 G）
2. **M3FD 数据**：先看实验室是否已挂载公共数据集
   ```bash
   find /data /datasets /home/share /mnt -name "M3FD*" -type d 2>/dev/null
   ```
   有就 `ln -s /shared/M3FD_Detection ~/efficient_loftr/data/M3FD_Detection`；没有再 scp `data/M3FD_Detection/{Ir,Vis,Annotation,index}` (~1.5GB)
3. **PC 缓存**（`Ir_pc/` / `Vis_pc/`）：4200×2 小文件，rsync 比 scp 快
   ```bash
   rsync -avz --progress \
     /c/Users/<本机用户名>/Desktop/毕设/efficient_loftr/data/M3FD_Detection/Ir_pc/ \
     vlrlab:~/efficient_loftr/data/M3FD_Detection/Ir_pc/
   # 同样 rsync Vis_pc/
   ```
   或服务器自己跑 `python MyScripts/precompute_pc_edges.py --dataset M3FD`（~35 min，CPU 多反而更快）
4. **v7 ep6 ckpt**（v8 resume 必需，~192MB）：scp 到 `~/efficient_loftr/logs/tb_logs/m3fd_v7_pcclahe/version_0/checkpoints/`
5. （可选）OOD eval：RoadScene PC cache 也一并同步，~2 min

### 5.4 v9 启动 SOP（模式 A）

```bash
ssh vlrlab
tmux new -s v9                                    # 防 SSH 断开
conda activate eloftr
cd ~/efficient_loftr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. R8 sanity（[eloftr-v8-msbn §5](../eloftr-v8-msbn/SKILL.md) 4 子 gate 必须全过）
python MyScripts/v8_compat_quickcheck.py

# 2. v9 debug（~3 min, Gate 14-17 全过）
CUDA_VISIBLE_DEVICES=0 bash MyScripts/run_m3fd_v9_msbn_debug.sh

# 3. v9 完整训练（~3.3h, ES 在 ep14-18 之间停）
CUDA_VISIBLE_DEVICES=0 bash MyScripts/run_m3fd_v9_msbn.sh

# 4. eval（in-domain + OOD）
bash MyScripts/eval_m3fd_finetuned.sh 9          # 期望 p@1 ≈ 0.515-0.525
bash MyScripts/eval_roadscene_finetuned.sh 9     # 期望 p@1 ≈ 0.195-0.210

# Ctrl+B D 离开 tmux；下次 ssh vlrlab && tmux attach -t v9 回来
```

TensorBoard 端口转发到本地浏览器：
```powershell
# 本地 PowerShell 新开一个 SSH 隧道
ssh -L 6006:localhost:6006 vlrlab
# 服务器侧
tensorboard --logdir=logs/tb_logs --port=6006
# 本地浏览器 http://localhost:6006
```

### 5.5 v9 验收标准（vs v8 实测）

| 指标 | v8 实测 | v9 模式 A 接受范围 | 失败处置 |
|---|---|---|---|
| M3FD test p@1 | 0.5494 | [0.515, 0.525] | < 0.515：MSBN inflate hook / 数据 / cfg 不一致，查 [eloftr-v8-msbn §11](../eloftr-v8-msbn/SKILL.md) |
| M3FD test p@3 | 0.9007 | [0.890, 0.905] | 同上 |
| OOD p@1 | 0.2025 | [0.195, 0.210] | < 0.195：检查 RoadScene PC cache 是否同步 |
| 见顶 epoch | 16 | 14-18 | 单卡 Linux 与 Windows 的 cuDNN 非确定性可能让见顶提前/推迟 |

模式 B / C 的接受范围放宽到 ±0.01（DDP 非确定性）。

## 6. v0..v10 全链路多卡注意点速查

| 版本 | 起点 ckpt | 单卡 vs 多卡风险点 |
|---|---|---|
| v0 (baseline) | 官方 outdoor.ckpt | 无 ckpt resume 逻辑，多卡只看 §4.1 LR 缩放 |
| v1 contrast | v0 | 同上 |
| v2 modemb | v1 | learnable emb 多卡时 DDP all-reduce 自动平均，无影响 |
| v3/v4 freeze | v2 | FREEZE_BACKBONE_BN=True 多卡时 backbone BN 不更新，与单卡一致；FREEZE_BN=True 时 sync_bn 替换的 SyncBN 仍 frozen |
| v5 m3fd | v4 ep0 | **§4.3 RandomConcatSampler 不分片首次出现**（v0-v4 是 RoadScene 也命中但样本少不明显，v5 M3FD 3780 才显著；2026-05-08 v10 修复后不再是问题） |
| v6/v6.1 | v5 ep9 / v6 ep6 | spillover cure（§2.3）服务器 24GB 不需要；schedule 继承 v5 |
| v7 pcclahe | v6.1 ep4 | PC cache 服务器同步（§5.3）；stage0 inflate hook 无多卡问题 |
| v8 msbn | v7 ep6 | **§4.1 LR 灾 + §4.2 sync_bn × MSBN，本仓库多卡风险最大的版本**；v9_server_repro = 在服务器复现 v8 |
| v9_e2e | outdoor.ckpt | 单卡训练，多卡风险不直接命中；如果想用 4 卡复现 v9_e2e 80 ep 同样需要 §4.1 LR 反向缩放 |
| **v10_msyn** | outdoor.ckpt | **首个 4 卡 DDP ship run + aligned IR-VIS 数据集**；§4.1-4.5 五个地雷全部命中并各自处理（§4.3 真修复）；wall-clock 实测见 §6.5 |

### 6.5 wall-clock 实战修订（v10 mode B 实测，2026-05-08）

**v10 是本仓库首个 4 卡 DDP + aligned IR-VIS + sync_bn = True 的 ship run**。实测速度跟 plan §10 估算（"3-4h"）出现重大偏差，根因是 sync_bn 在 RepVGG 多 BN 架构上的开销被严重低估了。

**实测分解**（mode B 4 卡 3090, bs=4 effective_bs=16, sync_bn=True）：

| 操作 | 单 step 耗时 | 来源 |
|---|---|---|
| forward (16 sample × 480² × 16M params) | ~0.15s | GPU 计算 |
| backward | ~0.30s | 反向比 forward 慢 2× |
| NCCL allreduce 16M grads | ~0.05s | 64MB / 25GB/s × 4 rank |
| **sync_bn 跨卡同步**（**主要被低估的开销**）| **~0.15s** | RepVGG backbone ~30 个 BN + MSBN 4 个 BN，每个 BN forward + backward 都要 4-rank sync mean/var |
| 其他（optimizer / dataloader prefetch / 等）| ~0.05s | — |
| **总** | **~0.7s/step** | **= 1.44 it/s** ✓ |

**对照单卡 v9_e2e（M3FD bs=4，无 sync_bn）**：~0.07s/step (13.69 it/s val phase)

| 阶段 | v10 单 step | v10 step 数（per rank）| v10 时长 |
|---|---|---|---|
| train | ~0.7s | 7187（28750/4）| ~83 min |
| val（forward only, no allreduce, no sync_bn update）| ~0.15s | 3025（12101/4）| ~8 min |
| **per epoch 总** | | | **~90 min ≈ 1.5h** |
| **10 epoch（不 ES）** | | | **~15h** |
| **8 epoch（ES patience=3 触发）** | | | **~12h** |

→ ship run 实际 **12-15h**（OS page cache 在 ep1+ 加速 IO 后可能压到 10-12h）。

**plan §10 估算 "3-4h" 偏乐观 4× 的根因**：
1. 我估"NCCL + sync_bn 开销 ~0.1s/step"基于"DDP 通用经验"，未考虑 RepVGG backbone 的 BN 数量
2. 没有先验数据点：v0-v9 全部单卡跑 + v9_server_repro 没真做完
3. 实际 sync_bn 占总 step 时间 ~25%（0.15s / 0.7s），是 plan 估算与现实的最大差异源

**v11+ 多卡 DDP 训练时间估算公式（修订版）**：

```
per_rank_step_time ≈ 0.7s   (RepVGG + sync_bn + 4 卡 NCCL, bs=4 effective_bs=16, 480² resolution)
per_rank_train_steps_per_epoch = N_SAMPLES_PER_SUBSET / batch_size
per_rank_val_steps_per_epoch = ceil(val_pairs / world_size)   (val_batch_size=1)
epoch_time ≈ per_rank_train_steps × 0.7 + per_rank_val_steps × 0.15  (in seconds)
total_wall_clock = max_epochs × epoch_time × 1.05 (overhead)  (不考虑 ES early stop)
```

未来加新 v_x 4 卡 ship run 之前，先用上面公式算 wall-clock，再决定 max_epochs 是否合理（避免又被 plan 估算偏乐观坑）。

## 7. 何时用本 skill

- 第一次接服务器 / 配 SSH / 配 Cursor Remote-SSH（§1）
- 把本地 .bat 翻成服务器 .sh，搞不清哪些 Windows 补丁要回滚（§2.1，答案：都不回滚）
- 计划多卡训练前评估 LR / WARMUP / sampler / sync_bn 风险（§4）
- v9 启动前对照必改清单（§5.2）+ 验收标准（§5.5）
- v10+ aligned IR-VIS 4 卡 ship run 启动前估算 wall-clock（§6.5 修订公式）+ 验证 §4.3 pre-shard log 是否出现
- 跑出来数字与本地差很多时反查（§4 + §6）
