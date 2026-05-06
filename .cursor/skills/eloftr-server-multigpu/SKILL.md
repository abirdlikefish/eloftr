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

### 2.3 ★ PYTORCH_CUDA_ALLOC_CONF 的 order-of-eval 隐藏 bug

[train.py:51-52](../../../train.py)：

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
```

这行 **强制覆盖** 任何 .bat / .sh 里通过 `set` / `export` 设置的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。换言之：v6.1 / v7 / v8 写在 .bat 里的 spillover cure（[eloftr-v6-finetune §6](../eloftr-v6-finetune/SKILL.md)）**实际上没有生效**——bat 设值 → train.py 启动 → train.py 立刻覆盖。Windows 上"看起来生效"是因为 PyTorch 在第一次 CUDA op 之前才读这个变量，巧合下 .bat 设的值偶尔保留。

服务器 24GB 比本地 16GB 富裕，spillover 风险大幅降低，但建议同时做两件事：

1. 修 train.py 第 52 行：
   ```python
   os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024")
   ```
   或直接删除该行。
2. .sh 启动前 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，作为防御性兜底。

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

[src/datasets/sampler.py:16-17](../../../src/datasets/sampler.py)：

```python
NOTE: This sampler behaves differently with DistributedSampler.
      It assume the dataset is splitted across ranks instead of replicated.
```

[src/lightning/data.py:393-405](../../../src/lightning/data.py) `train_dataloader` **不**用 `DistributedSampler` 包装 `RandomConcatSampler`，配合 [train.py:221](../../../train.py) 的 `replace_sampler_ddp=False`，意味着：

- ScanNet/MegaDepth 路径：[data.py:282-283](../../../src/lightning/data.py) 的 `get_local_split(npz_names, world_size, rank, seed)` 把 npz 文件按 rank 分片，每张卡只构建自己那份 ConcatDataset（数据已分片，sampler 对全集采样无重复）→ DDP 正确。
- **Aligned IR-VIS（M3FD / RoadScene）路径**：[data.py:250-277](../../../src/lightning/data.py) 短路分支**直接把全集 ConcatDataset 给所有 rank**，**不调用 `get_local_split`**。配合不分片的 RandomConcatSampler，4 张卡可能：
  - 同 seed → 4 张卡抽到完全相同的 indices → 4× 重复 step + 实际 effective_bs 只有 4（不是 16）
  - 不同 seed（PL 1.3.5 dataloader worker seed 行为依实现）→ 4 张卡抽到不同 945 indices → 实际看了 4 倍数据/epoch

**4 卡首次跑 M3FD 必做 sanity**：
1. TB 第一个 step 的 `train_loss` 与单卡接近（zero-shift v9 时）→ 排除 sampler 重叠
2. 看 4 张卡的 rank 0/1/2/3 启动 log 的 `[rank N]: building RoadSceneDataset (...) from .../train_pairs.txt`，应该都看到 3780 → 确认数据集**未分片**（这是预期）
3. 跑 1 epoch 后看 `len(train_loader)` 实际产出 step 数

如果观察到 step 数显著少于单卡 / 实际看的数据多于全集，必须显式给 RandomConcatSampler 加 DDP 分片逻辑（fork 一个 DDP-aware 版本），或在 cfg 里把 `N_SAMPLES_PER_SUBSET` 调成 `3780/world_size`。

### 4.4 复现性退化（DDP 非确定性）

DDP 引入的非确定性来源：
- NCCL all-reduce 浮点累加顺序（4 卡求和 ≠ 1 卡求和的 FP32 表示）
- 各 worker 的 dataloader 种子初始化（PL 1.3.5 对 DDP 不完美）
- sync_bn 统计聚合顺序

后果：多卡跑 v9 复现 v8，**±0.005 是合理范围**，不要期望严格 0.5494。严格复现就用模式 A。

### 4.5 大 batch generalization gap（Keskar 2017）

effective_batch 4 → 16/32 后，模型容易陷入 sharp minima，val precision 可能不升反降。文献观察阈值 batch>8K，effective_bs=16 仍在安全区，但 32 已有迹象。

**v8 见顶 ep16 p@1=0.520 是 bs=4 stochastic 噪声搜到的好点**，大 batch 可能直接错过。模式 B 跑出来的 v9 可能见顶提前到 ep10-12 但峰值仅 0.510，**不一定是 bug 而是 Keskar 效应**。

## 5. v9 实操：在 4 卡 3090 服务器复现 v8 ep16

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
| 激活 conda | `call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr` | `source ~/miniconda3/bin/activate eloftr` | 环境名按 §2.2 |
| 路径分隔符 | `configs\loftr\...` | `configs/loftr/...` | 续行 `^` → `\` |
| 环境变量 | `set X=Y` | `export X=Y` | |
| 末尾 | `pause` | 删掉 | |
| `--ckpt_path` | `logs\tb_logs\m3fd_v7_pcclahe\version_0\checkpoints\epoch=6-...ckpt` | 同名相对路径，**等号需引号**：`--ckpt_path='logs/.../epoch=6-....ckpt'` | v7 ep6 ckpt 必须传到服务器同位置 |
| `--gpus` | `1` | `1`（模式 A） | 模式 B/C 见 §3.2 |
| `--num_workers` | `6` | `8` | 服务器 CPU 核多 |
| `--pin_memory` | `false`（本地内存紧） | `true` | 服务器内存充足，DataLoader→GPU 拷贝更快 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`（被 train.py 覆盖，§2.3） | 同 + 顺手修 [train.py:52](../../../train.py) `setdefault` | |
| `--exp_name` | `m3fd_v8_msbn` | `m3fd_v9_msbn_repro` | 不要混进 v8 的 tb_logs |

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

## 6. v0..v8 全链路多卡注意点速查

| 版本 | 起点 ckpt | 单卡 vs 多卡风险点 |
|---|---|---|
| v0 (baseline) | 官方 outdoor.ckpt | 无 ckpt resume 逻辑，多卡只看 §4.1 LR 缩放 |
| v1 contrast | v0 | 同上 |
| v2 modemb | v1 | learnable emb 多卡时 DDP all-reduce 自动平均，无影响 |
| v3/v4 freeze | v2 | FREEZE_BACKBONE_BN=True 多卡时 backbone BN 不更新，与单卡一致；FREEZE_BN=True 时 sync_bn 替换的 SyncBN 仍 frozen |
| v5 m3fd | v4 ep0 | **§4.3 RandomConcatSampler 不分片首次出现**（v0-v4 是 RoadScene 也命中但样本少不明显，v5 M3FD 3780 才显著） |
| v6/v6.1 | v5 ep9 / v6 ep6 | spillover cure（§2.3）服务器 24GB 不需要；schedule 继承 v5 |
| v7 pcclahe | v6.1 ep4 | PC cache 服务器同步（§5.3）；stage0 inflate hook 无多卡问题 |
| v8 msbn | v7 ep6 | **§4.1 LR 灾 + §4.2 sync_bn × MSBN，本仓库多卡风险最大的版本**；v9 = 在服务器复现 v8 |

## 7. 何时用本 skill

- 第一次接服务器 / 配 SSH / 配 Cursor Remote-SSH（§1）
- 把本地 .bat 翻成服务器 .sh，搞不清哪些 Windows 补丁要回滚（§2.1，答案：都不回滚）
- 计划多卡训练前评估 LR / WARMUP / sampler / sync_bn 风险（§4）
- v9 启动前对照必改清单（§5.2）+ 验收标准（§5.5）
- 跑出来数字与本地差很多时反查（§4 + §6）
