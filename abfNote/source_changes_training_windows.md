# EfficientLoFTR Windows 单卡训练源码修改记录

本文记录为了在 Windows + 单 GPU 环境下使用 `data/MyTrainData` 跑通训练，对源码做过的兼容性修改。

## 修改背景

原项目主要面向 Linux 多卡/DDP 环境，且依赖版本较旧。实际运行时遇到以下问题：

- `pytorch-lightning==1.3.5` 与 `NumPy 2.x` 不兼容，旧代码会访问已移除的 `np.Inf`。
- Windows 单卡环境没有 NCCL，直接使用 `DDPPlugin` 会报 `Distributed package doesn't have NCCL built in`。
- 单卡非分布式模式下，直接调用 `torch.distributed` 或 `DistributedSampler` 会报 `Default process group has not been initialized`。
- 训练阶段 TensorBoard 绘图时，带梯度的 tensor 直接 `.numpy()` 会报 `Can't call numpy() on Tensor that requires grad`。
- PyTorch 2.6+ 将 `torch.load()` 的 `weights_only` 默认值改为 `True`，加载官方 Lightning checkpoint 时可能报 `Weights only load failed`。
- 当使用 `--resume_from_checkpoint` 续训时，PyTorch Lightning 1.3.5 内部的 `pl_load`（`pytorch_lightning/utilities/cloud_io.py`）也会调用 `torch.load`，且没有传 `weights_only=False`，会触发同样的报错；此时只补 `src/lightning/lightning_loftr.py` 那一处不够，需要在 `train.py` 入口处对 `torch.load` 做全局兼容性 monkey-patch。

## `train.py`

### 1. 兼容 NumPy 2.x

新增：

```python
import numpy as np
np.Inf = np.inf
```

原因：旧版 PyTorch Lightning 内部仍使用 `np.Inf`，而 NumPy 2.x 已移除该别名。

### 2. 延后创建 `ModelCheckpoint`

将 `ModelCheckpoint` 的创建移动到：

```python
if not args.disable_ckpt:
```

内部。

原因：debug 训练时使用 `--disable_ckpt`，不应仍然初始化 checkpoint callback。这样可以避免不保存 checkpoint 时仍触发旧版 Lightning 兼容问题。

### 3. 单卡时不启用 DDPPlugin

原始代码无条件传入：

```python
DDPPlugin(...)
```

已保留为注释，并改为：

```python
plugins = [NativeMixedPrecisionPlugin()]
if config.TRAINER.WORLD_SIZE > 1:
    plugins.insert(0, DDPPlugin(...))
```

同时将：

```python
sync_batchnorm=config.TRAINER.WORLD_SIZE > 0
```

改为：

```python
sync_batchnorm=config.TRAINER.WORLD_SIZE > 1
```

原因：Windows 单 GPU 不需要 DDP，也没有 NCCL；只有真正多卡时才启用分布式插件和同步 BN。

### 4. 全局兼容 PyTorch 2.6 的 `weights_only=True` 默认行为

新增（紧跟 `import torch`、在 `pl.Trainer.from_argparse_args` 调用之前）：

```python
import torch
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat
```

原因：

- `src/lightning/lightning_loftr.py` 里那条手写 `torch.load(..., weights_only=False)` 只覆盖了 `--ckpt_path` 旁路。
- 一旦走 `--resume_from_checkpoint`、`Trainer` 自带的恢复逻辑或 `ModelCheckpoint` 内部校验，就会落到 `pytorch_lightning/utilities/cloud_io.py:33` 的 `torch.load(f, map_location=...)`，那里没传 `weights_only`，PyTorch 2.6+ 默认 `True` 时会报：

  ```text
  _pickle.UnpicklingError: Weights only load failed ...
  Unsupported global: pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
  ```

- 因为 PL 1.x 把 `ModelCheckpoint / EarlyStopping / LRScheduler` 等多个类的实例也 pickle 进了 ckpt，逐个 `add_safe_globals` 不现实，最干净的做法就是在入口处把 `torch.load` 的默认值改回 `False`。
- 用 `kwargs.setdefault("weights_only", False)` 而不是直接覆盖，保留 “调用方显式传 `weights_only=True` 时仍尊重” 的语义。
- 仅安全地用于本项目自训 + 官方发布的 ckpt；不要拿来加载未知来源的 `.ckpt / .pt`。

## `src/lightning/data.py`

### 1. 分布式进程组初始化检查

原代码直接调用：

```python
dist.get_world_size()
dist.get_rank()
```

已改为先判断：

```python
if dist.is_available() and dist.is_initialized():
    ...
else:
    self.world_size = 1
    self.rank = 0
```

原因：单卡非 DDP 训练时没有初始化默认 process group，直接调用会报错。

### 2. 验证和测试阶段按需使用 `DistributedSampler`

`val_dataloader()` 和 `test_dataloader()` 已改为：

```python
use_distributed = dist.is_available() and dist.is_initialized() and self.world_size > 1
```

只有 `use_distributed=True` 时才使用 `DistributedSampler`；否则直接使用普通 `DataLoader`。

原因：`DistributedSampler` 会访问默认分布式进程组，单卡非分布式模式下会报 `Default process group has not been initialized`。

## `src/utils/plotting.py`

绘图函数 `_make_evaluation_figure()` 中，将用于可视化的 tensor 转 numpy 前加了 `detach()`：

```python
data['mkpts1_f'][b_mask].detach().cpu().numpy()
```

同类修改包括：

- `image0`
- `image1`
- `mkpts0_f`
- `mkpts1_f`
- `scale0`
- `scale1`
- `epi_errs`

原因：训练阶段这些 tensor 可能仍在计算图中，直接 `.numpy()` 会报错；绘图不参与反向传播，应该先 `detach()`。

## `src/lightning/lightning_loftr.py`

### 1. 兼容 PyTorch 2.6+ 加载官方 checkpoint

原代码：

```python
state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
```

已改为：

```python
state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)['state_dict']
```

原因：PyTorch 2.6+ 中 `torch.load()` 默认 `weights_only=True`，加载旧版 PyTorch Lightning 保存的 `.ckpt` 时，可能拒绝反序列化 checkpoint 内的 Lightning 对象并报：

```text
Weights only load failed
Unsupported global: pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
```

本项目使用的是 EfficientLoFTR 官方权重 `weights/eloftr_outdoor.ckpt`，来源可信，因此显式设置 `weights_only=False` 以保持旧版本加载行为。不要对未知来源的 checkpoint 随意使用该选项。

## 当前训练状态

debug 训练已能跑通，输出中出现：

```text
Epoch 0: 100%
loss=1.64
Aggregating metrics over 2 unique items...
```

说明：

- 数据集可以正常读取。
- 模型 forward / loss / backward 可以运行。
- validation 可以运行。
- Windows 单卡训练链路已经基本打通。

## 推荐后续训练命令

先做小规模训练：

```bat
python train.py data\MyTrainData\config\my_train_debug.py configs\loftr\eloftr_full.py --exp_name=my_train_small --gpus=1 --num_nodes=1 --batch_size=1 --num_workers=0 --pin_memory=false --check_val_every_n_epoch=1 --log_every_n_steps=20 --limit_train_batches=100 --limit_val_batches=20 --num_sanity_val_steps=0 --max_epochs=3 --disable_mp --thr 0.1
```

确认稳定后再去掉 `--limit_train_batches` 和 `--limit_val_batches`，开始更长训练。
