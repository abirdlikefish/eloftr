---
name: eloftr-windows-setup
description: 'Run EfficientLoFTR training and evaluation on Windows + single GPU + PyTorch 2.6 + NumPy 2.0+ + PL 1.3.5; six compatibility patches plus VRAM spillover cure (expandable_segments). Use when running, debugging, or setting up training/eval on Windows. Triggers: Windows 单卡 / 单卡训练 / NCCL 错误 / np.Inf 错误 / weights_only 报错 / DistributedSampler / 加载 ckpt 报错 / 显存碎片 / spillover / VRAM 16GB / GPU-Util 100 但功耗低 / PCIe 退化 / expandable_segments / PYTORCH_CUDA_ALLOC_CONF / cmd vs PowerShell / 激活 conda 环境 / eff_loftr 环境 / 双击 bat 启动, English ''Default process group has not been initialized'', ''Can''''t call numpy() on Tensor that requires grad'', ''Distributed package doesn''''t have NCCL'', ''Weights only load failed'', ''Unsupported global ModelCheckpoint'', ''GPU spillover'', ''PyTorch allocator fragmentation'', ''CUDA Virtual Memory API''. Spillover cure details for v6.1 see eloftr-v6-finetune.'
---

# EfficientLoFTR Windows + 单卡兼容补丁

> 本仓库官方代码假设 Linux + 多卡 + NCCL + PyTorch ≤ 2.5 + NumPy < 2.0 + PyTorch Lightning 1.3.5。
> 在 Windows 单卡 + PyTorch 2.6 + NumPy ≥ 2.0 上必须打 6 处补丁才能跑通 train / val / test / resume。
> 详细的原始改动日志：[abfNote/source_changes_training_windows.md](../../../abfNote/source_changes_training_windows.md)。
> 本 skill 是它的简化速查版。

## 1. 环境与激活

- 推荐 conda 环境名：`eff_loftr`（与 README 建议一致）。
- 激活 + 跑脚本必须在 **Anaconda Prompt (cmd)** 里：
  ```bat
  call conda activate eff_loftr
  set PYTHONPATH=%CD%;%PYTHONPATH%
  python train.py configs\data\... configs\loftr\... ...
  ```
- 不要用 PowerShell 拼 `&&`，会报 `ParserError: missing '(' after 'if'`、`'&&' invalid`。
  PowerShell 也不认 `set XXX=...`，要写 `$env:XXX = "..."`。最稳妥就是切 cmd。
- `.bat` 脚本（`MyScripts/run_roadscene_*.bat`、`MyScripts/eval_roadscene_*.bat`）已经默认走 cmd 语法，直接双击或在 cmd 里 `call MyScripts\run_roadscene_v3_combined.bat` 即可。

## 2. 六处必打补丁速查

| # | 文件 | 症状 / 原因 | 修复 |
|---|------|-------------|------|
| 1 | [train.py](../../../train.py)（`import numpy as np` 之后） | `AttributeError: np.Inf was removed in the NumPy 2.0 release` | 加 `np.Inf = np.inf`。PL 1.3.5 的 `EarlyStopping` / `ModelCheckpoint` 内部还在引用 `np.Inf`。 |
| 2 | [train.py](../../../train.py) `plugins=[...]` 处 | `RuntimeError: Distributed package doesn't have NCCL built in` | 把 `DDPPlugin` 包到 `if config.TRAINER.WORLD_SIZE > 1:` 里；同时 `pl.Trainer(..., sync_batchnorm=config.TRAINER.WORLD_SIZE > 1, ...)` 把阈值从 `> 0` 改成 `> 1`。 |
| 3 | [src/lightning/data.py](../../../src/lightning/data.py) `setup()` + `val/test_dataloader()` | `ValueError: Default process group has not been initialized` | 在所有 `dist.get_world_size()` / `DistributedSampler(...)` 前加 `if dist.is_available() and dist.is_initialized():` 守卫，单卡走 `world_size=1, rank=0` 分支。 |
| 4 | [src/utils/plotting.py](../../../src/utils/plotting.py) `_make_evaluation_figure*` | `RuntimeError: Can't call numpy() on Tensor that requires grad` | 所有 `tensor.cpu().numpy()` 改成 `tensor.detach().cpu().numpy()`，验证阶段被 hook 调用时 tensor 还挂着 autograd 图。 |
| 5 | [src/lightning/lightning_loftr.py](../../../src/lightning/lightning_loftr.py) `__init__` 加载 ckpt 处 | `_pickle.UnpicklingError: Weights only load failed ...`（**只发生在 `--ckpt_path` 旁路**） | `torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)['state_dict']`。PyTorch 2.6 把 `weights_only` 默认改成了 `True`，但官方 ELoFTR ckpt 含 PL 元数据。 |
| 6 | [train.py](../../../train.py) 紧跟 `import torch` 之后 | 切换到 `--resume_from_checkpoint` 后再次报 `Weights only load failed ... Unsupported global: pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`（PL 内部 `pl_load` 没传 `weights_only=False`） | 全局 monkey-patch `torch.load`：<br/>`_orig_torch_load = torch.load`<br/>`def _torch_load_compat(*a, **kw):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`kw.setdefault('weights_only', False)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`return _orig_torch_load(*a, **kw)`<br/>`torch.load = _torch_load_compat`<br/>用 `setdefault` 而不是直接覆盖，保留显式传 `True` 时的语义；只对自己训练 / 官方下载的 ckpt 用，不要拿来加载未知来源的 `.ckpt`。 |

## 3. 还会遇到的 Edge Cases

- **`reload_dataloaders_every_epoch=False`**（[train.py](../../../train.py)）：避免 PL 在 epoch 边界重建 sampler，否则训练集会重新 shuffle/抽样，单卡也会触发上面的 `DistributedSampler` 路径。
- **`ModelCheckpoint` 仅在 `not args.disable_ckpt` 时构造**（[train.py](../../../train.py)）：`--disable_ckpt` 调试时不建 callback，避免在 RoadScene 这种没有 `auc@10` 的任务上报 monitor 找不到。
- **EarlyStopping 已经从 `--disable_ckpt` 分支里独立出来**（[train.py](../../../train.py) 第 151-168 行）：所以 debug 脚本即便 `--disable_ckpt` 也会触发 EarlyStopping，详见 [eloftr-v3-v4-freeze §4](../eloftr-v3-v4-freeze/SKILL.md)。
- **TensorBoard 在 Windows 默认会把 tag 里的 `/` 当目录**：项目里所有自定义 metric 用 `precision@3px` 这种 `@` 命名而不是 `precision/3px`。

## 3.1 VRAM Spillover Cure：expandable_segments（PyTorch allocator 碎片化）

长时间 finetune（如 v6 50 epoch）在 16GB VRAM + 多个非训练 GPU 进程并存时，会出现"GPU-Util 100% + 功耗远低于 TDP + per-step 时间从 7s 飙到 80s"指纹，是 PyTorch caching allocator (slab-based) 的内部碎片化导致 fallback 到 shared GPU memory（PCIe 退化 22×）。

**Cure**：在 .bat 里 `python` 之前加：

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

把 PyTorch allocator 切到 CUDA Virtual Memory API（虚拟连续段，物理页可不连续），消除碎片可能性。代价：每个 segment 第一次分配多 ~2-3 ms。

完整诊断与 v6.1 实测验证见 [eloftr-v6-finetune §6 v6.1 spillover hotfix](../eloftr-v6-finetune/SKILL.md)。配套操作 SOP：用任务管理器关掉非训练 GPU 应用（Cursor / Edge / Steam / NVIDIA App / 浏览器 WebGL 标签页），从独立 WindowsTerminal 双击 .bat 启动而不要从 Cursor 集成终端启动。

## 4. 常见症状 → 这里找答案

- `np.Inf was removed`：表 1。
- `Default process group has not been initialized`：表 3。
- `Distributed package doesn't have NCCL built in`：表 2。
- `Can't call numpy() on Tensor that requires grad`：表 4。
- `Weights only load failed` + 报错来自你自己的 `--ckpt_path` 加载（栈上有 `src/lightning/lightning_loftr.py`）：表 5。
- `Weights only load failed` + 报错来自 PL 内部（栈上有 `pytorch_lightning/utilities/cloud_io.py` / `checkpoint_connector.restore`），通常发生在 `--resume_from_checkpoint`：表 6。
- `--disable_ckpt` 下 EarlyStopping 没生效：见上一节第 3 条。

> 注意：`--ckpt_path` 与 `--resume_from_checkpoint` 是两个完全不同的 flag。前者只把权重灌进 `matcher`（warm start），不恢复 optimizer / scheduler / global_step；后者才是 PL 真正的断点续训。详见 [eloftr-v6-finetune §3 --ckpt_path 在本仓的精确语义](../eloftr-v6-finetune/SKILL.md)。

## 5. 想加新 Windows 兼容补丁时

- 新补丁在 [abfNote/source_changes_training_windows.md](../../../abfNote/source_changes_training_windows.md) 末尾追加一节，按 “症状 / 原因 / 修复 / 受影响文件” 四段写。
- 同步在本 skill 第 2 节表格里加一行，并把行号链接保持指向当前提交的位置（用相对路径而不是绝对路径）。
