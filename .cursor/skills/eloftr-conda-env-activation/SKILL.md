---
name: eloftr-conda-env-activation
description: 'EfficientLoFTR 项目本地 Windows + 服务器 vlrlab 双端 conda 环境激活协议。规定本地用 eff_loftr (PyTorch 2.6 + NumPy 2.0+，conda 在 %USERPROFILE%\miniconda3) 与服务器用 eloftr_yurupeng (PyTorch 1.12.1，conda 在 /home/xyjiang/anaconda3)，提供两套即取即用的脚本头部模板（cmd .bat / bash .sh），以及 PYTHONPATH / CUDA_VISIBLE_DEVICES / PYTORCH_CUDA_ALLOC_CONF 的强制顺序。Use when 写新训练 / 评测脚本 / debug AI 找不到 python / 报 conda not found / ModuleNotFoundError src.lightning / activate.bat 找不到 / which python 指向 base / 跑脚本时模块导入失败 / 切换本地与服务器 / PowerShell 报 set 错误 / pip install 之前要不要 activate / PYTORCH_CUDA_ALLOC_CONF 顺序 / Unrecognized CachingAllocator option / weights_only TypeError / expandable_segments 在 PyTorch 1.12 不可用. Triggers: conda activate / conda not found / 找不到 conda / 找不到 python / 哪个环境 / 哪个 env / eff_loftr / eloftr_yurupeng / eloftr_training (禁用) / activate.bat / source activate / PYTHONPATH 设置 / set PYTHONPATH / export PYTHONPATH / ModuleNotFoundError src / cmd vs PowerShell / 双击 bat / tmux 跑训练 / CUDA_VISIBLE_DEVICES / PYTORCH_CUDA_ALLOC_CONF / expandable_segments / PyTorch 1.12.1 / PyTorch 2.6 / 服务器 PyTorch 版本 / pip install 污染 base / 服务器共用账号 / 不要碰导师 env, English "conda activate failed", "conda not found", "ModuleNotFoundError", "wrong python interpreter", "which python", "activate.bat not found", "PYTHONPATH not set", "PowerShell does not recognize set", "Unrecognized CachingAllocator option". 与 eloftr-windows-setup §1 (cmd vs PowerShell) / eloftr-vlrlab-server §1 (共用账号) / eloftr-yurupeng-workspace §0 (实际部署路径) 互补。'
---

# EfficientLoFTR conda 环境激活协议（双端）

> 本 skill 解决 AI 反复出现的"找不到 conda 环境 / 找不到 python / ModuleNotFoundError"问题。
> 项目两端环境：
>   - 本地 Windows（开发） → `eff_loftr`（PyTorch 2.6 + NumPy 2.0+）
>   - 服务器 vlrlab Linux（运行） → `eloftr_yurupeng`（PyTorch 1.12.1，克隆自导师 env）
>
> 不同点见 §3 对照表。本 skill 与下列 skill 互补，**禁止重复造轮子**：
>   - `eloftr-windows-setup` §1：cmd vs PowerShell 区别
>   - `eloftr-vlrlab-server` §1：共用账号守则
>   - `eloftr-yurupeng-workspace`：服务器实际部署路径与守卫

## 0. agent 自检：当前在哪一端？

session 开始时第一次写脚本/激活前，**必须**先判断当前环境，缓存到 session 内：

| 信号 | 本地 Windows | 服务器 vlrlab |
|---|---|---|
| 仓库根前缀 | `c:\` 或 `C:\` | `/home/xyjiang/` |
| `os.name` | `'nt'` | `'posix'` |
| `user_info.OS Version` | `win32 ...` | `linux ...` |
| Shell | cmd / Anaconda Prompt | bash + tmux |

判断后**不要每个 tool call 重复确认**，只在 session 第一次激活/写脚本前判断一次。

服务器端额外约束：根据 `.cursor/rules/01-server-ask-only.mdc`，agent 在服务器侧**仅 ask mode**，本 skill 的 §2 模板由本地 agent 写好后通过 `git push` 同步，服务器只 `git pull --ff-only` + 手动 `bash MyScripts/run_xxx.sh` 跑。

## 1. 本地 Windows 标准模板（cmd 唯一支持）

仓库根：`c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\`

任何新 `.bat` 必须以下列六行开头（37 个现有脚本统一这样写）：

```bat
@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%
```

- `%USERPROFILE%\miniconda3\Scripts\activate.bat` 是 Windows miniconda 的标准激活入口（不依赖 PATH）
- `%~dp0..` 是当前 .bat 所在目录的上一级 → 仓库根
- 必须用 `call`，否则 .bat 会停在 activate.bat 末尾不返回

**禁止**：

- 直接 `conda activate eff_loftr`：非 Anaconda Prompt 起的 cmd 里 `conda` 不在 PATH 上
- PowerShell 跑 .bat 或拼 `&&`：`set` 不通、`&&` 不通；硬要在 PowerShell 里跑改成 `& cmd /c MyScripts\xxx.bat` 或者就**切回 cmd**
- `set PYTHONPATH=%CD%`：吞掉已有的 PYTHONPATH，必须 `;%PYTHONPATH%`

Python 解释器最终位置：`C:\Users\abirdlikefish\miniconda3\envs\eff_loftr\python.exe`

`where python` 应该指向上面这条；如果指向 `miniconda3\python.exe`（base）或 `C:\Windows\...`，说明 activate 没生效，回 §1 检查。

## 2. 服务器 vlrlab 标准模板（bash + tmux）

仓库根：`/home/xyjiang/Desktop/yurupeng/eloftr/`

任何新 `.sh` 必须以下列七行开头（见 `MyScripts/run_m3fd_v9_e2e.sh`、`MyScripts/run_m3fd_v9_e2e_debug.sh`、`MyScripts/precompute_pc_edges.sh`、`MyScripts/view_training.sh`）：

```bash
#!/bin/bash
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
```

- `source /home/xyjiang/anaconda3/bin/activate <env>` 是导师装的 anaconda 的激活入口；agent root 身份默认 PATH 里**没有** anaconda/bin，必须显式 source
- `set -euo pipefail` 防止 activate 失败后硬跑
- `${PYTHONPATH:-}` 的 `:-` 是 `set -u` 下 PYTHONPATH 未定义时退化为空字符串，否则 unbound variable
- 长训练任务**必须**进 tmux：`tmux new -s <task>`，detach `Ctrl+B D`

**禁止**：

- `conda activate eloftr_yurupeng`：未 init 的 root shell 里 `conda` 不在 PATH 上
- `source activate eloftr_yurupeng`：旧 conda 语法，2.0+ 已废
- 用 `eloftr` / `eloftr_training` / `loftr` / `Xoftr` / `LightGlue` / `RoMa` / `gim` / `glue-factory` / `minima` / `dust3r` 任何之一：**全部是导师只读环境**（`AGENTS.md §1`）
- `pip install xxx` 之前没 activate：会污染 base，影响导师 env
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：服务器 PyTorch 1.12.1 不认这个 key，会报 `Unrecognized CachingAllocator option`，让 `train.py:52` 的 `setdefault('max_split_size_mb:1024')` 自然兜底（见 `MyScripts/run_m3fd_v9_e2e.sh:40-45`）

Python 解释器最终位置：`/home/xyjiang/anaconda3/envs/eloftr_yurupeng/bin/python`

`which python` 应该指向上面这条；如果指向 `/usr/bin/python` 或 `/home/xyjiang/anaconda3/bin/python`（base），说明 activate 没生效。

## 3. 两端对照速查

| 项 | 本地 Windows | 服务器 vlrlab |
|---|---|---|
| 仓库根 | `c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\` | `/home/xyjiang/Desktop/yurupeng/eloftr/` |
| conda 安装路径 | `%USERPROFILE%\miniconda3\` | `/home/xyjiang/anaconda3/` |
| 激活命令 | `call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr` | `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` |
| 环境名 | `eff_loftr` | `eloftr_yurupeng` |
| PyTorch | 2.6 | 1.12.1 |
| NumPy | ≥ 2.0（需 `np.Inf=np.inf` 补丁） | < 2.0 |
| Shell | cmd（禁 PowerShell） | bash + tmux |
| GPU 限制 | 默认单卡（`--gpus=1`） | `export CUDA_VISIBLE_DEVICES=N[,M]` 限定，运行前 `nvidia-smi` 看占用 |
| `expandable_segments` | 可用 | 禁用（PyTorch 1.12.1 不认） |
| 6 处兼容补丁 | 必须打齐（见 `eloftr-windows-setup`） | 同样需要（PyTorch 1.12.1 也兼容这 6 处补丁，见 `eloftr-windows-setup §2` 第 6 项的 try/except fallback） |
| 写边界 | 仓库整树自由 | 仅 `/home/xyjiang/Desktop/yurupeng/`，`/data/` 整盘只读 |
| agent 模式 | agent mode 自由开发 | **仅 ask mode**（`.cursor/rules/01-server-ask-only.mdc`） |
| 训练入口 | 双击 .bat 或 cmd 里 `MyScripts\run_xxx.bat` | tmux 内 `bash MyScripts/run_xxx.sh` |
| ckpt / 日志 | `logs/` `weights/`（仓库内，已 `.gitignore`） | 同上（仓库内系统盘 NVMe，单实验 ≤ 5 GB） |

## 4. 环境变量顺序（强约束）

无论哪一端，脚本里 export 的顺序是**有强约束**的，写错就会"环境激活了但 python 还是错的"：

```
[1] activate                      ← 必须最先
[2] cd 仓库根                      ← 解决相对路径
[3] PYTHONPATH=$PWD               ← 让 src/ 模块可 import
[4] CUDA_VISIBLE_DEVICES=N
[5] PYTORCH_CUDA_ALLOC_CONF=...   ← 仅本地（PyTorch ≥ 2.1），服务器 (1.12.1) 跳过
[6] python train.py ...           ← 必须最后
```

依据：

- `MyScripts/run_m3fd_v8_msbn.bat:29` 注释明写 "MUST be set BEFORE python invocation, AFTER conda activate"
- `MyScripts/run_m3fd_v6_1_finetune.bat:21` 注释 "AFTER conda activate (to avoid pollution of other envs)"
- `train.py:52` 的 `setdefault` 兜底：脚本没设 `PYTORCH_CUDA_ALLOC_CONF` 时退到 `max_split_size_mb:1024`

## 5. AI 反模式速查（"找不到环境"根因）

| 现象 | 根因 | 修复 |
|---|---|---|
| `'conda' is not recognized` (Windows) | 在普通 cmd 里直接调 `conda activate` | 改成 `call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr` |
| `bash: conda: command not found` (Linux) | root shell 默认 PATH 里没有 anaconda/bin | 改成 `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` |
| `ModuleNotFoundError: src.lightning` | 没 set/export PYTHONPATH | 添加 §1 / §2 模板里的第 5/6 行 |
| `which python` 指向 base | activate 没生效 / 顺序错 | 检查 §4 顺序，并确认 §1/§2 模板完整 |
| `Unrecognized CachingAllocator option: expandable_segments` | 服务器 PyTorch 1.12.1 不识别 | 服务器侧**不要** export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；让 `train.py:52` 兜底 |
| `TypeError: 'weights_only' is an invalid keyword` | 服务器 PyTorch < 1.13 | 走 `eloftr-windows-setup §2` 第 6 处的 try/except fallback monkey-patch |
| PowerShell 报 `&&` invalid 或 `set` parser error | PowerShell 不是 cmd | 切 cmd 或写 `$env:X="Y"; python ...` |
| 服务器 `pip install` 装到 base | 没 activate 就装 | 先 `source ...activate eloftr_yurupeng` 再装；**禁止**装到 `eloftr_training` 等导师 env |

## 6. 写新脚本时的 checklist

写一个新 `MyScripts/run_xxx.{bat,sh}` 前 agent 必须自查：

- [ ] 平台判断完成（§0），选对模板（§1 或 §2）
- [ ] 头部六/七行原样照搬，**不要自创**激活方式
- [ ] env 名写对（Windows = `eff_loftr` / Linux = `eloftr_yurupeng`，**不是** `eloftr` `eloftr_training`）
- [ ] PYTHONPATH 拼接而不是覆盖
- [ ] `cd` 用相对脚本位置（`%~dp0..` / `$(dirname "$0")/..`），不写死绝对路径
- [ ] 顺序遵守 §4：activate → cd → PYTHONPATH → CUDA_VISIBLE_DEVICES → [PYTORCH_CUDA_ALLOC_CONF] → python
- [ ] 服务器侧**不**写 `expandable_segments`
- [ ] 长任务在脚本注释里提示 "must be inside tmux"（见 `MyScripts/run_m3fd_v9_e2e.sh:11`）

## 7. 引用

- 本地标准范本：`MyScripts/run_m3fd_v8_msbn.bat`、`MyScripts/run_m3fd_v7_pcclahe.bat`
- 服务器标准范本：`MyScripts/run_m3fd_v9_e2e.sh`、`MyScripts/precompute_pc_edges.sh`
- 全 37 个 .bat / .sh 都遵守本协议（grep 验证：`MyScripts/` 内 `conda|activate|miniconda3|anaconda3|PYTHONPATH` 出现位置 100% 在头部 3-7 行）
- 边界守则：`AGENTS.md` §1, §3, §4, §10
- 服务器只读纪律：`.cursor/rules/01-server-ask-only.mdc`
