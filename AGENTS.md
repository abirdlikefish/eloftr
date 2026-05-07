# Agent 工作守则（强约束，必读）

> 本文件由项目维护者定义，所有 AI agent 在本仓库工作时必须遵守。同等约束的机器可读版本见 `.cursor/rules/00-scope-and-output.mdc` 与 `.cursor/rules/01-server-ask-only.mdc`。`.cursor/hooks.json` 当前**仅做对话归档**（跨平台 Python），不再做边界守卫——guard hook 因 Linux 路径硬编码 + Windows 不兼容已废止。
>
> **守卫职责**：agent 自律 + 用户审 diff + 服务器 ask-mode 纪律（见 `abfNote/dev-workflow.md`）。

## 1. 当前环境快照（与 skill `eloftr-vlrlab-server` 对照差异）

- **仓库根**：
  - 服务器：`/home/xyjiang/Desktop/yurupeng/eloftr/`（**非** skill 推荐的 `~/projects/yurupeng/efficient_loftr/`）
  - 本地 Windows：`c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\`
- **服务器 agent 身份**：`uid=0(root)`，但 `USER=xyjiang`、`HOME=/home/xyjiang`、`CURSOR_ORIG_UID=1001`
- **服务器**：vlrlab 4×RTX 3090（详见 `.cursor/skills/eloftr-vlrlab-server/SKILL.md` §0）
- **conda（服务器）**：使用克隆出的 `eloftr_yurupeng`。**禁止**修改 `eloftr_training` / `eloftr` / `loftr` / `Xoftr` / `gim` / `LightGlue` / `RoMa` / `glue-factory` / `minima` / `dust3r` 等导师环境
- **当前的边界**：guard hook 已删除，agent 必须严格自律遵守 §2/§3/§4。**服务器侧仅 ask mode**（见 `.cursor/rules/01-server-ask-only.mdc`），所有写入与 shell 由本地 agent mode + git push 完成；服务器只 `git pull --ff-only` + 手动 tmux 跑训练。最后一道防线是你/导师人工把关。

## 2. 可写区（白名单）

**仅** `/home/xyjiang/Desktop/yurupeng/` 整个子树可写。

**只读软链复用**（仅作软链目标，禁止写/新建/删除）：
- `/data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection/`（M3FD：Annotation/  Ir/  Vis/  test_generate_file/Vis/）
- `/data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/roadscene/`（RoadScene：ir/  vi/  meta/）
- `/data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/tno/`（TNO，OOD 备用）
- `/data/xyjiang/Datasets/`、`/data/xyjiang/Matching/EfficientLoFTR/`（备查参考，仅读）

**训练产物**（写入仓库内系统盘 NVMe，单实验 ≤ 5 GB，跑 5–10 个 ablation 总占用 < 50 GB 可控）：
- `logs/`（TB 日志）
- `weights/`（ckpt）
- `data/pc_cache/`（v7+ PC 边缘缓存，**不能写源只读目录**，必须落到这里）

## 3. 黑名单（禁止任何写入，agent 自律拒绝 + 服务器 ask-mode 纪律守住）

- `/home/xyjiang/anaconda3/`、`/home/xyjiang/.bashrc`、`/home/xyjiang/.condarc`、`/home/xyjiang/.gitconfig`
- `/home/xyjiang/.cursor/`、`/home/xyjiang/.cursor-server/`（已被 root 污染过一次，禁止二次写入）
- `/home/xyjiang/projects/`（如果存在，是导师工作目录）
- `/home/xyjiang/` 下除 `Desktop/yurupeng/` 之外的所有子目录
- **整个 `/data/`**（包括 `/data/xyjiang/`、`/data/student2/`、`/data/llt/`、`/data/underwater/`），仅作只读软链目标

## 4. 危险操作禁令

- 禁止 `sudo` / `su` 二次提权（已是 root）
- 禁止 `rm -rf` 任何不在白名单内的路径
- 禁止 `pip install` 落到 `base` / `eloftr` / `eloftr_training` 等导师 env（先 `conda activate eloftr_yurupeng`）
- 禁止 `chmod -R` / `chown -R` 整目录
- 禁止 `dd of=` / `mkfs` / `shutdown` / `reboot` 等系统级命令
- 禁止占满 4 张 3090；训练前先 `nvidia-smi`，用 `CUDA_VISIBLE_DEVICES=N[,M]` 限定卡
- 不要在 sandbox shell 内断言 `nvidia-smi` 不可用——sandbox 屏蔽 `/dev/nvidia*`，应在 tmux 真实 shell 验证

## 5. 输出物路径表（agent 必须遵守）

| 类别 | 路径 | 入版本库? |
|---|---|---|
| 新 skill | `.cursor/skills/<name>/SKILL.md` | ✓ |
| 新 rule | `.cursor/rules/<name>.mdc` | ✓ |
| 新 hook | `.cursor/hooks.json` + `.cursor/hooks/<name>.{py,sh,bat}` | ✓ |
| Plan / Todo 阶段总结 | `~/.cursor/plans/<name>_<hash>.plan.md`（Cursor 默认全局位置，不双写到项目）| — |
| 对话记录归档 | `.cursor/chat-logs/<session-uuid>.jsonl` | ✗（gitignore，由 `.cursor/hooks/archive_transcript.py` 在 stop/sessionEnd 时自动归档，跨平台） |
| 人工对话备忘 | `abfNote/chat-logs/<YYYYMMDD>.md` | ✗（gitignore） |
| 工程笔记 / 实验记录 | `abfNote/<topic>.md` | ✓ |
| 训练脚本 | `MyScripts/run_*.sh` | ✓ |
| 训练日志 / TB | `logs/`（仓库内实体目录，系统盘 NVMe） | ✗（已 gitignore） |
| 权重 ckpt | `weights/`（仓库内实体目录） | ✗（已 gitignore） |
| 数据集 | `data/<dataset>/` 软链到只读源 | ✗（已 gitignore） |
| PC-CLAHE 缓存 | `data/pc_cache/<dataset>/{Ir_pc,Vis_pc}/` | ✗（已 gitignore） |
| 跨版本 eval 汇总 | `results/eval_summary.md`（维护规则见 `.cursor/skills/eloftr-results/SKILL.md`） | ✓ |
| 跨实验 TB 训练侧汇总 | `results/tb_summary.md`（维护规则见 `.cursor/skills/eloftr-tb-summary/SKILL.md`） | ✓ |

**禁止**写入用户级 Cursor 目录（`/home/xyjiang/.cursor/skills-cursor/`、`/home/xyjiang/.cursor/rules/` 等）。

### 5.1 plan 双写规则（**已废弃，2026-05-07**）

> **DEPRECATED**：原"CreatePlan + Write 项目内副本"双写规则已废弃。
>
> 废弃理由：
>
> - **服务器侧不会触发 `CreatePlan`**：按 `.cursor/rules/01-server-ask-only.mdc`，服务器侧仅 ask mode，不调写工具，agent 不会在服务器生成 plan。原"plan 散落在 `/home/xyjiang/.cursor/plans/` 污染共用账号"的担忧场景不存在。
> - **跨机器可见性不再是 plan 的设计目标**：plan 是开发会话内的工作记忆（IDE plan-card UI），完成后被 commit/对话归档替代；不需要 git 跨机同步。
> - **双写本身有维护成本**：每次 plan 修改要改两份，容易漂移。
>
> 现行约定：
>
> - plan 只走 `CreatePlan` 默认位置 `~/.cursor/plans/<name>_<hash>.plan.md`（Cursor 全局）
> - 后续维护用文件编辑工具直接改全局位置那份
> - 项目内 `.cursor/plans/` 目录保留 `.gitignore` 黑名单（防御性，万一手动 drop 不会污染版本库）
> - 重要的工程决策 / 阶段总结落 `abfNote/<topic>.md`（入版本库），不是 plan

## 6. 数据集软链表（执行 `ln -s` 时使用）

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr/data
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection M3FD_Detection
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/roadscene      RoadScene
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/tno            TNO
```

> 现有 `data/megadepth/` 与 `data/scannet/` 占位目录（含 `.gitignore`）保留不动，cross_modal_v1 分支不训这两个；将来需要再说。

## 7. skill 索引（按使用频率）

- **写脚本 / 找不到 conda env / ModuleNotFoundError** → `.cursor/skills/eloftr-conda-env-activation/SKILL.md`（双端激活协议，新写 .bat / .sh **必读**；与本文件 §10 等价）
- 首次接服务器 → `.cursor/skills/eloftr-vlrlab-server/SKILL.md`
- 多卡 / DDP / v9 复现 v8 → `.cursor/skills/eloftr-server-multigpu/SKILL.md`
- 评测流程 → `.cursor/skills/eloftr-eval-pipeline/SKILL.md`
- **跨版本 eval 数字 / SOTA 排名 / 综合通用性** → `.cursor/skills/eloftr-results/SKILL.md`（指向 `results/eval_summary.md`，新 eval 跑完必走的追加流程）
- **跨实验训练侧 KPI / 训练成本 / best val epoch / modemb-MSBN 诊断** → `.cursor/skills/eloftr-tb-summary/SKILL.md`（指向 `results/tb_summary.md`，新训练跑完必走的 aggregate 流程；与 eval 表用途互补）
- 数据集（M3FD / RoadScene） → `.cursor/skills/eloftr-{m3fd,roadscene}-data/SKILL.md`
- 各版本 ablation → `.cursor/skills/eloftr-v{1..9}-*/SKILL.md`、`eloftr-cross-modal-experiments/SKILL.md`
- 仅本地 Windows 单卡相关 → `.cursor/skills/eloftr-windows-setup/SKILL.md`（**服务器忽略**这条）

## 8. 已知工程改动 TODO（动训练 cfg 时再做）

- [ ] **v7+ PC 边缘缓存输出路径**：cfg/脚本默认写到 `data/<dataset>/Ir_pc|Vis_pc/`，但 `data/M3FD_Detection` 等是只读软链。需改 `MyScripts/precompute_pc_edges.py` 输出位置 → `data/pc_cache/<dataset>/{Ir_pc,Vis_pc}/`，并修对应 cfg 的 `cache_dir`。
- [ ] **train/val/test 划分文件**：M3FD/RoadScene 源目录里如果没有 `index/`，需要在仓库内 `data/index/<dataset>/{train,val,test}_pairs.txt` 单独建，**不要写到只读源目录**。
- [ ] **未来如训 MegaDepth/ScanNet**：先 `find /data/xyjiang/Datasets/` 看是否有现成可软链的；预期分别在 `LargeData4DL/megadepth1500/` 与 `ScanNet_data/`。

## 9. 工作流速记（每次开 agent 前自检）

> 整体流程详见 `abfNote/dev-workflow.md`：本地 Windows agent mode 改 → `git push` → 服务器 ask mode + 手动 tmux 跑训练。

### 9.A 本地 Windows（开发端）

1. `pwd` 确认在 `c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\`
2. `git status` 干净，`git pull --ff-only` 拉服务器侧热修
3. agent mode 改代码（cfg / 模型 / 脚本）
4. `git add . && git commit -m "..." && git push`

### 9.B 服务器 vlrlab（运行端，**仅 ask mode**）

1. `pwd` 确认在 `/home/xyjiang/Desktop/yurupeng/eloftr/`
2. `git pull --ff-only`（不允许 merge commit；如果 fail 说明服务器有未提交改动，需要先在服务器 commit + push 一次）
3. `nvidia-smi` 看卡占用（在真实 tmux shell 里，sandbox 里看不到）
4. 长任务进 tmux：`tmux new -s <task>`，detach 用 `Ctrl+B D`
5. `bash MyScripts/run_<exp>.sh`（脚本头部已自动 `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` + `cd` + `PYTHONPATH` + `CUDA_VISIBLE_DEVICES`，详见 §10 / `.cursor/skills/eloftr-conda-env-activation/SKILL.md`）
   - 仅在不走 .sh、要直接 `python` 时手敲：`source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng && export CUDA_VISIBLE_DEVICES=N && export PYTHONPATH="$PWD:${PYTHONPATH:-}"`
   - **不要**写 `conda activate eloftr_yurupeng`（root shell 默认 PATH 没 anaconda/bin，会 `bash: conda: command not found`）
6. 训练产物已 `.gitignore`（`logs/` `weights/` `data/`），无需 push；只 push `abfNote/<topic>.md` 这类实验笔记

## 10. 脚本头部协议（conda 环境激活，**写新 .bat / .sh 必读**）

> 完整速查、反模式表、PyTorch 1.12 vs 2.6 差异见 `.cursor/skills/eloftr-conda-env-activation/SKILL.md`。本节是强约束模板，agent 写新脚本必须**原样照搬**，不允许自创激活方式。

### 10.A 双端激活协议对照

| 项 | 本地 Windows | 服务器 vlrlab |
|---|---|---|
| 仓库根 | `c:\Users\abirdlikefish\Desktop\毕设\efficient_loftr\` | `/home/xyjiang/Desktop/yurupeng/eloftr/` |
| conda 安装路径 | `%USERPROFILE%\miniconda3\` | `/home/xyjiang/anaconda3/` |
| 激活命令 | `call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr` | `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` |
| 环境名 | `eff_loftr` | `eloftr_yurupeng`（**禁** `eloftr` / `eloftr_training` / `loftr` 等导师 env） |
| PyTorch / NumPy | 2.6 / ≥2.0 | 1.12.1 / <2.0 |
| Shell | cmd（**禁 PowerShell**） | bash + tmux |
| `expandable_segments` | 可用 | 禁用（PyTorch 1.12.1 不识别，会报 `Unrecognized CachingAllocator option`） |

### 10.B 本地 Windows `.bat` 头部模板（六行原样照搬）

```bat
@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%
```

参考范本：`MyScripts/run_m3fd_v8_msbn.bat`、`MyScripts/run_m3fd_v7_pcclahe.bat`。

### 10.C 服务器 Linux `.sh` 头部模板（七行原样照搬）

```bash
#!/bin/bash
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
```

参考范本：`MyScripts/run_m3fd_v9_e2e.sh`、`MyScripts/precompute_pc_edges.sh`。

### 10.D 环境变量顺序（强约束，写错就"环境激活了但 python 还是错的"）

```
[1] activate                      ← 必须最先
[2] cd 仓库根
[3] PYTHONPATH=$PWD               ← 让 src/ 模块可 import
[4] CUDA_VISIBLE_DEVICES=N
[5] PYTORCH_CUDA_ALLOC_CONF=...   ← 仅本地（PyTorch ≥ 2.1），服务器跳过
[6] python train.py ...           ← 必须最后
```

### 10.E 高频反模式（AI 找不到环境的根因，**禁用**）

| 反模式 | 正确写法 |
|---|---|
| `conda activate eff_loftr` （Windows 普通 cmd） | `call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr` |
| `conda activate eloftr_yurupeng` （服务器 root shell） | `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` |
| `source activate eloftr_yurupeng`（旧 conda 1.x 语法） | 同上（用 `source <prefix>/bin/activate <env>`） |
| `cd /d %CD%\..` / 写死绝对路径 | `cd /d "%~dp0.."` / `cd "$(dirname "$0")/.."` |
| `set PYTHONPATH=%CD%`（吞掉旧值） | `set PYTHONPATH=%CD%;%PYTHONPATH%` |
| `export PYTHONPATH=$PWD`（无 `:-` fallback） | `export PYTHONPATH="$PWD:${PYTHONPATH:-}"` |
| 服务器导出 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 删除该行；让 `train.py:52` 兜底成 `max_split_size_mb:1024` |
| `pip install xxx` 在 activate 之前 | 先 activate 再装；**禁止**装到 base / `eloftr_training` 等导师 env |
| PowerShell 跑 `set X=Y && python` | 切 cmd；或写 `$env:X="Y"; python ...`
