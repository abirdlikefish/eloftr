---
name: eloftr-yurupeng-workspace
description: 'yurupeng 在 vlrlab 4×RTX 3090 服务器（root 共用账号 xyjiang）上的 EfficientLoFTR 工作区 SOP：实际仓库路径 /home/xyjiang/Desktop/yurupeng/eloftr/、可写区/黑名单边界、Cursor agent 三层防越界守卫（AGENTS.md + .cursor/rules/00-scope-and-output.mdc + .cursor/hooks/{guard-write-scope,guard-shell-scope,archive-transcript}.sh + hooks.json）、M3FD/RoadScene/TNO 在 /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/ 下的精确只读源路径与软链命令、产物落位约定（logs/weights/data/pc_cache/.cursor/{plans,chat-logs}/abfNote/chat-logs/）、failClosed 自我锁外事故的解法、grep 字面量匹配的激进副作用、root 身份残留风险与 hook 软约束的边界。Use when 在 yurupeng 工作区开发 / 部署/审计 agent 守卫 / 排查 hook 拦截 / 建数据集软链 / 与 eloftr-vlrlab-server 推荐路径对照 / 解释为什么本仓库不在 ~/projects/ 下 / chat-logs 自动归档失败 / pip install 被拦 / sudo 被拦 / Write 工具被自家 hook 锁外面 / Read-only file system on .cursor / agent 写文件 owner=root 但目录 owner=xyjiang. Triggers: yurupeng / Desktop/yurupeng / root 守卫 / hooks.json 自我锁 / failClosed / guard-write-scope / guard-shell-scope / archive-transcript / Cursor sandbox readonly / chat-logs 镜像 / agent transcripts / .cursor/plans / .cursor/chat-logs / abfNote/chat-logs / AGENTS.md / 00-scope-and-output.mdc / TarDAL/M3FD_Detection / TarDAL/roadscene / TarDAL/tno / Infrared_image_datasets / 数据集只读 / 软链 / ln -s / dr-xr-xr-x / pip install 拦截 / sudo 拦截 / pc_cache 写源目录 / 与 eloftr-vlrlab-server 路径偏移. 与 eloftr-vlrlab-server 的关系：本 skill 是 yurupeng 实际部署的差异层；vlrlab-server 是通用 SOP。两者并存，遇路径冲突以本 skill 为准。'
---

# yurupeng 工作区差异层与 agent 守卫体系

> 本 skill 是在 [eloftr-vlrlab-server](../eloftr-vlrlab-server/SKILL.md)（通用 SOP）与 [eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md)（多卡 DDP）之上、专门记录 **yurupeng 在 vlrlab 服务器上的实际部署差异** 与 **Cursor agent 边界守卫工程化资产** 的 skill。
>
> **遇路径/身份/conda env 名称冲突时以本 skill 为准。** 通用机制（DDP、TensorBoard 端口转发、tmux 等）仍以另两条 skill 为准。

## 0. 与 vlrlab-server skill 的具体差异（先看）

| 维度 | [eloftr-vlrlab-server](../eloftr-vlrlab-server/SKILL.md) 推荐 | yurupeng 实际部署 |
|---|---|---|
| 仓库根 | `/home/xyjiang/projects/<your-name>/efficient_loftr/` | **`/home/xyjiang/Desktop/yurupeng/eloftr/`** |
| 大文件位置 | `/data/xyjiang/<your-name>/efficient_loftr/{data,logs,weights}` 软链 | **全部仓库内实体目录**（系统盘 NVMe，单实验 < 5 GB 可控） |
| conda env | `eloftr_<your-name>` | `eloftr_yurupeng`（克隆自 `eloftr_training`） |
| Cursor agent 身份 | `xyjiang` 普通用户（uid=1001） | **`uid=0(root)`**（CURSOR_ORIG_UID=1001，agent 启动时被提权） |
| 数据盘 `/data/` 写入 | 自己个人区 `/data/xyjiang/<your-name>/` 可写 | **整盘只读**（hook 黑名单），数据集仅作软链目标 |
| 防越界机制 | 自觉 + skill §1 守则 | **三层防越界**（AGENTS.md + 项目 rule + hook，详见 §3） |

## 1. 实际环境信息（首次接手必读）

### 1.1 服务器与身份

```
HostName    222.20.94.235  (备 222.20.99.26)
Port        8708
User        xyjiang （与导师共用，CURSOR_ORIG_UID=1001）
当前 shell  whoami=root, USER=xyjiang, HOME=/home/xyjiang  ← 已被提权
```

`whoami=root` + `HOME=/home/xyjiang` 这个组合意味着：
- 文件系统层面**没有保护**（root 能写整盘）
- skill `eloftr-vlrlab-server` §1 的所有"自觉"守则**只能靠 hook 软约束 + 人工把关**
- **绝对不要**在 agent 里跑 `sudo` / `su` 二次提权（hook 已拦）

### 1.2 磁盘与 GPU

| 盘 | 设备 | 总量 | 已用 | 剩余 | 用途 |
|---|---|---|---|---|---|
| 系统盘 | `/dev/nvme0n1p4` (NVMe ext4) | 819 GB | 199 GB | **579 GB** | 代码 + 训练日志 + ckpt + 小数据集 |
| 数据盘 | `/dev/sda1` (SATA ext4) | 15 TB | 6.0 TB | **7.8 TB** | 仅软链复用，**只读** |

GPU：4× RTX 3090 24GB 共享。Driver 570.153.02 / CUDA driver 12.8 / 系统 nvcc 11.7（PyTorch 自带 runtime，不依赖系统 nvcc）。

> Cursor agent 内 sandbox 屏蔽 `/dev/nvidia*`，`nvidia-smi` 在 agent shell 里会报 `couldn't communicate with driver`，**这是正常的**——真正训练在 tmux 真实 shell 里跑。

### 1.3 Cursor 本地配置（owner 已被污染）

`/home/xyjiang/.cursor/` 整个目录 owner 是 `root:root`（之前 root agent 写脏过）。**禁止再次写入**——hook 已把它列入黑名单。Cursor server 写 transcripts 用的是 xyjiang 身份，这部分目录 owner 仍是 xyjiang，归档脚本 cp 跨 owner 没问题。

## 2. 目录布局（最终态）

```
/home/xyjiang/Desktop/yurupeng/eloftr/        ← 仓库根（唯一可写区根）
├── AGENTS.md                                  ← 自然语言守则
├── .gitignore                                 ← 已追加 .cursor/{plans,chat-logs}/、abfNote/chat-logs/、.cursor/hooks/*.log
├── .cursor/
│   ├── hooks.json                             ← 注册 4 类 hook
│   ├── hooks/
│   │   ├── guard-write-scope.sh   (chmod 755) ← preToolUse 守卫
│   │   ├── guard-shell-scope.sh   (chmod 755) ← beforeShellExecution 守卫
│   │   └── archive-transcript.sh  (chmod 755) ← stop/sessionEnd 自动归档
│   ├── rules/
│   │   └── 00-scope-and-output.mdc            ← alwaysApply 边界规则
│   ├── skills/                                ← 14 条 eloftr-* skill + 本 skill
│   ├── plans/  (gitignore)                    ← agent 写的阶段总结
│   └── chat-logs/  (gitignore)                ← 自动归档的 transcripts
├── abfNote/
│   ├── source_changes_training_windows.md
│   ├── chat-logs/  (gitignore)                ← 人工对话备忘
│   └── <topic>.md                             ← 工程笔记
├── data/                                      ← gitignore（包含软链 + 写入子目录）
│   ├── M3FD_Detection → /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection  (只读软链)
│   ├── RoadScene      → /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/roadscene       (只读软链)
│   ├── TNO            → /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/tno             (只读软链)
│   ├── megadepth/  scannet/                   ← 现有 git 占位（含 .gitignore 子文件）
│   ├── pc_cache/<dataset>/{Ir_pc,Vis_pc}/     ← v7+ PC 边缘缓存（写入仓库内，**不能写源软链**）
│   └── index/<dataset>/{train,val,test}_pairs.txt  ← 若源目录无 index/ 时仓库内自建
├── logs/                                      ← gitignore，仓库内实体（NVMe 跑得快）
├── weights/                                   ← gitignore，仓库内实体
├── configs/  src/  MyScripts/  notebooks/
├── train.py  test.py  README.md
└── ...
```

**关键约定**：所有训练产物（logs/weights/data/pc_cache/）都落仓库内系统盘，不再走 `/data/xyjiang/yurupeng/` 软链（与 vlrlab-server skill §5 不同）。这是因为：
- 单实验 < 5 GB，5–10 个 ablation 总占用 < 50 GB，系统盘 579 GB 完全够
- NVMe 速度优势对 dataloader I/O 有边际收益
- 心智简洁：一个 Desktop/yurupeng/ 子树管所有自己的东西，hook 白名单单一项

> **未来**若训 MegaDepth/ScanNet（`data/megadepth/`、`data/scannet/` 占位目录已存在），单实验日志可能突破 50 GB，那时再把 `logs/` `weights/` 改软链到 `/data/xyjiang/yurupeng/` 个人区——届时需要导师授权 `/data/xyjiang/yurupeng/` 写权限并相应放宽 hook。

## 3. Cursor agent 三层防越界守卫体系

### 3.1 设计原则

| 层 | 文件 | 失效后果 |
|---|---|---|
| 1. 自然语言层 | `AGENTS.md` | agent 自律不被 enforce，但每次会话都被读 |
| 2. 项目 rule | `.cursor/rules/00-scope-and-output.mdc` (`alwaysApply: true`) | 同上，但更短、更近代码 |
| 3. Hook 守卫（**唯一硬约束**） | `.cursor/hooks/{guard-write-scope,guard-shell-scope}.sh` + `hooks.json` (`failClosed: true`) | 拦下 agent 工具调用，违规时直接 deny |
| 4. 自动归档 | `.cursor/hooks/archive-transcript.sh` (`failClosed: false`) | 失败仅少一份备份，不卡会话 |

**root 身份下 hook 仍然是软约束**——agent 自身的 Write/Shell 工具调用走 hook，但人工 SSH 登录跑 shell 完全绕过 hook。这是设计上的限制，不是 bug。

### 3.2 边界规则（与 hook 实际匹配的版本）

**白名单**（`guard-write-scope.sh:WHITELIST`）：
```
/home/xyjiang/Desktop/yurupeng
```

**黑名单**（`guard-write-scope.sh:BLACKLIST` + `guard-shell-scope.sh` 关键词）：
```
/home/xyjiang/anaconda3
/home/xyjiang/.bashrc
/home/xyjiang/.condarc
/home/xyjiang/.gitconfig
/home/xyjiang/.cursor
/home/xyjiang/.cursor-server
/home/xyjiang/projects
/data                         ← 整盘
```

**Shell 关键词拦截**（`guard-shell-scope.sh` 9 条规则）：
1. `sudo` / `su` 二次提权
2. `rm -rf` 命中只读/系统路径（`/`, `/home/xyjiang/{anaconda3,...}`, `/data`, `~`, `$HOME`）
3. 重定向 `>` `>>` 写入黑名单文件
4. `chmod -R` / `chown -R` 命中 `/`, `/data`, `/home/xyjiang`
5. `cp` / `mv` / `touch` / `mkdir` / `install` / `tee` / `dd of=` / `rsync` / `scp` 写入 `/data/`
6. `wget -O` / `curl -o` 下载到黑名单
7. `pip install` 时 `CONDA_DEFAULT_ENV ∈ {空, base, eloftr, eloftr_training, loftr, Xoftr, gim, LightGlue, RoMa, glue-factory, minima, dust3r}`
8. `conda env remove` / `conda install -n` 命中导师 env
9. `mkfs` / `shutdown` / `reboot` / `poweroff` / `halt` / `init 0` / `init 6`

### 3.3 archive-transcript 镜像约定

```
源:   /home/xyjiang/.cursor/projects/home-xyjiang-Desktop-yurupeng-eloftr/agent-transcripts/<uuid>/<uuid>.jsonl
目标: /home/xyjiang/Desktop/yurupeng/eloftr/.cursor/chat-logs/<uuid>.jsonl
```

stop / sessionEnd hook 触发，幂等（mtime 比较，不重复 cp）。`.cursor/chat-logs/` 已 gitignore，**永远不进版本库**——transcript 含完整对话明文，外发 GitHub 有泄密与体积风险。

> Agent 进程在 sandbox 内时（`Write access limited to the workspace directory`）`mkdir .cursor/chat-logs/` 可能失败（sandbox 给 `.cursor/` 加了额外只读保护防 agent 篡改自家配置）。`archive-transcript.sh` 在真实 stop/sessionEnd 触发时由 Cursor 主进程拉起，应能正常 mkdir + cp。占位目录已用 `.gitkeep` 落地避免该问题。

## 4. 数据集只读源路径表（已实证存在，权限 `dr-xr-xr-x`）

```
/data/xyjiang/Datasets/Infrared_image_datasets/
├── TarDAL/
│   ├── M3FD_Detection/         ← 子目录 Annotation/  Ir/  Vis/  test_generate_file/Vis/
│   ├── M3FD_Fusion/            ← Ir/  Vis/（OOD 备用，300 对）
│   ├── roadscene/              ← ir/  vi/  meta/   ★ skill `eloftr-roadscene-data` 期望布局
│   └── tno/                    ← ir/  vi/  meta/   备用 OOD
├── road-scene-infrared-visible-images/   ← 同名 git 仓库版本（含 crop_HR_visible 等增强变体；与 eloftr 仓库默认 `crop_LR_visible/cropinfrared` 命名不一致，**先不软链**）
├── LLVIP/  MSRS/  FLIR_ADAS_v2/  Freiburg_Thermal/  VVTUAV_Dataset/   ← 其它跨模态备用
```

参考实现：`/data/xyjiang/Matching/EfficientLoFTR/`（导师的 EfficientLoFTR 工程实体，**禁止写**，可读作 cfg 对照）。

### 4.1 必要的软链命令（首次配数据时跑一次）

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr/data
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection M3FD_Detection
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/roadscene      RoadScene
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/tno            TNO
```

> hook 不拦 `ln -s`，因为 `ln -s` 写的是 **软链本体**（在仓库内白名单），源是只读复用。

### 4.2 名称差异 → cfg 适配

[eloftr-roadscene-data](../eloftr-roadscene-data/SKILL.md) skill 期望 `data/RoadScene/{cropinfrared, crop_LR_visible}/`，但 TarDAL 的 `roadscene/` 用的是 `{ir, vi, meta}/`。两种处理：

1. **快速法**：在 `configs/data/roadscene_trainval.py` 里 override：
   ```python
   cfg.DATASET.ROAD_IR_SUBDIR  = "ir"   # 不是 cropinfrared
   cfg.DATASET.ROAD_VIS_SUBDIR = "vi"   # 不是 crop_LR_visible
   ```
   并把 `make_roadscene_splits.py` 默认 `--ir_subdir/--vis_subdir/--ext` 同步改。
2. **保守法**：仍想用 RoadScene 原始的 `cropinfrared/crop_LR_visible/` 命名，就软链到 `road-scene-infrared-visible-images/` 而不是 `TarDAL/roadscene/`，但要确认两份数据是否一致（前者是 git 仓库下载的；后者是 TarDAL 整理后的）。

M3FD 的 `Ir/Vis/` 与 [eloftr-m3fd-data](../eloftr-m3fd-data/SKILL.md) skill 期望布局完全一致，**无需 cfg 改动**。

### 4.3 train/val/test 划分文件

源软链目录是只读的，**不能在源目录写 `index/`**。需要在仓库内独立维护：

```
data/index/M3FD/{train,val,test}_pairs.txt        ← 仓库内可写
data/index/RoadScene/{train,val,test}_pairs.txt   ← 仓库内可写
```

cfg 里的 `LIST_PATH` 指到仓库内 `data/index/<dataset>/<split>_pairs.txt`，**不要**指到 `data/<dataset>/index/`。如果 split 脚本默认输出位置是 `<dataset>/index/`（例如 `make_m3fd_splits.py`），**必须加 `--index_dir data/index/M3FD` 参数 override**。

## 5. v7+ PC-CLAHE 缓存路径冲突（重要！）

[eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md) 与 [eloftr-m3fd-data §5.5](../eloftr-m3fd-data/SKILL.md) 默认假设：

```
data/M3FD_Detection/Ir_pc/    ← 写入源目录（**只读**，会失败）
data/M3FD_Detection/Vis_pc/
```

**yurupeng 工作区必须改成**：

```
data/pc_cache/M3FD/Ir_pc/
data/pc_cache/M3FD/Vis_pc/
data/pc_cache/RoadScene/Ir_pc/
data/pc_cache/RoadScene/Vis_pc/
```

需要的代码改动（动训练前必做）：

1. **`MyScripts/precompute_pc_edges.py`**：增加 `--out_dir` 参数，把输出从 `<root>/Ir_pc/` 改到 `data/pc_cache/<dataset>/Ir_pc/`。
2. **cfg 字段**（`configs/loftr/eloftr_full_v7_pcclahe.py` 与 v8/v9 等）：
   - 找到 `cfg.DATASET.ROAD_IR_PC_SUBDIR` / `ROAD_VIS_PC_SUBDIR`
   - 改成绝对路径或仓库相对路径，例如：
     ```python
     cfg.DATASET.ROAD_IR_PC_PATH  = "data/pc_cache/M3FD/Ir_pc"   # 不再用 SUBDIR 拼接
     cfg.DATASET.ROAD_VIS_PC_PATH = "data/pc_cache/M3FD/Vis_pc"
     ```
   - 若 cfg 字段约定是 `_SUBDIR`（拼接到 `data/M3FD_Detection/<SUBDIR>/`），需要先改 [src/datasets/roadscene.py](../../../src/datasets/roadscene.py) 让它支持外部路径。

> 这是 [AGENTS.md §8 工程 TODO](../../../AGENTS.md) 第 1 条遗留事项。动训练前 grep 当前 cfg：`grep -rn "PC_SUBDIR\|Ir_pc\|Vis_pc" configs/ src/ MyScripts/` 看影响面。

## 6. failClosed 自我锁外事故复盘（Cursor hooks 部署的重要教训）

### 6.1 事故时间线

1. agent 用 `Write` 写好 `.cursor/hooks/{guard-*,archive-*}.sh` 三个脚本（**未 chmod +x**）
2. agent 用 `Write` 写 `.cursor/hooks.json`（注册 4 类 hook，`failClosed: true`）
3. Cursor 热加载 hooks.json
4. agent 下一个 `StrReplace` 调用 → hook 触发 → 脚本无 +x 权限 → exit 126 → `failClosed=true` → **拒绝 agent 的写工具**
5. agent 下一个 `Shell` 调用 → 同样被拒
6. agent 此时**无法**自我修复（Write/StrReplace/Delete/Shell 全被拦），只剩 Read/Glob/Grep/AskQuestion/TodoWrite

### 6.2 解锁动作（人工 SSH）

```bash
chmod +x /home/xyjiang/Desktop/yurupeng/eloftr/.cursor/hooks/*.sh
ls -l   /home/xyjiang/Desktop/yurupeng/eloftr/.cursor/hooks/   # 确认 -rwxrwxr-x
```

### 6.3 教训：Cursor hook 部署正确顺序

| 步骤 | 顺序 | 备注 |
|---|---|---|
| 1 | 写脚本到 `.cursor/hooks/*.sh` | Write 工具 |
| 2 | **chmod +x** | Shell 工具 |
| 3 | 用 `bash hook.sh < /dev/null` 验证脚本能返回合法 JSON | Shell 工具 |
| 4 | **最后** 写 `hooks.json` 注册 hook | Write 工具 |
| 5 | 触发一次小动作（如 `ls`）确认 hook 已 loaded 且 allow case 通过 | Shell 工具 |
| 6 | 切换 `failClosed: false → true`（如果想阶段性收紧） | StrReplace |

> 替代策略：初次部署时 `hooks.json` 先写 `failClosed: false`，验证无误再改 true。但 yurupeng 这次直接上 true 是接受过这个风险的（边界守卫的目的就是宁可锁外面也不漏）。

## 7. grep 字面量匹配的激进副作用

`guard-shell-scope.sh` 用 `grep -qE` 在整条命令字符串里搜关键字（不分上下文）。例如想测试 hook 是否拦 `pip install` + base env，写：

```bash
echo '{"command":"pip install foo"}' | bash .cursor/hooks/guard-shell-scope.sh
```

整个**外层命令**字符串里包含 `pip install` 字面量 → 即便它在 `'...'` 引号字面量里 → 也被 grep 命中 → 测试者本身被拦。

**结论**：
- 这是**安全偏向保守**的特性，不是 bug。它意味着即便 agent 写 `echo "pip install foo"` 这种纯打印命令也会被拦——agent 的所有意图（不管是真执行还是假装演示）都被同一规则覆盖。
- **测试 hook 的 deny case 时**，不要在外层命令字符串里嵌入敏感关键字。改用：
  - Write 把 payload 写到仓库内 `tmp/test_inputs/case.json`（白名单允许）
  - `bash hook.sh < tmp/test_inputs/case.json`（外层命令本身不含 sudo/pip install/rm -rf 字面量）
- **实际部署中**：hook 已多次实证有效（事故 §6 + 含 `rm -rf /data` 字面量的测试命令 + 含 `pip install + base` 的测试命令均被拦），证据已足，不再追求完整组合 smoke test。

## 8. git 作者身份与 root 限制

当前 `whoami=root`，root 没有 `~/.gitconfig`（导师的 `~/.gitconfig` 是 xyjiang 用户的，`HOME=/home/xyjiang` 但 EUID=0 时 git 行为依实现）。`git config --global` **被 git safety protocol 禁止**，因为会污染导师 dotfile。

**解决方案**：每次 commit 用 `-c` 一次性指定 author：

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr
git -c user.name=abirdlikefish -c user.email=abirdlikefish@qq.com commit -m "..."
```

仓库历史 commit 全部 author 是 `abirdlikefish <abirdlikefish@qq.com>`（与 GitHub 远程 `abirdlikefish/eloftr` 一致），保持一致即可。`-c` 参数不写入任何 config 文件。

## 9. 工作流清单（每天打开 agent 后做什么）

1. agent shell 自检：`pwd && whoami && ls .cursor/hooks/`
   - 期望：cwd = `/home/xyjiang/Desktop/yurupeng/eloftr/`、`whoami=root`、3 个 hook 脚本都是 `-rwxrwxr-x`
2. `git status && git pull`
3. 尝试一次受控的越界写：让 agent `Write /tmp/foo.txt` —— 应被拦
4. 如计划训练：在 SSH 真实终端 `tmux new -s <task>` + `nvidia-smi` 看卡占用 + 启动训练
5. 长任务后 detach（Ctrl+B D），把 ckpt/log 留在 `logs/` `weights/`
6. 收工前 `git add . && git -c user.name=... -c user.email=... commit -m ... && git push`
   - **检查**：不要 push `.cursor/chat-logs/` `.cursor/plans/` `abfNote/chat-logs/`（已 gitignore，正常情况下不会）

## 10. 相关 skill 索引

- 通用 SOP（与本 skill 路径有冲突时以本 skill 为准）→ [eloftr-vlrlab-server](../eloftr-vlrlab-server/SKILL.md)
- 多卡 DDP 风险与 v9 复现 v8 → [eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md)
- 数据集集成 → [eloftr-m3fd-data](../eloftr-m3fd-data/SKILL.md) / [eloftr-roadscene-data](../eloftr-roadscene-data/SKILL.md)
- 评测流程 → [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)
- 各版本 ablation → `eloftr-v{1..8}-*` + [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
- v7+ PC-CLAHE 输入端优化 → [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)
- 仅本地 Windows 单卡 → [eloftr-windows-setup](../eloftr-windows-setup/SKILL.md)（**服务器忽略**）

## 11. 工程化资产 commit 历史

| commit | 内容 |
|---|---|
| `03d732f` | 添加 agent 工作边界守卫，限制仅 Desktop/yurupeng 内可写。AGENTS.md + .cursor/rules/00-scope-and-output.mdc + .cursor/hooks/{guard-write,guard-shell,archive-transcript}.sh + hooks.json + .gitignore 追加自动产物 |
