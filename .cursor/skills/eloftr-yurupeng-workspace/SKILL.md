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
├── data/                                      ← gitignore（实体可写目录 + 子目录软链方案，2026-05-06 实施）
│   ├── M3FD_Detection/                        ← 实体可写目录（仓库内，不是整目录软链）
│   │   ├── Ir         → /data/xyjiang/Datasets/.../TarDAL/M3FD_Detection/Ir         (子目录软链, 只读)
│   │   ├── Vis        → /data/xyjiang/Datasets/.../TarDAL/M3FD_Detection/Vis        (子目录软链, 只读)
│   │   ├── Annotation → /data/xyjiang/Datasets/.../TarDAL/M3FD_Detection/Annotation (子目录软链, 只读)
│   │   ├── Ir_pc/                              ← 仓库内实体, v7+ PC 缓存写入
│   │   ├── Vis_pc/                             ← 仓库内实体, v7+ PC 缓存写入
│   │   └── index/                              ← 仓库内实体, make_m3fd_splits.py 写入
│   ├── RoadScene/                             ← 实体可写目录（**用 road-scene-infrared-visible-images, 不是 TarDAL/roadscene**，§4 详释）
│   │   ├── cropinfrared    → /data/xyjiang/Datasets/.../road-scene-infrared-visible-images/cropinfrared    (子目录软链, 只读)
│   │   ├── crop_LR_visible → /data/xyjiang/Datasets/.../road-scene-infrared-visible-images/crop_LR_visible (子目录软链, 只读)
│   │   ├── cropinfrared_pc/                    ← 仓库内实体, v7+ PC 缓存
│   │   ├── crop_LR_visible_pc/                 ← 仓库内实体, v7+ PC 缓存
│   │   └── index/                              ← 仓库内实体, make_roadscene_splits.py 写入
│   └── megadepth/  scannet/                   ← 现有 git 占位（含 .gitignore 子文件）
├── logs/                                      ← gitignore，仓库内实体（NVMe 跑得快）
├── weights/                                   ← gitignore，仓库内实体
├── configs/  src/  MyScripts/  notebooks/
├── train.py  test.py  README.md
└── ...
```

**关键约定**：所有训练产物（`logs/`、`weights/`、`data/<dataset>/{Ir_pc,Vis_pc,index}/`）都落仓库内系统盘，不再走 `/data/xyjiang/yurupeng/` 软链（与 vlrlab-server skill §5 不同）。这是因为：
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
│   ├── M3FD_Detection/         ← 子目录 Annotation/  Ir/  Vis/  test_generate_file/  (4200 对 PNG, ★ M3FD 用这个)
│   ├── M3FD_Fusion/            ← Ir/  Vis/（OOD 备用，300 对）
│   ├── roadscene/              ← ir/  vi/  meta/   (★ 仅 41 对 PNG, 与 v0-v8 实测的 220 对不兼容, **不要用作 RoadScene OOD**)
│   └── tno/                    ← ir/  vi/  meta/   备用 OOD
├── road-scene-infrared-visible-images/  ← 子目录 cropinfrared/  crop_LR_visible/  crop_HR_visible/  infrared/  (220 对 JPG, ★ RoadScene 用这个, 命名匹配 cfg 默认)
├── LLVIP/  MSRS/  FLIR_ADAS_v2/  Freiburg_Thermal/  VVTUAV_Dataset/   ← 其它跨模态备用
```

参考实现：`/data/xyjiang/Matching/EfficientLoFTR/`（导师的 EfficientLoFTR 工程实体，**禁止写**，可读作 cfg 对照）。

### 4.0 ★ RoadScene 两份分发陷阱（2026-05-06 踩过坑）

服务器同时有 TarDAL 整理版与论文官方版，命名 / 文件数 / 后缀全部不同：

| 来源 | 路径 | 子目录命名 | 文件数 | 后缀 | 命名规则 | 与 v0-v8 兼容? |
|---|---|---|---|---|---|---|
| TarDAL 整理版 | `TarDAL/roadscene/` | `ir / vi / meta` | 41 对 | `.png` | `001.png` ~ `041.png` | ✗ 数据量太少, 与 v6.1/v7/v8 实测的 OOD test 数据集不同 |
| **RoadScene 论文官方版** | `road-scene-infrared-visible-images/` | `cropinfrared / crop_LR_visible / crop_HR_visible` | **220 对** | **`.jpg`** | `FLIR_*.jpg` | **✓ 与 cfg 默认值匹配, 与 v0-v8 实测 OOD 数据可比** |

**用 TarDAL/roadscene 的后果**：cfg 默认 `ROAD_IR_SUBDIR='cropinfrared'` 在 TarDAL 那边 dangling；即使改成 `ir`，41 对也不够做 OOD test（v6.1/v7/v8 SOTA 判定基于 22-pair val + 22-pair test，11/11 切分会让 σ 翻倍）。**TarDAL/roadscene 仅适合极快速 sanity，不能做正式 OOD**。

### 4.1 必要的软链命令（首次配数据时跑一次, 子目录软链方案）

`data/<dataset>/` 是**仓库内实体可写目录**（不是整目录软链），里面分两类：图像目录（`Ir/Vis` 或 `cropinfrared/crop_LR_visible`）软链到只读源；预处理 + 索引子目录（`Ir_pc/Vis_pc/index/`）是项目内实体。

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr/data

# M3FD: 实体目录 + 3 个子目录软链 + 3 个实体子目录
mkdir -p M3FD_Detection
cd M3FD_Detection
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection/Ir          Ir
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection/Vis         Vis
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection/Annotation  Annotation
mkdir -p Ir_pc Vis_pc index
cd ..

# RoadScene: 用 road-scene-infrared-visible-images (不是 TarDAL/roadscene!)
mkdir -p RoadScene
cd RoadScene
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/road-scene-infrared-visible-images/cropinfrared      cropinfrared
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/road-scene-infrared-visible-images/crop_LR_visible   crop_LR_visible
mkdir -p cropinfrared_pc crop_LR_visible_pc index
cd ..

# Megadepth_Syn (v10): 实体目录 + train 子目录 + 2 个 train 子目录软链 + 2 个 _pc 实体子目录
mkdir -p Megadepth_Syn/{train,index}
cd Megadepth_Syn
mkdir -p train/{infrared_pc,phoenix_pc}
ln -s /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/train/infrared  train/infrared
ln -s /data/xyjiang/image_style_transfer/4090_data/Megadepth_Syn/train/phoenix   train/phoenix
# test/Undistorted_SfM 不软链 (v10 不用, 留 v11+ B 路线 真 epipolar pose AUC eval)
cd ..

# 验证
ls M3FD_Detection/Ir | head -3                           # 期望: 00000.png 等
ls RoadScene/cropinfrared | head -3                       # 期望: FLIR_00006.jpg 等
ls RoadScene/cropinfrared | wc -l                         # 期望: 220
ls -L Megadepth_Syn/train/infrared/phoenix/S6/zl548/MegaDepth_v1 | head -3   # 期望: 0000  0001  0003 等
ls -L Megadepth_Syn/train/infrared/phoenix/S6/zl548/MegaDepth_v1 | wc -l     # 期望: ~195 scene
```

> hook 不拦 `ln -s`，因为 `ln -s` 写的是软链本体（在仓库内白名单），源是只读复用。

> **修错软链**：如果之前按整目录软链方案建过 `data/RoadScene -> TarDAL/roadscene` 类的错软链，用 `ln -sfn <new_target> <link_name>`（symbolic, force, no-dereference）一条命令换目标，不需要先 rm。

### 4.2 为什么走子目录软链而不是 cfg 路径解耦

历史考虑过两种数据组织方案：

| 方案 | data/<ds>/ | 代码改动 | 与 windows 兼容 |
|---|---|---|---|
| A. 子目录软链（**当前采用**） | 实体可写, 内部 Ir/Vis 软链, Ir_pc/index 实体 | **0 行** | ✓ windows 上 Ir 是实体目录, Linux 上是软链, dataset 类透明 |
| B. cfg 路径解耦（曾考虑） | 整目录软链到只读源 | 需加 `ROAD_PC_ROOT` cfg 字段, dataset 类 / data.py 各 ~10 行 | ⚠️ R3 三层默认值兜底, 风险大 |

最终选 A：cfg 完全不动（windows 原版的 `cropinfrared/crop_LR_visible/Ir/Vis/Ir_pc/index` 命名直接 work）+ `cv2.imread` 透明跟随软链 + 误写 IR/VIS 源时 OS 直接拒绝（写到只读源会 `EROFS`）+ v0-v8 字节级兼容自动满足。

M3FD 子目录命名 `Ir/Vis` 与 [eloftr-m3fd-data](../eloftr-m3fd-data/SKILL.md) 期望完全一致，无需任何 cfg 改动。RoadScene 子目录命名 `cropinfrared/crop_LR_visible` 与 [eloftr-roadscene-data](../eloftr-roadscene-data/SKILL.md) 期望一致。

### 4.3 train/val/test 划分文件（落仓库内 dataset root 子目录）

`make_m3fd_splits.py` / `make_roadscene_splits.py` 默认参数 `--root data/<ds> --out_subdir index` → 写入 `data/<ds>/index/`，**正好就是子目录软链方案下的实体可写目录**，无需任何 override：

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr
python MyScripts/make_m3fd_splits.py        # 写入 data/M3FD_Detection/index/
python MyScripts/make_roadscene_splits.py   # 写入 data/RoadScene/index/

wc -l data/M3FD_Detection/index/*.txt        # 期望: 3780 / 210 / 210
wc -l data/RoadScene/index/*.txt              # 期望: 176 / 22 / 22 (220 张 80/10/10)
```

> 历史曾考虑 `data/index/<ds>/...` 独立 root，配合"整目录软链"方案。子目录软链方案下不再需要这一层抽象，删除 `--out_root` 等 cli 参数也可以（保留为可选）。

## 5. v7+ PC-CLAHE 缓存路径（子目录软链方案下已无冲突）

`data/<ds>/` 是仓库内实体可写目录（不是整目录软链），所以 `data/<ds>/{Ir_pc,Vis_pc}/` 自然落在项目内可写位置，**与 windows 本地完全一致，cfg / dataset / precompute 脚本零改动**。

```
data/M3FD_Detection/Ir_pc/                                          ← 仓库内实体, precompute_pc_edges.py 默认输出
data/M3FD_Detection/Vis_pc/
data/RoadScene/cropinfrared_pc/                                     ← 同样
data/RoadScene/crop_LR_visible_pc/
data/Megadepth_Syn/train/infrared_pc/S6/zl548/MegaDepth_v1/...      ← v10 新增, M3FD-style source/_pc 同级
data/Megadepth_Syn/train/phoenix_pc/S6/zl548/MegaDepth_v1/...
```

`MyScripts/precompute_pc_edges.py` 默认 `ir_out = root / job["ir_pc_subdir"]` 直接落对位置。Linux 入口脚本：

```bash
bash MyScripts/precompute_pc_edges.sh        # 默认 M3FD + RoadScene 一起算, ~35-50 min
# 或单独
python MyScripts/precompute_pc_edges.py --dataset M3FD
python MyScripts/precompute_pc_edges.py --dataset RoadScene

# v10 Megadepth_Syn 走 L1+L2+L3 三层加速 (~46 min, vs baseline ~14h)
python MyScripts/precompute_pc_edges.py --dataset Megadepth_Syn \
       --recursive --max_long_edge 640 --pc_nscale 3 --workers 24
```

> AGENTS.md §8 第 1 条遗留 TODO（"PC 缓存输出位置 → `data/pc_cache/<dataset>/`"）在子目录软链方案下作废，可在毕业前 cleanup AGENTS.md 时一并删除。

### 5.1 phasepack 与 pyfftw（速度优化）

`eloftr_yurupeng` env (clone 自 `eloftr_training`) 已装 `phasepack 1.5`（用户手动 `pip install` 过）。但 `pyfftw` 没装，phasepack import 时会 warning：

```
UserWarning: Module 'pyfftw' (FFTW Python bindings) could not be imported.
Falling back on the slower 'fftpack' module for 2D Fourier transforms.
```

后果：M3FD 8400 张 PC 预计算从 ~35 min 拖到 ~50 min。**v10 Megadepth_Syn ~258K cache 强烈建议装 pyfftw**（占 PC 总耗时 30%）：

```bash
# 在 eloftr_yurupeng env 里 (不要装到导师 env)
source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
pip install pyfftw                                # 一次性, libfftw3-dev 系统已装
python -c "import pyfftw; print(pyfftw.__version__)"   # 验证
```

### 5.2 v10 PC cache fix script 工作流（Megadepth_Syn 专用救场）

[MyScripts/fix_pc_cache_alignment.py](../../../MyScripts/fix_pc_cache_alignment.py) 是 **v10 一次性补救脚本**：当 precompute 用 L3 优化（`--max_long_edge 640 --df 32`）时，部分 raw aspect 触发 df 截断破坏 cache 跟 raw 在 dataset 端 `_resize_keep_aspect` 的 target shape 不一致 → np.stack 崩。fix script 把所有 cache 强 resize 到 raw 的 dataset target shape，让 dataset 加载时 cv2.resize 是 no-op。

工作流（v10 ship 路径）:
```bash
# 1. precompute 跑完 (~46 min)
python MyScripts/precompute_pc_edges.py --dataset Megadepth_Syn \
       --recursive --max_long_edge 640 --pc_nscale 3 --workers 24

# 2. dry-run 看有多少 cache 需要修 (~15 min, IR + VIS 两路)
python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --dry-run --workers 24
# 期望输出: fixed: ~257790 (= 128895 IR + 128895 VIS, 100% 都需要修)

# 3. 实跑 (~30 min, inplace 写回, idempotent)
python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --workers 24
```

跑完后 dataset 端 [src/datasets/roadscene.py L353-404](../../../src/datasets/roadscene.py) 的 target-shape check 会通过, mode A/B debug 不再报 `RuntimeError: PC cache shape ... resizes to ...`。

**注意**：fix script 不是 v0-v9 的需求。M3FD/RoadScene 用 default precompute（`--max_long_edge 0`，不缩放）→ cache 跟 raw 同 shape → 不需要 fix。仅当用 v10 风格的 L3 加速（`--max_long_edge > 0`）时才需要 fix script。

详见 [eloftr-v10-msyn §4.3](../eloftr-v10-msyn/SKILL.md) + [eloftr-megadepth-syn-data §7](../eloftr-megadepth-syn-data/SKILL.md) + [eloftr-v7-pcclahe §13.3](../eloftr-v7-pcclahe/SKILL.md)。

### 5.3 conda env 实际位置

`eloftr_yurupeng` 不在标准 `~/anaconda3/envs/`，而在 `/data/xyjiang/envs/eloftr_yurupeng`（系统盘紧时常见做法）。conda 通过 `envs_dirs` 配置识别，所以：

- `conda activate eloftr_yurupeng`（按名字）✓ work
- `source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng` ✓ work（精确路径解析到 `/data/xyjiang/envs/`）
- Python 版本 3.8.20（不是 3.10），但与 v0-v8 训练栈兼容

`MyScripts/precompute_pc_edges.sh` 第 31 行用的就是后一种 `source ... activate` 写法，已实测可工作。

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
