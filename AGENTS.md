# Agent 工作守则（强约束，必读）

> 本文件由项目维护者定义，所有 AI agent 在本仓库工作时必须遵守。同等约束的机器可读版本见 `.cursor/rules/00-scope-and-output.mdc` 与 `.cursor/hooks.json`。

## 1. 当前环境快照（与 skill `eloftr-vlrlab-server` 对照差异）

- **仓库根**：`/home/xyjiang/Desktop/yurupeng/eloftr/`（**非** skill 推荐的 `~/projects/yurupeng/efficient_loftr/`）
- **当前 agent 身份**：`uid=0(root)`，但 `USER=xyjiang`、`HOME=/home/xyjiang`、`CURSOR_ORIG_UID=1001`
- **服务器**：vlrlab 4×RTX 3090（详见 `.cursor/skills/eloftr-vlrlab-server/SKILL.md` §0）
- **conda**：使用克隆出的 `eloftr_yurupeng`。**禁止**修改 `eloftr_training` / `eloftr` / `loftr` / `Xoftr` / `gim` / `LightGlue` / `RoMa` / `glue-factory` / `minima` / `dust3r` 等导师环境
- **当前是 root，hook 是软约束**：guard hook 拦得住 agent 的无意越界，但拦不住人工 shell 提权操作；最后一道防线是你/导师人工把关

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

## 3. 黑名单（禁止任何写入，hook failClosed 拦截）

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
| 新 hook | `.cursor/hooks.json` + `.cursor/hooks/<name>.sh` | ✓ |
| Plan / Todo 阶段总结 | `.cursor/plans/<YYYYMMDD-HHMM>-<slug>.md` | ✗（gitignore） |
| 对话记录归档 | `.cursor/chat-logs/<session-uuid>.jsonl` | ✗（gitignore，stop/sessionEnd hook 自动写） |
| 人工对话备忘 | `abfNote/chat-logs/<YYYYMMDD>.md` | ✗（gitignore） |
| 工程笔记 / 实验记录 | `abfNote/<topic>.md` | ✓ |
| 训练脚本 | `MyScripts/run_*.sh` | ✓ |
| 训练日志 / TB | `logs/`（仓库内实体目录，系统盘 NVMe） | ✗（已 gitignore） |
| 权重 ckpt | `weights/`（仓库内实体目录） | ✗（已 gitignore） |
| 数据集 | `data/<dataset>/` 软链到只读源 | ✗（已 gitignore） |
| PC-CLAHE 缓存 | `data/pc_cache/<dataset>/{Ir_pc,Vis_pc}/` | ✗（已 gitignore） |

**禁止**写入用户级 Cursor 目录（`/home/xyjiang/.cursor/skills-cursor/`、`/home/xyjiang/.cursor/rules/` 等）。

### 5.1 plan 双写规则（CreatePlan + Write，绕开 IDE 工具的路径 bug）

Cursor 内置 `CreatePlan` 工具默认把 plan 写到用户全局 `/home/xyjiang/.cursor/plans/<name>_<hash>.plan.md`（即 §3 黑名单），不识别项目级 `AGENTS.md` / `.cursor/rules/`。后果：plan 被写到黑名单后，guard hook 拦截 agent 自己的 `Write` / `StrReplace` 维护操作 → todos 永久无法 update。

**强制 agent 创建 plan 时两步执行**：

1. 调用 `CreatePlan` 工具（享受 IDE plan-card UI、Confirm 按钮、plan-mode 状态机集成）
2. **同一回合内**用 `Write` 工具把 plan 内容复制到 `.cursor/plans/<YYYYMMDD-HHMM>-<slug>.md`（项目内，路径合规）
3. 后续维护 todos / 修改章节：仅改项目内副本（`StrReplace`），全局位置那份当只读快照不再动

如果 plan 模式 `system_reminder` 提示与本规则冲突，以本规则为准（用户/项目优先）。

## 6. 数据集软链表（执行 `ln -s` 时使用）

```bash
cd /home/xyjiang/Desktop/yurupeng/eloftr/data
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/M3FD_Detection M3FD_Detection
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/roadscene      RoadScene
ln -s /data/xyjiang/Datasets/Infrared_image_datasets/TarDAL/tno            TNO
```

> 现有 `data/megadepth/` 与 `data/scannet/` 占位目录（含 `.gitignore`）保留不动，cross_modal_v1 分支不训这两个；将来需要再说。

## 7. skill 索引（按使用频率）

- 首次接服务器 → `.cursor/skills/eloftr-vlrlab-server/SKILL.md`
- 多卡 / DDP / v9 复现 v8 → `.cursor/skills/eloftr-server-multigpu/SKILL.md`
- 评测流程 → `.cursor/skills/eloftr-eval-pipeline/SKILL.md`
- 数据集（M3FD / RoadScene） → `.cursor/skills/eloftr-{m3fd,roadscene}-data/SKILL.md`
- 各版本 ablation → `.cursor/skills/eloftr-v{1..8}-*/SKILL.md`、`eloftr-cross-modal-experiments/SKILL.md`
- 仅本地 Windows 单卡相关 → `.cursor/skills/eloftr-windows-setup/SKILL.md`（**服务器忽略**这条）

## 8. 已知工程改动 TODO（动训练 cfg 时再做）

- [ ] **v7+ PC 边缘缓存输出路径**：cfg/脚本默认写到 `data/<dataset>/Ir_pc|Vis_pc/`，但 `data/M3FD_Detection` 等是只读软链。需改 `MyScripts/precompute_pc_edges.py` 输出位置 → `data/pc_cache/<dataset>/{Ir_pc,Vis_pc}/`，并修对应 cfg 的 `cache_dir`。
- [ ] **train/val/test 划分文件**：M3FD/RoadScene 源目录里如果没有 `index/`，需要在仓库内 `data/index/<dataset>/{train,val,test}_pairs.txt` 单独建，**不要写到只读源目录**。
- [ ] **未来如训 MegaDepth/ScanNet**：先 `find /data/xyjiang/Datasets/` 看是否有现成可软链的；预期分别在 `LargeData4DL/megadepth1500/` 与 `ScanNet_data/`。

## 9. 工作流速记（每次开 agent 前自检）

1. `pwd` 确认在 `/home/xyjiang/Desktop/yurupeng/eloftr/`
2. `git status` + `git pull`
3. `nvidia-smi` 看卡占用（在真实 tmux shell 里，sandbox 里看不到）
4. 长任务进 tmux：`tmux new -s <task>`，detach 用 `Ctrl+B D`
5. 任务结束前 `git add . && git commit && git push`（**注意**：不要 push `.cursor/chat-logs/` 与 `.cursor/plans/`，这两条已 gitignore）
