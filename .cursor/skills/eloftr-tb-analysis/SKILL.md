---
name: eloftr-tb-analysis
description: Read TensorBoard event files of EfficientLoFTR training runs and emit AI-friendly JSON/CSV without opening TensorBoard UI. Single entry-point script MyScripts/read_tb_metrics.py with 4 sub-commands (tags / summary / export / best). Auto-detects dataset type (aligned_irvis vs megadepth), auto-grades v9 acceptance (strong/medium/weak/fail), reports MSBN drift verdict + modemb verdict, handles huge tfevents (v9 = 748 MB) safely by skipping figure decode. Path resolver accepts events file / version_x dir / experiment dir; missing --logdir triggers interactive input(). Use when 分析训练结果 / 看 tensorboard / 读 tfevents / 评估 v9 训练好坏 / 自动评级 / 找最佳 epoch / best epoch / 拿 precision@1px / 拿 AUC / 训练曲线 / val 指标 / MSBN 是否生效 / modemb 是否学到 / 训练耗时 / 比较多个 version / 给 AI 看训练日志 / 把 tb 数据导出 CSV / 不想开 tensorboard UI / events.out.tfevents 太大打不开 / OOM 加载 tfevents / Windows 单卡训完之后看结果 / 服务器训完拉回来分析. Triggers: read_tb_metrics / tb_logs / tfevents / EventAccumulator / tensorboard scalar / v9_acceptance / MSBN drift verdict / aligned_irvis dataset_type / precision@1px / metrics_0 / val_match figure 占体积 / size_guidance images=1 / interactive input logdir / summary.json / best epoch / per_epoch list, English 'tensorboard event file', 'parse tfevents', 'extract scalars', 'auto-grade training', 'best validation epoch', 'training stats JSON', 'avoid OOM tensorboard', 'figure decode skip'. Companion skills: eloftr-eval-pipeline (independent test-time eval, not the same as TB val), eloftr-v8-msbn / eloftr-v9 (which scalars are written by which version).
---

# TensorBoard Analysis（一行命令把训练日志变成 AI 可读 JSON）

> 训练侧把所有指标写进 `logs/tb_logs/<exp>/version_<N>/events.out.tfevents.*`。这套 skill 把"打开 TensorBoard UI 人眼看曲线"换成"一条 CLI 出 JSON / CSV，AI 直接消化"。
>
> 入口脚本：[MyScripts/read_tb_metrics.py](../../../MyScripts/read_tb_metrics.py)。
>
> **何时用**：
> - 训练跑完，想知道"成不成功 / 最佳 epoch 在哪 / 关键指标是多少"
> - AI 需要程序化读训练结果做后续决策（比如自动选 best ckpt 跑 eval）
> - tfevents 文件太大（v9 = 748 MB），用 TensorBoard UI 打开会卡甚至 OOM

## 0. 一行最常用命令（AI 默认入口）

```bash
python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/<exp>
```

输出一份完整 JSON 到 stdout，schema 稳定。三种 `--logdir` 写法都接受：

```text
logs/tb_logs/m3fd_v9_e2e_outdoor                                # 实验目录 (auto picks newest version_*)
logs/tb_logs/m3fd_v9_e2e_outdoor/version_0                      # version 目录
logs/tb_logs/m3fd_v9_e2e_outdoor/version_0/events.out.tfevents.1778078682.3090.382918.0  # 文件本身
```

**不传 `--logdir` 时会 `input()` 提示用户粘贴**，方便手动指定。

## 1. 四个子命令速查

| 子命令 | 用途 | 输出形态 |
|---|---|---|
| `tags` | 列出所有 scalar tag + 样本数 + 首末值 + 步号范围；image / histogram tag 只列名不解码 | 终端表格 |
| `summary` | **AI 主入口**：dataset 类型 + train/val 关键统计 + per-epoch 全量 val 指标 + best epoch + （v9 时）自动评级 | JSON（stdout 或 `--out file.json`） |
| `export <tag> --out p1.csv` | 单个 scalar 全量 (step, wall_time, value) 导出 | CSV |
| `best <tag> [--mode max\|min]` | 找指定 scalar 的最佳值 + 对应 step + wall_time | JSON |

### 1.1 完整示例

```bash
# 最常用：吐 JSON 到 stdout
python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/m3fd_v9_e2e_outdoor

# 落盘 + 同时打印一行 verdict 到 stderr
python MyScripts/read_tb_metrics.py summary \
    --logdir logs/tb_logs/m3fd_v9_e2e_outdoor \
    --out logs/tb_logs/m3fd_v9_e2e_outdoor/version_0/summary.json
# stderr: [v9] grade=strong  p@1px=0.6881  best_epoch=52

# 只看一眼 tag 列表
python MyScripts/read_tb_metrics.py tags --logdir logs/tb_logs/m3fd_v9_e2e_outdoor

# 把 precision@1px 全 80 epoch 数据导成 CSV 喂 matplotlib
python MyScripts/read_tb_metrics.py export "metrics_0/precision@1px" \
    --logdir logs/tb_logs/m3fd_v9_e2e_outdoor \
    --out dump/v9_p1px.csv

# 验证最佳 epoch（与 summary 里的 best 应一致）
python MyScripts/read_tb_metrics.py best "metrics_0/precision@1px" \
    --logdir logs/tb_logs/m3fd_v9_e2e_outdoor
```

### 1.2 平台运行入口

| 平台 | 推荐命令 |
|---|---|
| 本地 Windows（PowerShell） | `& "C:\Users\abirdlikefish\miniconda3\envs\eff_loftr\python.exe" MyScripts\read_tb_metrics.py summary --logdir <path>` |
| 本地 Windows（cmd） | `conda activate eff_loftr && python MyScripts\read_tb_metrics.py summary --logdir <path>` |
| 服务器 vlrlab | `conda activate eloftr_yurupeng && python MyScripts/read_tb_metrics.py summary --logdir <path>` |

> PowerShell 不会自动初始化 conda hook，所以本地优先用绝对路径调 python.exe。脚本本身只依赖 `tensorboard` 包（PL 1.3.5 自带），两套 env 都已具备。

## 2. summary JSON schema（v1）

下面是 v9 真实跑出来的 JSON（精简了 per_epoch 中间项），字段含义见注释：

```jsonc
{
  "schema_version": 1,                                  // bump 时回头改这一段
  "logdir": "...\\logs\\tb_logs\\m3fd_v9_e2e_outdoor\\version_0",
  "events_file": "events.out.tfevents.1778078682.3090.382918.0",
  "events_size_mb": 748.33,
  "exp_name": "m3fd_v9_e2e_outdoor",
  "version": 0,
  "dataset_type": "aligned_irvis",                      // aligned_irvis | megadepth | unknown
  "wall_clock": {
    "start_iso": "2026-05-06T22:44:42",
    "end_iso":   "2026-05-07T09:41:59",
    "elapsed_hours": 10.955                             // 实际训练耗时
  },
  "training": {
    "total_global_steps": 75550,
    "final_train_loss": 0.208,
    "min_train_loss": {"value": 0.109, "step": 67100},
    "total_epochs": 80,
    "final_avg_loss_on_epoch": 0.168,
    "modality_emb": {                                   // 仅当 USE_MODALITY_EMB=True
      "ir_norm_first": 0.0, "ir_norm_last": 0.1077,
      "vis_norm_first": 0.0, "vis_norm_last": 0.0951,
      "verdict": "modemb learning (norm grew >0.05)"   // 或 "modemb dead (...)"
    },
    "msbn": {                                           // 仅当 USE_MSBN=True
      "layer1_drift_first": 0.0031, "layer1_drift_last": 0.0622, "layer1_drift_max": 0.2388,
      "layer2_drift_first": 0.0145, "layer2_drift_last": 0.1133, "layer2_drift_max": 0.1641,
      "verdict": "MSBN active (drift_last > 0.05)"     // 或 "MSBN dead (...)"
    }
  },
  "validation": {
    "monitor_key": "metrics_0/precision@1px",           // aligned_irvis 走 p@1, megadepth 走 auc@5
    "monitor_mode": "max",
    "num_validations": 80,
    "best": {
      "epoch": 52,
      "value": 0.688116,
      "all_metrics_at_best": {
        "precision@1px": 0.688116,
        "precision@3px": 0.975305,
        "precision@5px": 0.992865,
        "mean_pixel_error": 0.734341,
        "num_matches": 2194.43,
        "mean_conf": ...,
        "avg_loss": 0.846216
      }
    },
    "per_epoch": [                                      // 每个 val epoch 一行；可直接画曲线
      {"epoch": 0, "precision@1px": 0.347, "precision@3px": 0.754, ...},
      ...
      {"epoch": 79, "precision@1px": 0.662, "precision@3px": 0.975, ...}
    ]
  },
  "v9_acceptance": {                                    // 仅当 exp_name 含 "v9"
    "grade": "strong",                                  // strong | medium | weak | fail
    "p@1px": 0.6881,
    "best_epoch": 52,
    "rule_source": "MyScripts/run_m3fd_v9_e2e.sh L26-L30"
  },
  "all_scalar_tags": ["epoch", "lr-AdamW", ..., "val_0/avg_n_pos"],
  "image_tags_count": 35
}
```

### 2.1 dataset_type 嗅探规则

| 检测条件 | dataset_type | 数据集 |
|---|---|---|
| 任一 tag 以 `metrics_0/precision@` 开头 | `aligned_irvis` | RoadScene / M3FD / TNO（任何 aligned IR-VIS） |
| 任一 tag 以 `metrics_0/auc@` 开头 | `megadepth` | MegaDepth / ScanNet |
| 都没有 | `unknown` | 训练只跑了 sanity check，没产出 val 指标 |

不读 cfg 文件，纯靠 tag 形态判别，未来加 v10/v11 也无需改代码。

### 2.2 monitor key 自动选取

| dataset_type | monitor_key | mode |
|---|---|---|
| `aligned_irvis` | `metrics_0/precision@1px` | max |
| `megadepth` | `metrics_0/auc@5` | max |
| `unknown` | （跳过 validation 段） | — |

### 2.3 v9 自动评级规则（来源：[MyScripts/run_m3fd_v9_e2e.sh](../../../MyScripts/run_m3fd_v9_e2e.sh) L26-L30）

| grade | best_epoch ∈ | best p@1px ∈ |
|---|---|---|
| strong | [35, 55] | ≥ 0.51 |
| medium | [35, 65] | [0.45, 0.51) |
| weak | [35, 78] | [0.40, 0.45) |
| fail | otherwise | otherwise |

仅当 `exp_name` 字符串里含 `"v9"`（不分大小写）时启用。其它 v 想加自动评级 → 在 `cmd_summary` 函数末尾加判断块即可。

## 3. AI 助手怎么用（推荐套路）

**默认套路 A：要回答"v_X 训得怎么样"**

1. 拼出 `--logdir logs/tb_logs/<exp>` 调 `summary`，得到 JSON
2. 读 `out["v9_acceptance"]["grade"]`（v9 专属）或 `out["validation"]["best"]`
3. 读 `out["training"]["msbn"]["verdict"]` 判 MSBN 起没起作用
4. 读 `out["training"]["modality_emb"]["verdict"]` 判 modemb 学没学到
5. 读 `out["wall_clock"]["elapsed_hours"]` 报训练耗时
6. **不要自己再 grep 终端日志**，summary JSON 已经把所有训练侧 KPI 抽完

**套路 B：要画收敛曲线 / 比较 N 次实验**

```bash
for exp in m3fd_v8_msbn m3fd_v9_e2e_outdoor; do
    python MyScripts/read_tb_metrics.py export "metrics_0/precision@1px" \
        --logdir logs/tb_logs/$exp \
        --out dump/${exp}_p1px.csv
done
```

然后用 matplotlib / pandas 在 ipython 里 overlay 多条曲线。

**套路 C：要给训完的 ckpt 跑独立 eval（衔接 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)）**

1. 用 `summary` 拿到 `validation.best.epoch`
2. 配合 `eval_*finetuned.bat <X> <Y> <Z>` 的 Z 参数（见 eval-pipeline §4.1，Z=1 是 precision@3px 倒序第 1 个）
3. 注意训练 val 的 best epoch ≠ 独立 eval 的 best ckpt（eval 选 ckpt 是按文件名 `precision@3px=` 字段排序，跟训练 val 走同一份 RoadSceneDataset）

## 4. 实现要点（修脚本时回头看）

### 4.1 大 tfevents 不 OOM 的关键

```python
EventAccumulator(
    str(events_path),
    size_guidance={
        'scalars': 0,        # 0 = unlimited，全部加载
        'images': 1,         # 只索引、不解码 PNG
        'tensors': 1,
        'histograms': 1,
        'audio': 1,
        'graph': 0,          # 0 = 关闭
        'compressedHistograms': 1,
    },
)
```

v9 的 748 MB 里 99% 是 35 个 `val_match_0/evaluation/pair-*` 图（每 epoch 多张），AI 用不到。`images=1` 后实测加载 + summary < 5 秒。

### 4.2 路径归一化

`resolve_events()` 三层 fallback：
1. 直接是 `events.out.tfevents.*` → 直接返回
2. 是 `version_x/`（含 events 文件）→ glob `events.out.tfevents.*` 取最大的（体积最大 = 跑得最久）
3. 是实验目录 → 找最大编号的 `version_*` 子目录，再递归 1 / 2

实验目录里有多个 `version_*`（断点续训会产生）时**只取最新那个**。如果想分析早期 version，必须手动指定 `--logdir logs/tb_logs/<exp>/version_<N>`。

### 4.3 PL 自带的"双写" tag

PL 训练时通过 `self.log(k, ...)`（见 [src/lightning/lightning_loftr.py:659](../../../src/lightning/lightning_loftr.py)）会自动把 `precision@1px / precision@3px / precision@5px / mean_pixel_error / num_matches` 写一份到顶级（无 `metrics_0/` 前缀），用 `global_step` 作横轴。

`summary` 故意忽略这些"双写副本"，只读 `metrics_0/*` 系列（横轴是 epoch），因为：
- per-epoch 列表必须能按 epoch 对齐多个指标
- `global_step` 横轴的 4 行（epoch 0/26/52/79 但 step 0/944/...）没法跟其它 tag 配对

### 4.4 wall_clock 计算

只采样 `train/avg_loss_on_epoch` + `train/loss` 的 `wall_time`，计算 min/max。如果两者都不存在就退化到任意 3 个 scalar。这避开了图 tag 的 wall_time（可能是 figure 写盘时间，比 step 写时间晚很多，会高估训练时长）。

## 5. 真实跑通过的样例（v9_e2e = 第一次 e2e cold start，2026-05-07 跑完）

```text
exp_name:        m3fd_v9_e2e_outdoor
events:          748 MB（35 张 val_match figure 占大头）
elapsed:         10.96 h（cfg 注释预期 13.3 h，实际单卡 3090 跑得更快）
total epochs:    80（无 EarlyStopping 触发, ES monitor 选 p@3 不是 p@1）
best epoch:      52  ← 落在 strong 区间 [35, 55] 末端
best p@1px:      0.6881  ← 远超 strong 阈值 0.51
best p@3px:      0.9753
best p@5px:      0.9929
best mpe:        0.73 px（已突破 RoadScene 类 1-2 px 物理对齐残差）
modemb verdict:  modemb learning (ir_norm 0→0.108, vis_norm 0→0.095)
msbn verdict:    MSBN active (layer1 drift 0→0.062, layer2 drift 0→0.113)
v9 grade:        strong  ← e2e cold start 直接成功，跳过 v0-v7 链路可行
PL ckpt picked:  ep 75 (top-1 by precision@3px monitor, ≠ ep 52 best p@1)
```

### 5.1 训练 val 结论被独立 eval 验证（2026-05-07 后续）

| 指标 | 训练 val ep75 (本脚本) | 独立 eval ep75 (eval_roadscene.py) | 差值 |
|---|---|---|---|
| M3FD p@1px | 0.6699 | **0.6863** | +0.016（独立 eval 更高）|
| M3FD p@3px | 0.9757 | 0.9792 | +0.003 |
| M3FD p@5px | 0.9933 | 0.9960 | +0.003 |
| M3FD mpe | 0.769 | 0.7283 | -0.041 |
| OOD p@1px (RoadScene) | — (训练时只测 in-domain) | **0.2128** | — |
| OOD p@3px | — | 0.7209 | — |

**结论**: 训练 val 与独立 eval 的差距完全在抽样噪声内, 且独立 eval 数字略高 — **没有 train-test gap, 不是过拟合**. v9 同时达成 in-domain SOTA + OOD 不退化（对比 v8 OOD p@1=0.2025 退化 -4.7%）, 是 v0-v9 全程综合通用性 SOTA. 详见 [eloftr-v9-e2e §4](../eloftr-v9-e2e/SKILL.md).

> **结论速记**：v9 e2e cold start 不仅验证通过（strong）+ in-domain 大幅 SOTA, 还**意外解决了 v8 SKILL §12 列为 future work 的 MSBN OOD trade-off** — 单纯靠训练策略层面的改动（80 ep + cold start）就能解, 不需要新算法. 这是 v9 的关键论文 finding.

## 6. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'tensorboard'` | 用了系统 python 而非 conda env python | 见 §1.2，用 env 里 python.exe 的绝对路径 |
| `[err] no events.out.tfevents.* found under <path>` | `--logdir` 写错，或者训练根本没启动到写第一笔日志 | 看 `ls logs/tb_logs/<exp>/version_*/` 里有没有 events 文件 |
| `MemoryError` / 加载几分钟没动 | size_guidance 被改回 0，PNG 在被全量解码 | 检查 `load_ea()` 的 `size_guidance['images']` 仍是 1 |
| `dataset_type: unknown` | 训练只跑了 sanity check，从未走完一个 val epoch | 检查训练日志，多半是 epoch 0 就 NaN crash 了 |
| v9 grade 出 `fail` 但人眼看曲线是好的 | best_epoch 落在 [0, 35) 提前收敛了（学习率没 warmup 完就过早达到峰值） | 调小 `WARMUP_STEP` 或检查 LR scheduler |
| best epoch 比 EarlyStopping patience 之后还远 | EarlyStopping 没触发，训练用满 max_epochs；可能学习率 schedule 太平 | 看 `lr-AdamW` 曲线确认 MSLR 是否生效 |
| 多个 version 想同时分析 | 当前实现只取最新 version | 手动循环：`for v in version_*; do --logdir <exp>/$v; done` |

## 7. 与其它 skill 的关系

- **[eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)**：跑训完后的独立 eval（用 `eval_roadscene.py`），输出 `dump/<exp>_eval_*/overall.txt`。本 skill 只看训练时 val（PL 的 `validation_step` → tfevents），**两条管线在 dataset / mask / 优化路径上完全镜像**（见 eval-pipeline §1）但产物路径不同。AI 应先用本 skill 拿训练 KPI，再决定是否需要独立 eval 重测。
- **[eloftr-results](../eloftr-results/SKILL.md)**：独立 eval 跑出来的跨版本数字汇总在 `results/eval_summary.md`。**本 skill 的训练 val 数字 ≠ results 表的独立 eval 数字**（口径接近但 dataset random ordering / RepVGG mode 等微差异可能差 0.01-0.05），不要混填。要做"v8 vs v9 对比"答辩 / 论文，引用 results 表；要解释"v9 训练曲线为什么 ep52 见顶"，引用本 skill summary JSON。
- **[eloftr-tb-summary](../eloftr-tb-summary/SKILL.md)**：本 skill 的 `aggregate` 子命令产出的跨实验 Markdown 表存在 `results/tb_summary.md`。**两 skill 的关系类似 eval-pipeline（产生 overall.txt） vs results（汇总）**：本 skill 是"如何对单个实验抽训练 KPI"，eloftr-tb-summary 是"v1..v9 横向汇总表的位置 + 维护流程"。新训练跑完 → 跑 `aggregate` → 把 raw 表贴进 `tb_summary.md` 替换基础三表行。详见该 skill §2 工作流。
- **[eloftr-v8-msbn](../eloftr-v8-msbn/SKILL.md)** / **[eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)**：解释每个 v_X 写了哪些 train/* tag。本 skill 的 summary 自动适配 `train/bn_drift_ratio_*`（v8+ MSBN）、`train/mod_emb_*_norm`（v2+ modemb），不需要 AI 自己识别。
- **[eloftr-vlrlab-server](../eloftr-vlrlab-server/SKILL.md)** / **[eloftr-server-multigpu](../eloftr-server-multigpu/SKILL.md)**：服务器跑训练后，可以直接在服务器 `git pull` 后用 ask-mode agent + 本 skill 抽 summary，再 `git push abfNote/<exp>.md` 备份结果（不要 push tfevents 自身，logs/ 已 gitignore）。
