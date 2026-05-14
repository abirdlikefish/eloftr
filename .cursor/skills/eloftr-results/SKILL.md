---
name: eloftr-results
description: Single source-of-truth index for EfficientLoFTR cross-modal experiment-chain v5..v9 independent eval results (M3FD in-domain + RoadScene OOD), one Markdown table at results/eval_summary.md aggregating all dump/<dataset>_eval_v<X>_*/overall.txt files. Use when asked for "the results table" / a side-by-side comparison across versions / current SOTA / which version to ship / 综合通用性排名 / writing a thesis comparison section / drafting answer to "is v9 better than v8" without re-reading per-version skills, OR when a NEW eval run finishes (eval_*finetuned.bat / eval_roadscene.py outputs new overall.txt) and the table needs a fresh row. Triggers: results / 结果汇总 / 总表 / 评测对照表 / 跨版本比较 / SOTA 排名 / 综合通用性 / 谁最好 / v5 v6 v7 v8 v9 数字 / overall.txt 集合 / dump 汇总 / 实验结果中心 / which ckpt to ship / 选哪个版本 / 答辩用对比表 / append new row to results / new eval finished / "我刚跑完 v10 eval, 把数字填进去", English 'aggregate eval results', 'cross-version comparison table', 'all overall.txt summary', 'append new eval row', 'in-domain vs OOD matrix', 'composite generalization score', 'which checkpoint should I ship', 'thesis-ready comparison'. Companion skills: eloftr-eval-pipeline (HOW each overall.txt is produced), eloftr-tb-analysis (training-side per-epoch metrics, NOT independent eval), eloftr-v5..v9 (per-version implementation context), eloftr-cross-modal-experiments (overall chain narrative).
---

# 实验结果中心（results/eval_summary.md 的工作流）

> **本 skill 不存数字、不解释模型**。它只规定一件事：**所有 v5..v9 独立 eval 结果，唯一汇总位在 `results/eval_summary.md`**——读取、追加、引用都走这一个文件。
>
> 数字本身、跨版本对照表、综合通用性排名、各版本 ckpt 选择决策，全部在 [`results/eval_summary.md`](../../../results/eval_summary.md)。

## 0. 一句话用法

| 场景 | 动作 |
|---|---|
| 想看"现在跨版本谁最强 / 选哪个 ckpt ship" | 直接读 [`results/eval_summary.md`](../../../results/eval_summary.md) §3 综合通用性排名 |
| 想看某版本的具体 in-domain / OOD 数字 | [`results/eval_summary.md`](../../../results/eval_summary.md) §1 / §2 |
| 跑完一次新 eval（产生新的 `dump/<dataset>_eval_v<X>_*/overall.txt`） | 按本 skill §2 工作流追加一行 |
| 想引用某版本数字到论文 / 答辩 PPT | 直接 copy `results/eval_summary.md` 表 |
| 想知道每个 overall.txt 是怎么来的（哪个 cfg / ckpt 选择规则 / 为什么训练 val 与独立 eval 不同口径） | 跳 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)，本 skill 只管"汇总"不管"产生" |
| 想看某版本训练侧 per-epoch 曲线（与独立 eval 不同口径） | 跳 [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md) |

## 1. 为什么要单独做这个 skill

- **去重**：v5/v6/v7/v8/v9 五个 per-version skill 各自零散写过自己的"实测结果"段，跨版本对照需要人脑拼接，且有过 v6 vs v6.1 数字哪份是新版的混淆历史。
- **单点更新**：新 eval 跑完只需要改一个文件 + 在 `cross-modal-experiments` 顶部 SOTA 指针处改一行，不再每个 v_x SKILL 一次。
- **答辩 / 论文友好**：表是论文 / PPT 直接能用的形态，三张表都是定列定行，便于版本控制 diff。
- **历史命名兼容**：v5 / v6 用的是历史 dump 目录名（`m3fd_eval_v5_combined_*` / `m3fd_eval_v6_finetune_*`，详见 [eval-pipeline §4.5](../eloftr-eval-pipeline/SKILL.md)），新版 eval 用新命名（`m3fd_eval_v<X>_version<Y>_<topZ|last>`）。汇总表的"数据来源文件索引"段（§5）专门记录这个映射，避免下次 agent 又走错目录。

## 2. 新 eval 跑完后的追加工作流（强约束，3 步）

1. **确认 overall.txt 已经写出**：跑完 `eval_*finetuned.bat <X>` 或 `eval_roadscene.py` 后，目标目录里应同时有 `overall.txt` + `summary.csv` + 若干 `*_match.png`。如果只有 PNG 没 overall.txt，eval 实际没跑完（早期被 Ctrl+C 之类）。
2. **在 [`results/eval_summary.md`](../../../results/eval_summary.md) 的对应表追加一行**：
    - §1 (M3FD in-domain) → 新增 `m3fd_eval_*` overall.txt 的字段
    - §2 (RoadScene OOD) → 新增 `roadscene_eval_*` overall.txt 的字段
    - §3 综合通用性 → 必须等 §1 + §2 都有同一版本 ckpt 数字才能填，"综合 = M3FD P@3 + OOD P@3"，并按数值重排"排名"列、回填 `Δ vs prev`
    - §5 数据来源索引 → 把新 overall.txt 路径加进去（用新命名约定）
3. **更新指针**（仅当新版本进入综合通用性 top1 / 新版本结论改变 v9 SOTA 地位时）：
    - 改 [`.cursor/skills/eloftr-cross-modal-experiments/SKILL.md`](../eloftr-cross-modal-experiments/SKILL.md) 顶部 mermaid 图最右节点的高亮 / "★ 推荐" 标注
    - 改 [`.cursor/skills/eloftr-v9-e2e/SKILL.md`](../eloftr-v9-e2e/SKILL.md) §4 / §10（如果 v10+ 替代 v9 ship）
    - 改 [`.cursor/skills/eloftr-v<new>/SKILL.md`](../) 顶部"Results 索引"行，让指向 `results/eval_summary.md` 的链接落到位
    - 在 `results/eval_summary.md` 顶部 "最近一次更新" 改日期 + 下面每段 ⭐ 高亮换行（如有 SOTA 易主）

> **不要**直接在 v_x SKILL 里贴新数字然后忘了同步到 `results/eval_summary.md`。**所有跨版本对比的"事实"只许 `results/eval_summary.md` 一个文件持有。**v_x SKILL 里的"实测"段可以保留**该版本自己的内部分析**（如训练曲线、ablation 假设），但**对照表只引用本 skill**。

## 3. 表的数据规范（追加新行时务必遵守）

### 3.1 RoadScene / M3FD prec@N px overall.txt 字段（§1 §2 §3）

每个 overall.txt 共写 8 个字段，全部如实抄入：

```text
ckpt:               -> ckpt 列（取 epoch 编号 + val P@3 写成 "ep=N (P@3=X.XXX)"）
main_cfg:           -> main_cfg 列（短文件名）
data_cfg:           -> 决定写到 §1 (m3fd_*) 还是 §2 (roadscene_*)
pairs:              -> pairs 隐含（M3FD 永远 210, RoadScene 永远 22, 异常时显式标注）
apply_homography:   -> 始终 False（独立 eval 不开 H 增强），异常时显式标注
total_matches:      -> total_matches 列
mean_pixel_error:   -> mpe 列（保留 4 位小数）
precision@1px:      -> P@1px 列（保留 4 位小数）
precision@3px:      -> P@3px 列（保留 4 位小数）
precision@5px:      -> P@5px 列（保留 4 位小数）
```

**派生列**（不在 overall.txt 里，需手算）：
- `matches/pair = total_matches / pairs`，取整
- §3 综合 = M3FD P@3 + OOD P@3，保留 4 位
- §3 排名 = 按"综合"降序
- §3 Δ vs prev = 与排序后上一行的"综合"差值

**列宽对齐**：保持 §1 / §2 的列顺序与现有表一致（main_cfg / ckpt / total_matches / matches/pair / mpe / P@1 / P@3 / P@5）；§3 列顺序固定（M3FD P@1 / M3FD P@3 / M3FD mpe / OOD P@1 / OOD P@3 / OOD mpe / 综合 / 排名 / Δ）。

### 3.2 METU-VisTIR pose-AUC overall.txt 字段（§6，2026-05-13 新增）

**METU 跟 RoadScene/M3FD 是不同的评测指标**——RoadScene/M3FD 是 epipolar prec@1/3/5px，METU 是 pose-based AUC@5/10/20°。**两个不同表，不要混着填**。完整 METU 协议见 [eloftr-metu-vistir-eval](../eloftr-metu-vistir-eval/SKILL.md)。

METU overall.txt 三段块，应抄入字段：

```text
[overall]
all_class_mean  auc@5: 2.962   auc@10: 8.093   auc@20: 18.125
num_matches: 253.84

→ §6 表填这 4 列（auc@5 / auc@10 / auc@20 / num_matches），单位百分数 (overall.txt 已 ×100)
```

per-class 两行 (`cloudy_cloudy` / `cloudy_sunny`) 是 diagnostic，不入 §6 主表；若需细分到光照条件，加 §6.1 附表。per-scene 10 行更细，不入汇总。

**派生列**：
- `Δ vs baseline` = 该 ckpt auc@10 - ELoFTR outdoor.ckpt baseline auc@10（cross-modal 训练增益）
- 排名按 `auc@10` 降序（或 `auc@5 + auc@10 + auc@20` 加和，跟 §3 综合通用性同思路）

**协议变更注意**（2026-05-13）：旧的 METU 行（如果有）基于「ransac_thr=2.0 / 5-restart / pool aggregation / T_0to1=inv(P1)@P0」旧协议，**跟新协议的 ELoFTR baseline 不可直接比较**，详见 [eloftr-metu-vistir-eval §5](../eloftr-metu-vistir-eval/SKILL.md)。追加新行时确认是新协议跑出来的 (overall.txt 顶部 `protocol: MINIMA / XoFTR` 标记)。

## 4. 与其它 skill 的关系（边界）

| skill | 它管什么 | 不管什么 |
|---|---|---|
| **本 skill (eloftr-results)** | 「跨版本独立 eval 数字的汇总在哪、怎么追加新行、怎么解释 v9⭐ / v8⚠️ 这种标记」 | 数字怎么来的（→ eval-pipeline / metu-vistir-eval）、模型为什么这样设计（→ v_x）、训练曲线（→ tb-analysis） |
| [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md) | 「`eval_roadscene.py` / `eval_*finetuned.bat` 怎么跑、ckpt cfg 怎么自动配对、新旧目录命名映射、`unexpected_keys` 调试」 — RoadScene / M3FD precision@N px 路径 | 跨版本数字对照表（→ 本 skill）；METU pose-AUC（→ metu-vistir-eval） |
| [eloftr-metu-vistir-eval](../eloftr-metu-vistir-eval/SKILL.md) | METU-VisTIR pose-AUC eval：MINIMA 协议精确参数、bit-perfect 复现论文表 3、T_0to1 公式、side 不对称、故障树 | 跨版本数字对照表（→ 本 skill）；RoadScene / M3FD（→ eval-pipeline） |
| [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md) | 「训练 tfevents 抽 per-epoch JSON、v9_acceptance 自动评级、MSBN 是否 active」 | **训练 val ≠ 独立 eval**，本 skill 表里的所有数字都来自独立 eval 不来自 tb |
| [eloftr-v5-m3fd](../eloftr-v5-m3fd/SKILL.md) ... [eloftr-v9-e2e](../eloftr-v9-e2e/SKILL.md) | 单版本设计动机、配置、训练 schedule、ablation 想法 | 跨版本横向对比（→ 本 skill） |
| [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md) | 链路总览、继承图、候选路径 A-G、加新版本 playbook | 当前 SOTA 数字（→ 本 skill） |

## 5. 历史与未来扩展

- **v1..v4 暂不纳入本 skill 表**：那四个是 RoadScene-trained，独立 eval 在 `dump/roadscene_eval_v{1..4}_*/`，"in-domain"对它们而言是 RoadScene 而非 M3FD，跟 v5+ 不可同表对比。如果以后做 v1..v4 的"跨数据集 eval 矩阵"（在 M3FD 上测 RoadScene-trained ckpt），新建 `results/eval_v1_v4_cross_dataset.md`，本 skill 不管。
- **METU-VisTIR (2026-05-13 加入)**：作为 §6 新表，**协议跟 RoadScene/M3FD 完全不同**（pose-AUC@5/10/20° vs prec@N px）。dataset 协议 + 跑法 + 故障树详见 [eloftr-metu-vistir-eval](../eloftr-metu-vistir-eval/SKILL.md)。**§3 综合通用性公式不扩到 METU**（混入不同指标会破坏 single-number ranking），METU 单独看 `auc@10` 排序。
- **未来超 15 行后**：本 skill §0 "数据来源索引" 表会变长，那时可以补一个 `MyScripts/aggregate_eval_results.py` 自动扫 `dump/*/overall.txt` 出 CSV，避免手抄漏字段。当前 6 行规模写脚本 ROI 不高，**先手维护**。
- **未来加 LLVIP / KAIST / TNO**：根据数据集是 epipolar prec 还是 pose-AUC，加到 §1 / §2 一族 或 §6 一族。本 skill 工作流不变。

## 6. AGENTS.md / 路径规则关系

- `results/` 已加入 [AGENTS.md §5 输出物路径表](../../../AGENTS.md) 与 [.cursor/rules/00-scope-and-output.mdc](../../../.cursor/rules/00-scope-and-output.mdc)，**入版本库**（不在 `.gitignore`，与 `abfNote/` 同等位)。本仓核心实验结论与 `abfNote/<topic>.md` 一同纳入版本控制，便于答辩 / 复现 / 跨机器同步。
- `results/eval_summary.md` 是"事实表"；`abfNote/<topic>.md` 是"思考过程 / 故事线"。**两者不重叠**：表里的每一行如果需要展开论证，去 `abfNote/<topic>.md` 写散文，把数字 link 到 `results/eval_summary.md`。
