---
name: eloftr-tb-summary
description: Single source-of-truth index for EfficientLoFTR cross-experiment training-side TensorBoard summary (v1..v9), one Markdown table at results/tb_summary.md aggregating all logs/tb_logs/<exp>/*/events.out.tfevents.* via MyScripts/read_tb_metrics.py aggregate. Use when asked for "training cost & convergence comparison" / "best val epoch across all versions" / "modemb / MSBN diagnostics across versions" / "wall-clock per experiment" / "train val vs independent eval gap" / "regenerate tb summary after a new run" / 训练成本对比 / 训练曲线总览 / 多实验对照 / aggregate tb_logs / regenerate tb_summary.md / new training finished, fold into tb_summary, OR when interpreting why train val ≠ independent eval (use eval_summary.md for shipping numbers, this skill for training process diagnostics). Triggers: tb_summary / tb 总结 / 训练侧汇总 / aggregate / read_tb_metrics aggregate / 训练 val 对比 / best epoch 对比 / wall-clock 对比 / modemb verdict 对比 / MSBN drift 对比 / train val vs eval gap / train val 与独立 eval 差距 / 训练 ep 与 ship ckpt ep 不一致 / PL ModelCheckpoint monitor=p@3 vs best p@1 / v9 ep52 vs ep75 / 重新生成训练总结 / append new exp to tb_summary, English 'aggregate tensorboard summaries', 'cross-experiment training table', 'train val vs eval gap', 'wall-clock comparison', 'best validation epoch table', 'modemb / MSBN verdict matrix', 'PL ckpt monitor mismatch'. Companion skills: eloftr-tb-analysis (HOW to read individual tfevents, NOT cross-experiment), eloftr-results (sibling, owns results/eval_summary.md = independent eval not training val), eloftr-v1..v9 (per-version implementation context), eloftr-cross-modal-experiments (overall chain narrative).
---

# 训练侧 TB 总结中心（results/tb_summary.md 的工作流）

> **本 skill 不存数字、不重复模型分析**。它只规定一件事：**所有 v1..v9 训练时 PL `metrics_0/*` + `train/*` 的横向对照，唯一汇总位在 `results/tb_summary.md`**——读取、追加、引用都走这一个文件。
>
> 数字本身、训练成本、best val epoch、modemb / MSBN 诊断、与独立 eval 的对照（§4 表）全部在 [`results/tb_summary.md`](../../../results/tb_summary.md)。

## 0. 一句话用法

| 场景 | 动作 |
|---|---|
| 想看"v_X 训了多久 / 最好 epoch 在哪 / MSBN 起没起作用" | 直接读 [`results/tb_summary.md`](../../../results/tb_summary.md) §1/§2/§3 |
| 想看"训练 val 与独立 eval 差多少（系统偏移 vs 漂移）" | [`results/tb_summary.md`](../../../results/tb_summary.md) §4 联动表 |
| 跑完新训练（产生新的 `logs/tb_logs/<exp>/`） | 按本 skill §2 工作流：跑 `aggregate` → 把 raw 内容贴进表 → 删 raw |
| 想看某实验 per-epoch 完整曲线 / tag 列表 / 导出 CSV | 跳 [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md)，本 skill 只管"跨实验汇总"不管"单实验深读" |
| 想看独立 eval 数字（论文 / 答辩用） | 跳 [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)，**两个表用途完全不同** |

## 1. 为什么要单独做这个 skill（与其它 skill 的边界）

| skill | 它管什么 | 不管什么 |
|---|---|---|
| **本 skill (eloftr-tb-summary)** | 「跨实验训练侧 KPI 表的位置 / 维护流程 / 与 eval_summary.md 的联动」| 单实验 tfevents 怎么读（→ tb-analysis）、独立 eval 数字（→ results）、模型为啥这样设计（→ v_x） |
| [eloftr-tb-analysis](../eloftr-tb-analysis/SKILL.md) | 「`MyScripts/read_tb_metrics.py` 的 `tags` / `summary` / `export` / `best` 子命令、单实验 JSON schema、size_guidance 防 OOM、v9 自动评级」| 跨实验对照表（→ 本 skill） |
| [eloftr-results](../eloftr-results/SKILL.md) | 「跨版本独立 eval 数字汇总在 `results/eval_summary.md`」| 训练侧曲线（→ 本 skill） |
| [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md) | 「`eval_roadscene.py` 怎么跑产生 `overall.txt`」| 训练侧曲线（→ 本 skill） |
| [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md) | 「实验链总览、继承图、加新版本 playbook」| 当前训练成本数字（→ 本 skill）、当前 SOTA（→ results） |

**核心区分**：
- **`tb_summary.md` (本 skill)** = 训练时 PL `validation_step` 写到 tfevents 的**训练 val 数字** + 训练成本（wall-clock、step 数）+ 训练侧诊断（modemb / MSBN drift）
- **`eval_summary.md` ([eloftr-results](../eloftr-results/SKILL.md))** = 训练完后用 `eval_roadscene.py` 跑出来的**独立 eval 数字** + 综合通用性排名

两者**口径接近但不同**：训练 val 走 RepVGG multi-branch + dataset random ordering，独立 eval 完全镜像但 random seed 可能不同 → 系统偏移恒为 +0.012..+0.030（独立 eval 更高），见 `tb_summary.md` §4 + §5 观察 5。**论文 / 答辩永远引用 eval_summary.md，本表用来解释训练动态**。

## 2. 新训练跑完后的追加工作流（强约束，4 步）

**前提**：新训练已产出 `logs/tb_logs/<new_exp>/version_<N>/events.out.tfevents.*`。

1. **确认新实验目录名匹配 aggregate include 模式**：
   - 默认 include = `^(roadscene|m3fd)_v\d`
   - 默认 exclude = `(_debug|_small|_compat_test)`
   - 新加 dataset (例 `llvip_v10_xxx`) 时，跑 `aggregate --include "^(roadscene|m3fd|llvip)_v\d"` 临时扩展，并把更宽的 include 反写到本 skill §2.1
2. **跑 aggregate 重新生成基础三表**（v10 后默认 include = `^(roadscene|m3fd|msyn)_v\d`，无需手传 `--include`）：
    ```bash
    # 本地 Windows PowerShell:
    & "$env:USERPROFILE\miniconda3\envs\eff_loftr\python.exe" `
        MyScripts\read_tb_metrics.py aggregate `
        --out results\tb_summary_raw.md

    # 本地 Windows cmd:
    %USERPROFILE%\miniconda3\envs\eff_loftr\python.exe ^
        MyScripts\read_tb_metrics.py aggregate ^
        --out results\tb_summary_raw.md

    # 服务器 vlrlab:
    conda activate eloftr_yurupeng && \
        python MyScripts/read_tb_metrics.py aggregate \
        --out results/tb_summary_raw.md
    ```

   如果需要临时扩展（如未来加 LLVIP）：PowerShell 写 `--include '^(roadscene|m3fd|msyn|llvip)_v\d'`（**单引号**包住，否则 `|` 被解析成管道）；cmd 直接用双引号 `"..."` 也可。
3. **把 raw 三表内容贴进 `results/tb_summary.md` §1/§2/§3 替换对应行**：
    - **保留** `results/tb_summary.md` 顶部 intro、§4 联动表、§5 观察、§6 复现命令——这些是人工策展，**不**应被 raw 覆盖
    - 仅替换 §1/§2/§3 表格行（基础数字）
    - 新版本如果是 v10+，按 §2.2 排序规则插到正确位置（`v9` 之后）
    - 重算 §4 联动表的"Δ P@1 / Δ P@3"列（如果该新版本也有 `eval_summary.md` 入口）
    - 在顶部"最近一次更新"改日期 + 简述（"v10 e2e cold start 跑完，新增第 11 行"）
4. **删除 raw 文件**：`del results\tb_summary_raw.md`（或 `rm`）
5. **(可选) 更新对应 v_x SKILL** 的 "TB 总结指针" 行，让其指到本 skill；并在 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md) mermaid 图末段加新节点

> **不要**直接手填 raw 三表的数字进 `results/tb_summary.md`，永远走 `aggregate` 自动算 → 复制粘贴。手填易错（小数位 / 千分位 / 中英标点 / time iso 截断到分），脚本输出统一格式。

### 2.1 当前 include/exclude regex

| 项 | regex |
|---|---|
| include | `^(roadscene\|m3fd\|msyn)_v\d` |
| exclude | `(_debug\|_small\|_compat_test)` |

历史扩展记录：
- 2026-05-04 ~ 2026-05-07：v1..v9 时期 include = `^(roadscene\|m3fd)_v\d`（v1..v4 RoadScene + v5..v9 M3FD）
- 2026-05-08（v10）：扩到 `^(roadscene\|m3fd\|msyn)_v\d`，新增 Megadepth_Syn 训练（exp_name `msyn_v10_ddp`）。**默认值已同步改 `MyScripts/read_tb_metrics.py:645`**，所以现在不传 `--include` 也能命中 v10。

未来添 LLVIP / KAIST / TNO 数据集训练时，include 改成 `^(roadscene\|m3fd\|msyn\|llvip\|kaist\|tno)_v\d`，**同步改 `read_tb_metrics.py:645` 默认值 + 本节**。

### 2.2 版本号排序规则（aggregate 内置 + 本表行序）

`MyScripts/read_tb_metrics.py:_version_sort_key` 用 `(major, minor)` 元组排序：
- v1 → (1, 0), v2 → (2, 0), ..., v6 → (6, 0), **v6.1 → (6, 1)**, v7 → (7, 0), ..., v9 → (9, 0)
- 不能解析的 (例 `v9_e2e_outdoor` 后缀部分) 不影响主序，只用 `v9` 标签插到 (9, 0)
- 新增 v10 / v11 / v6.2 时直接按这个规则插

如果将来加 `v9_server_repro`（DDP 复现 v8 的 infra 实验），可能与 `m3fd_v9_e2e_outdoor` 同 major，需要扩 `_version_sort_key` 加第三键（如实验"路径 G/H"或时间戳）。届时回头改脚本 + 本节。

## 3. 数据规范（脚本默认 + 人工补段的边界）

### 3.1 `aggregate` 自动产出（§1/§2/§3 三表）

- §1 训练成本：从 `train/loss` + `train/avg_loss_on_epoch` 取 step / epoch 数 + min/final loss；从 wall_time 取 start/end ISO + elapsed_hours
- §2 训练 val 最佳 epoch：从 `metrics_0/precision@1px` 取 max epoch + 该 epoch 的 P@1/P@3/P@5/mpe/num_matches/avg_loss
- §3 modemb / MSBN：仅当对应 tag 存在才填，否则 `—`；verdict 由 `read_tb_metrics.py` 内置阈值判断 (modemb >0.05, MSBN drift_last >0.05)

### 3.2 人工补段（§4/§5/§6 + 各表脚注）

- §4 联动表：手抄 `eval_summary.md` §1 M3FD test 的 ckpt epoch + P@1/P@3，算 Δ
- §5 关键观察：6-8 条事实陈述（如 "v9 cold start MSBN drift max=0.24 vs v8 resume max=0.09"），与 `eval_summary.md` §4 风格一致——**只陈述事实不带分析，分析留 v_x SKILL**
- 各表脚注 ¹²³⁴⁵⁶⁷：标注异常 (v6 spillover、v3 全冻、v9 ep52 vs ep75 ship ckpt 不一致 等)，与对应 v_x SKILL 反向链接

人工补段的核心约束：**只陈述事实 + 反向链接到 v_x SKILL，不重复 v_x SKILL 的设计动机/机制猜测**。

## 4. 与 `tb_summary.md` §4 联动表的维护协议

§4 表的左半部分（tb best ep + tb val P@1/P@3）来自本 skill 的 `aggregate` 输出；右半部分（eval ep + eval P@1/P@3）来自 [eloftr-results](../eloftr-results/SKILL.md) 维护的 `eval_summary.md` §1 M3FD test。

**两表必须保持一致**的字段：
- `eval ep`（§4 右）= `eval_summary.md` §1 ckpt 列的 epoch（如 v9 = 75 不是 52）
- `eval P@1` / `eval P@3`（§4 右）= `eval_summary.md` §1 P@1px / P@3px 列

**变更触发联动**：
- `eval_summary.md` 改了某行 ckpt（罕见，比如重选 ship ckpt） → 同步改本表 §4 对应行
- `tb_summary.md` 改了 §2（重训某实验改了 best epoch） → 同步改本表 §4 对应行

如果 §4 表与 §2/eval_summary 数字不一致，**§4 表的数字以 §2 + eval_summary 的最新数字为准**（§4 是派生表）。

## 5. 历史与未来扩展

- **未来加 LLVIP / KAIST / TNO 训练**：先按 §2.1 扩 include regex；如果 `_version_sort_key` 需扩展见 §2.2
- **未来超 15 行（v15+）**：表会变长，可以考虑把 §1/§2/§3 拆成多张表（按 dataset / 按 epoch 数 / 按 SOTA 候选分），但**始终单文件**，避免再分裂
- **未来 aggregate 加列**：例如 `train/total_param_count`（如果训练时 log）、`val/best_p3_epoch` (single ckpt 选择校验)。修改 `cmd_aggregate` + 本 skill §3 同步描述

## 6. 与 AGENTS.md / 路径规则关系

- `results/tb_summary.md` 已加入 [AGENTS.md §5 输出物路径表](../../../AGENTS.md) 与 [.cursor/rules/00-scope-and-output.mdc](../../../.cursor/rules/00-scope-and-output.mdc)，与 `results/eval_summary.md` 同级，**入版本库**
- `tb_logs/` 整树**仍 gitignore**（events 文件 100MB-748MB 不入版本库），所以本表是"对 gitignore 黑盒里数据的版本受控视图"——这是为什么需要这个 skill 的根本原因
