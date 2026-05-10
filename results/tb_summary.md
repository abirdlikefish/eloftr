# EfficientLoFTR 训练侧 TensorBoard 汇总（v1..v10）

> 数据来源：每个实验的 `logs/tb_logs/<exp>/version_<N>/events.out.tfevents.*`，由 `MyScripts/read_tb_metrics.py aggregate` 一次扫全部产出基础 3 张表，再人工补 §4 联动 + §5 观察 + §6 复现命令。
>
> 本表只覆盖**训练时 PL `validation_step` → tfevents** 写出的指标（PL `metrics_0/*` + `train/*`），**与独立 eval 不是同口径**：独立 eval 数字在 [`results/eval_summary.md`](./eval_summary.md)，两条管线对照见 §4。
>
> `tb_logs/` 整树已 gitignore（仅本地 / 服务器持有），表里的"复现命令"假设你跑过对应训练或拉到了 tfevents。
>
> 维护规则与新实验追加流程见 [.cursor/skills/eloftr-tb-summary/SKILL.md](../.cursor/skills/eloftr-tb-summary/SKILL.md)。
>
> 最近一次更新：2026-05-08（v10 Megadepth_Syn 4 卡 DDP ship 完成 → aggregate include 扩到 `^(roadscene|m3fd|msyn)_v\d`，扫 v1..v10 共 11 个实验；同步追加独立 eval 双数据集结果到 [`eval_summary.md`](./eval_summary.md)）。

---

## §1 训练成本与收敛

| 版本 | exp_name | version | dataset | total_epochs | total_steps | wall (h) | final loss | min loss | start | end |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| v1 | `roadscene_v1_contrast` | 2 | aligned_irvis | 20 | 1 950 | 0.38 | 0.4265 | 0.3915 | 2026-05-04T17:58 | 2026-05-04T18:20 |
| v2 | `roadscene_v2_modemb` | 1 | aligned_irvis | 79 | 7 900 | 0.46 | 0.1646 | 0.1293 | 2026-05-04T16:44 | 2026-05-04T17:12 |
| v3 | `roadscene_v3_combined` | 3 | aligned_irvis | 12 | 550 | 0.27 | 0.6204 | 0.5669 | 2026-05-05T00:17 | 2026-05-05T00:33 |
| v4 | `roadscene_v4_combined` | 2 | aligned_irvis | 18 | 850 | 0.32 | 0.4888 | 0.4939 | 2026-05-04T22:24 | 2026-05-04T22:43 |
| v5 | `m3fd_v5_combined` | 0 | aligned_irvis | 10 | 9 400 | 3.91 | 0.4157 | 0.3289 | 2026-05-05T01:28 | 2026-05-05T05:23 |
| v6 ¹ | `m3fd_v6_finetune` | 0 | aligned_irvis | 8 | 7 550 | 2.73 | 0.3888 | 0.2841 | 2026-05-05T13:45 | 2026-05-05T16:29 |
| v6.1 | `m3fd_v6_1_finetune` | 0 | aligned_irvis | 8 | 8 000 | 1.52 | 0.3688 | 0.2612 | 2026-05-05T16:59 | 2026-05-05T18:30 |
| v7 | `m3fd_v7_pcclahe` | 0 | aligned_irvis | 10 | 9 400 | 2.72 | 0.3570 | 0.2756 | 2026-05-05T21:54 | 2026-05-06T00:37 |
| v8 | `m3fd_v8_msbn` | 0 | aligned_irvis | 20 | 18 850 | 3.98 | 0.3225 | 0.2272 | 2026-05-06T03:43 | 2026-05-06T07:42 |
| v9 | `m3fd_v9_e2e_outdoor` | 0 | aligned_irvis | 80 | 75 550 | 10.96 | 0.1683 | 0.1094 | 2026-05-06T22:44 | 2026-05-07T09:41 |
| **v10** ⁸ | `msyn_v10_ddp` | 0 | aligned_irvis | **12** | **86 250** | **17.11** | **0.2116** | **0.1208** | 2026-05-07T23:04 | 2026-05-08T16:10 |

¹ v6 实测 ep6 spillover 触发训练中断（per [eloftr-v6-finetune SKILL §spillover](../.cursor/skills/eloftr-v6-finetune/SKILL.md)），表中 `total_epochs=8` 是 PL 写到 tfevents 的最后一个 epoch 数（包含 ES patience 相关的尾巴），不是真正的"完整收敛"epoch 数。v6.1 是 v6 的 hotfix（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`），cfg 不变，但 wall=1.52h（v6 用了 2.73h）说明 PERSISTENT_WORKERS + expandable_segments 的双重收益。

⁸ v10 是首个跨数据集 + 4 卡 DDP 实验：训练集换 Megadepth_Syn（~106K train pair，约 28× M3FD），4 卡 sync_bn DDP，故 step/ep 7 188（vs v9 944, 7.6×）+ wall 17.11h（v9 10.96h）。**plan §10 原估"3-4h"完整 ship run 实际偏乐观 5×**（v10 SKILL §wall-clock 修订写"4×"，按 17.11/3.5 h ≈ 4.9× 应改"5×"）；主因：sync_bn 在 RepVGG ~30 BN + MSBN 4 BN 架构上跨 4 卡同步 mean/var 占 step 时间 ~25%，4 卡名义 4× 加速吃到 ~0.7× 实际。详见 [eloftr-v10-msyn SKILL §6](../.cursor/skills/eloftr-v10-msyn/SKILL.md) 与 [eloftr-server-multigpu SKILL §6.5](../.cursor/skills/eloftr-server-multigpu/SKILL.md)。

> 多 version 实验（v1/v2/v3/v4 都有 version_0..2/3 多次重训）：本表的 version 列是 `aggregate` 自动选取的最新版本号（与 `eval_*finetuned.bat` 默认 Y 选最高版本号一致）。其它历史 version 不在本表。

---

## §2 训练 val 最佳 epoch + 主指标（PL `metrics_0/*`，监控 max P@1px）

| 版本 | best_epoch | val P@1px | val P@3px | val P@5px | val mpe | val num_matches | val avg_loss | num_validations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 19 | 0.5726 | 0.7463 | 0.7609 | 2.7735 | 1 348 | 1.2763 | 20 |
| v2 ² | 62 | **0.7413** | 0.7932 | 0.7980 | 2.2520 | 1 548 | 0.6411 | 29 |
| v3 ³ | 7 | 0.3077 | 0.6774 | 0.7776 | 3.3849 | 1 011 | 2.4382 | 12 |
| v4 ³ | 16 | 0.3741 | 0.7132 | 0.7840 | 2.9665 | 1 395 | 1.8341 | 18 |
| v5 | 9 | 0.4181 | 0.7891 | 0.8644 | 2.2460 | 1 896 | 1.8275 | 10 |
| v6 | 6 | 0.4253 | 0.8012 | 0.8728 | 2.1271 | 1 947 | 1.7259 | 8 |
| v6.1 | 4 | 0.4428 | 0.8143 | 0.8789 | 2.0286 | 1 964 | 1.6528 | 8 |
| v7 | 6 | 0.4754 | 0.8546 | 0.9126 | 1.7064 | 2 013 | 1.4906 | 10 |
| v8 | 16 | 0.5196 | 0.8885 | 0.9367 | 1.4147 | 2 084 | 1.3224 | 20 |
| v9 | 52 ⁴ | 0.6881 | 0.9753 | 0.9929 | 0.7343 | 2 194 | 0.8462 | 80 |
| **v10** ⁹ | **11** | **0.8344** | **0.9976** | **0.9991** | **0.4545** | 2 059 | **0.4271** | 12 |

² **v2 RoadScene val P@1=0.74 是过拟合**：22 张 RoadScene val pair 训了 79 ep，best ep 在 62（接近末段），v2 SKILL §"22 张过拟合"已论证。注意这是**RoadScene val** 数字，**不能**与 v5..v9 的 M3FD val P@1 同表对比（数据集不同）。v5 选 M3FD 后 val P@1 暂时降到 0.42 但 OOD 通用性反而更强（详见 [eloftr-v5-m3fd SKILL §1](../.cursor/skills/eloftr-v5-m3fd/SKILL.md)）。

³ **v3 P@1=0.31 是 freeze stack 误激进**：v3 全冻 backbone + freeze BN + 强 LR 衰减，fine BN 没机会更新→fine sub-pixel 学不到。v4 REVISION 2 释放 fine BN 后 P@1 0.37 (+21% rel)。详见 [eloftr-v3-v4-freeze SKILL](../.cursor/skills/eloftr-v3-v4-freeze/SKILL.md)。

⁴ **v9 best_epoch=52 但 PL 实际 ship ckpt = ep75**：原因是 PL `ModelCheckpoint(monitor='precision@3px')`（不是 P@1），ep75 P@3=0.9758 比 ep52 P@3=0.9753 高 0.0005（统计噪声内）→ 选了 ep75。完整解释见 [eloftr-v9-e2e SKILL §6](../.cursor/skills/eloftr-v9-e2e/SKILL.md)。本表 §4 给的对照仍以"训练 val best_epoch + 该 epoch 指标"为列，**不**是"PL ship ckpt 的 epoch"。

⁹ **v10 训练 val P@3=0.998 / P@5=0.999 是 evaluator 退化产物，不能当真分**：v10 训练数据 Megadepth_Syn 的 (IR_i, VIS_i) 是 style-transfer 产物，**像素一一对齐**；同时 [`src/lightning/data.py` L288](../src/lightning/data.py) `homography_aug=(self.road_homography_aug and mode == 'train')` 在 val/test 时强制关闭 homography aug → val IR-VIS 直接以 identity homography 进 metric → P@3/P@5 在 ep0（cold start）就 0.979/0.992，ep11 饱和到 0.998/0.999。**v10 的 P@3 单 epoch 增量 < 0.001 时，PL ckpt monitor=P@3 选 ckpt 几乎在统计噪声内挑**（v10 ep11 P@3=0.9976 最高，但 ep10=0.9973、ep8=0.9966 仅差 0.0003-0.0010），**真分必须看独立 eval [`eval_summary.md`](./eval_summary.md) §1/§2**（M3FD 0.9643 + RoadScene 0.9377）。本表保留 0.998/0.999 仅作"训练动态 reference"，论文 / 答辩永远引用 eval_summary。

---

## §3 modemb / MSBN 诊断（仅相应版本写出）

| 版本 | modemb ir_norm | modemb vis_norm | modemb verdict | MSBN drift L1 (last/max) | MSBN drift L2 (last/max) | MSBN verdict |
|---|---:|---:|---|---|---|---|
| v1 | — | — | — | — | — | — |
| v2 ⁵ | 0.0433→0.0491 | 0.0422→0.0468 | **modemb dead** (norm ≤0.05；MODALITY_EMB_INIT='normal_0.02' 起点已大、增长被压) | — | — | — |
| v3 | 0.0000→0.0857 | 0.0000→0.0913 | modemb learning | — | — | — |
| v4 | 0.0000→0.0538 | 0.0000→0.0496 | modemb learning（vis 临界，刚过 0.05） | — | — | — |
| v5 | 0.0000→0.0957 | 0.0000→0.0905 | modemb learning ✓ | — | — | — |
| v6 | 0.0957→0.0981 | 0.0905→0.0926 | modemb learning（resume 起点 = v5 终态，仅微调） | — | — | — |
| v6.1 | 0.0972→0.0996 | 0.0918→0.0939 | modemb learning（resume 起点 = v6 终态）| — | — | — |
| v7 | 0.0988→0.1003 | 0.0932→0.0949 | modemb learning（resume 起点 = v6.1 终态） | — | — | — |
| v8 | 0.0999→0.1004 | 0.0945→0.0946 | modemb learning（resume 起点 = v7 终态，几乎不动） | 0.0722 / 0.0882 | 0.1064 / 0.1389 | MSBN active ✓ |
| v9 ⁶ | 0.0000→0.1077 | 0.0000→0.0951 | modemb learning（cold start 真零起步） | 0.0622 / 0.2388 | 0.1133 / 0.1641 | MSBN active ✓（drift max 高于 v8、last 接近 v8）|
| **v10** ¹⁰ | **0.0000→0.0861** | **0.0000→0.0863** | modemb learning（cold start 真零起步，**ir/vis 终态对称**）| **0.1895 / 0.2045** | **0.0551 / 0.1210** | MSBN active ✓（**drift_last 比 v9 高 3×、max 接近 v9，弧形未走完**）|

⁵ v2 'modemb dead' 来自 v2 SKILL 后续修复：v2 `MODALITY_EMB_INIT='normal_0.02'` 起点 norm ≈ √(C·0.02²) ≈ 0.04+，本来就在 0.05 阈值附近，"dead" verdict 阈值偏严，**不代表 v2 modemb 完全没学**——只是相对起点的增长很小，且 v2 fine 路径有"模态盲"问题（v2 SKILL "残余信号在 fine 路径被两次吃掉"）让 modemb 在 fine 阶段被结构性吃掉。v3+ 切到 `MODALITY_EMB_INIT='zeros'` 起点真为 0，verdict 才能稳定为 learning。

⁶ v9 vs v8 的 MSBN drift 对比是 **v9 解决 OOD trade-off 的关键证据**：v8 max drift 0.0882/0.1389 → v9 max drift **0.2388**/0.1641 (L1 翻 ~2.7×)。v9 80-ep cold start 下 MSBN drift 一度冲到 0.24 后回落到 0.06（last），呈"先大幅分化、后回归温和"轨迹；v8 20-ep finetune 没有时间走完这个弧。详见 [eloftr-v9-e2e SKILL §5(1)](../.cursor/skills/eloftr-v9-e2e/SKILL.md)。

¹⁰ **v10 modemb / MSBN 弧形未走完是 12-ep 不够长的指纹**：
- modemb ir 0.0861 / vis 0.0863 vs v9 终态 0.1077/0.0951 → v10 仍在上升期（ir/vis 几乎对称是 cold-start zero-init + 弱模态差 + 强 homography aug 的预期态）。
- MSBN L1 drift_last=0.1895 vs v9 last=0.0622（高 3×），但 max=0.2045 已接近 v9 max=0.2388。说明 v10 处于 v9 描述的"先大幅分化"上升尾端，**还没开始回归温和**。
- 工程含义：如果以后做 v10.x 在 Megadepth_Syn 上把 max_ep 加到 30-40，理论上 MSBN drift 会沿 v9 弧形继续走完"分化 → 回归"循环，模态适配可能再深一层。但成本太高（每多 10 ep ~14h），且独立 eval 已经 SOTA（综合 1.9020），边际收益不明，**不建议为 ablation 之外的目的延 ep**。详见 [eloftr-v10-msyn SKILL §8.1](../.cursor/skills/eloftr-v10-msyn/SKILL.md)。

---

## §4 训练 val（本表 §2）vs 独立 eval（[eval_summary.md §1](./eval_summary.md)）对照

> 仅 v5..v9 有独立 eval（v1..v4 是 RoadScene-trained，独立 eval 在 `dump/roadscene_eval_v{1..4}_*/`，未纳入 eval_summary 主表）。
>
> 注意：本表 "tb best_epoch P@1" 取自 §2 训练 val 数字；"独立 eval P@1" 取自 [eval_summary.md §1](./eval_summary.md) M3FD test。**eval 选 ckpt 走 PL ModelCheckpoint top-1 by P@3**，所以 eval ckpt 的 epoch 可能与本表 best_epoch（按 P@1）不一致——v9 这点最明显，eval 用的是 ep75 不是 ep52。

| 版本 | tb best ep | tb val P@1 | tb val P@3 | eval ep ⁷ | eval P@1 | eval P@3 | Δ P@1 (eval−tb) | Δ P@3 (eval−tb) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 | 9 | 0.4181 | 0.7891 | 9 | 0.4352 | 0.8072 | +0.0171 | +0.0181 |
| v6 | 6 | 0.4253 | 0.8012 | 6 | 0.4426 | 0.8208 | +0.0173 | +0.0196 |
| v6.1 | 4 | 0.4428 | 0.8143 | 4 | 0.4639 | 0.8318 | +0.0211 | +0.0175 |
| v7 | 6 | 0.4754 | 0.8546 | 6 | 0.4975 | 0.8672 | +0.0221 | +0.0126 |
| v8 | 16 | 0.5196 | 0.8885 | 16 | 0.5494 | 0.9007 | +0.0298 | +0.0122 |
| v9 | 52 | 0.6881 | 0.9753 | 75 ⁴ | 0.6863 | 0.9792 | −0.0018 | +0.0039 |
| **v10** ¹¹ | **11** | **0.8344** | **0.9976** | **11** | **0.5129** | **0.9643** | **−0.3215** | **−0.0333** |

⁷ "eval ep" = `dump/m3fd_eval_v<X>_*/overall.txt` 的 ckpt 文件名 epoch 编号。v5..v8 时 best (P@1) epoch 与 PL 选的 (P@3 top-1) ckpt epoch 巧合一致，v9 不一致（见 §2 注 ⁴）。v10 一致（ep11 同时是 by-P@1 和 by-P@3 的最高）。

¹¹ **v10 Δ P@1 = −0.32 是 evaluator 口径完全不同**（不是漂移、不是模型问题）：
- v5..v9 的训练 val 与独立 eval **同口径** = 都在 M3FD aligned IR-VIS 上 + val 时关 homography aug，仅 RepVGG 模式 / sampler ordering / mask 处理细节微差，所以 Δ 稳定在 +0.012..+0.030（独立 eval 略宽松）。
- **v10 训练 val 是 Megadepth_Syn aligned IR-VIS（pixel-perfect by style-transfer 构造）+ val 关 homography → IR-VIS = identity matching**，任务上限本身就极高（ep0 就 P@3=0.979）；**v10 独立 eval 是 M3FD test 真 LWIR + ~1px 标定残差**，任务难度回到 v9 同水平 → 自然 P@1 大幅下跌（0.83 → 0.51）、P@3 小幅下跌（0.998 → 0.964）。
- **这条 Δ 不是 v9 那种"训练 val ≈ 独立 eval 误差 +0.02 内"的指纹**，而是"训练 val 任务退化到 identity"的副作用。论文 / 答辩时**绝对不要引用 v10 训练 val 数字**，全部以 [`eval_summary.md`](./eval_summary.md) 独立 eval 为准。

**Δ 解读（v5..v9 部分仍成立）**：所有 v5..v8 的独立 eval P@1 比训练 val 高 +0.017..+0.030（独立 eval **更宽松**），P@3 高 +0.012..+0.020。原因是两条管线在 RepVGG 模式 / dataset random ordering / mask 处理细节有微差异（详见 [eloftr-eval-pipeline §1](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)），是固定偏移**不是**漂移；v9 ep75（不是 ep52）独立 eval 反而 P@1 略低 0.002（因为 ep75 训练 val P@1=0.670 比 ep52 P@1=0.6881 低 0.018）—— 这进一步印证 **"独立 eval 比训练 val 高 +0.02 左右"是稳定指纹，不依赖 ckpt**。论文 / 答辩时引用 eval_summary.md 的独立 eval 数字（更接近真实部署效果），引用本表 §2 的 tb 数字解释"训练曲线为什么 ep52 见顶 / ep75 才被 ship"。

---

## §5 关键观察（来自上面四节的事实，不再带分析）

1. **训练成本陡增**：v5 全集 M3FD 后 step/epoch ≈ 945（vs v1..v4 RoadScene ≈ 22-50 step/epoch），单 epoch wall-clock ≈ 23 min（v5 单卡）。v9 80 ep 共 75 550 step，wall=10.96h ≈ 8.2 min/ep（**比 v5 23 min/ep 快近 3 倍**），**主因是 PERSISTENT_WORKERS=True**（v6+ 引入），dataloader worker 不再每 epoch 重启。**v10 step/ep ≈ 7 188（28× v9）+ 4 卡 sync_bn DDP wall ≈ 85.5 min/ep**，4 卡名义 4× 加速被 sync_bn 吃到 ~0.7×，跟 [eloftr-server-multigpu §6.5](../.cursor/skills/eloftr-server-multigpu/SKILL.md) wall-clock 修订一致。
2. **best_epoch 在 cold start vs resume 的位置规律**：cold start (v5/v9/v10) best_epoch 在末段（10/10=100%, 52/80=65%, **11/12=92%**）；resume (v6/v6.1/v7/v8) best_epoch 在前段（6/8, 4/8, 6/10, 16/20，都 ≤ 80%）。v10 进一步验证 cold start 链规律——但 v10 best_epoch 在末尾且 P@1 ep10→ep11 仍涨 +0.007、avg_loss 仍降 4.8% rel，说明 12 ep 偏紧；不延 ep 的判断在 §3 注 ¹⁰ + [eloftr-v10-msyn SKILL §8.1](../.cursor/skills/eloftr-v10-msyn/SKILL.md)。
3. **v6 → v6.1 同 cfg 仅环境差**：tb val P@1 0.4253 → 0.4428 (+4.1% rel)、P@3 0.8012 → 0.8143 (+1.6%)、wall 2.73h → 1.52h (44% 快)。说明 spillover 不仅吃显存，也实质性影响精度收敛。
4. **v8 vs v9 MSBN 训练动态强烈不同**：v8 (20ep, resume v7) MSBN drift L1 max=0.0882，v9 (80ep, cold) max=**0.2388**。v9 给了双 BN 充分时间分化 + 回归到平均值，v8 只够"快速适应 M3FD"。这正对应 v8 OOD trade-off 出现 / v9 trade-off 消失的**训练侧机制证据**。**v10 (12ep, cold, msyn) MSBN drift L1 max=0.2045 接近 v9 max 但 last=0.1895 还在分化上升期**——12 ep 还没走完 v9 那种"分化 → 回归"弧形，但独立 eval 已 SOTA（综合 1.9020），说明 OOD 鲁棒不需要 MSBN 走完整弧形（数据多样性 + 强 homography aug 主导，详见 [eloftr-v10-msyn SKILL §9](../.cursor/skills/eloftr-v10-msyn/SKILL.md)）。
5. **train val ↔ independent eval 系统偏移**：v5..v9 在 M3FD 同口径下 Δ 稳定 +0.012..+0.030（独立 eval 更宽松，固定偏移），是数据管线微差异的指纹；**v10 Δ P@1=−0.32 是不同 evaluator 口径**（训练 val 是 Megadepth_Syn identity 任务、独立 eval 是 M3FD 真 LWIR），跟 v5..v9 的 +0.02 不可比。**所有论文图表只引用独立 eval（eval_summary.md）一份，绝对不引用 v10 训练 val 0.834**。
6. **v3 vs v4**：v3 全冻 + 强 LR 衰减 → P@1 0.31（最差），v4 REVISION 2 释放 fine BN → P@1 0.37 (+21% rel)。这条"freeze 不是越多越好"的教训是 v5+ 全部默认 `FREEZE_BACKBONE_BN=True / 释放 fine BN` 的设计依据；**v10 进一步在 31× 大数据上 unfreeze 全部 backbone BN（v10 cfg `FREEZE_BACKBONE_BN=False`）**，最终在独立 eval 综合通用性上 SOTA → 验证"小数据冻 BN、大数据解冻 BN"是合理 design rule。
7. **v9 train_loss min = 0.1094**（80 ep）vs v8 0.2272（20 ep）vs v5 0.3289（10 ep）vs **v10 0.1208**（12 ep, msyn）：min loss 与 epoch 数 + 数据集难度都相关，v10 12 ep 就达到 v9 80 ep 的 min loss 水平 → 反映 Megadepth_Syn 任务对 outdoor.ckpt 起点更友好（[eloftr-v10-msyn §9 H1](../.cursor/skills/eloftr-v10-msyn/SKILL.md)）。**不能跨版本直接比 loss 数值的绝对意义**——只是各自训练的"内部收敛深度"指标。
8. **v10 独立 eval 数字 vs 训练 val 数字的强解耦**是本表新增的"训练动态 vs 部署效果不一致"案例：训练 val P@1=0.83 看起来比 v9 ep52 0.69 强 +0.14，但实际独立 eval M3FD P@1 v10 0.51 比 v9 ep75 0.69 弱 −0.18。这是教科书级"训练 val saturate / evaluator 口径不一致 / 任务难度差"的失配，毕设论文里值得作为 evaluator 设计的注意点单独写一段（与"独立 eval 系统偏移 +0.02"的稳定指纹形成对比）。

---

## §6 复现命令 / 数据来源

### 6.1 一次性重新生成本表的基础三表

`aggregate` 默认 `--include "^(roadscene|m3fd|msyn)_v\d"`（v10 后扩展加 `msyn` 容纳 Megadepth_Syn 训练；详见 [eloftr-tb-summary SKILL §2.1](../.cursor/skills/eloftr-tb-summary/SKILL.md)）。如果走默认仍然命中所有 v1..v10 实验。

```bash
# 本地 Windows (PowerShell 用绝对路径调 env python，注意 regex 必须单引号包住，否则 PowerShell 把 `|` 当管道)
& "$env:USERPROFILE\miniconda3\envs\eff_loftr\python.exe" `
    MyScripts\read_tb_metrics.py aggregate `
    --include '^(roadscene|m3fd|msyn)_v\d' `
    --out results\tb_summary_raw.md

# 服务器 vlrlab (bash)
conda activate eloftr_yurupeng && \
    python MyScripts/read_tb_metrics.py aggregate \
    --include '^(roadscene|m3fd|msyn)_v\d' \
    --out results/tb_summary_raw.md
```

输出会覆盖 `results/tb_summary_raw.md` 的三张基础表（§1/§2/§3 列与本文一致）。然后人工把 raw 内容贴到本文件 §1/§2/§3 替换对应行（保留 §4/§5/§6 的注释、观察、命令段），最后 `rm results/tb_summary_raw.md`。

### 6.2 单实验深读

任何单个实验的 per-epoch 完整曲线、tag 列表、CSV 导出，用 `summary` / `tags` / `export` 三个子命令，详见 [eloftr-tb-analysis SKILL](../.cursor/skills/eloftr-tb-analysis/SKILL.md)。

### 6.3 选取的 version 列表（aggregate 自动按 highest version_N）

| 版本 | 实验目录 | 选取的 version_N |
|---|---|---:|
| v1 | `logs/tb_logs/roadscene_v1_contrast/` | 2 |
| v2 | `logs/tb_logs/roadscene_v2_modemb/` | 1 |
| v3 | `logs/tb_logs/roadscene_v3_combined/` | 3 |
| v4 | `logs/tb_logs/roadscene_v4_combined/` | 2 |
| v5 | `logs/tb_logs/m3fd_v5_combined/` | 0 |
| v6 | `logs/tb_logs/m3fd_v6_finetune/` | 0 |
| v6.1 | `logs/tb_logs/m3fd_v6_1_finetune/` | 0 |
| v7 | `logs/tb_logs/m3fd_v7_pcclahe/` | 0 |
| v8 | `logs/tb_logs/m3fd_v8_msbn/` | 0 |
| v9 | `logs/tb_logs/m3fd_v9_e2e_outdoor/` | 0 |
| v10 | `logs/tb_logs/msyn_v10_ddp/` | 0 |

> 想强制读老 version：`python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/roadscene_v1_contrast/version_0` 直接指到具体目录，aggregate 子命令暂不支持 per-experiment version override（如果以后需要，扩展 `--include` regex 即可，比如 `^roadscene_v1_contrast/version_0$` 形态）。
>
> `tb_logs/` 整树已 gitignore，本表的数字若与你本机 / 服务器跑出的不同（多次重训会改 `version_N` 编号），以本机最新 aggregate 输出为准。
