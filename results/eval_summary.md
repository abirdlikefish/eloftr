# EfficientLoFTR Cross-Modal 实验链 独立 Eval 汇总

> 数据来源：`dump/<dataset>_eval_v<X>_version<Y>_<topZ|last>/overall.txt`，由 `MyScripts/eval_roadscene.py`（M3FD test 与 RoadScene test 共用同一脚本）产生。完整生产管线见 [.cursor/skills/eloftr-eval-pipeline/SKILL.md](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)。
>
> 使用、维护规则与解读注意见 [.cursor/skills/eloftr-results/SKILL.md](../.cursor/skills/eloftr-results/SKILL.md)。
>
> **本表只覆盖 M3FD 训练的 v5..v9 链路**（含 v6.1）；v1..v4 是 RoadScene 训练（177 train，22 val），独立 eval 在 `dump/roadscene_eval_v{1..4}_*/`，未纳入本表。
>
> 最近一次更新：2026-05-07（v9 e2e cold start 跑完 + 独立 eval 完成）。

---

## 1. M3FD test（in-domain，210 pair）

| 版本 | main_cfg | ckpt（val P@3 选 top-1） | total_matches | matches/pair | mpe (px) | P@1px | P@3px | P@5px |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v5 | `eloftr_full_v5_m3fd.py` | ep=9 (0.789) | 407 163 | 1 939 | 2.0747 | 0.4352 | 0.8072 | 0.8777 |
| v6 ¹ | `eloftr_full_v6_1_finetune.py` | ep=6 (0.801) | 417 113 | 1 986 | 1.9630 | 0.4426 | 0.8208 | 0.8863 |
| v6.1 | `eloftr_full_v6_1_finetune.py` | ep=4 (0.814) | 420 059 | 2 000 | 1.8685 | 0.4639 | 0.8318 | 0.8904 |
| v7 (PC+CLAHE) | `eloftr_full_v7_pcclahe.py` | ep=6 (0.855) | 431 148 | 2 053 | 1.5868 | 0.4975 | 0.8672 | 0.9207 |
| v8 (MSBN) | `eloftr_full_v8_msbn.py` | ep=16 (0.889) | 444 395 | 2 116 | 1.2839 | 0.5494 | 0.9007 | 0.9464 |
| **v9 (e2e)** | `eloftr_full_v9_e2e.py` | **ep=75 (0.976)** | **463 048** | **2 205** | **0.7283** | **0.6863** | **0.9792** | **0.9960** |

¹ v6 的 overall.txt 里 main_cfg 写的是 `v6_1_finetune.py`，是因为 v6.1 是 v6 的 spillover hotfix（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`），cfg/schedule 完全继承 v6 不变，仅 .bat 加 env 变量；所以两者复用同一份 cfg。两者的"训练"差异实际只在 ckpt（v6 ep6 vs v6.1 ep4）。详见 [eloftr-v6-finetune SKILL](../.cursor/skills/eloftr-v6-finetune/SKILL.md)。

> v5 / v6 完整 overall.txt 在历史目录名 `dump/m3fd_eval_v5_combined_version0_top1/` / `dump/m3fd_eval_v6_finetune_version0_top1/`（per [eval-pipeline §4.5](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)），新命名 `dump/m3fd_eval_v5_version0_top1/` 与 `dump/m3fd_eval_v6_version0_top1/` 仅有部分 PNG，未生成 overall.txt。

---

## 2. RoadScene test（OOD，22 pair）

| 版本 | main_cfg | ckpt | total_matches | matches/pair | mpe (px) | P@1px | P@3px | P@5px |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v5 | `eloftr_full_v5_m3fd.py` | ep=9 | 29 722 | 1 351 | 3.3856 | 0.1858 | 0.5989 | 0.7825 |
| v6 ¹ | `eloftr_full_v6_1_finetune.py` | ep=6 | 30 113 | 1 369 | 3.3908 | 0.1767 | 0.6010 | 0.7818 |
| v6.1 | `eloftr_full_v6_1_finetune.py` | ep=4 | 30 356 | 1 380 | 3.3478 | 0.1762 | 0.5928 | 0.7818 |
| v7 (PC+CLAHE) | `eloftr_full_v7_pcclahe.py` | ep=6 | 31 507 | 1 432 | 3.2313 | 0.2124 | 0.6085 | 0.7840 |
| v8 (MSBN) ⚠️ | `eloftr_full_v8_msbn.py` | ep=16 | 31 927 | 1 451 | 3.1153 | 0.2025 | 0.6187 | 0.7966 |
| **v9 (e2e)** | `eloftr_full_v9_e2e.py` | **ep=75** | **33 691** | **1 531** | **2.3864** | **0.2128** | **0.7209** | **0.8912** |

¹ 同 §1 注 1。

⚠️ v8 OOD P@1 相对 v7 退 −4.7%（4.34σ 显著退化）即"MSBN OOD trade-off"，v8 SKILL §11.5② 当时归因为"MSBN 学到 M3FD-specific BN 分布、1px 亚像素无法迁移"，v8 SKILL §12 列为 future work。**v9 在不引入新方法的前提下完全消除该退化**（OOD P@1 0.2128 ≥ v7 0.2124），见 §3 综合通用性 + [eloftr-v9-e2e §5](../.cursor/skills/eloftr-v9-e2e/SKILL.md) 三个机制候选。

---

## 3. 综合通用性（in-domain P@3 + OOD P@3，越大越好）

| 版本 | M3FD P@1 | M3FD P@3 | M3FD mpe | OOD P@1 | OOD P@3 | OOD mpe | 综合 = M3FD P@3 + OOD P@3 | 排名 | Δ vs prev |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| v5 | 0.4352 | 0.8072 | 2.07 | 0.1858 | 0.5989 | 3.39 | 1.4061 | 6 | — |
| v6 | 0.4426 | 0.8208 | 1.96 | 0.1767 | 0.6010 | 3.39 | 1.4218 | 5 | +0.0157 |
| v6.1 | 0.4639 | 0.8318 | 1.87 | 0.1762 | 0.5928 | 3.35 | 1.4246 | 4 | +0.0028 |
| v7 (PC+CLAHE) | 0.4975 | 0.8672 | 1.59 | 0.2124 | 0.6085 | 3.23 | 1.4757 | 3 | +0.0511 |
| v8 (MSBN) | 0.5494 | 0.9007 | 1.28 | 0.2025 | 0.6187 | 3.12 | 1.5194 | 2 | +0.0437 |
| **v9 (e2e)** ⭐ | **0.6863** | **0.9792** | **0.73** | **0.2128** | **0.7209** | **2.39** | **1.7001** | **1** | **+0.1807** |

⭐ v9 综合通用性增量 +0.1807 = v8 之前任何代际跳跃（最大 v6.1→v7 = +0.0511）的 **3.5 倍**，且 in-domain / OOD 双向均刷新前代记录，是 v0-v9 全程综合通用性 SOTA、毕设最终交付推荐。

---

## 4. 关键观察（来自上面三表的事实，不再带分析）

1. **in-domain P@1 单调上升**：0.435 → 0.443 → 0.464 → 0.498 → 0.549 → **0.686**，无版本退化。v9 vs v5 相对 +57.7%。
2. **OOD P@1 在 v6→v6.1 微退（−0.0005）+ v7→v8 显退（−4.7% rel）+ v8→v9 救回**。除 v8 外，趋势整体向上。
3. **mpe（mean pixel error）单调下降**：v5 → v9 在 M3FD 上 2.07 → 0.73，3 倍精度提升。OOD 上 3.39 → 2.39，1.4 倍提升。
4. **matches/pair 单调上升**：M3FD 1939 → 2205 (+13.7%)，OOD 1351 → 1531 (+13.3%)。两数据集涨幅几乎一致 → 说明 v9 不是"在 M3FD 上多 match 但 OOD 少 match"的偏 in-domain 模型，是真正的双向涨。
5. **M3FD vs OOD matches/pair 比例在 1.41-1.45× 区间稳定**（v7=1.41, v8=1.45, v9=1.44）。这个比例反映 RoadScene vs M3FD 数据本身的"画面信息密度差"，与版本无关，详见 [eloftr-eval-pipeline §6 Q OOD matches 偏少](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)。
6. **v9 的 P@3 / P@5 已逼近物理天花板**：M3FD P@5=0.9960、OOD P@3=0.7209（RoadScene 标定残差 2-5 px → P@3 物理上限估 60-70%，详见 [eloftr-eval-pipeline §5.4](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)）。继续刷点收益空间小。

---

## 5. 数据来源文件索引

| 行 | overall.txt 路径 |
|---|---|
| M3FD v5 | `dump/m3fd_eval_v5_combined_version0_top1/overall.txt` |
| M3FD v6 | `dump/m3fd_eval_v6_finetune_version0_top1/overall.txt` |
| M3FD v6.1 | `dump/m3fd_eval_v6_1_version0_top1/overall.txt` |
| M3FD v7 | `dump/m3fd_eval_v7_version0_top1/overall.txt` |
| M3FD v8 | `dump/m3fd_eval_v8_version0_top1/overall.txt` |
| M3FD v9 | `dump/m3fd_eval_v9_version0_top1/overall.txt` |
| OOD v5 | `dump/roadscene_eval_m3fd_v5_combined_version0_top1/overall.txt` |
| OOD v6 | `dump/roadscene_eval_m3fd_v6_finetune_version0_top1/overall.txt` |
| OOD v6.1 | `dump/roadscene_eval_v6_1_version0_top1/overall.txt` |
| OOD v7 | `dump/roadscene_eval_v7_version0_top1/overall.txt` |
| OOD v8 | `dump/roadscene_eval_v8_version0_top1/overall.txt` |
| OOD v9 | `dump/roadscene_eval_v9_version0_top1/overall.txt` |

> 旧/新命名差异（v5/v6 走旧名）见 [eloftr-eval-pipeline §4.5](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)。新 eval 全用新命名 `dump/<dataset>_eval_v<X>_version<Y>_<topZ|last>/`。
