# EfficientLoFTR Cross-Modal 实验链 独立 Eval 汇总

> 数据来源：`dump/<dataset>_eval_v<X>_version<Y>_<topZ|last>/overall.txt`，由 `MyScripts/eval_roadscene.py`（M3FD test 与 RoadScene test 共用同一脚本）产生。完整生产管线见 [.cursor/skills/eloftr-eval-pipeline/SKILL.md](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)。
>
> 使用、维护规则与解读注意见 [.cursor/skills/eloftr-results/SKILL.md](../.cursor/skills/eloftr-results/SKILL.md)。
>
> **本表覆盖 M3FD 训练的 v5..v9 链路 + Megadepth_Syn 训练的 v10**（v6.1 同 v6 cfg / v10 训练集换源，详见各行注释）；v1..v4 是 RoadScene 训练（177 train，22 val），独立 eval 在 `dump/roadscene_eval_v{1..4}_*/`，未纳入本表。
>
> **重要口径提示**：v5..v9 在 M3FD 上训练，因此 §1 M3FD test 对它们是 **in-domain**；v10 在 Megadepth_Syn 上训练，**没看过 M3FD 一帧**，所以 §1 M3FD test 对 v10 是 **OOD**。RoadScene 22 pair 对所有版本都是 OOD。看综合通用性时这一点很关键（详见 §3 + §4 观察 7）。
>
> 最近一次更新：2026-05-08（v10 Megadepth_Syn 4 卡 DDP ship 完成 + 双数据集独立 eval 跑完，新综合通用性 SOTA = 1.9020 vs v9 1.7001）。

---

## 1. M3FD test（v5..v9 in-domain / **v10 OOD**，210 pair）

| 版本 | main_cfg | ckpt（val P@3 选 top-1） | total_matches | matches/pair | mpe (px) | P@1px | P@3px | P@5px |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v5 | `eloftr_full_v5_m3fd.py` | ep=9 (0.789) | 407 163 | 1 939 | 2.0747 | 0.4352 | 0.8072 | 0.8777 |
| v6 ¹ | `eloftr_full_v6_1_finetune.py` | ep=6 (0.801) | 417 113 | 1 986 | 1.9630 | 0.4426 | 0.8208 | 0.8863 |
| v6.1 | `eloftr_full_v6_1_finetune.py` | ep=4 (0.814) | 420 059 | 2 000 | 1.8685 | 0.4639 | 0.8318 | 0.8904 |
| v7 (PC+CLAHE) | `eloftr_full_v7_pcclahe.py` | ep=6 (0.855) | 431 148 | 2 053 | 1.5868 | 0.4975 | 0.8672 | 0.9207 |
| v8 (MSBN) | `eloftr_full_v8_msbn.py` | ep=16 (0.889) | 444 395 | 2 116 | 1.2839 | 0.5494 | 0.9007 | 0.9464 |
| **v9 (e2e)** | `eloftr_full_v9_e2e.py` | **ep=75 (0.976)** | **463 048** | **2 205** | **0.7283** | **0.6863** | **0.9792** | **0.9960** |
| **v10 (msyn)** ² | `eloftr_full_v10_msyn_ddp.py` | **ep=11 (0.998)** ³ | 462 667 | 2 203 | 1.0394 | 0.5129 | 0.9643 | 0.9927 |

¹ v6 的 overall.txt 里 main_cfg 写的是 `v6_1_finetune.py`，是因为 v6.1 是 v6 的 spillover hotfix（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`），cfg/schedule 完全继承 v6 不变，仅 .bat 加 env 变量；所以两者复用同一份 cfg。两者的"训练"差异实际只在 ckpt（v6 ep6 vs v6.1 ep4）。详见 [eloftr-v6-finetune SKILL](../.cursor/skills/eloftr-v6-finetune/SKILL.md)。

² **v10 对 M3FD 是 OOD**：v10 训练集是 Megadepth_Syn（195 个 MegaDepth scene + style-transfer 合成 IR，~106K train pair），**0 ep 看过 M3FD**。即便如此，v10 在 M3FD 上的 P@1=0.5129 仍超过早期 in-domain 训练的 v5/v6/v6.1（0.435/0.443/0.464），P@3=0.9643 与 v9（0.9792）仅差 0.015，P@5=0.9927 与 v9（0.9960）几乎持平 → 宏观结构匹配能力跨数据集稳定，仅 sub-pixel 不如 v9 ep75。详见 [eloftr-v10-msyn SKILL §8](../.cursor/skills/eloftr-v10-msyn/SKILL.md)。

³ v10 ckpt 文件名末段 `precision@3px=0.998` 对应训练 val 数字。注意 v10 训练 val 是 Megadepth_Syn val（pixel-aligned by construction + val 时 homography aug 关闭 → val 任务退化为 identity matching，详见 [eloftr-v10-msyn SKILL §8.1](../.cursor/skills/eloftr-v10-msyn/SKILL.md)），所以 ckpt 文件名上的 0.998 不能直接跟 v9 ep75 文件名上的 0.976 比；本表用的是真独立 eval 数字（0.9643）。

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
| v9 (e2e) | `eloftr_full_v9_e2e.py` | ep=75 | 33 691 | 1 531 | 2.3864 | 0.2128 | 0.7209 | 0.8912 |
| **v10 (msyn)** ⚡ | `eloftr_full_v10_msyn_ddp.py` | **ep=11** | 35 005 | 1 591 | **1.2004** | **0.4560** | **0.9377** | **0.9917** |

¹ 同 §1 注 1。

⚠️ v8 OOD P@1 相对 v7 退 −4.7%（4.34σ 显著退化）即"MSBN OOD trade-off"，v8 SKILL §11.5② 当时归因为"MSBN 学到 M3FD-specific BN 分布、1px 亚像素无法迁移"，v8 SKILL §12 列为 future work。**v9 在不引入新方法的前提下完全消除该退化**（OOD P@1 0.2128 ≥ v7 0.2124），见 §3 综合通用性 + [eloftr-v9-e2e §5](../.cursor/skills/eloftr-v9-e2e/SKILL.md) 三个机制候选。

⚡ **v10 在 RoadScene OOD 上的飞跃是 v0..v10 全程最大代际突破**：P@1 0.2128 → **0.4560**（+114% rel, +0.243 abs），P@3 0.7209 → **0.9377**（+30% rel, +0.217 abs），mpe 2.39 → **1.20** px（−50%）。这不是渐进改进而是质变；而且 v10 训练数据完全没有 RoadScene 一帧（M3FD 也没有），所以这是真正"训练分布外"的泛化。机制猜测见 [eloftr-v10-msyn SKILL §9](../.cursor/skills/eloftr-v10-msyn/SKILL.md)：Megadepth_Syn 195 scene 多样性 + 强 homography aug → 学到的是 viewpoint-invariant 几何特征而非 M3FD-specific BN 分布。

---

## 3. 综合通用性（M3FD P@3 + RoadScene OOD P@3，越大越好）

> 注意：v5..v9 是 in-domain (M3FD-trained) + OOD (RoadScene)；**v10 是 OOD + OOD**（训练集 = Megadepth_Syn，对 M3FD 也没看过）。所以 v10 这一行的"综合"在严格意义上比 v5..v9 那行更"双向 OOD"，泛化深度更高。

| 版本 | 训练集 | M3FD P@1 | M3FD P@3 | M3FD mpe | OOD P@1 | OOD P@3 | OOD mpe | 综合 = M3FD P@3 + OOD P@3 | 排名 | Δ vs prev |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| v5 | M3FD | 0.4352 | 0.8072 | 2.07 | 0.1858 | 0.5989 | 3.39 | 1.4061 | 7 | — |
| v6 | M3FD | 0.4426 | 0.8208 | 1.96 | 0.1767 | 0.6010 | 3.39 | 1.4218 | 6 | +0.0157 |
| v6.1 | M3FD | 0.4639 | 0.8318 | 1.87 | 0.1762 | 0.5928 | 3.35 | 1.4246 | 5 | +0.0028 |
| v7 (PC+CLAHE) | M3FD | 0.4975 | 0.8672 | 1.59 | 0.2124 | 0.6085 | 3.23 | 1.4757 | 4 | +0.0511 |
| v8 (MSBN) | M3FD | 0.5494 | 0.9007 | 1.28 | 0.2025 | 0.6187 | 3.12 | 1.5194 | 3 | +0.0437 |
| v9 (e2e) | M3FD | 0.6863 | 0.9792 | 0.73 | 0.2128 | 0.7209 | 2.39 | 1.7001 | 2 | +0.1807 |
| **v10 (msyn)** ⭐ | **Megadepth_Syn** | 0.5129 | 0.9643 | 1.04 | **0.4560** | **0.9377** | **1.20** | **1.9020** | **1** | **+0.2019** |

⭐ **v10 是新的综合通用性 SOTA**：综合 P@3 = 1.9020，**比 v9 高 +0.2019**（甚至比 v8→v9 跳跃 +0.1807 还高 +12%）。关键证据：

1. **in-domain ↔ OOD gap 从 v5..v9 的 0.21–0.28 区间砸到 0.027**（v10: 0.9643 − 0.9377 = 0.027 vs v9: 0.9792 − 0.7209 = 0.2583）→ v10 学到的是真跨数据集稳定的几何特征，不是 dataset-specific hack。
2. **OOD P@1 翻倍 + mpe 砍半** = 真泛化能力跃迁，不是刷指标。
3. **v10 在 M3FD 上没看过一帧也能 P@1=0.51**，超过早期 in-domain 训练的 v5/v6/v6.1。
4. **代价**：v10 在 M3FD 上 P@1 落后 v9 ep75 0.173，sub-pixel 精度（mpe）落后 0.31 px；这是用 in-domain 极致换 OOD 鲁棒的合理 trade-off。

毕设最终交付建议：**v9 / v10 双轨**——v9 = M3FD 在线监控专用（in-domain SOTA），v10 = 通用跨模态部署（综合通用性 SOTA + 真 OOD 鲁棒）。

---

## 4. 关键观察（来自上面三表的事实，不再带分析）

1. **v5..v9 链路 in-domain P@1 单调上升**：0.435 → 0.443 → 0.464 → 0.498 → 0.549 → **0.686**，无版本退化。v9 vs v5 相对 +57.7%。**v10 因换训练集回落到 0.513**，跟 v7 持平（v10 对 M3FD 是 OOD）。
2. **OOD P@1 趋势**：v6→v6.1 微退（−0.0005）+ v7→v8 显退（−4.7% rel）+ v8→v9 救回 + **v9→v10 翻倍**（0.2128 → 0.4560，+114% rel）。v10 是历史最大 OOD 跳跃。
3. **mpe（mean pixel error）单调下降**：v5 → v9 在 M3FD 上 2.07 → 0.73（3 倍提升），v9 → v10 退到 1.04（v10 对 M3FD OOD，sub-pixel 不如 v9 ep75）。OOD 上 3.39 → v9 2.39 → **v10 1.20**（v10 在 OOD 上反而最锐，3 倍提升）。
4. **matches/pair 单调上升**：M3FD 1939 → v9 2205 (+13.7%) → v10 2203（持平 v9）。OOD 1351 → v9 1531 (+13.3%) → **v10 1591**（继续涨 +3.9% vs v9）。v10 在两数据集 matches/pair 都没掉。
5. **M3FD vs OOD matches/pair 比例在 1.39-1.45× 区间稳定**（v7=1.41, v8=1.45, v9=1.44, v10=1.39）。这个比例反映 RoadScene vs M3FD 数据本身的"画面信息密度差"，与版本无关，详见 [eloftr-eval-pipeline §6 Q OOD matches 偏少](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)。
6. **P@3 / P@5 物理天花板**：v9 M3FD P@5=0.9960 已贴上限；v10 OOD P@3=**0.9377** 直接突破"RoadScene 标定残差 2-5 px → P@3 物理上限估 60-70%"的旧上限（详见 [eloftr-eval-pipeline §5.4](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)），需要重新核校 RoadScene 标定残差到底是不是 2-5 px 量级，或是"v10 学到了亚像素跨数据集稳定的几何先验"打破估算前提。
7. **in-domain ↔ OOD gap (M3FD P@3 − OOD P@3) 从 v5..v9 0.21..0.28 区间塌到 v10 0.027**：

   | 版本 | M3FD P@3 | OOD P@3 | gap |
   |---|---:|---:|---:|
   | v5 | 0.8072 | 0.5989 | 0.2083 |
   | v6 | 0.8208 | 0.6010 | 0.2198 |
   | v6.1 | 0.8318 | 0.5928 | 0.2390 |
   | v7 | 0.8672 | 0.6085 | 0.2587 |
   | v8 | 0.9007 | 0.6187 | 0.2820 |
   | v9 | 0.9792 | 0.7209 | 0.2583 |
   | **v10** | **0.9643** | **0.9377** | **0.0266** |

   这是 v10 在综合通用性上的核心证据：**几乎消除了 in-domain / OOD 的鸿沟**，而不是把 in-domain 数字推得更高。
8. **v10 训练 val P@3 / P@5 接近饱和（0.998 / 0.999）但独立 eval 数字明显低于这两者**：原因是 Megadepth_Syn 训练 val 的 IR-VIS 是 pixel-aligned by construction（style-transfer 产物）+ val 时 homography aug 关掉（[`src/lightning/data.py` L288](../src/lightning/data.py)：`homography_aug=(... and mode == 'train')`），val 任务退化为 identity matching。**论文 / 答辩永远引用本表（独立 eval）数字，不要引用 ckpt 文件名上的 0.998**。详见 [eloftr-v10-msyn SKILL §8.1 注](../.cursor/skills/eloftr-v10-msyn/SKILL.md)。

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
| M3FD v10 | `dump/m3fd_eval_v10_version0_top1/overall.txt` |
| OOD v10 | `dump/roadscene_eval_v10_version0_top1/overall.txt` |

> 旧/新命名差异（v5/v6 走旧名）见 [eloftr-eval-pipeline §4.5](../.cursor/skills/eloftr-eval-pipeline/SKILL.md)。新 eval 全用新命名 `dump/<dataset>_eval_v<X>_version<Y>_<topZ|last>/`。
