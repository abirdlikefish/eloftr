# v9_e2e Cold Start 实验登记 / 结果摘要

> 完整实现细节、机制解释、Q&A 见 [.cursor/skills/eloftr-v9-e2e/SKILL.md](../.cursor/skills/eloftr-v9-e2e/SKILL.md)。
> 横向对照（v7 / v8 / v9）见 [.cursor/skills/eloftr-cross-modal-experiments/SKILL.md §5](../.cursor/skills/eloftr-cross-modal-experiments/SKILL.md)。
> 本文是实验登记册式的简短记录，永久化关键数字 + 时间线 + 决策。

---

## 1. 实验元信息

| 字段 | 值 |
|---|---|
| 实验编号 | v9_e2e |
| exp_name | `m3fd_v9_e2e_outdoor` |
| version | 0 |
| 训练 cfg | [`configs/loftr/eloftr_full_v9_e2e.py`](../configs/loftr/eloftr_full_v9_e2e.py) |
| 训练脚本 | [`MyScripts/run_m3fd_v9_e2e.sh`](../MyScripts/run_m3fd_v9_e2e.sh) |
| 数据 cfg | [`configs/data/m3fd_trainval.py`](../configs/data/m3fd_trainval.py) |
| 起始权重 | `weights/eloftr_outdoor.ckpt`（MegaDepth 户外 baseline，**真冷启动，不从 v_x finetune**）|
| 硬件 | 单卡 RTX 3090（vlrlab 服务器）|
| 训练时长 | **10.96 h**（cfg 注释预估 13.3 h，PERSISTENT_WORKERS 收益 -18%）|
| 训练时间 | 2026-05-06 22:44 → 2026-05-07 09:42 |
| 完成日期 | 2026-05-07 |
| max_epochs | 80（实跑满，无 EarlyStopping 触发）|
| total global steps | 75550 |
| tfevents 文件 | `logs/tb_logs/m3fd_v9_e2e_outdoor/version_0/events.out.tfevents.1778078682.3090.382918.0` (748 MB) |
| ship ckpt | `logs/.../version_0/checkpoints/epoch=75-precision@1px=0.670-precision@3px=0.976-precision@5px=0.993.ckpt` |

## 2. v9 设计动机（一句话）

> 验证两件事：(1) v0→v5→v6→v6.1→v7→v8 的 5 步 finetune 链是否必要？是否可以单次 80-ep cold start 复现 v8 ep16 的 in-domain SOTA？(2) v8 SKILL §12 列为 future work 的 MSBN OOD trade-off (in-domain p@1 +10.4% / OOD p@1 -4.7%) 能否单纯靠训练策略（不引入域不变正则）解决？

## 3. 配置摘要

```python
# configs/loftr/eloftr_full_v9_e2e.py 关键 flag
# Style A 继承: from configs.loftr.eloftr_full import cfg  # 直接继承 v0 baseline, 不继承 v8

cfg.LOFTR.LOSS.USE_CONTRASTIVE = True            # v1
cfg.LOFTR.USE_MODALITY_EMB = True                # v2 (zeros init)
cfg.LOFTR.FREEZE_BACKBONE_BN = True              # v3/v4 (其余 freeze 全关)
cfg.LOFTR.BACKBONE_IN_CHANNELS = 2               # v7 A1
cfg.LOFTR.USE_EDGE_INPUT = True                  # v7 A1
cfg.LOFTR.USE_CLAHE_IR = True                    # v7 A2
cfg.LOFTR.USE_MSBN = True                        # v8 E1
cfg.TRAINER.CANONICAL_LR = 2e-3                  # bs=4 -> TRUE_LR=1.25e-4 (v5 level fast LR, vs v8 = 2.5e-5)
cfg.TRAINER.WARMUP_STEP = 60                     # bs=4 -> actual 960 step ~ 1 ep
cfg.TRAINER.MSLR_MILESTONES = [35, 55, 70]
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 20
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 3780          # v5 sampler full M3FD
```

启动 sanity gate (run_m3fd_v9_e2e.sh L18-24) 全部通过：stage0 inflate 1ch→2ch / MSBN inflate single→dual_BN / modemb missing keys 自动初始化。

## 4. 训练时间线（按 LR schedule 切段）

| 区间 | epoch | TRUE_LR | 实测 p@1 范围 | 备注 |
|---|---|---|---|---|
| warmup | 0–1 | 0 → 1.25e-4 | 0.347 → 0.411 | 起步比 v8 resume 起点 0.55 低（cold start 必然）|
| full LR 早期 | 1–7 | 1.25e-4 | 0.411 → 0.558 | +0.024/ep, MegaDepth-domain → M3FD 适应 |
| full LR 中段 | 7–26 | 1.25e-4 | 0.540 → 0.631 | +0.005/ep, 平台震荡; **ep 26 已追平 v8 ep16 = 0.61** |
| full LR 末段 | 27–34 | 1.25e-4 | 0.617 → 0.628 | 平台震荡 |
| **MSLR #1** | 35–55 | 6.25e-5 | 0.628 → **0.6881** | **质变区段**, best @ **ep 52 = 0.6881** |
| MSLR #2 | 55–70 | 3.13e-5 | 0.661 → 0.677 | 微震荡, 不再创新高 |
| MSLR #3 | 70–80 | 1.56e-5 | 0.660 → 0.673 | 末段稳定, 完全无进步; **ep 79 = 0.6623 比 ep 52 退 -0.026** |

> **关键观察**: 质变红利集中在 MSLR #1, 后两段几乎只是维持 + 缓慢退化。最佳 epoch 是 52 但 PL ckpt monitor=p@3 选了 ep 75; 详见 §7。

## 5. 训练侧诊断（read_tb_metrics.py summary）

| 模块 | 起 → 终 | max | verdict |
|---|---|---|---|
| modemb ir_norm | 0 → 0.1077 | — | learning ✓（>0.05）|
| modemb vis_norm | 0 → 0.0951 | — | learning ✓ |
| MSBN drift layer1 | 0.0031 → 0.0622 | **0.2388** | active ✓（先大幅分化、后回归温和）|
| MSBN drift layer2 | 0.0145 → 0.1133 | 0.1641 | active ✓ |
| train/loss | 1.473 → 0.208 | min 0.109 @ step 67100 | 单调降 |
| train/avg_loss_on_epoch | 0.7272 → 0.1683 | — | 单调降 |
| `read_tb_metrics.py` v9_acceptance grade | — | — | **strong** |

## 6. 独立 eval 结果（毕设 ship final 数字）

### 6.1 in-domain test（M3FD 210 pairs）

| ckpt | p@1 | p@3 | p@5 | mpe | total_matches |
|---|---|---|---|---|---|
| v7 ep6 | 0.4975 | 0.8672 | 0.9207 | 1.59 | 431K |
| v8 ep16 | 0.5494 | 0.9007 | 0.9464 | 1.28 | 444K |
| **v9 ep75** | **0.6863** | **0.9792** | **0.9960** | **0.7283** | **463K** |
| Δ vs v8 | **+24.9% rel** | +8.7% | +5.2% | -43.3% | +4.2% |

p@3 / p@5 已**接近物理天花板**（M3FD 标定残差 ~1 px），mpe 0.73 px **突破** ~1 px 残差。

### 6.2 OOD test（RoadScene 22 pairs）

| ckpt | p@1 | p@3 | p@5 | mpe | total_matches |
|---|---|---|---|---|---|
| v7 ep6 | 0.2124 | 0.6085 | 0.7840 | 3.23 | 31927 |
| v8 ep16 | 0.2025⚠️ | 0.6187 | 0.7966 | 3.12 | ~32000 |
| **v9 ep75** | **0.2128** | **0.7209** | **0.8912** | **2.39** | **33691** |
| Δ vs v8 | **+5.1% rel** ✅ 救回 | **+16.5%** | **+11.9%** | -23.4% | +5.5% |

**OOD p@1 = 0.2128 ≥ v7 ep6 = 0.2124**，**完全消除 v8 的 4.7% OOD 退化**。OOD p@3 = 0.7209 已触及 RoadScene 物理天花板（~60-70%）。

### 6.3 综合通用性（in-domain p@3 + OOD p@3）

| ckpt | M3FD p@3 | OOD p@3 | 综合 | 排名 | Δ vs prev |
|---|---|---|---|---|---|
| v7 ep6 | 0.8672 | 0.6085 | 1.4757 | 4 | +0.0511 |
| v8 ep16 | 0.9007 | 0.6187 | 1.5194 | 3 | +0.0437 |
| **v9 ep75** | **0.9792** | **0.7209** | **1.7001** | **1** | **+0.1807** |

**v9 综合通用性增量 +0.1807 = 历史最大跳跃幅度（v6.1→v7 = +0.0511）的 3.5 倍**。

### 6.4 跨版本指纹（OOD/in 涨幅比 = "真模态不变" 判别）

| 跨版本跳跃 | in-domain p@1 Δ | OOD p@1 Δ | 比例 | 类型 |
|---|---|---|---|---|
| v5 → v6 / v6 → v6.1 | +1.7% / +4.8% | -4.9% / -0.3% | 反向 / 微跌 | 典型 in-domain overfit |
| v6.1 → v7 | +7.2% | +20.5% | +2.85 | **真模态不变（输入端）**|
| v7 → v8 | +10.4% | -4.7% | -0.45 | dataset-specific 适应（fine 架构）|
| **v7 → v9 (cold start)** | **+37.9%** | **+0.2% 持平** | **+0.005 中性** | **训练策略让 MSBN 通用化** |
| v7 → v9 在 p@3 上 | +12.9% | **+18.5%** | +1.43 | **真模态不变指纹复现** |

## 7. PL ckpt monitor 偏差（务必记录）

- 训练 val p@1 全局最佳 epoch = **ep 52, p@1=0.6881**
- PL ModelCheckpoint 实际保存的 top-1 ckpt = **ep 75**（按 `precision@3px` monitor 选, ep75 p@3=0.9758 比 ep52 p@3=0.9753 高 0.0005）
- ES patience=20 没在 ep 72 触发，因为 ES monitor 同样选 p@3，p@3 在 ep52 之后还在微涨永远刷新 patience
- **后果**: 多训了 28 ep（35% 算力浪费）但**不影响最终 ship 数字**（实际选的就是 ep 75 ckpt，已经是后段了）
- **修复方案**: 改 `lightning_loftr.py` 里 ModelCheckpoint 的 monitor 为 `precision@1px` → 留给 v10+
- **当前不重训 v9**: 性价比低（11h 换 ~0.01-0.02 p@1 提升）

## 8. 关键 Finding（论文 discussion 章节用）

**v8 SKILL §12 把 "MSBN + 域不变正则" 列为 future work, v9 实测推翻这一结论**:

> v8 在 v7 ckpt resume 训练 20 ep 时观察到的 MSBN OOD trade-off (in-domain p@1 +10.4% / OOD p@1 -4.7%) 被 v9 80-ep cold start 训练完全消除。这表明 v8 OOD trade-off 的根因不是 MSBN 架构本身, 而是 finetune 链路中 ckpt 的 dataset bias 累积 + schedule 不足导致 MSBN running_stats 未充分收敛到模态不变状态。延长训练 schedule + 从 cross-domain baseline (MegaDepth) 冷启动这两个**训练策略层面**的改动, 无需引入域不变正则即可解决 MSBN 的 OOD trade-off。

机制候选（待 v9.1/v9.2/v9.3 ablation 拆分）:

1. **80 ep 长 schedule + 三段 MSLR**: 让 MSBN drift 经历"先大幅分化 max=0.2388, 后回归温和 last=0.0622"轨迹, dual BN running_stats 收敛到接近"真平均"
2. **Cold start from outdoor.ckpt**: MSBN 从 MegaDepth-domain 单 BN inflate（更 raw, 通用）, 而不是从 v7 ep6 的 M3FD-tuned 单 BN inflate（v8 是这种, 起点已带 dataset bias）
3. **`MODALITY_EMB_INIT='zeros'` + 整网络真冷启动**: modemb 与 MSBN 联合分布更平衡

## 9. Ship 决定 + Follow-up

### Ship 决定

**毕设最终交付从原计划的 v7 切换到 v9_e2e**。理由：
- v9 同时具备 v8 的 in-domain 战斗力（p@1 = 0.69）和 v7 的 OOD 通用性（p@1 = 0.21 持平）
- 综合通用性 #1（v0-v9 全程）
- 论文叙事更强：训练策略层面的改动（不需要新算法）就能解决 v8 的 OOD trade-off

v7 / v8 在论文中保留为 ablation 章节，论证"训练策略 vs 架构改动 vs 输入端处理"的边界。

### Follow-up（按优先级）

| P | 动作 | 目的 |
|---|---|---|
| P0（已完成 ✓） | 写本份登记 + 新建 `eloftr-v9-e2e` SKILL + 更新 cross-modal-experiments / eval-pipeline / tb-analysis SKILL | 永久化 v9 经验 |
| P1 | （可选）跑 `read_tb_metrics.py summary --out logs/.../summary.json` 把 JSON 落盘 | 永久快照 |
| P1 | v9.1/v9.2/v9.3 ablation（§8 三机制拆分）| 论文章节强化 v9 解 v8 OOD trade-off 的 root cause 解释 |
| P2 | 测更多 OOD 数据集（TNO / LLVIP / KAIST）| 验证 v9 通用性 |
| P2 | （可选）改 ckpt monitor 为 p@1 重训 v9 | 拿 ep52 ckpt 测试，估计 in-domain p@1 多 0.01-0.02 |
| P3 | v10 加 path E2 (FiLM) 或 E3 (cosine fine matching) | 在 v9 训练策略基础上榨更高 in-domain SOTA |

---

## 附录 A. v9 训练 80 epoch 关键节点 p@1 速查

| epoch | p@1px | epoch | p@1px | epoch | p@1px | epoch | p@1px |
|---|---|---|---|---|---|---|---|
| 0 | 0.347 | 20 | 0.605 | 40 | 0.638 | 60 | 0.671 |
| 5 | 0.506 | 23 | 0.627 | 45 | 0.652 | 65 | 0.666 |
| 10 | 0.584 | 26 | 0.631 | 46 | 0.671 | 70 | 0.671 |
| 15 | 0.590 | 30 | 0.617 | 50 | 0.677 | 73 | 0.674 |
| 17 | 0.586 | 35 (MSLR#1) | — | **52** | **0.6881** ← best | **75** | **0.670** ← ship ckpt |
| 19 | 0.615 | — | — | 55 (MSLR#2) | — | 79 | 0.662 |

完整 80 epoch 数据通过 `python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/m3fd_v9_e2e_outdoor` 读 JSON 重现。
