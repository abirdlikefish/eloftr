---
name: eloftr-v15-pose-contrast
description: |
  EfficientLoFTR v15 = v14 + v1 cross-modal symmetric InfoNCE contrastive
  loss. v15 vs v14 是 1 项改动 ablation——cfg.LOFTR.LOSS.USE_CONTRASTIVE
  False -> True, 其他 (IMG_RESIZE=640, NPE 走 train.py:130 fallback
  [832,832,832,832] 跟 v0-v13 一致 -- v14 初版误设 [832,832,640,640] 已修复
  见 v14 cfg "NPE bug post-mortem", bs=4, CANONICAL_LR 5e-4, WARMUP 450,
  MSLR [6,10,14], ES patience 5, N_SAMPLES=100, max_ep=18, EVAL_TIMES=1,
  limit_val_batches=0.2, ENABLE_PLOTTING=False, log_every_n_steps=500)
  全部继承 v14.

  v1 contrastive loss 简介 (复用 v1 contrast 实现, 见 .cursor/skills/eloftr-
  v1-contrast/SKILL.md):
  - transformer 出口对 IR/VIS coarse tokens 做 symmetric InfoNCE
    (loftr.py:129-131 在 use_contrastive=True 时存 feat_c0/c1_tokens)
  - 锚点 = spvs_coarse 提供的 GT match (data['spv_b_ids/i_ids/j_ids'])
  - 负样本 = 整个 batch 跨场景拉平的 B*L 范围 (=25600 tokens at bs=4 +
    6400 coarse cells)
  - L2-normalize 后做点积, loss 跟特征绝对模长无关
  - bs=4 满足 v1 设计的 bs >= 2 硬性要求, 25600 跨场景 negative pool 充足

  默认超参 (default.py:186-188 已声明, v15 cfg 仅 toggle 不改 hyperparams):
  - cfg.LOFTR.LOSS.CONTRASTIVE_WEIGHT = 0.01
  - cfg.LOFTR.LOSS.CONTRASTIVE_TEMP = 0.1
  跟 v1 设计原值完全一致, 让 v15 跟 v1 在两个超参上同步, 仅 dataset path
  跟 v1 (RoadScene H 监督) 不同 (v15 走 Megadepth_Syn pose 监督).

  为什么 v14 加 contrastive 可能有效:
  - v14 已经走 pose 监督 (cross-view + cross-modal), model 已经被强制学
    modality-invariant features.
  - contrastive loss 提供"额外的显式跨模态正则化": 让 cosine similarity 在
    GT match 上最大化 + 在非 match 上最小化.
  - 跟 pose 监督正交: pose loss 用反投影 GT, contrastive 用对应 token 配对.
  - v1 设计初衷是治 dual-softmax focal 饱和; v14 走 pose 监督路径, coarse
    loss 也是基于 conf_matrix 的 focal, 可能受益于 InfoNCE 的"持续给梯度"
    特性 (focal 在概率接近 1 时梯度指数级衰减; InfoNCE 是 (K+1) 路 cross-
    entropy, 只要负样本里出现伪近邻就持续给梯度).

  MegaDepth pose path vs RoadScene H path 兼容性 (v1 原设计是 RoadScene):
  - contrastive loss 实现 (loftr_loss.py:134-180) 仅依赖 data['spv_b_ids/
    i_ids/j_ids'], 这 3 个字段 MegaDepth path 的 spvs_coarse 也会 update
    (src/loftr/utils/supervision.py:125-129). 所以 v15 在 v14 上零代码
    改动直接可用, 仅 cfg toggle.
  - v14 走 MegaDepth pose path, b_ids 来自 mutual NN check + depth_consistent
    双约束, 数量比 RoadScene H path 偏少但充足 (v13 实测 50-200 / pair
    在 spvs_coarse).

  2 个新建文件 (零 src/ 改动):
  (1) configs/loftr/eloftr_full_v15_pose_msyn_ddp.py (~90 行, 继承 v14 ddp
      + 1 行 cfg override: cfg.LOFTR.LOSS.USE_CONTRASTIVE = True; docstring
      含 v1 contrastive 设计原理 + v14 v15 ablation 矩阵 + 监控点)
  (2) MyScripts/run_msyn_v15_pose_ddp.sh (~112 行, 基于 v14 ddp.sh 改 3 处:
      main_cfg / exp_name + 注释更新; data_cfg 复用 v14 的 megadepth_syn_pose_
      640.py)
  
  注: v15 没建 singlecard cfg/sh, 因为 contrastive 行为在 single-card +
  bs=4 也工作良好, smoke 直接复用 v14 singlecard cfg + 临时 toggle (但
  实际 ship 前 smoke 一般用 4 卡 sanity-only mode, 不必建 v15 singlecard).

  Ship 状态 (2026-05-14): NOT YET shipped, plan-only.
  cfg + sh 已建好但未 commit. ship 依赖 v14 ship 完成 + METU eval 拿到 v14
  baseline 数字, 才能跑 v15 跟 v14 比 ablation.

  v14 实测对照基础 (待 v14 ship 完成填; v14 又依赖 v13 ship 完成 + METU
  eval 决策):
    v14 best ep / auc@10 / METU auc@20: TODO (依赖 v14 ship)

  v15 schedule 推算 (sample-pass 跟 v14 完全等效, 仅多 contrastive 项):
    v14 best ep, ES 触发 ep, max_ep 18, MSLR [6,10,14] 全部继承

  理论预期 (METU all auc@20 vs v14):
    加 contrastive 通常 +1-5% relative AUC (cross-modal 文献常见)
    但 v14 已强制 modality-invariant (pose 监督), 边际收益可能 ±1-3%
    净: 大概率 ±0~+5% (跟 RANSAC 噪声同量级)

  工程预期 (wall-clock vs v14):
    显存: feat_c0/c1_tokens [4, 6400, 256] fp32 = 25 MB / batch + autograd
      ~50-75 MB / 卡. 占 v14 ~17 GB / 卡 < 1%.
    速度: contrastive matmul 6400^2 * 4 = 160M FLOP fp32, ~5-10 ms / step.
      v14 baseline ~0.83 s/step, +0.5-1%.
    TB: 多 4 个 scalar (loss_contrast / loss_i2v / loss_v2i / n_pos).
    12+ ep wall-clock: 跟 v14 ~9-10h 基本一致 (+0.5%).
    events.tfevents: 跟 v14 ~220 MB 基本一致.

  Acceptance gate (METU all auc@20 vs v14):
    Strong  : v15 >= v14 + 1.0pp -> 论文 "InfoNCE in pose supervision works"
    Medium  : v15 in [v14 + 0.3, v14 + 1.0)pp -> 写 ablation 表小贡献
    Flat    : v15 in [v14 - 0.3, v14 + 0.3]pp -> contrastive 边际, 论文叙事可省
    Negative: v15 in [v14 - 1.0, v14 - 0.3)pp -> contrastive 跟 pose 监督
              干扰; 调 CONTRASTIVE_WEIGHT 0.01 -> 0.005 重试
    Fail    : v15 < v14 - 1.0pp -> 严重负效应, 回退 v14
    Crash   : NaN loss -> InfoNCE temp 过低 / n_pos 为 0
              (检查 loftr_loss.py:156-158 zero-fallback)

  辅助监控 (TB scalar, ENABLE_PLOTTING=False 仍然保留 scalar 不影响):
    train/loss_contrast 应单调下降 (cross-modal 对齐学到了)
    train/n_pos 应 >= 30 / batch (足够 InfoNCE 信号; v13 实测 b_ids ~50-200
      / pair, bs=4 -> n_pos ~200-800 / batch)
    train/loss_i2v vs train/loss_v2i 应接近 (symmetric); 长期不对称暗示
      模态偏向 (image0=IR vs image1=VIS 学到不对称信号)

  Use when running/debugging/extending v15 / 解释 v15 vs v14 ablation /
  contrastive loss 在 pose 监督下的兼容性 / v1 contrastive loss 跨数据集
  (RoadScene H -> Megadepth_Syn pose) 复用 / 监控 train/loss_contrast 单
  调下降 / n_pos 不足解决 / InfoNCE temp 调参 / CONTRASTIVE_WEIGHT 调参 /
  写论文 contrastive ablation 章节.

  Triggers: v15 / v15_pose / v15_pose_contrast / msyn_v15_pose_ddp /
  USE_CONTRASTIVE True / cross-modal symmetric InfoNCE / v1 contrastive
  in pose path / contrastive in MegaDepth pose / spv_b_ids / feat_c0_tokens /
  feat_c1_tokens / CONTRASTIVE_WEIGHT 0.01 / CONTRASTIVE_TEMP 0.1 / v1
  default 复用 / loftr_loss.py:134-180 zero-fallback / supervision.py:125-129
  spv_b_ids update / dual-softmax focal 饱和治理 / cross-modal regularizer /
  正交于 pose 监督 / v14 baseline + 1 项 toggle / loss_i2v vs loss_v2i 对称
  / n_pos >= 30 / cross-modal 文献常见 +1-5% / GPU 散热 v15 复用 v14 风险,
  English 'v14 plus contrastive', 'cross-modal symmetric InfoNCE in pose
  supervision', 'reuse v1 contrastive loss across datasets', 'orthogonal
  to pose supervision', 'one-line cfg toggle USE_CONTRASTIVE'.

  v15 = path J 第二级衍生 (v13 -> v14 -> v15 链), 在 v14 工程加速框架上
  叠加 v1 contrastive loss. 测试 "InfoNCE 跟 pose 监督是否正交可叠加".
  Inheritance graph:
    eloftr_full.py (v0 baseline)
      -> v13 (B-pure pose 监督, 832, bs=2, eloftr-v13-pose)
        -> v14 (640 + 工程加速, bs=4, eloftr-v14-resolution-640)
          -> v15 (THIS skill, + USE_CONTRASTIVE=True)

  Companion:
    eloftr-v14-resolution-640 (parent / v15 acceptance gate 起点 = v14
      实测数字, 待 v14 ship 完成填)
    eloftr-v13-pose (祖父 / v15 间接继承的 B-pure stack)
    eloftr-v1-contrast (v1 contrastive loss 实现 + 设计原理 + RoadScene
      H 监督下的 ablation 数字)
    eloftr-megadepth-syn-data (Megadepth_Syn dataset SOP)
    eloftr-cross-modal-experiments (chain 路线总览)
    eloftr-server-multigpu (DDP runtime + 5 traps)
    eloftr-eval-pipeline (eval_metu_vistir_finetuned.bat 协议)
    eloftr-results (跨版本数字总表; v15 数字待 backfill)
    eloftr-tb-summary (训练侧 KPI 总表; v15 wall-clock / best_ep /
      loss_contrast 曲线 待 backfill)
---

# v15: v14 + v1 Cross-Modal Symmetric InfoNCE Contrastive Loss (Plan-Only)

> 继承链：`eloftr_full.py` → **`v13` (path J B-pure)** → **`v14` (640 + 工程加速)** → **`v15` (THIS, + InfoNCE)**
> 父对照 → [eloftr-v14-resolution-640](../eloftr-v14-resolution-640/SKILL.md)
> contrastive 实现 → [eloftr-v1-contrast](../eloftr-v1-contrast/SKILL.md)
> 跨版本数字 → [eloftr-results](../eloftr-results/SKILL.md) → [`results/eval_summary.md`](../../../results/eval_summary.md)
> **状态**：plan-only, ship pending. 依赖 v14 ship 完成 + METU eval 拿到 baseline 数字, 才能跟 v15 比 ablation.

## 1. 设计意图：InfoNCE 跟 pose 监督是否正交可叠加？

v1 在 RoadScene H 监督上加 contrastive loss 拿到 +X% AUC (见 [eloftr-v1-contrast](../eloftr-v1-contrast/SKILL.md))。但 v14 已经走 pose 监督 (model 已经被强制学 modality-invariant features), contrastive loss 是不是冗余？

v15 想回答：

> **加 contrastive loss (一个显式的跨模态正则化项) 是否在 pose 监督路径下仍有边际收益？**

3 个可能的答案：
- **可叠加正交**: contrastive 跟 pose 监督正交 (用对应 token 配对 vs 反投影 GT), 各自补对方梯度信号 -> v15 > v14
- **冗余饱和**: pose 监督已经把 modality 学透了, contrastive 边际收益 0 -> v15 ≈ v14
- **干扰**: contrastive 跟 pose 监督学到的"哪些 token 该相似"不一致, 互相抵消 -> v15 < v14

## 2. 2 个新建文件 (零 src/ 改动) + cfg 继承链

| 文件 | 关键设计 |
|---|---|
| [`configs/loftr/eloftr_full_v15_pose_msyn_ddp.py`](../../../configs/loftr/eloftr_full_v15_pose_msyn_ddp.py) | 90 行。继承 v14 ddp + **1 行 cfg override**: `cfg.LOFTR.LOSS.USE_CONTRASTIVE = True`. docstring 含 v1 contrastive 设计原理 + v14 v15 ablation 矩阵 + 监控点 + acceptance gate. |
| [`MyScripts/run_msyn_v15_pose_ddp.sh`](../../../MyScripts/run_msyn_v15_pose_ddp.sh) | 112 行。基于 v14 ddp.sh **改 3 处**: main_cfg / exp_name + 注释更新; data_cfg 复用 v14 的 [megadepth_syn_pose_640.py](../../../configs/data/megadepth_syn_pose_640.py). |

**0 src/ 改动**。继承链:

```
eloftr_full.py (v0 baseline)
  └─ v13_pose_msyn_ddp (B-pure, 832, bs=2)
       └─ v14_pose_msyn_ddp (8 项 overrides, 640, bs=4)
            └─ v15_pose_msyn_ddp (THIS, + 1 行 USE_CONTRASTIVE=True)
```

注: v15 没建 singlecard cfg/sh, 因为 contrastive 行为在 single-card + bs=4 也工作, smoke 一般用 4 卡 sanity-only mode 不必建独立 singlecard.

## 3. 唯一改动 + v1 默认超参复用

```python
# eloftr_full_v15_pose_msyn_ddp.py 唯一改动 (相对 v14 ddp)
cfg.LOFTR.LOSS.USE_CONTRASTIVE = True
```

CONTRASTIVE_WEIGHT / CONTRASTIVE_TEMP 走 [`src/config/default.py:186-188`](../../../src/config/default.py) 默认值:
- `CONTRASTIVE_WEIGHT = 0.01`
- `CONTRASTIVE_TEMP = 0.1`

跟 v1 原值完全一致 (v1 设计目的是治 dual-softmax focal 饱和)。让 v15 跟 v1 在两个超参上完全同步, 仅 dataset path 跟 v1 (RoadScene H 监督) 不同 (v15 走 Megadepth_Syn pose 监督)。

## 4. v1 contrastive loss 兼容 MegaDepth pose path

**关键代码点** ([`src/losses/loftr_loss.py:134-180`](../../../src/losses/loftr_loss.py)):

```python
# v1 contrastive 实现仅依赖 data['spv_b_ids/i_ids/j_ids']
b_ids = data['spv_b_ids']
i_ids = data['spv_i_ids']
j_ids = data['spv_j_ids']
# ... InfoNCE on (B*L, D) flatten + L2-normalize + cosine sim ...
```

这 3 个字段 MegaDepth path 的 [`spvs_coarse`](../../../src/loftr/utils/supervision.py) 也会 update ([L125-129](../../../src/loftr/utils/supervision.py)):

```python
data.update({
    'spv_b_ids': b_ids,
    'spv_i_ids': i_ids,
    'spv_j_ids': j_ids
})
```

**所以 v15 在 v14 上零代码改动直接可用**, 仅 cfg toggle。

v14 走 MegaDepth pose path, `b_ids` 来自 spvs_coarse 的 mutual NN check + depth_consistent 双约束 ([supervision.py:99-102](../../../src/loftr/utils/supervision.py))。数量比 RoadScene H path 偏少但充足 (v13 实测 50-200 GT match / pair, bs=4 -> n_pos 200-800 / batch, 足够 InfoNCE 信号)。

## 5. 工程开销 (相对 v14 几乎可忽略)

| 项 | v14 baseline | v15 增量 | v15 总 |
|---|---|---|---|
| 显存 / 卡 | ~17 GB | feat_c0/c1_tokens [4, 6400, 256] fp32 = 25 MB/batch + autograd ~50-75 MB | **~17.05 GB** (+0.4%) |
| step time | ~0.83 s/step | contrastive matmul 6400² × 4 = 160M FLOP fp32 ~5-10 ms / step | **~0.84 s/step** (+0.5-1%) |
| TB scalar | 4 个 (loss/auc) | +4 个 (loss_contrast / loss_i2v / loss_v2i / n_pos) | 8 个 |
| events.tfevents | ~220 MB | scalar 增量可忽略 | **~220 MB** |
| ep wall-clock | ~33 min | +0.5% | ~33 min |
| 12-18 ep ship | ~9-10 h | +0.5% | **~9-10 h** |

## 6. acceptance gate (待 v14 ship 完成 + METU eval 拿到 baseline 数字后填)

| 档位 | METU all auc@20 vs v14 | 解读 |
|---|---|---|
| Strong | ≥ v14 + 1.0pp | 论文 "InfoNCE in pose supervision works", InfoNCE 跟 pose 监督正交可叠加 |
| Medium | in [v14 + 0.3, v14 + 1.0)pp | 写 ablation 表小贡献 |
| Flat | in [v14 − 0.3, v14 + 0.3]pp | contrastive 边际, 论文叙事可省略此 ablation |
| Negative | in [v14 − 1.0, v14 − 0.3)pp | contrastive 跟 pose 监督干扰, 调 CONTRASTIVE_WEIGHT 0.01 → 0.005 重试 |
| Fail | < v14 − 1.0pp | 严重负效应, 回退 v14 |
| Crash | NaN loss | InfoNCE temp 过低 / n_pos 为 0; 检查 [`loftr_loss.py:156-158`](../../../src/losses/loftr_loss.py) zero-fallback |

## 7. 训练时关键监控点 (TB scalar)

ENABLE_PLOTTING=False 关掉 train figure, 但**所有 scalar 仍保留**, v15 多出的 4 个 scalar 都能在 TB 看到:

| scalar | 健康判据 | 病理信号 |
|---|---|---|
| `train/loss_contrast` | 单调下降 (cross-modal 对齐学到了) | 一直 plateau / 升 -> contrastive 跟主 loss 干扰 |
| `train/loss_i2v` vs `train/loss_v2i` | 接近 (symmetric InfoNCE 双向均衡) | 长期不对称 -> 模态偏向 (image0=IR vs image1=VIS 学到不对称信号) |
| `train/n_pos` | ≥ 30 / batch (够 InfoNCE 信号) | < 10 / batch -> spv_b_ids 不足, contrastive 退化, 检查 spvs_coarse 输出 |

## 8. ship 时序 (依赖 v14 完成)

```mermaid
flowchart LR
  v13["v13 ship 实测<br/>9 ep, ES 即将触发<br/>best ep6 auc@10=0.4096"]
  v13 --> v13Eval["v13 METU eval<br/>(eval_metu_vistir_finetuned.bat 13)"]
  v13Eval --> v14Decide["v14 跑 vs 跳过"]
  
  v14Decide -->|跑| v14Ship["v14 ship<br/>~9h"]
  v14Ship --> v14Eval["v14 METU eval<br/>(eval_metu_vistir_finetuned.bat 14)"]
  v14Eval --> v15Decide["v15 跑 vs 跳过<br/>(看 v14 数字 strong/medium 才上 v15)"]
  
  v15Decide -->|跑| v15Ship["v15 ship<br/>~9-10h"]
  v15Ship --> v15Eval["v15 METU eval<br/>vs v14 比 ablation"]
```

**推荐流程**：
1. **现在 (本 skill)**: cfg + sh 已建好, plan-only 状态, 等 v14 ship 完成
2. **v14 ship 完成 + METU eval 后**: 看 v14 数字
   - v14 数字 strong/medium → v15 跑确认 contrastive 是否正交可叠加
   - v14 数字 weak/fail → 跳过 v15, 反思 v14 设计
3. **v15 ship 完成后**: backfill 数字到 [eloftr-results](../eloftr-results/SKILL.md) + 本 skill description

## 9. v15 ship 后的 backfill 工作

```
python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/msyn_v15_pose_ddp/version_<N>
```

backfill 进:
- 本 skill description 的 "Ship 状态" + 实测曲线
- [eloftr-results](../eloftr-results/SKILL.md) -> [`results/eval_summary.md`](../../../results/eval_summary.md)
- [eloftr-tb-summary](../eloftr-tb-summary/SKILL.md) -> [`results/tb_summary.md`](../../../results/tb_summary.md)
- 关键: 对比 v14 vs v15 在 METU all auc@20 上的差异, 写论文 contrastive ablation 章节
