---
name: eloftr-v1-contrast
description: EfficientLoFTR v1 cross-modal experiment - symmetric InfoNCE contrastive loss on coarse features for IR/VIS matching. Use when user mentions v1, v1_contrast, eloftr_full_v1_contrast, USE_CONTRASTIVE, CONTRASTIVE_WEIGHT, CONTRASTIVE_TEMP, _compute_contrastive_loss, loss_i2v / loss_v2i / loss_contrast / n_pos, run_roadscene_v1_*.bat, symmetric InfoNCE, dual-softmax focal saturation, "bs=1 时 InfoNCE 退化", or wants to debug / extend v1. Triggers: v1 / v1_contrast / 对比损失 / 跨模态对比 / InfoNCE / 对比学习 / 实现对比损失 / 加 contrastive / 看 v1 / 跑 v1 / v1 实现细节, English 'add contrastive loss', 'InfoNCE for IR-VIS', 'symmetric contrastive', 'why InfoNCE not focal', 'L2-normalize before dot product', 'why bs >= 2 for v1'. v1 is the first link of v1->v2->v3->v4->v5->v6->v6.1->v7 inheritance chain; see eloftr-cross-modal-experiments for chain overview.
---

# v1：Symmetric InfoNCE Cross-Modal Contrastive Loss

> 继承链：v0 (baseline) → **v1 (contrast)** → v2 (modemb) → v3 (freeze stack) → v4 (REVISION 2) → v5 (M3FD) → v6 / v6.1 (resume) → v7 (PC + CLAHE)
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
> 评估见 [eloftr-eval-pipeline](../eloftr-eval-pipeline/SKILL.md)

## 配置

[configs/loftr/eloftr_full_v1_contrast.py](../../../configs/loftr/eloftr_full_v1_contrast.py)：

```python
from configs.loftr.eloftr_full import cfg
cfg.LOFTR.LOSS.USE_CONTRASTIVE    = True
cfg.LOFTR.LOSS.CONTRASTIVE_WEIGHT = 0.01
cfg.LOFTR.LOSS.CONTRASTIVE_TEMP   = 0.1
```

**Default 中的开关**（[src/config/default.py:107-109](../../../src/config/default.py)）：默认 `USE_CONTRASTIVE = False`，所以 baseline / 旧实验完全不受影响。

## 为什么是 InfoNCE 而不是继续调 dual-softmax focal

- focal 在概率接近 1 时梯度指数级衰减，跨模态相似度始终偏低反而很快饱和；InfoNCE 是 (K+1) 路 cross-entropy，只要负样本里出现伪近邻就持续给梯度。
- dual-softmax 隐含 "互为最近邻才算正" 的硬约束，IR 中信息密度低的格子很难单向匹配；symmetric InfoNCE 把方向拆成两个独立的 (K+1) 分类，i2v 和 v2i 各自约束。
- L2-normalize 后做点积让 loss 与特征绝对模长无关，避免下游 v2 的 modemb 被"通过放大模长降 loss"的捷径抢走优化方向。

## 实现要点

1. [src/loftr/loftr.py:125-131](../../../src/loftr/loftr.py)：transformer 出口处把 `feat_c0 / feat_c1` 同时存到 `data` dict（**只在 `use_contrastive=True` 时存**，避免 baseline 多占显存），保留 autograd 图，让对比 loss 能反传到 transformer + backbone。
2. [src/losses/loftr_loss.py:134-180](../../../src/losses/loftr_loss.py) `_compute_contrastive_loss`：用 `data['spv_b_ids/i_ids/j_ids']`（粗匹配 GT）取出锚点，对 L2-norm 后的特征做 i→v 与 v→i 两个方向的 InfoNCE，负样本池是整个 batch 拉平的所有 token (`B*L`)。
3. [src/losses/loftr_loss.py:277-286](../../../src/losses/loftr_loss.py) 在 `forward` 里加权求和：
   `loss = loss + CONTRASTIVE_WEIGHT * (loss_i2v + loss_v2i) / 2`，并 log `loss_contrast / loss_i2v / loss_v2i / n_pos`。

## Bat 入口

- [`MyScripts/run_roadscene_v1_debug.bat`](../../../MyScripts/run_roadscene_v1_debug.bat) / [`_small.bat`](../../../MyScripts/run_roadscene_v1_small.bat) / [`_contrast.bat`](../../../MyScripts/run_roadscene_v1_contrast.bat)
- **`bs >= 2` 是硬性要求**：`bs=1` 时 InfoNCE 的负样本池退化为 in-image，等价于 dual-softmax，对比损失等于白加。

## TensorBoard 监控

训练日志会出现：
- `train/loss_contrast`：对比损失数值
- `train/loss_i2v` / `train/loss_v2i`：两个方向各自损失
- `train/n_pos`：本 step 用的锚点对数（= 粗 GT 数）

`n_pos` 接近 0 说明 coarse GT 太稀疏，contrast 项基本没贡献——属于 LR 太高 / mask 错 / 数据 H 增强太狠的诊断信号。

## 实测

v1 没单独跑完整实验（v1 直接被 v2 inherit 后跑 v2），结果数字见 [eloftr-cross-modal-experiments §7 实验对应表](../eloftr-cross-modal-experiments/SKILL.md)。
