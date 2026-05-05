---
name: eloftr-v2-modemb
description: EfficientLoFTR v2 cross-modal experiment - learnable modality embedding (modemb_ir / modemb_vis) added to coarse features before transformer. Use when user mentions v2, v2_modemb, eloftr_full_v2_modemb, USE_MODALITY_EMB, MODALITY_EMB_INIT, modality_emb_ir / modality_emb_vis, mod_emb_ir_norm / mod_emb_vis_norm, run_roadscene_v2_modemb.bat, "modemb only affects coarse not fine", "fine_preprocess BN eats modemb bias", "fine_matching argmax immune to bias", or wants to debug / extend v2 / understand why modemb fails on fine. Triggers: v2 / v2_modemb / 模态嵌入 / modemb / 可学习模态向量 / 模态身份 / 加 modemb / 看 v2 / 跑 v2 / fine 路径模态盲 / 为什么 modemb 对 fine 没用 / coarse vs fine modemb / 信号路径分析, English 'modality embedding', 'learnable modality token', 'why modemb fails on fine', 'BN eats bias', 'argmax immune to constant bias', 'why coarse-only injection'. v2 is the second link of v1->v2->v3->v4->v5->v6->v6.1->v7 chain, accumulates v1's contrastive loss.
---

# v2：Learnable Modality Embedding

> 继承链：v1 (contrast) → **v2 (modemb)** → v3 → v4 → v5 → v6 → v7
> 总览见 [eloftr-cross-modal-experiments](../eloftr-cross-modal-experiments/SKILL.md)
> v1 见 [eloftr-v1-contrast](../eloftr-v1-contrast/SKILL.md)

## 配置

[configs/loftr/eloftr_full_v2_modemb.py](../../../configs/loftr/eloftr_full_v2_modemb.py)：

```python
from configs.loftr.eloftr_full_v1_contrast import cfg
cfg.LOFTR.USE_MODALITY_EMB  = True
cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'   # 起点等价于 v1
```

## 实现要点

[src/loftr/loftr.py:41-62, 112-118](../../../src/loftr/loftr.py)：

- `__init__` 里按 `d_model = config['coarse']['d_model']` 注册两个 `nn.Parameter(torch.zeros(d_model))`：`modality_emb_ir / modality_emb_vis`。
- `forward` 在 backbone 输出之后、送入 `loftr_coarse` 之前广播相加：
  ```python
  feat_c0 = feat_c0 + self.modality_emb_ir.view(1, -1, 1, 1)   # IR
  feat_c1 = feat_c1 + self.modality_emb_vis.view(1, -1, 1, 1)  # VIS
  ```
  只在输入端注入一次，残差会把模态身份逐层传递下去。

## 关键设计选择

| 选择 | 决策 | 原因 |
|------|------|------|
| 加在哪里 | backbone 输出之后 / transformer 之前 | LoFTR 后续是 `feat → q/k/v` 投影，输入端加偏置 = Q/K/V 都被影响，注意力机制本身才能 "意识到" 模态。只加在 V 上只能改输出聚合，注意力分布不变。 |
| 初值 | `zeros` | 起点与 v1 完全等价（finetune 安全网），梯度推动它远离 0 才说明真的有用。 |
| 与 RoPE 关系 | RoPE 只在 self-attention、只乘到 Q/K，是相对位置；modemb 在输入端、影响 Q/K/V，是绝对模态身份。两者正交。详见 [src/loftr/loftr_module/transformer.py:71-72](../../../src/loftr/loftr_module/transformer.py)（`if self.rope:` 分支只在 self 阶段执行）。 |

## 监控

[src/lightning/lightning_loftr.py:250-258](../../../src/lightning/lightning_loftr.py)：每个 `training_step` 把 `mod_emb_ir_norm / mod_emb_vis_norm` 写到 TensorBoard，配合 val 曲线判断是否过拟合（早期 v2 实验里两者持续涨 + val 在 epoch 3-4 见顶后回落，是 v3 出现的根本动机）。

## modemb 信号路径：只影响 coarse，对 fine 几乎完全失效

v2 的 modemb 设计是 "在 coarse transformer 入口加 256-d 偏置"。这个设计对 coarse 匹配非常有效（v2 的 p@3 涨到 0.80 就是证据），但**对 fine refinement 几乎 0 影响**——这是 v5 短板 p@1px 的根本原因之一，也是未来 v7 的优化空间（实际 v7 走了输入端而非 fine MSBN，见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)）。

### 注入位置回顾

```python
# src/loftr/loftr.py:112-118
if self.use_modality_emb:
    feat_c0 = feat_c0 + self.modality_emb_ir.view(1, -1, 1, 1)
    feat_c1 = feat_c1 + self.modality_emb_vis.view(1, -1, 1, 1)
feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)  # transformer 之前
```

之后 transformer 输出的 `feat_c0/feat_c1` **带 modemb 残余**，会随着第 148 行 `fine_preprocess(feat_c0, feat_c1, data)` 进入 fine 路径。

### 残余信号在 fine 路径被两次"吃掉"

**第一次吃：fine_preprocess 的 2 个 BN**

```python
# src/loftr/loftr_module/fine_preprocess.py:28-40
self.layer2_outconv2 = nn.Sequential(
    conv3x3(...), nn.BatchNorm2d(...), nn.LeakyReLU(), conv3x3(...),
)
self.layer1_outconv2 = nn.Sequential(
    conv3x3(...), nn.BatchNorm2d(...), nn.LeakyReLU(), conv3x3(...),
)
```

FPN 链路是 `feat_c → upsample → +x2 → layer2_outconv2 → upsample → +x1 → layer1_outconv2 → upsample`。其中：

- `x2` / `x1` 来自 backbone 中间层，**不带 modemb**
- `+x2` / `+x1` 把 modemb 信号稀释一半
- 紧跟着的 `BatchNorm2d` 把 channel-wise 常数偏置完全归零（BN 数学性质：`BN(x+b) = BN(x)`）

**第二次吃：fine_matching 的 inner-product + argmax**

```python
# src/loftr/utils/fine_matching.py:50-57
feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5     # 注意：不是 L2-normalize
conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
```

即使 fine_preprocess 没有 BN（假设把 BN 都拿掉），fine_matching 找的是 `argmax_{l,r} softmax(conf_matrix)`：

- 给 `feat_f0 += b_ir`、`feat_f1 += b_vis`，inner product 多出 3 项：`<b_ir, feat_f1>` 只跟 r 有关、`<feat_f0, b_vis>` 只跟 l 有关、`<b_ir, b_vis>` 是常数
- row-wise 和 col-wise softmax 各自把 row-only / col-only / 全局常数完全消除
- **argmax 位置不变** → fine peak 不变 → p@1px 不变

### 直接结论与 v7+ 候选方向

| 注入位置 | 被消除机制 | 实际有效 |
|---|---|---|
| 原图 / backbone 内任一处 | 后续 BN 归零 | ❌ 完全无效 |
| backbone 输出 / coarse transformer 前（**当前 v2 位置**） | LN 部分压缩，但 attention 对绝对值敏感 | ✅ **有效** |
| coarse transformer 输出 / fine_preprocess 前 | fine_preprocess 2 BN 归零 | ❌ 无效 |
| fine_preprocess 输出 / fine_matching 前 | fine_matching argmax 对常数偏置免疫 | ❌ **无效** |

**所以"再加两个常数向量给 fine"是无效设计**。要让 fine 真懂模态，必须破解"BN 吃 bias + argmax 吃 bias"两个机制——三条候选见 [eloftr-cross-modal-experiments §4 候选路径 E](../eloftr-cross-modal-experiments/SKILL.md)（MSBN / FiLM / cosine + modemb）。v7 选了正交路径（输入端 PC + CLAHE），见 [eloftr-v7-pcclahe](../eloftr-v7-pcclahe/SKILL.md)。

> v5 的 p@1px=0.435 在 in-domain 已是 v4 REV2 (0.329) 的 1.32×，证明 fine BN 收敛了，但 fine 路径对模态盲是结构性短板，**单靠继续训练 v5 无法把 p@1 推过 ~0.55**。这是 v6 (resume + 慢 LR) 与 v7+ (fine-modemb) 的分工边界。

## 实测

v2 在 RoadScene val 上最佳 ep 49: p@1=**0.741** / p@3=**0.800** / p@5=0.806（bs=2，effective TRUE_LR ~2.67e-5，~6400 step）。详细 ckpt-cfg 对照与 v2 vs v5 OOD 矩阵见 [eloftr-v3-v4-freeze](../eloftr-v3-v4-freeze/SKILL.md) §5 与 [eloftr-v5-m3fd](../eloftr-v5-m3fd/SKILL.md) OOD 矩阵节。
