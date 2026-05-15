"""v16 = v14 + v2 cross-modal learnable modality embedding (modemb).

v16 vs v14 是 1-2 项改动 ablation:
- 训练 fingerprint 仅由 cfg.LOFTR.USE_MODALITY_EMB 1 个 cfg 区分
  (MODALITY_EMB_INIT='zeros' 是 v2 default, 显式声明保 step 0 byte-identical
   v14 -- 起点跟 v14 完全一致, 仅梯度推动 modemb 远离 0 才有效果)
- 其他全部继承 v14 (训练 cfg 7 项 + sh 7 项):
  IMG_RESIZE=640, bs=4, CANONICAL_LR 5e-4, WARMUP 450, MSLR [6,10,14],
  ES patience 5, N_SAMPLES=100, max_ep=18, EVAL_TIMES=1,
  limit_val_batches=0.2, ENABLE_PLOTTING=False, log_every_n_steps=500.
- NPE 继承 v14 修复后状态: 不显式设, train.py:129-130 fallback
  [832, 832, 832, 832] (ratio=1.0 不 stretch). v14 初版曾设 [832,832,640,640]
  踩 stretch bug (ratio=1.3 让 RoPE position phase 错位 50%, ep0 auc@10
  从 0.376 跌到 0.214), 已删 -- 见 v14 cfg "NPE bug post-mortem".
- 跟 v15 (contrastive) 正交: v16 测 modemb 单独效果, v15 测 contrastive
  单独效果, 未来 v17 = v14 + USE_MODALITY_EMB + USE_CONTRASTIVE 测组合.

v2 modemb 简介 (见 .cursor/skills/eloftr-v2-modemb/SKILL.md):
- transformer 入口对 IR/VIS coarse features 加 256-d 可学习偏置
  (loftr.py:116-118, broadcast over H, W)
- 让注意力机制 (Q/K/V 投影) 显式知道模态身份, 学到 modality-aware
  attention pattern
- 跟 NPE/RoPE 正交: RoPE 是相对位置 (只乘 Q/K), modemb 是绝对模态身份
  (加在 feature, Q/K/V 都受影响). v14/v15/v16 NPE 都走 fallback
  [832,832,832,832] 不 stretch, modemb 跟 RoPE 完全独立, 不冲突.
- 'zeros' init 让 step 0 == v14 baseline (safe finetune, finetune 不会因为
  modemb 反而变差)

为什么对 v14 加 modemb 可能有效:
- v14 已经走 pose 监督 (cross-view + cross-modal), 但 model 是隐式学
  modality-invariance (通过 BN running stats + transformer 注意力)
- modemb 提供"显式跨模态信号": transformer 自己知道哪个 token 是 IR /
  哪个是 VIS, 可以学 modality-specific attention bias
- 跟 v15 contrastive 不同: v15 在 loss 端约束 cross-modal alignment,
  v16 在 input 端给 transformer 模态身份. 两者正交.

v2 在 RoadScene 上的实测 (历史参考, 跟 v16 不直接可比):
- ep 49 best: p@1=0.741 / p@3=0.800 / p@5=0.806 (RoadScene val 22 张, 过拟合)
- mod_emb_ir_norm / vis_norm 从 0 单调涨到 ~0.5-1.0 (健康信号)
- v2 失败模式: norm 卡在 0 几乎不动 -> modemb dead, 改 INIT='normal_0.02'

注意 fine path 限制 (来自 v2 SKILL §"modemb 信号路径"):
- modemb 加在 transformer 入口, 残差让信号传到 fine_preprocess
- 但 fine_preprocess 的 2 个 BN 把 channel-wise 常数偏置完全归零
  (BN(x+b) = BN(x))
- 即使 BN 不吃, fine_matching argmax 对常数偏置免疫 (row/col softmax 各自
  消除 row-only/col-only 偏置)
- 所以 v16 modemb 主要影响 coarse 阶段, fine refinement 几乎不受影响
- 这对 v16 不是问题: v14 用 MegaDepth pose 监督, fine 阶段也是 pose 监督
  下的 sub-pixel regression, 不依赖 modemb

v14 实测对照 (待 v14 ship 完成填):
  best ep / auc@10 / METU auc@20: TODO

理论预期 (METU all auc@20 vs v14):
- modemb 让 transformer 显式知道模态身份, 注意力学 modality-aware pattern
- 但 v14 已强制隐式 modality-invariant, 边际收益可能 +-1-3%
- 净: 大概率 +-0~+5% relative (跟 RANSAC 噪声 + v15 同量级)

工程预期 (wall-clock vs v14):
- 显存: 2 个 nn.Parameter(256) fp32 = 2 KB / 卡 + 同样大小梯度. **可忽略**
  (相比 v14 16 GB / 卡 baseline)
- 速度: forward 加 1 次 broadcast add (256-d), 微秒级. **0% step time**
- TB: 多 2 个 scalar (mod_emb_ir_norm / mod_emb_vis_norm)
- 18 ep wall-clock: 跟 v14 ~9-10h **完全一致**

Acceptance gate (v16 vs v14 实测):
  Strong  : METU auc@20 >= v14 + 1.0pp -> 论文 "modemb in pose supervision works"
  Medium  : v14 + [0.3, 1.0)pp -> 写 ablation 表小贡献
  Flat    : v14 +- 0.3pp -> modemb 边际, 论文叙事可省
  Negative: v14 - [0.3, 1.0)pp -> modemb 跟 pose 监督干扰, 罕见
            (zeros init 起点等价 v14, 应不会 negative)
  Fail    : < v14 - 1.0pp -> 严重负效应, 检查 modemb dead vs runaway
  Crash   : NaN loss -> 极少发生 (zeros init 保护)

辅助监控 (TB scalar 来自 lightning_loftr.py:522-530):
  train/mod_emb_ir_norm 应单调上涨 (从 0 涨到 ~0.5-1.0)
  train/mod_emb_vis_norm 同上
  ir/vis norm 长期接近 0 -> modemb dead, 下次 v16b 改 INIT='normal_0.02'
  ir/vis norm 持续上涨 > 5.0 -> modemb runaway, 加 weight_decay 或减 LR
"""
from configs.loftr.eloftr_full_v14_pose_msyn_ddp import cfg

# === v16 唯一改动: 启用 v2 cross-modal modality embedding ===
# loftr.py:47-62 注册 nn.Parameter(zeros(256)), L116-118 forward 加偏置
# (broadcast over H, W). 让 transformer 输入端显式知道模态身份, attention
# 机制可学 modality-aware Q/K/V projection.
# zeros init: step 0 == v14 baseline byte-identical (safe finetune); 仅梯度
# 推开 0 才有效果. 失败模式 = norm 卡 0 不动 -> 改 'normal_0.02' (v16b).
cfg.LOFTR.USE_MODALITY_EMB = True
cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'

# bat / sh: --max_epochs=18, --batch_size=4, --gpus=4, --disable_mp, --thr 0.1,
#          --limit_val_batches=0.2, --log_every_n_steps=500
