"""v15 = v14 + v1 cross-modal symmetric InfoNCE contrastive loss.

v15 vs v14 是 1 项改动 ablation:
- 训练 fingerprint 仅由 cfg.LOFTR.LOSS.USE_CONTRASTIVE 区分
- 其他全部继承 v14 (7 项 cfg + 7 项 sh):
  IMG_RESIZE=640 (NPE 走 train.py fallback [832,832,832,832], 跟 v0-v13 一致;
  v14 初版 NPE bug 已修复, 见 eloftr_full_v14_pose_msyn_ddp.py "NPE bug
  post-mortem"), bs=4, CANONICAL_LR 5e-4, WARMUP 450, MSLR [6,10,14],
  ES patience 5, N_SAMPLES=100, max_ep=18, EVAL_TIMES=1,
  limit_val_batches=0.2, ENABLE_PLOTTING=False, log_every_n_steps=500.

v1 contrastive loss 简介 (见 .cursor/skills/eloftr-v1-contrast/SKILL.md):
- transformer 出口对 IR/VIS coarse tokens 做 symmetric InfoNCE
  (loftr.py:129-131 在 use_contrastive=True 时存 feat_c0/c1_tokens)
- 锚点 = spvs_coarse 提供的 GT match (data['spv_b_ids/i_ids/j_ids'])
- 负样本 = 整个 batch 跨场景拉平的 B*L (=25600 tokens at bs=4 + 6400 coarse cells)
- L2-normalize 后做点积, 让 loss 跟特征绝对模长无关
- bs=4 满足 bs >= 2 硬性要求, 25600 跨场景 negative pool 充足

为什么对 v14 加 contrastive 可能有效:
- v14 已经走 pose 监督 (cross-view + cross-modal), model 已经被强制学
  modality-invariant features
- contrastive loss 提供"额外的显式跨模态正则化": 让 cosine similarity 在
  GT match 上最大化 + 在非 match 上最小化
- 跟 pose 监督正交: pose loss 用反投影 GT, contrastive 用对应 token 配对
- v1 设计初衷就是治 dual-softmax focal 饱和; v14 走 pose 监督路径, coarse
  loss 也是基于 conf_matrix 的 focal, 可能受益于 InfoNCE 的"持续给梯度"特性

为什么 InfoNCE 而不是继续调 focal (v1 SKILL):
- focal 在概率接近 1 时梯度指数级衰减, 跨模态相似度始终偏低反而很快饱和
- InfoNCE 是 (K+1) 路 cross-entropy, 只要负样本里出现伪近邻就持续给梯度
- dual-softmax 隐含"互为最近邻才算正"的硬约束, IR 中信息密度低的格子很
  难单向匹配; symmetric InfoNCE 把方向拆成两个独立的 (K+1) 分类

MegaDepth pose path vs RoadScene H path 兼容性 (v1 原设计是 RoadScene):
- contrastive loss 实现 (loftr_loss.py:134-180) 仅依赖 data['spv_b_ids/
  i_ids/j_ids'], 这 3 个字段 MegaDepth path 的 spvs_coarse 也会 update
  (src/loftr/utils/supervision.py:125-129).
- v14 走 MegaDepth pose path, b_ids 来自 mutual NN check + depth_consistent
  双约束, 数量比 RoadScene 偏少但充足 (v13 实测 50-200 / pair).
- 所以 v15 在 v14 上零代码改动直接可用, 仅 cfg toggle.

v14 实测对照 (待 v14 ship 完成填):
  best ep / auc@10 / METU auc@20: TODO

v14 ship 完成后通过 read_tb_metrics.py 拿数据回来填:
  python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/msyn_v14_pose_ddp/version_<N>

v15 schedule 推算 (sample-pass 跟 v14 完全等效, 仅多 contrastive 项):
  v14 best ep, ES 触发 ep, max_ep 18, MSLR [6,10,14] 全继承

理论预期 (METU all auc@20 vs v14):
  加 contrastive 通常 +1-5% relative AUC (cross-modal 文献常见)
  但 v14 已强制 modality-invariant, 边际收益可能 ±1-3%
  净: 大概率 ±0~+5% (跟 RANSAC 噪声同量级)

工程预期 (wall-clock vs v14):
  - 显存: feat_c0/c1_tokens [4, 6400, 256] fp32 = 25 MB/batch + autograd
    ~50-75 MB / 卡. 占 v14 16 GB / 卡 < 1%
  - 速度: contrastive matmul 6400^2 * 4 = 160M FLOP fp32, ~5-10 ms / step.
    v14 baseline ~0.83 s/step, +0.5-1%
  - TB: 多 4 个 scalar (loss_contrast / loss_i2v / loss_v2i / n_pos)
  - 12+ ep wall-clock: 跟 v14 ~9-10h 基本一致 (+0.5%)

Acceptance gate (v15 vs v14 实测):
  Strong  : METU auc@20 >= v14 + 1.0pp -> 论文 "InfoNCE in pose supervision works"
  Medium  : v14 + [0.3, 1.0)pp -> 写 ablation 表小贡献
  Flat    : v14 +- 0.3pp -> contrastive 边际, 论文叙事可省
  Negative: v14 - [0.3, 1.0)pp -> contrastive 跟 pose 监督干扰, 调
            CONTRASTIVE_WEIGHT 0.01 -> 0.005 重试
  Fail    : < v14 - 1.0pp -> 严重负效应, 回退 v14
  Crash   : NaN loss -> InfoNCE temp 过低 / n_pos 为 0
            (检查 loftr_loss.py:156-158 zero-fallback)

辅助监控 (TB scalar):
  train/loss_contrast 应该单调下降 (cross-modal 对齐学到了)
  train/n_pos 应该 >= 30 / batch (够 InfoNCE 信号; v13 b_ids ~50-200 / pair)
  train/loss_i2v vs train/loss_v2i 应该接近 (对称); 长期不对称 -> 模态偏向
"""
from configs.loftr.eloftr_full_v14_pose_msyn_ddp import cfg

# === v15 唯一改动: 启用 v1 cross-modal symmetric InfoNCE contrastive loss ===
# default.py:186-188 已声明 CONTRASTIVE_WEIGHT=0.01 / CONTRASTIVE_TEMP=0.1
# (= v1 原值, 设计目的是治 dual-softmax focal 饱和). 这里仅 toggle USE_CONTRASTIVE;
# weight / temp 保 default 让 v15 跟 v1 在两个超参上完全一致, 仅 dataset path
# 跟 v1 (RoadScene H) 不同 (v15 走 MegaDepth pose, 仍兼容 -- 见 docstring).
cfg.LOFTR.LOSS.USE_CONTRASTIVE = True

# bat / sh: --max_epochs=18, --batch_size=4, --gpus=4, --disable_mp, --thr 0.1,
#          --limit_val_batches=0.2, --log_every_n_steps=500
