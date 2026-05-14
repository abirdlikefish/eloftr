"""v14 = v13 + IMG_RESIZE 640 + NPE [832,832,640,640] + bs=4 (LR/WARMUP 反向缩放保
TRUE_LR=1.25e-4 不变) + EVAL_TIMES=1 (val 加速) + N_SAMPLES_PER_SUBSET 200->100
(LoFTR paper default) + 配套 schedule 等比例放大 (max_ep 12->18, MSLR x2,
ES x~1.7) + ENABLE_PLOTTING=False (TB 文件 1.1GB->220MB).

v14 vs v13 ablation 矩阵 (13 项改动 / 6 组逻辑):
- 训练核心 ablation 变量 (1 项): MGDPT_IMG_RESIZE 832->640 (in data cfg, not here)
- NPE 配套校准 (1 项): cfg.LOFTR.COARSE.NPE = [832, 832, 640, 640]
- bs + LR/WARMUP 反向缩放 (3 项): bs 2->4, CANONICAL_LR 1e-3->5e-4,
  WARMUP_STEP 225->450 (TRUE_LR / actual WARMUP step 跟 v13 完全一致)
- val 加速 (2 项): EVAL_TIMES 5->1, --limit_val_batches 0.5->0.2 (val 协议跟
  eval_metu_vistir_finetuned.bat 的 --ransac_times 1 完全一致)
- N + schedule 等比例放大 (4 项): N_SAMPLES_PER_SUBSET 200->100 (LoFTR paper
  default), max_ep 12->18, MSLR [3,5,7]->[6,10,14], ES patience 3->5
  (sample-pass 总量跟 v13 ES 早停后实际值 ~158K 等效, v14 估 ~156K)
- TB 精简 (2 项): ENABLE_PLOTTING True->False (events 1.1GB->220MB),
  --log_every_n_steps 50->500 (scalar 10x 稀疏 + 训练 step 加速 3-5%)

训练 fingerprint 仅由 IMG_RESIZE 主导 (其他都是"语义等效"或"工程加速"),
v14 vs v13 是干净的训练分辨率 ablation.

DDP LR / WARMUP 反向缩放数学 (4-GPU bs=4, effective_batch=16):
  _scaling = (4 * 4) / 64 = 0.25
  TRUE_LR  = CANONICAL_LR * _scaling = 5e-4 * 0.25 = 1.25e-4 (== v10 == v9 == v13)
  actual WARMUP step = WARMUP_STEP / _scaling = 450 / 0.25 = 1800 step (== v10/v13)

v13 实测完整曲线 (read_tb_metrics.py summary --logdir
logs/tb_logs/msyn_v13_pose_ddp/version_1, 9 ep x 25h wall-clock):
  ep0  auc@10=0.376  warmup 末 (1800 step)
  ep1  auc@10=0.387  full LR
  ep2  auc@10=0.391
  ep3  auc@10=0.396  MSLR1 LR->6.25e-5
  ep4  auc@10=0.400
  ep5  auc@10=0.405  MSLR2 LR->3.13e-5
  ep6  auc@10=0.4096 *** best *** (monitor=auc@10)
  ep7  auc@10=0.4069 MSLR3 LR->1.56e-5, 过拟合
  ep8  auc@10=0.4060 ES 即将触发 (3 ep no-improve)

v14 schedule 推算 (sample-pass 等效 v13 ES 早停后实际值):
  v13 best ep=6 (sample-pass 6 * 4400 step * bs=4 = 105K / rank)
    -> v14 best ep ~12 (sample-pass 12 * 2300 * bs=4 = 110K / rank, 等效)
  v13 ES 触发 ep~9 -> v14 ES 触发 ep~15-17
  max_ep 18 给 ES 留余地

理论预期 (METU all auc@20):
  消除 832->640 train/eval mismatch (NPE 频率自动校准 + AGG attention spatial
  pattern + fine_window 物理覆盖率): +5~15% relative
  640 训练分辨率本身比 832 弱: -5% relative
  N=100 sample-pass 略减: -2% relative
  净 -2 ~ +8% relative auc@20 (大概率持平或微升)

工程预期 (wall-clock, 含 ENABLE_PLOTTING=False + log_every_n_steps=500 加速):
  - sim_matrix 单 batch peak: 832@bs=2 0.93GB -> 640@bs=4 0.66GB (-29%)
  - single step time: v13 实测 ~1.0 s/step (bs=2 832) -> v14 ~0.83 s/step
    (bs=4 640 + 关 train figure -5% + log 频率砍 10x -3%)
  - train per ep: 167 min @v13 -> ~32 min @v14 (-81%)
  - val per ep: 23 min @v13 -> ~1.4 min @v14 (-94%)
  - ep total: 167 min @v13 -> ~33 min @v14
  - v13 实测 9 ep x 167 min = 25h vs v14 估 17 ep x 33 min ~= 9h (~2.8x 加速)
  - events.tfevents 大小: v13 1.1GB -> v14 ~220MB (-80%, 关 train figure)

显存预算 (基于 v13 ship 实测 832+bs=2 = 20.2 GB / 卡 +4 GB vs plan 估算):
  cuDNN benchmark workspace + DDP NCCL bucket buffer 实测吃 +4 GB.
  v14 640+bs=4: sim_matrix 砍 29%, 但 bs 翻倍 + 4GB workspace 不变.
  实际预期 ~17 GB / 卡 (vs 24 GB), 留 7 GB 余量, 安全.

GPU 散热 (v13 ship 实测 GPU 0/2/3: 81-87 C, GPU 1: 66 C; 2 接近 88 C 阈值):
  v14 跑前可选: sudo nvidia-smi -i 2 -pl 280 (限 GPU 2 power, 温度 87->78 C,
  step time -5% 但避免 thermal throttling 净持平). 若 GPU 2 实测 < 86 C 不需限.

Acceptance gate (v14 vs v13 实测 best by auc@10 monitor):
  Strong : val auc@10 >= 0.415 (v13 + 0.005 abs / +1.2% rel)
           METU all auc@20 >= v13 数字 + 0.5pp
  Medium : val auc@10 in [0.405, 0.415)
  Weak   : val auc@10 < 0.405
  Crash  : NaN loss or METU auc < outdoor.ckpt -> NPE 设错 / cfg merge
           顺序错; 查 train.py:129-130
"""
from configs.loftr.eloftr_full_v13_pose_msyn_ddp import cfg

# === v14 唯一架构性改动: NPE 显式设跨分辨率校准 ===
# v13 走 train.py:130 fallback [832,832,832,832], 当前数据是 832 也对; v14 当前
# 数据是 640, 必须显式设, 否则 RoPE 频率没按比例校准 = 等于 v0-v12 那种潜在 bug
# 状态. position_encoding.py:21-22 用 train_res / test_res 比例 stretch RoPE
# frequency, 让 outdoor.ckpt (832 训) 在 640 input 上看到的位置距离尺度跟训练
# 时一致. 第一对 [832, 832] = outdoor.ckpt 训练分辨率 (不变); 第二对 [640, 640]
# = v14 当前数据长边.
cfg.LOFTR.COARSE.NPE = [832, 832, 640, 640]

# === bs=4 显存安全 (sim_matrix 0.66GB 比 v13 bs=2 的 0.93GB 还小) ===
# 反向缩放 LR/WARMUP 保 TRUE_LR + actual WARMUP step 跟 v13 / v10 / v9 完全
# 一致. 见 docstring 数学.
cfg.TRAINER.CANONICAL_LR = 5e-4   # _scaling = (4*4)/64 = 0.25, TRUE_LR = 5e-4 * 0.25 = 1.25e-4
cfg.TRAINER.WARMUP_STEP = 450     # actual = 450 / 0.25 = 1800 step (== v13 == v10 mode B)

# === val 加速 ===
# EVAL_TIMES 5->1 (eloftr_full.py 默认 5, paper 报告 5x mean +- std).
# v14 train val 跟 eval_metu_vistir_finetuned.bat (--ransac_times 1) 协议一致.
# best-ckpt 选择基准 auc@10 仍是 1 次 RANSAC 的值, 噪声 0.3-1.5% AUC 但 trend 不变.
cfg.LOFTR.EVAL_TIMES = 1

# === N_SAMPLES + schedule 等比例放大 ===
# N_SAMPLES_PER_SUBSET 200->100 (LoFTR paper default). 配套 schedule 等比例
# 放大让 sample-pass 总量接近 v13 ES 早停后实际值 (v13 ES ep~9, sample-pass
# ~158K; v14 max_ep=18, ES 估 ep17 停, sample-pass ~156K, 等效).
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100
cfg.TRAINER.MSLR_MILESTONES = [6, 10, 14]      # v13 [3,5,7] 等比例 x2 (sample-pass 角度等效)
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 5         # v13 patience=3 等比例略压 (3/12=25% vs 5/18=28%)

# === TB 数据精简 ===
# 关 train figure (lightning_loftr.py:562 受这个 flag 控制, val figure 不受影响).
# v13 events.tfevents 1.1GB 实测分解: 1754 个 train figure x ~500KB ~= 880MB
# (80%) + 145MB val figure (ENABLE_PLOTTING 不影响 val) + 10MB scalar +
# 70MB overhead. 关掉后 events ~220MB, ckpt 选择 / val 可视化全保留, 训练
# step 还省 ~5% (matplotlib + PNG 编码 + protobuf 序列化是个不小的 CPU spike,
# 每 50 step 一次).
cfg.TRAINER.ENABLE_PLOTTING = False

# bat / sh: --max_epochs=18, --batch_size=4, --gpus=4, --disable_mp, --thr 0.1,
#          --limit_val_batches=0.2, --log_every_n_steps=500
