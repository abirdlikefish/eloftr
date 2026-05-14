#!/bin/bash
# ============================================================================
# run_msyn_v14_pose_ddp.sh
# v14 entry: v13 + IMG_RESIZE 832->640 + bs=4 +
# val 加速 (EVAL_TIMES=1 / limit_val_batches=0.2) + N_SAMPLES 200->100 +
# schedule 等比例放大 (max_ep 12->18, MSLR [3,5,7]->[6,10,14], ES 3->5) +
# TB 精简 (ENABLE_PLOTTING=False / log_every_n_steps 50->500).
#
# NOTE: v14 cfg 不显式设 cfg.LOFTR.COARSE.NPE (跟 v0-v13 一致, 走 train.py:130
# fallback [832,832,832,832]). v14 初版误设 NPE=[832,832,640,640] 让 RoPE
# position stretch 1.3x, attention pattern 被破坏, ep0 起步 0.214 < outdoor.ckpt
# baseline (~0.30+). 修复见 cfg docstring "NPE bug post-mortem".
#
# v14 vs v13 是 13 项改动 / 6 组逻辑动机的 ablation, 见
# configs/loftr/eloftr_full_v14_pose_msyn_ddp.py docstring.
#
# 训练 fingerprint 仅由 IMG_RESIZE 主导, 其他都是"语义等效"或"工程加速",
# v14 vs v13 是干净的训练分辨率 ablation.
#
# Diff vs run_msyn_v13_pose_ddp.sh (7 处):
#   1. data_cfg : configs/data/megadepth_syn_pose_trainval.py (832)
#                 -> configs/data/megadepth_syn_pose_640.py (640)
#   2. main_cfg : configs/loftr/eloftr_full_v13_pose_msyn_ddp.py
#                 -> configs/loftr/eloftr_full_v14_pose_msyn_ddp.py
#   3. --exp_name : msyn_v13_pose_ddp -> msyn_v14_pose_ddp
#   4. --batch_size : 2 -> 4 (640 path sim_matrix 0.66GB / batch, 显存够)
#   5. --limit_val_batches : 0.5 -> 0.2 (val 实跑 4493 * 0.2 = 899 pair,
#                                         跟 LoFTR 业界 1500 pair 量级一致)
#   6. --max_epochs : 12 -> 18 (N=100 sample-pass 砍半 -> ep 翻倍补偿 + ES 余地)
#   7. --log_every_n_steps : 50 -> 500 (TB scalar 10x 稀疏 + step 加速 3-5%)
# 其他全部一致: --num_workers=12, --pin_memory=true, --disable_mp,
# --thr 0.1, prereq check 路径 (LoFTR src + small val).
#
# Usage:
#   bash MyScripts/run_msyn_v14_pose_ddp.sh
#   bash MyScripts/run_msyn_v14_pose_ddp.sh --sanity-only   # pre-flight only
#
# Prerequisites (v14 复用 v13 的 split / 软链 / sanity 脚本, 不用重做):
#   1. Symlinks (v13 已建过, 复用):
#      data/Megadepth_Syn/index/scene_info_pose -> /data/.../scene_info_0.1_0.7
#      data/Megadepth_Syn/index/trainvaltest_list_src -> /data/.../trainvaltest_list
#   2. val_list_loftr_small.txt (v13 已 commit, 复用)
#   3. Numeric sanity gate (v14 复用 v13 sanity 脚本):
#      python MyScripts/sanity_megadepth_syn_pose.py
#
# DDP schedule (bs=4 反向缩放 TRUE_LR=1.25e-4 跟 v10/v13 一致; ~9h ship):
#   max_epochs=18, ES patience=5, MSLR=[6,10,14], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4
#   WARMUP_STEP=450  -> actual 1800 step (~0.4 ep at v14)
#   N_SAMPLES_PER_SUBSET=100 (LoFTR paper default), SB_SUBSET_SAMPLE_REPLACEMENT=False
#   --limit_val_batches=0.2 (val 4493 pair * 0.2 = 899 effective)
#   --log_every_n_steps=500 (TB scalar 10x 稀疏)
#
# v13 实测对照 (read_tb_metrics.py summary, 9 ep x 25h):
#   v13 best ep=6, auc@10=0.4096, auc@5=0.2508, auc@20=0.5730
#   v14 strong gate: auc@10 >= 0.415 (v13 + 0.005 abs / +1.2% rel)
#
# Why bs=4 in v14: 640 path sim_matrix at coarse_matching.py:122 is
# (bs=4, 6400, 6400) fp32 = 0.66 GB / batch (vs v13 bs=2 + 832 = 0.93 GB).
# bs=4 + 640 实际更小, 安全开 bs=4.
#
# 显存预算 (v13 ship 实测 832+bs=2 = 20.2 GB / 卡, +4 GB vs plan):
#   cuDNN benchmark workspace + DDP NCCL bucket buffer 实测吃 +4 GB.
#   v14 640+bs=4 实际预期 ~17 GB / 卡 (vs 24 GB), 留 7 GB 余量, 安全.
#   首次跑前 nvidia-smi 看稳态值确认.
#
# GPU 散热 (v13 ship 实测 GPU 0/2/3: 81-87 C, GPU 2 接近 88 C 阈值):
#   v14 跑前可选限 GPU 2 power 避免 thermal throttling:
#     sudo nvidia-smi -i 2 -pl 280
#   (温度 87 -> 78 C, step time -5% 但避免 throttle 净持平)
#
# Acceptance (METU all auc@20, eval_metu_vistir_finetuned.bat 跑 v14 ckpt):
#   Strong : METU auc@20 >= v13 数字 + 0.5pp -> 论文写"分辨率 ablation 成功"
#   Medium : METU auc@20 in [v13 - 0.5pp, v13 + 0.5pp] -> 832/640 都行
#   Weak   : METU auc@20 < v13 - 1.0pp -> 回退用 v13 832 path
#   Crash  : NaN loss / v14 ep0 auc@10 < outdoor.ckpt baseline (~0.30-0.35)
#            -> 再次检查 cfg 是否误加 NPE override (v14 初版 NPE bug 已修复)
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (复用 v13 sanity 脚本)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v14 sanity] running numeric + visual pre-flight checks..."
    python MyScripts/sanity_megadepth_syn_pose.py
    python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
    echo "[v14 sanity] done; check dump/v13_visualize_pairs_train_seed42/ eyeball before ship"
    exit 0
fi

# Prerequisite checks: pose-side symlinks + split lists + base ckpt (跟 v13 一致)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v14 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v14_pose_msyn_ddp.py \
  --exp_name=msyn_v14_pose_ddp \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
  --gpus=4 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=12 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=500 \
  --limit_train_batches=1.0 \
  --limit_val_batches=0.2 \
  --num_sanity_val_steps=0 \
  --max_epochs=18 \
  --disable_mp \
  --thr 0.1
