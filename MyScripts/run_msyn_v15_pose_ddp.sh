#!/bin/bash
# ============================================================================
# run_msyn_v15_pose_ddp.sh
# v15 entry: v14 + v1 cross-modal symmetric InfoNCE contrastive loss.
#
# v15 vs v14 是 1 项改动 ablation: cfg.LOFTR.LOSS.USE_CONTRASTIVE False -> True.
# 见 configs/loftr/eloftr_full_v15_pose_msyn_ddp.py docstring 完整 rationale.
#
# v15 训练 fingerprint 仅由 USE_CONTRASTIVE 1 个 cfg 区分, 其他全部继承 v14:
#   IMG_RESIZE=640, NPE [832,832,640,640], bs=4, CANONICAL_LR 5e-4,
#   WARMUP 450, MSLR [6,10,14], ES patience 5, N_SAMPLES=100, max_ep=18,
#   EVAL_TIMES=1, limit_val_batches=0.2, ENABLE_PLOTTING=False,
#   log_every_n_steps=500.
#
# Diff vs run_msyn_v14_pose_ddp.sh (3 处, 其他全部一致):
#   1. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_ddp.py
#                 -> configs/loftr/eloftr_full_v15_pose_msyn_ddp.py
#   2. --exp_name : msyn_v14_pose_ddp -> msyn_v15_pose_ddp
#   3. (注释更新; data_cfg 复用 v14 的 640.py)
# data_cfg : configs/data/megadepth_syn_pose_640.py (v14 已建, 直接复用)
#
# Usage:
#   bash MyScripts/run_msyn_v15_pose_ddp.sh
#   bash MyScripts/run_msyn_v15_pose_ddp.sh --sanity-only   # pre-flight only
#
# Prerequisites (跟 v14 完全相同, 不用重做):
#   1. Symlinks: data/Megadepth_Syn/index/scene_info_pose +
#                data/Megadepth_Syn/index/trainvaltest_list_src
#   2. val_list_loftr_small.txt (v13 已 commit)
#   3. weights/eloftr_outdoor.ckpt
#   4. Numeric sanity gate: python MyScripts/sanity_megadepth_syn_pose.py
#
# DDP schedule (跟 v14 完全一致, ~9-10h ship):
#   max_epochs=18, ES patience=5, MSLR=[6,10,14], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4
#   WARMUP_STEP=450  -> actual 1800 step
#   N_SAMPLES_PER_SUBSET=100, SB_SUBSET_SAMPLE_REPLACEMENT=False
#   USE_CONTRASTIVE=True (CONTRASTIVE_WEIGHT=0.01, TEMP=0.1, v1 default)
#   --limit_val_batches=0.2 (val 4493 pair * 0.2 = 899 effective)
#   --log_every_n_steps=500 (TB scalar 10x 稀疏)
#
# v15 vs v14 实际差异 (期望训练时观察):
#   - 显存 +50-75 MB / 卡 (feat_c0/c1_tokens autograd graph), 可忽略
#   - step time +0.5-1% (contrastive matmul 6400^2 fp32), 可忽略
#   - TB 多 4 个 scalar:
#     * train/loss_contrast 应单调下降 (cross-modal 对齐学到了)
#     * train/loss_i2v vs train/loss_v2i 应接近 (symmetric)
#     * train/n_pos 应 >= 30 / batch (足够 InfoNCE 信号)
#   - n_pos = 0 几乎所有 step -> contrastive 退化, 检查 spv_b_ids
#
# Acceptance (METU all auc@20, eval_metu_vistir_finetuned.bat 跑 v15 ckpt):
#   Strong   : METU auc@20 >= v14 + 1.0pp -> 论文 "InfoNCE in pose works"
#   Medium-up: v14 + [0.3, 1.0)pp -> ablation 表小贡献
#   Flat     : v14 +- 0.3pp -> contrastive 边际, 论文叙事可省
#   Negative : v14 - [0.3, 1.0)pp -> 干扰; 调 CONTRASTIVE_WEIGHT 重试
#   Fail     : < v14 - 1.0pp -> 严重负效应, 回退 v14
#   Crash    : NaN -> 检查 src/losses/loftr_loss.py:156-158 zero-fallback
#
# 显存预算 (v14 实测 16-17 GB / 卡 + v15 feat_tokens autograd +50-75 MB):
#   v15 ~17-18 GB / 卡 (vs 24 GB), 留 6-7 GB 余量, 安全.
#
# GPU 散热 (继承 v14 risk; v14 ship 实测 GPU 2 接近 88 C):
#   v15 跑前可选 sudo nvidia-smi -i 2 -pl 280 限 GPU 2 power.
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (复用 v13/v14 sanity 脚本)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v15 sanity] running numeric + visual pre-flight checks..."
    python MyScripts/sanity_megadepth_syn_pose.py
    python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
    echo "[v15 sanity] done; check dump/v13_visualize_pairs_train_seed42/ eyeball before ship"
    exit 0
fi

# Prerequisite checks (跟 v14 完全相同)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v15 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v15_pose_msyn_ddp.py \
  --exp_name=msyn_v15_pose_ddp \
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
