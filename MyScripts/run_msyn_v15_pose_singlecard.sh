#!/bin/bash
# ============================================================================
# run_msyn_v15_pose_singlecard.sh
# v15 single-card smoke run (~5 min wall-clock). Validates the full
# train.py path on 1 GPU + 10 batches before committing to 9-10h DDP ship.
#
# Gate: ep0 step 0 train_loss is finite (no NaN/Inf), spvs_coarse 输出
# b_ids >= 50 on average, AND v15 新增 contrastive 路径正常工作:
#   - TB 出现 train/loss_contrast (初始 ~10, smoke 末应 < 10.5)
#   - TB 出现 train/n_pos > 30 / batch (足够 InfoNCE 信号)
#   - train/loss_i2v vs train/loss_v2i 数值接近 (symmetric InfoNCE)
#
# 若 ep0 step 0 立刻 NaN 在 loss_contrast 上, 检查
# src/losses/loftr_loss.py:156-158 zero-fallback 是否生效.
#
# Diff vs run_msyn_v14_pose_singlecard.sh (3 处, 其他全部一致):
#   1. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_singlecard.py
#                 -> configs/loftr/eloftr_full_v15_pose_msyn_singlecard.py
#   2. --exp_name : msyn_v14_pose_singlecard -> msyn_v15_pose_singlecard
#   3. (注释更新; data_cfg 复用 v14 的 640.py)
# data_cfg : configs/data/megadepth_syn_pose_640.py (v14 已建, 直接复用)
# 其他全部一致: --gpus=1, CUDA_VISIBLE_DEVICES=0, --num_workers=4,
# --pin_memory=true, --max_epochs=1 --limit_train_batches=10
# --limit_val_batches=4 --disable_ckpt --disable_mp --thr 0.1,
# --log_every_n_steps=1 (smoke 每 step 看).
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0

# Prerequisite checks (跟 v14 完全相同)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v15 smoke prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v15_pose_msyn_singlecard.py \
  --exp_name=msyn_v15_pose_singlecard \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
  --gpus=1 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=4 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=1 \
  --limit_train_batches=10 \
  --limit_val_batches=4 \
  --num_sanity_val_steps=0 \
  --max_epochs=1 \
  --disable_mp \
  --disable_ckpt \
  --thr 0.1
