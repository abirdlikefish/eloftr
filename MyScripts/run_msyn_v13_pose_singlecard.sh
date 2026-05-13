#!/bin/bash
# ============================================================================
# run_msyn_v13_pose_singlecard.sh
# v13 single-card smoke run (~5 min wall-clock). Validates the full
# train.py path on 1 GPU + 10 batches before committing to 17h DDP ship.
#
# Gate: ep0 step 0 train_loss is finite (no NaN/Inf), and per-batch
# spvs_coarse logs "b_ids >= 50" on average. If train_loss > 5 at step 0
# something is wrong (compare v10 first-step ~0.57; v13 starts from
# outdoor.ckpt on K+pose+depth data so first-step loss should be in the
# 0.5-2.0 range).
#
# Usage:
#   bash MyScripts/run_msyn_v13_pose_singlecard.sh
#
# Diff vs run_msyn_v13_pose_ddp.sh:
#   - main_cfg_path  : eloftr_full_v13_pose_msyn_singlecard.py
#                     (reverse-scales LR to keep TRUE_LR=1.25e-4 same as DDP)
#   - --exp_name     : msyn_v13_pose_singlecard
#   - --gpus=1, CUDA_VISIBLE_DEVICES=0
#   - --max_epochs=1 --limit_train_batches=10 --disable_ckpt (smoke only)
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0

# Prerequisite checks (same as ddp; needs split lists + base ckpt)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v13 smoke prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_trainval.py \
  configs/loftr/eloftr_full_v13_pose_msyn_singlecard.py \
  --exp_name=msyn_v13_pose_singlecard \
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
