#!/bin/bash
# ============================================================================
# run_msyn_v18_pose_rot50_only_singlecard.sh
# v18 single-card smoke run (~5 min wall-clock). Validates the full
# train.py path on 1 GPU + 10 batches with H aug enabled before committing
# to 9h DDP ship.
#
# Gate: ep0 step 0 train_loss is finite (no NaN/Inf), per-batch
# spvs_coarse logs "b_ids >= 30" (lower than v14's >=50 because H aug
# cuts covisibility ~30%), no IndexError on supervision.spvs_coarse
# backward path (Bug #8 clamp(depth1.shape) catches rotated cells outside
# canvas). first-step loss should be in 0.6-2.5 range (slightly higher
# than v14's 0.5-2.0 due to outdoor.ckpt fine head shock from anchor
# convention switch, see v18 cfg "cold-start risk" docstring).
#
# Usage:
#   bash MyScripts/run_msyn_v18_pose_rot50_only_singlecard.sh
#
# Diff vs run_msyn_v14_pose_singlecard.sh (3 处):
#   1. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_singlecard.py
#                 -> configs/loftr/eloftr_full_v18_pose_msyn_rot50_only_singlecard.py
#   2. --exp_name : msyn_v14_pose_singlecard -> msyn_v18_pose_rot50_only_singlecard
#   3. comment block updated
# All other args identical to v14 singlecard.
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0

# Prerequisite checks (复用 v13/v14 split / 软链 / list)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v18 smoke prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v18_pose_msyn_rot50_only_singlecard.py \
  --exp_name=msyn_v18_pose_rot50_only_singlecard \
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
