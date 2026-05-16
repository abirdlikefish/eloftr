#!/bin/bash
# ============================================================================
# run_msyn_v18_pose_rot50_only_ddp.sh
# v18 entry: v14 + single-side rotation-only Homography augmentation on
# image1 (VIS, MegaDepth-style cross-view pose supervision). image0 (IR)
# untouched, image1 warped by H_vis sampled in resized pixel space;
# rot_deg=50, scale=1, trans=0, persp=0, prob=1.0.
#
# This is a 1-variable ablation on top of v14 (only the H aug knob differs;
# data path, schedule, LR, WARMUP, bs, val protocol all inherited byte-
# identical from v14 ddp). Plus 1 supporting cfg flag (ABSOLUTE_FINE_IDX)
# required for geometric correctness under H_vis rotation, see v18 cfg
# docstring "ABSOLUTE_FINE_IDX rationale".
#
# Diff vs run_msyn_v14_pose_ddp.sh (3 处):
#   1. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_ddp.py
#                 -> configs/loftr/eloftr_full_v18_pose_msyn_rot50_only_ddp.py
#   2. --exp_name : msyn_v14_pose_ddp -> msyn_v18_pose_rot50_only_ddp
#   3. comment block (this header) reflects v18 ablation rationale
# data_cfg, --gpus=4, --batch_size=4, --num_workers=12, --max_epochs=18,
# --limit_val_batches=0.2, --log_every_n_steps=500, --disable_mp, --thr 0.1,
# prereq checks: ALL inherited from v14 ddp.sh byte-identical.
#
# Usage:
#   bash MyScripts/run_msyn_v18_pose_rot50_only_ddp.sh
#   bash MyScripts/run_msyn_v18_pose_rot50_only_ddp.sh --sanity-only
#
# Acceptance gate (METU all auc@20 vs v14 baseline):
#   Strong   : v18 >= v14 + 1.0 pp -> rot=50 H aug is a clean win
#   Medium   : v18 in [v14 + 0.3, v14 + 1.0) pp -> small benefit
#   Flat     : v18 in [v14 - 0.3, v14 + 0.3] pp -> rot=50 too aggressive or
#                                                   marginal; try rot=25
#   Negative : v18 in [v14 - 1.0, v14 - 0.3) pp -> drop prob to 0.5
#   Fail     : v18 < v14 - 1.0 pp -> regression, check sanity tests pass
#
# cold-start expectation (outdoor.ckpt fine head shock):
#   ep0 val auc@10 may drop 1-3 pp vs v14 ep0 = 0.34 because outdoor.ckpt's
#   fine head was trained with the v0 -3.5 trick; v18 inference uses
#   absolute idx convention so stage-1 mkpts1_f systematically offset by
#   ~3.5*scale1 ~= 1 orig pixel. RANSAC tolerates this; fine head should
#   re-anchor in ep1-2 (conf_matrix_f_gt is anchor-invariant).
#
# Wall-clock: same as v14 (~9 h for 17 ep on 4-GPU DDP). H aug overhead is
# ~+0.5% per step (cv2.warpPerspective + 2 _warp_pts_homography calls).
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (复用 v13 sanity 脚本; v18 不需新 sanity, 因 H aug
# 几何正确性走 spvs_coarse / spvs_fine 内 'H_vis' in data 守卫, sanity 脚本
# 不输出 H_vis -> 走 byte-identical v14 路径)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v18 sanity] running numeric + visual pre-flight checks..."
    python MyScripts/sanity_megadepth_syn_pose.py
    python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
    echo "[v18 sanity] done; check dump/v13_visualize_pairs_train_seed42/ eyeball before ship"
    exit 0
fi

# Prerequisite checks: pose-side symlinks + split lists + base ckpt
# (inherited from v14 ddp.sh; v18 reuses v13/v14 split/symlink completely)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v18 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v18_pose_msyn_rot50_only_ddp.py \
  --exp_name=msyn_v18_pose_rot50_only_ddp \
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
