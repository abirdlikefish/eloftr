#!/bin/bash
# ============================================================================
# run_msyn_v19_pose_rot25_only_ddp.sh
# v19 entry: v18 with rot_deg lowered 50 -> 25. Single-side rotation-only
# Homography augmentation on image1 (VIS, MegaDepth-style cross-view pose
# supervision). image0 (IR) untouched, image1 warped by H_vis sampled in
# resized pixel space; rot_deg=25, scale=1, trans=0, persp=0, prob=1.0.
#
# This is THE retry path explicitly recommended by v18 cfg's "Flat" branch
# (try rot=25 if v18 ends up flat at v14 +- 0.3 pp). v19 is a 1-knob
# ablation on top of v18 (only rot_deg differs; data path, schedule, LR,
# WARMUP, bs, val protocol, ABSOLUTE_FINE_IDX flag all inherited byte-
# identical from v18 ddp).
#
# Diff vs run_msyn_v18_pose_rot50_only_ddp.sh (3 处):
#   1. main_cfg : configs/loftr/eloftr_full_v18_pose_msyn_rot50_only_ddp.py
#                 -> configs/loftr/eloftr_full_v19_pose_msyn_rot25_only_ddp.py
#   2. --exp_name : msyn_v18_pose_rot50_only_ddp -> msyn_v19_pose_rot25_only_ddp
#   3. comment block (this header) reflects v19 ablation rationale (rot=25)
# data_cfg, --gpus=4, --batch_size=4, --num_workers=12, --max_epochs=18,
# --limit_val_batches=0.2, --log_every_n_steps=500, --disable_mp, --thr 0.1,
# prereq checks: ALL inherited from v18 ddp.sh byte-identical.
#
# Usage:
#   bash MyScripts/run_msyn_v19_pose_rot25_only_ddp.sh
#   bash MyScripts/run_msyn_v19_pose_rot25_only_ddp.sh --sanity-only
#
# Acceptance gate (METU all auc@20 vs v14 baseline):
#   Strong   : v19 >= v14 + 1.0 pp -> rot=25 H aug is a clean win
#   Medium   : v19 in [v14 + 0.3, v14 + 1.0) pp -> small benefit
#   Flat     : v19 in [v14 - 0.3, v14 + 0.3] pp -> rotation aug 在 pose
#                                                   监督下不工作; 转向 v15/v16
#   Negative : v19 in [v14 - 1.0, v14 - 0.3) pp -> 25 度也干扰; drop prob 0.5
#   Fail     : v19 < v14 - 1.0 pp -> regression, check sanity tests pass
#
# cold-start expectation (outdoor.ckpt fine head shock):
#   Same as v18 (~3.5*scale1 ~= 1 orig pixel offset because shock magnitude
#   depends on the ABSOLUTE_FINE_IDX convention switch, NOT on rot_deg).
#   ep0 val auc@10 may drop 1-3 pp vs v14 ep0 = 0.34. Fine head should
#   re-anchor in ep1-2 (conf_matrix_f_gt is anchor-invariant).
#
# Wall-clock: same as v14/v18 (~9 h for 17 ep on 4-GPU DDP). H aug overhead
# is ~+0.5% per step (cv2.warpPerspective + 2 _warp_pts_homography calls)
# regardless of rot_deg.
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (复用 v13 sanity 脚本; v19 不需新 sanity, 因 H aug
# 几何正确性走 spvs_coarse / spvs_fine 内 'H_vis' in data 守卫, sanity 脚本
# 不输出 H_vis -> 走 byte-identical v14 路径; H_vis 数值健康在 25 / 50 度
# 都是 pure-rotation matrix, 检测 H_vis[:,2,:2].abs().max() < 1e-6 跟
# rot_deg 大小无关)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v19 sanity] running numeric + visual pre-flight checks..."
    python MyScripts/sanity_megadepth_syn_pose.py
    python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
    echo "[v19 sanity] done; check dump/v13_visualize_pairs_train_seed42/ eyeball before ship"
    exit 0
fi

# Prerequisite checks: pose-side symlinks + split lists + base ckpt
# (inherited from v14/v18 ddp.sh; v19 reuses v13/v14/v18 split/symlink completely)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v19 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v19_pose_msyn_rot25_only_ddp.py \
  --exp_name=msyn_v19_pose_rot25_only_ddp \
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
