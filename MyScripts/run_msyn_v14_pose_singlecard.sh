#!/bin/bash
# ============================================================================
# run_msyn_v14_pose_singlecard.sh
# v14 single-card smoke run (~5 min wall-clock). Validates the full
# train.py path on 1 GPU + 10 batches before committing to 9h DDP ship.
#
# Gate: ep0 step 0 train_loss is finite (no NaN/Inf), and per-batch
# spvs_coarse logs "b_ids >= 50" on average. If train_loss > 5 at step 0
# something is wrong (compare v13 first-step ~0.65; v14 starts from
# outdoor.ckpt at IMG_RESIZE=640 + NPE 走 train.py:130 fallback
# [832,832,832,832] (interpolation 不 stretch, 见 v14 cfg "NPE bug
# post-mortem"), first-step loss should be in the 0.5-2.0 range).
#
# Usage:
#   bash MyScripts/run_msyn_v14_pose_singlecard.sh
#
# Diff vs run_msyn_v13_pose_singlecard.sh (4 处, smoke 不需改 schedule/log_every):
#   1. data_cfg : configs/data/megadepth_syn_pose_trainval.py (832)
#                 -> configs/data/megadepth_syn_pose_640.py (640)
#   2. main_cfg : configs/loftr/eloftr_full_v13_pose_msyn_singlecard.py
#                 -> configs/loftr/eloftr_full_v14_pose_msyn_singlecard.py
#   3. --exp_name : msyn_v13_pose_singlecard -> msyn_v14_pose_singlecard
#   4. --batch_size : 4 -> 4 (其实跟 v13 一样 bs=4 因为 v13 singlecard 也用 bs=4
#      跑 smoke, 无 OOM 风险, 见 v13 singlecard.sh L49). 这里列出来只是为了完整
#      对照, 实际值未变.
# 其他全部一致: --gpus=1, CUDA_VISIBLE_DEVICES=0, --num_workers=4, --pin_memory=true,
# --max_epochs=1 --limit_train_batches=10 --limit_val_batches=4 --disable_ckpt
# --disable_mp --thr 0.1, --log_every_n_steps=1 (smoke 每 step 看)
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0

# Prerequisite checks (复用 v13 split / 软链 / list)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v14 smoke prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v14_pose_msyn_singlecard.py \
  --exp_name=msyn_v14_pose_singlecard \
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
