#!/bin/bash
# ============================================================================
# run_msyn_v13_pose_ddp.sh
# v13 entry: real K + W2C pose + depth supervision on cross-view + cross-modal
# Megadepth_Syn, 4-GPU DDP ship run from outdoor.ckpt cold start.
#
# B-pure stack (v1-v8 USE_* all OFF; BACKBONE_IN_CHANNELS=1). Goal: isolate
# the contribution of "real pose supervision + cross-view + cross-modal" by
# stripping all the modality-invariance inductive bias that v9..v12 added.
#
# Diff vs run_msyn_v12_dualh_baseline_ddp.sh:
#   - main_cfg_path : eloftr_full_v13_pose_msyn_ddp.py (NEW, short-circuit
#                     from eloftr_full.py same as v12)
#   - data_cfg_path : megadepth_syn_pose_trainval.py (NEW, pose source +
#                     pose-suffix list files)
#   - --exp_name    : msyn_v13_pose_ddp
#   - Prerequisites adjusted: pose lists + scene_info_pose softlinks
#     (NOT train_pairs.txt / val_pairs.txt; those are v10..v12's
#      M3FD-style aligned IR-VIS lists)
#   - sanity script: MyScripts/sanity_megadepth_syn_pose.py (NEW;
#                    not sanity_v11_dualh which assumes aligned IR-VIS)
#   - PC cache prereq checks: REMOVED (USE_EDGE_INPUT=False ignores PC).
#
# Usage:
#   bash MyScripts/run_msyn_v13_pose_ddp.sh
#   bash MyScripts/run_msyn_v13_pose_ddp.sh --sanity-only   # pre-flight only
#
# Prerequisites (run once on first ship; idempotent):
#   1. Symlinks:
#      cd data/Megadepth_Syn && mkdir -p index
#      ln -s /data/xyjiang/Datasets/LofTR/train-data/megadepth_indices/scene_info_0.1_0.7  index/scene_info_pose
#      ln -s /data/xyjiang/Datasets/LofTR/train-data/megadepth_indices/trainvaltest_list   index/trainvaltest_list_src
#   2. Split lists:
#      python MyScripts/build_megadepth_syn_pose_splits.py
#   3. Numeric sanity gate:
#      python MyScripts/sanity_megadepth_syn_pose.py
#   4. Eyes-on visual sanity (optional but recommended before ship):
#      python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
#
# DDP schedule (matches v10 ddp wall-clock ~17h):
#   max_epochs=12, ES patience=3, MSLR=[3,5,7], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4
#   WARMUP_STEP=450  -> actual 1800 step (~0.25 ep)
#   N_SAMPLES_PER_SUBSET=200 (LoFTR default), SB_SUBSET_SAMPLE_REPLACEMENT=False
#
# Acceptance (METU all auc@20):
#   Strong : >= 0.05 (5x of v10's 0.87%) -> thesis high point
#   Medium : in [0.02, 0.05] -> consider v13b (B-fused: stack on top of pose sup)
#   Weak   : < 0.02 -> revisit WARMUP / LR / N_SAMPLES_PER_SUBSET
#   Crash  : NaN loss or METU auc < outdoor.ckpt -> dispatch / K-T unit bug
#            (sanity scripts should have caught this; investigate)
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v13 sanity] running numeric + visual pre-flight checks..."
    python MyScripts/sanity_megadepth_syn_pose.py
    python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
    echo "[v13 sanity] done; check dump/v13_visualize_pairs_train_seed42/ eyeball before ship"
    exit 0
fi

# Prerequisite checks: pose-side symlinks + split lists + base ckpt
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_pose/train_list_pose.txt \
         data/Megadepth_Syn/index/trainvaltest_list_pose/val_list_pose.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v13 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_trainval.py \
  configs/loftr/eloftr_full_v13_pose_msyn_ddp.py \
  --exp_name=msyn_v13_pose_ddp \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
  --gpus=4 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=12 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=50 \
  --limit_train_batches=1.0 \
  --limit_val_batches=1.0 \
  --num_sanity_val_steps=0 \
  --max_epochs=12 \
  --disable_mp \
  --thr 0.1
