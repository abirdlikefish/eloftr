#!/bin/bash
# ============================================================================
# run_msyn_v11_dualh_ddp.sh
# v11 entry: dual-side aggressive Homography aug on Megadepth_Syn ~115K train,
# 4-GPU DDP ship run from outdoor.ckpt cold start.
#
# Diff vs run_msyn_v10_ddp.sh:
#   - main_cfg_path : eloftr_full_v10_msyn_ddp.py
#                  -> eloftr_full_v11_dualh_aggressive_msyn_ddp.py
#   - --exp_name   : msyn_v10_ddp -> msyn_v11_dualh_aggressive_ddp
#   ALL OTHER ARGS (gpus / batch / workers / max_ep / disable_mp / thr) IDENTICAL.
#
# Usage:
#   bash MyScripts/run_msyn_v11_dualh_ddp.sh
#   (must be inside tmux: tmux new -s msyn_v11; expected wall-clock ~17h
#    same as v10 because dual aug ~negligible CPU vs network forward)
#
# Prerequisites:
#   - All 4 GPUs free (check `nvidia-smi`); CUDA_VISIBLE_DEVICES=0,1,2,3 below
#   - sync_batchnorm WILL be enabled by train.py:237 because WORLD_SIZE > 1
#     (4x sample size for MSBN dual BN convergence; accept v9 byte-mismatch)
#   - data/Megadepth_Syn/ symlinks + PC cache + train/val pairs txt + outdoor.ckpt
#     (same prereq set as v10; identical filesystem state)
#
# v11-specific schedule (vs v10 only WARMUP differs):
#   max_epochs=12, ES patience=3, MSLR=[3,5,7], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4 (matches v9/v10)
#   WARMUP_STEP=900 -> actual 3600 step (~0.5 ep), DOUBLED from v10's 1800 step
#   N_SAMPLES_PER_SUBSET=28750 (4 rank * 28750 = 115K full set, no overlap)
#
# v11-specific aug (vs v10 single weak):
#   ROAD_HOMOGRAPHY_DUAL=True
#   ROAD_HOMOGRAPHY_PROB=0.7
#   ROAD_HOMOGRAPHY_KWARGS = rot 25 / scale (0.75,1.25) / trans 0.12 / persp 0.08
#
# Sanity gates (extends v10's 6 gates with 3 v11-specific gates):
#   1-6 same as v10 (TRUE_LR=1.25e-4, WARMUP=3600 instead of 1800)
#   v11.7 dataset: log shows "RoadSceneDataset (...) homography_dual=True"
#   v11.8 step 0-100 mean train_loss < 2.0 (cold start under aug shock OK)
#   v11.9 step 0-100 mean covisibility > 0.30 (per-batch GT count > 100)
#
# Pre-flight check (run BEFORE this script, on the server):
#   bash MyScripts/run_msyn_v11_dualh_ddp.sh --sanity-only
#     wires through to MyScripts/sanity_v11_dualh.py and exits before training
#
# Acceptance (v11 ship; thresholds match v10 because LR/MSLR/data identical):
#   strong : in-domain test p@1 >= 0.60, peak ep 2-7,  wall-clock <= 18h
#   medium : in-domain test p@1 in [0.50, 0.60], peak ep 2-9, <= 19h
#   weak   : in-domain test p@1 in [0.40, 0.50], peak ep 2-11, <= 20h
#   fail   : in-domain test p@1 < 0.40 or any DDP gate fails
#
# Fail mode recovery (per cfg docstring):
#   ep0 train_loss > 3.0 sustained -> kill, in v11 cfg bump WARMUP_STEP 900->1800
#                                    and PROB 0.7->0.5, retry
#   sanity gate fail               -> rollback to v10_msyn_ddp + investigate roadscene.py patch
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode: --sanity-only flag short-circuits to the sanity
# script and exits, useful for one-off pre-flight verification.
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v11 sanity] running pre-flight sanity check, skipping training..."
    python MyScripts/sanity_v11_dualh.py \
        configs/data/megadepth_syn_trainval.py \
        configs/loftr/eloftr_full_v11_dualh_aggressive_msyn_ddp.py
    exit $?
fi

# Prerequisite checks: subdir symlinks + PC cache + index + base ckpt
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/train/infrared_pc \
         data/Megadepth_Syn/train/phoenix_pc \
         data/Megadepth_Syn/index/train_pairs.txt \
         data/Megadepth_Syn/index/val_pairs.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v11 prereq] missing: $p"; exit 1; }
done

if [ -z "$(ls -A data/Megadepth_Syn/train/infrared_pc 2>/dev/null)" ]; then
    echo "[v11 prereq] data/Megadepth_Syn/train/infrared_pc is empty"
    echo "            run: bash MyScripts/precompute_pc_edges.sh --dataset Megadepth_Syn \\"
    echo "                     --recursive --max_long_edge 640 --pc_nscale 3 --workers 24"
    exit 1
fi

python train.py \
  configs/data/megadepth_syn_trainval.py \
  configs/loftr/eloftr_full_v11_dualh_aggressive_msyn_ddp.py \
  --exp_name=msyn_v11_dualh_aggressive_ddp \
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
