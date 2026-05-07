#!/bin/bash
# ============================================================================
# run_msyn_v10_ddp.sh
# v10 mode B entry: 4-GPU DDP ship run on Megadepth_Syn (~115K train).
#
# This is the v10 production run. Reverse-scaled cfg keeps effective TRUE_LR
# at 1.25e-4 (== v9 single-card) despite 4x effective batch.
#
# Usage:
#   bash MyScripts/run_msyn_v10_ddp.sh
#   (must be inside tmux: tmux new -s msyn_v10_b; expected wall-clock ~3-4h)
#
# Prerequisites in addition to mode A:
#   - All 4 GPUs free (check `nvidia-smi`); CUDA_VISIBLE_DEVICES=0,1,2,3 below
#   - sync_batchnorm WILL be enabled by train.py:220 because WORLD_SIZE > 1
#     (4x sample size for MSBN dual BN convergence; accept v9 byte-mismatch)
#
# Schedule (see configs/loftr/eloftr_full_v10_msyn_ddp.py for details):
#   max_epochs=10, ES patience=3, MSLR=[2,4,6]
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4 (matches v9 single-card)
#   WARMUP_STEP=0 -> actual 0 step (skip warmup entirely)
#   N_SAMPLES_PER_SUBSET=29000 (4 rank * 29K ~= 115K train full set)
#
# Sanity gates (same 6 as mode A; gate 6 prints WARMUP_STEP=0 here):
#   1-5 same as mode A
#   6. "TRUE_LR=1.25e-04, WARMUP_STEP=0" (cfg 0 / 0.25 = 0)
#
# DDP-specific debug checks (do these in mode B debug pass):
#   a. grep "[rank " logs/.../version_0/version_0.log -- expect 4 rank entries
#      each printing "building RoadSceneDataset (...) from .../train_pairs.txt"
#   b. PL progress bar "Epoch 0: ... <X>/<Y>" -- X = single-rank step/epoch.
#      Expected ~7187 (29K / 4 bs); if X = ~28750 (single-card density),
#      sampler is NOT sharded -> scenario 1 (same-seed dup), 4 rank see
#      identical batches -> effective_bs is actually 16 on 4x duplicated data
#   c. TB train_loss step 0 should be ~1.4-1.6 (same as v9 ep0)
#
# Acceptance (v10 ship; 4-card non-determinism allows +/- 0.005):
#   strong : in-domain test p@1 >= 0.60, peak ep 4-7, wall-clock <= 4h
#   medium : in-domain test p@1 in [0.50, 0.60], peak ep 4-8, <= 5h
#   weak   : in-domain test p@1 in [0.40, 0.50], peak ep 4-9, <= 6h
#   fail   : in-domain test p@1 < 0.40 or any DDP gate fails
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Prerequisite checks: subdir symlinks + PC cache + index + base ckpt
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/train/infrared_pc \
         data/Megadepth_Syn/train/phoenix_pc \
         data/Megadepth_Syn/index/train_pairs.txt \
         data/Megadepth_Syn/index/val_pairs.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v10 prereq] missing: $p"; exit 1; }
done

if [ -z "$(ls -A data/Megadepth_Syn/train/infrared_pc 2>/dev/null)" ]; then
    echo "[v10 prereq] data/Megadepth_Syn/train/infrared_pc is empty"
    echo "            run: bash MyScripts/precompute_pc_edges.sh --dataset Megadepth_Syn \\"
    echo "                     --recursive --max_long_edge 640 --pc_nscale 3 --workers 24"
    exit 1
fi

python train.py \
  configs/data/megadepth_syn_trainval.py \
  configs/loftr/eloftr_full_v10_msyn_ddp.py \
  --exp_name=msyn_v10_ddp \
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
  --max_epochs=10 \
  --disable_mp \
  --thr 0.1
