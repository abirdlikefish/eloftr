#!/bin/bash
# ============================================================================
# run_msyn_v10_singlecard.sh
# v10 mode A entry: single GPU validation run on Megadepth_Syn (~115K train).
#
# Mainly for the 5-min debug pass before mode B 4-card DDP, OR as a strict
# single-card baseline if mode B misbehaves and we need to bisect DDP issues.
#
# Usage:
#   bash MyScripts/run_msyn_v10_singlecard.sh
#   (must be inside tmux: tmux new -s msyn_v10_a; expected wall-clock ~10-12h
#    for full 10ep, but we typically only run debug here)
#
# Schedule (see configs/loftr/eloftr_full_v10_msyn_singlecard.py for details):
#   max_epochs=10, ES patience=3, MSLR=[2,4,6]
#   bs=4 single GPU -> TRUE_LR=1.25e-4, WARMUP=32 step (~0.001 ep)
#
# Sanity gates to watch in startup log (~30 lines, see plan SS6.1):
#   1. "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
#   2. "MSBN inflated layer{1,2}_outconv2.1 (...) -> bn_ir + bn_vis (zero-shift)"
#   3. "Missing keys: [...modality_embedding_ir, ...modality_embedding_vis...]"
#   4. PL model summary: fine_preprocess.*_bn_ir/vis.weight shape [128]/[256]
#   5. "RoadSceneDataset: CLAHE enabled (clipLimit=2.0, ..., ir=True, vis=False)"
#      x2 (train + val each instantiate dataset once)
#   6. "TRUE_LR=1.25e-04, WARMUP_STEP=32" (cfg 2 / 0.0625 scaling)
#
# Acceptance (cold start; vs v9 M3FD ep75 0.6863 only as rough reference):
#   strong : peak ep 4-7,  val p@1 >= 0.60
#   medium : peak ep 4-8,  val p@1 in [0.50, 0.60]
#   weak   : peak ep 4-9,  val p@1 in [0.40, 0.50]
#   fail   : val p@1 < 0.40
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
# Note: 'expandable_segments' is PyTorch 2.1+ only. eloftr_yurupeng has torch
# 1.12.1, which raises "Unrecognized CachingAllocator option" if we set it.
# Removed the explicit export -- train.py:52 will setdefault to
# 'max_split_size_mb:1024' (PyTorch 1.x supported, no spillover risk on 24GB
# single-GPU @ bs=4, leaving ~9GB headroom).

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

# infrared_pc / phoenix_pc must be non-empty (precompute_pc_edges.sh must have run)
if [ -z "$(ls -A data/Megadepth_Syn/train/infrared_pc 2>/dev/null)" ]; then
    echo "[v10 prereq] data/Megadepth_Syn/train/infrared_pc is empty"
    echo "            run: bash MyScripts/precompute_pc_edges.sh --dataset Megadepth_Syn \\"
    echo "                     --recursive --max_long_edge 640 --pc_nscale 3 --workers 24"
    exit 1
fi

python train.py \
  configs/data/megadepth_syn_trainval.py \
  configs/loftr/eloftr_full_v10_msyn_singlecard.py \
  --exp_name=msyn_v10_singlecard \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
  --gpus=1 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=8 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=50 \
  --limit_train_batches=1.0 \
  --limit_val_batches=1.0 \
  --num_sanity_val_steps=0 \
  --max_epochs=10 \
  --disable_mp \
  --thr 0.1
