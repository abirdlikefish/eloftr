#!/bin/bash
# ============================================================================
# run_m3fd_v9_e2e.sh
# v9 e2e cold start training entry (Linux, single GPU mode A).
#
# Comprehensive cold start from MegaDepth paper baseline (eloftr_outdoor.ckpt)
# with v1-v8 full feature stack enabled. Not a v8 reproduction; evaluates
# whether the v0-v7 intermediate finetune chain can be skipped.
#
# Usage:
#   bash MyScripts/run_m3fd_v9_e2e.sh
#   (must be inside tmux: tmux new -s v9_e2e; expected wall-clock ~13.3h)
#
# Schedule (see configs/loftr/eloftr_full_v9_e2e.py for full LR timeline):
#   max_epochs=80, ES patience=20, MSLR=[35,55,70]
#   bs=4 single GPU -> TRUE_LR=1.25e-4, actual WARMUP=960 step (~1 ep)
#
# Sanity gates to watch in startup log (~30 lines):
#   1. "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
#   2. "MSBN inflated layer{1,2}_outconv2.1 (...) -> bn_ir + bn_vis (zero-shift)"
#   3. "Missing keys: [...modality_embedding_ir, ...modality_embedding_vis...]"
#   4. PL model summary: fine_preprocess.*_bn_ir/vis.weight shape [128]/[256]
#   5. "RoadSceneDataset: CLAHE enabled (clipLimit=2.0, ..., ir=True, vis=False)"
#   6. "TRUE_LR=1.25e-04, WARMUP_STEP=960" (cfg 60 * 16x scaling)
#
# Acceptance (cold start; NOT measuring against v8 0.5494):
#   strong : peak ep 35-55, val p@1 >= 0.51
#   medium : peak ep 35-65, val p@1 in [0.45, 0.51]
#   weak   : peak ep 35-78, val p@1 in [0.40, 0.45]
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
# single-GPU @ bs=4, leaving ~9GB headroom). Re-add 'expandable_segments:True'
# only when running on PyTorch >= 2.1.

# Prerequisite checks: subdir symlinks + PC cache + index + base ckpt
for p in data/M3FD_Detection/Ir \
         data/M3FD_Detection/Vis \
         data/M3FD_Detection/Ir_pc \
         data/M3FD_Detection/Vis_pc \
         data/M3FD_Detection/index/train_pairs.txt \
         data/M3FD_Detection/index/val_pairs.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v9 prereq] missing: $p"; exit 1; }
done

# Ir_pc/Vis_pc must be non-empty (precompute_pc_edges.sh must have completed)
if [ -z "$(ls -A data/M3FD_Detection/Ir_pc 2>/dev/null)" ]; then
    echo "[v9 prereq] data/M3FD_Detection/Ir_pc is empty"
    echo "            run: bash MyScripts/precompute_pc_edges.sh"
    exit 1
fi

python train.py \
  configs/data/m3fd_trainval.py \
  configs/loftr/eloftr_full_v9_e2e.py \
  --exp_name=m3fd_v9_e2e_outdoor \
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
  --max_epochs=80 \
  --disable_mp \
  --thr 0.1
