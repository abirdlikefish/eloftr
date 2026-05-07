#!/bin/bash
# ============================================================================
# run_m3fd_v9_e2e_debug.sh
# Debug entry for v9 e2e cold start. Mirrors run_m3fd_v9_e2e.sh but with:
#   --limit_train_batches=10  (not 1.0)
#   --limit_val_batches=2     (not 1.0)
#   --max_epochs=1            (not 80)
#   --disable_ckpt            (skip ModelCheckpoint to keep tb_logs clean)
#
# Wall-clock ~3 min on RTX 3090. Validates the 6 sanity gates listed in
# run_m3fd_v9_e2e.sh and configs/loftr/eloftr_full_v9_e2e.py docstring.
# Run BEFORE starting the full 80-ep training to catch obvious bugs early.
#
# Usage:
#   bash MyScripts/run_m3fd_v9_e2e_debug.sh
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
# Note: 'expandable_segments' is PyTorch 2.1+ only; eloftr_yurupeng has 1.12.1.
# train.py:52 will setdefault to 'max_split_size_mb:1024' (1.x compatible).

for p in data/M3FD_Detection/Ir \
         data/M3FD_Detection/Vis \
         data/M3FD_Detection/Ir_pc \
         data/M3FD_Detection/Vis_pc \
         data/M3FD_Detection/index/train_pairs.txt \
         data/M3FD_Detection/index/val_pairs.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v9 debug prereq] missing: $p"; exit 1; }
done

if [ -z "$(ls -A data/M3FD_Detection/Ir_pc 2>/dev/null)" ]; then
    echo "[v9 debug prereq] data/M3FD_Detection/Ir_pc is empty"
    echo "                  run: bash MyScripts/precompute_pc_edges.sh"
    exit 1
fi

python train.py \
  configs/data/m3fd_trainval.py \
  configs/loftr/eloftr_full_v9_e2e.py \
  --exp_name=m3fd_v9_e2e_outdoor_debug \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
  --gpus=1 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=8 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=10 \
  --limit_train_batches=10 \
  --limit_val_batches=2 \
  --num_sanity_val_steps=0 \
  --max_epochs=1 \
  --disable_mp \
  --disable_ckpt \
  --thr 0.1
