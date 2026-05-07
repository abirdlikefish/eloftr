#!/bin/bash
# ============================================================================
# precompute_pc_edges.sh
# Linux entry for v7_pcclahe Phase Congruency edge pre-computation. Mirrors
# precompute_pc_edges.bat (Windows). Run ONCE before any v7+ training; output
# is cached and re-used across epochs.
#
# Usage:
#   bash MyScripts/precompute_pc_edges.sh                 # default: M3FD + RoadScene
#   bash MyScripts/precompute_pc_edges.sh --dataset M3FD
#   bash MyScripts/precompute_pc_edges.sh --dataset RoadScene --workers 4
#   bash MyScripts/precompute_pc_edges.sh --dataset M3FD --overwrite
#
# Outputs (sub-directory soft-link layout, see plan SS2.4):
#   data/M3FD_Detection/Ir_pc/*.png        (paired with Ir/, written in repo)
#   data/M3FD_Detection/Vis_pc/*.png       (paired with Vis/, written in repo)
#   data/RoadScene/cropinfrared_pc/*.png   (paired with cropinfrared/)
#   data/RoadScene/crop_LR_visible_pc/*.png
#
# Approximate runtime (workers=8, scipy fftpack fallback):
#   M3FD       4200 imgs x 2 modalities = 8400 imgs   ~30-40 min
#   RoadScene   222 imgs x 2 modalities =  444 imgs    ~1-2 min
#
# Idempotent: re-running skips already-existing PC outputs unless --overwrite
# is passed. Safe to abort with Ctrl+C and resume later.
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ "$#" -eq 0 ]; then
    echo "[precompute_pc_edges] No args - running default: M3FD + RoadScene"
    python MyScripts/precompute_pc_edges.py --dataset M3FD RoadScene
else
    echo "[precompute_pc_edges] Forwarding args: $*"
    python MyScripts/precompute_pc_edges.py "$@"
fi

echo "[precompute_pc_edges] All done."
