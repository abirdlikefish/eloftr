#!/bin/bash
# ============================================================================
# run_msyn_v12_dualh_baseline_ddp.sh
# v12 entry: pure baseline + v11 dual-side aggressive Homography aug on
# Megadepth_Syn ~115K train, 4-GPU DDP ship run from outdoor.ckpt cold start.
#
# Ablation goal: isolate the contribution of v11's dual H aug, with all
# v1-v8 USE_* (contrastive / modemb / PC / CLAHE / MSBN) explicitly OFF
# (default values). See cfg docstring for the full rationale.
#
# Diff vs run_msyn_v11_dualh_ddp.sh:
#   - main_cfg_path : eloftr_full_v11_dualh_aggressive_msyn_ddp.py
#                  -> eloftr_full_v12_dualh_baseline_msyn_ddp.py
#                  (cfg short-circuits to eloftr_full.py instead of going
#                   through v9_e2e -> v10 chain, so all USE_* default to False)
#   - --exp_name   : msyn_v11_dualh_aggressive_ddp -> msyn_v12_dualh_baseline_ddp
#   - PC cache prereq checks REMOVED (v12 USE_EDGE_INPUT=False does not read
#     PC cache files at training time; the train/{infrared,phoenix}_pc dirs
#     can be empty or even missing without affecting v12 dataset behaviour)
#   ALL OTHER ARGS (gpus / batch / workers / max_ep / disable_mp / thr) IDENTICAL.
#
# Usage:
#   bash MyScripts/run_msyn_v12_dualh_baseline_ddp.sh
#   (must be inside tmux: tmux new -s msyn_v12; expected wall-clock ~17h
#    same as v11 because dual aug ~negligible CPU vs network forward)
#
# Prerequisites:
#   - All 4 GPUs free (check `nvidia-smi`); CUDA_VISIBLE_DEVICES=0,1,2,3 below
#   - sync_batchnorm WILL be enabled by train.py:237 because WORLD_SIZE > 1
#     (no MSBN to worry about; sync_bn just averages the single BN per layer
#      across ranks, byte-clean behaviour)
#   - data/Megadepth_Syn/ symlinks (train/infrared, train/phoenix) +
#     train/val pairs txt + outdoor.ckpt
#     PC cache (train/{infrared,phoenix}_pc) is NOT required for v12.
#
# v12-specific schedule (vs v11 only USE_* differ; LR / WARMUP identical):
#   max_epochs=12, ES patience=3, MSLR=[3,5,7], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4 (matches v11)
#   WARMUP_STEP=900 -> actual 3600 step (~0.5 ep) (matches v11)
#   N_SAMPLES_PER_SUBSET=28750 (4 rank * 28750 = 115K full set, no overlap)
#
# v12 input-side aug (== v11):
#   ROAD_HOMOGRAPHY_DUAL=True
#   ROAD_HOMOGRAPHY_PROB=0.7
#   ROAD_HOMOGRAPHY_KWARGS = rot 25 / scale (0.75,1.25) / trans 0.12 / persp 0.08
#
# v12 model arch (== eloftr_full.py paper baseline):
#   single-channel raw image, single BN per layer, NO contrastive, NO modemb,
#   NO PC channel, NO CLAHE, NO MSBN, NO freeze.
#
# Sanity gates (reuses MyScripts/sanity_v11_dualh.py via getattr fallbacks):
#   1-5 same as v11 (shape/dtype, dual aug fires, covisibility >= 0.30,
#                    H composition correctness, backward compat under DUAL=False)
#   The sanity script reads cfg.LOFTR.USE_EDGE_INPUT etc via getattr defaults,
#   so it transparently builds a 1-channel raw-image dataset for v12 instead
#   of v11's 2-channel PC+CLAHE dataset. No script changes needed.
#
# Pre-flight check (run BEFORE this script, on the server):
#   bash MyScripts/run_msyn_v12_dualh_baseline_ddp.sh --sanity-only
#     wires through to MyScripts/sanity_v11_dualh.py and exits before training
#
# Acceptance (v12 ship; thresholds RELAXED vs v11 because v12 lacks v1-v8
# stack; see cfg docstring "Acceptance grading" section for full rationale):
#   H_isolated_strong : M3FD-OOD P@1 >= 0.30, RoadScene-OOD P@1 >= 0.25
#   H_isolated_medium : M3FD-OOD P@1 in [0.20, 0.30], RoadScene-OOD in [0.18, 0.25]
#   H_isolated_weak   : M3FD-OOD P@1 in [0.15, 0.20]
#   H_isolated_fail   : M3FD-OOD P@1 < 0.15 OR ep0 train_loss > 3.0 sustained
#
# Fail mode recovery (v12 has higher cold-start crash risk than v11 because
# no modemb/MSBN to absorb modality variance):
#   ep0 train_loss > 3.0 sustained -> kill, in v12 cfg bump WARMUP_STEP 900->1800
#                                    and PROB 0.7->0.5, retry
#   sanity gate fail               -> investigate; v12 cfg short-circuits
#                                    inheritance so most likely culprit is
#                                    a mistake in this script's args
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode: --sanity-only flag short-circuits to the sanity
# script and exits, useful for one-off pre-flight verification. Reuses the
# v11 sanity script because all dual-H code paths are shared; the script
# uses getattr fallbacks to handle v12's USE_*=False configuration.
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v12 sanity] running pre-flight sanity check, skipping training..."
    python MyScripts/sanity_v11_dualh.py \
        configs/data/megadepth_syn_trainval.py \
        configs/loftr/eloftr_full_v12_dualh_baseline_msyn_ddp.py
    exit $?
fi

# Prerequisite checks: subdir symlinks + index + base ckpt
# (PC cache dirs NOT required because USE_EDGE_INPUT=False)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/train_pairs.txt \
         data/Megadepth_Syn/index/val_pairs.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v12 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_trainval.py \
  configs/loftr/eloftr_full_v12_dualh_baseline_msyn_ddp.py \
  --exp_name=msyn_v12_dualh_baseline_ddp \
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
