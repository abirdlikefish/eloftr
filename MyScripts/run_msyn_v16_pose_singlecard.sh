#!/bin/bash
# ============================================================================
# run_msyn_v16_pose_singlecard.sh
# v16 single-card smoke run (~5 min wall-clock). Validates the full
# train.py path on 1 GPU + 10 batches before committing to 9-10h DDP ship.
#
# Gate: ep0 step 0 train_loss is finite (no NaN/Inf), spvs_coarse 输出
# b_ids >= 50 on average, AND v16 新增 modemb 路径正常工作:
#   - TB 出现 train/mod_emb_ir_norm + train/mod_emb_vis_norm
#   - step 0 两个 norm = 0.0 exactly (zeros init 数学保证)
#   - smoke 10 step 后 norm 微微 > 0 (gradient 推动开始, 健康信号)
#
# 若 mod_emb_ir/vis_norm 在 smoke 10 step 仍 = 0.0 -> 检查 USE_MODALITY_EMB
# cfg merge 是否真的传到 matcher (loftr.py:47-62 应该见 init_mode='zeros').
# 若 train_loss step 1 跟 v14 ep0 step 1 不一致 -> zeros init 起点不再
# byte-identical, 检查 modemb 是否被意外初始化为非零.
#
# Diff vs run_msyn_v14_pose_singlecard.sh (3 处, 其他全部一致):
#   1. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_singlecard.py
#                 -> configs/loftr/eloftr_full_v16_pose_msyn_singlecard.py
#   2. --exp_name : msyn_v14_pose_singlecard -> msyn_v16_pose_singlecard
#   3. (注释更新; data_cfg 复用 v14 的 640.py)
# data_cfg : configs/data/megadepth_syn_pose_640.py (v14 已建, 直接复用)
# 其他全部一致: --gpus=1, CUDA_VISIBLE_DEVICES=0, --num_workers=4,
# --pin_memory=true, --max_epochs=1 --limit_train_batches=10
# --limit_val_batches=4 --disable_ckpt --disable_mp --thr 0.1,
# --log_every_n_steps=1 (smoke 每 step 看).
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0

# Prerequisite checks (跟 v14 完全相同)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v16 smoke prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v16_pose_msyn_singlecard.py \
  --exp_name=msyn_v16_pose_singlecard \
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
