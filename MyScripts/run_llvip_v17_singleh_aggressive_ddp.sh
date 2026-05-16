#!/bin/bash
# ============================================================================
# run_llvip_v17_singleh_aggressive_ddp.sh
#
# v17 entry: finetune from v14 best ckpt (ep12, auc@10=0.273) on LLVIP with
# aggressive single-side H aug (rot=25 / scale=0.75-1.25 / trans=0.12 / persp=0.08,
# image1=VIS only, with eval_roadscene_all_singleh_finetuned.bat 协议一致).
#
# v17 vs v14 是 "多重 distribution shift transfer learning":
#   dataset (Megadepth_Syn -> LLVIP) + 监督 (pose -> H) + aug (无 -> aggressive)
#   + geometry (cross-view -> monocular pixel-aligned) + BN stats.
# Schedule V2 (中庸): TRUE_LR=1e-4 (= v14 cold start 0.8x), WARMUP=300 step,
# MSLR=[2,4,6], ES patience=5. 详见
# configs/loftr/eloftr_full_v17_llvip_singleh_aggressive_ddp.py docstring.
#
# Diff vs run_msyn_v14_pose_ddp.sh (5 处 + 删除一些):
#   1. data_cfg : configs/data/megadepth_syn_pose_640.py
#                 -> configs/data/llvip_trainval.py
#   2. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_ddp.py
#                 -> configs/loftr/eloftr_full_v17_llvip_singleh_aggressive_ddp.py
#   3. --exp_name : msyn_v14_pose_ddp -> llvip_v17_singleh_aggressive_ddp
#   4. + --ckpt_path 指向 v14 best (--ckpt_path 不是 --resume_from_checkpoint!
#         前者只 load weight, 后者会 resume optimizer/scheduler/monitor 状态;
#         v17 切到 precision@3px 监控会跟 v14 监控的 auc@10 冲突).
#   5. --max_epochs : 18 -> 12 (finetune 短 schedule)
#   6. --limit_val_batches : 0.2 -> 0.5 (val pool=LLVIP test 3463, *0.5=1731 pair,
#         业界 megadepth_val_1500 量级)
#   7. prereq check 列表换成 LLVIP (而不是 Megadepth_Syn pose 索引), v14 ckpt
#      也加入 prereq.
#
# Usage:
#   bash MyScripts/run_llvip_v17_singleh_aggressive_ddp.sh
#   bash MyScripts/run_llvip_v17_singleh_aggressive_ddp.sh --sanity-only
#
# Prerequisites (本地 git push 之后, 服务器 git pull 完成才能跑):
#   1. data/LLVIP/{infrared,visible}/{train,test}/*.jpg 存在
#      (服务器: 子目录软链到 /data/xyjiang/Datasets/Infrared_image_datasets/LLVIP/,
#       本地: 已有真实数据)
#   2. data/LLVIP/index/train_pairs.txt (12025 行, 'train/<id>.jpg' 形式)
#      data/LLVIP/index/test_pairs.txt  (3463 行,  'test/<id>.jpg'  形式)
#      生成命令: python MyScripts/make_llvip_splits.py
#   3. v14 best ckpt 存在 (从本地 push 的 logs/tb_logs/msyn_v14_pose_ddp/
#      复制到服务器, 或服务器 v14 ship 已完成):
#      logs/tb_logs/msyn_v14_pose_ddp/version_0/checkpoints/
#        epoch=12-auc@5=0.151-auc@10=0.273-auc@20=0.428.ckpt
#
# DDP schedule (V2 中庸 finetune, bs=4 反向缩放 TRUE_LR=1e-4):
#   max_epochs=12, ES patience=5, MSLR=[2,4,6], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=4e-4 -> TRUE_LR=1e-4
#   WARMUP_STEP=75 -> actual 300 step ~0.4 ep
#   N_SAMPLES_PER_SUBSET=3006 (= 12025 / world_size=4, IR-VIS short-circuit
#                              image-level pre-shard 后 per-rank size,
#                              不是 12025!)
#   SB_SUBSET_SAMPLE_REPLACEMENT=False (randperm 全覆盖)
#   --limit_val_batches=0.5 (val 3463 pair * 0.5 = 1731 effective)
#   --log_every_n_steps=500 (跟 v14 一致, TB scalar 10x 稀疏)
#
# Wall-clock estimate (服务器 4×3090 DDP, v14 实测 step time 1.0 s 推算):
#   per ep: train 12.5 min + val 1.1 min + ckpt save 0.4 min = ~14 min
#   max_ep=12 跑满: ~2.8 h (上限, 罕见)
#   ES @ ep 7-8 大概率 (best_ep ep 2-3): ~1.6-1.9 h, center ~1.7 h
#
# Acceptance gate (val precision@3px, sanity-only mode 1 ep 后校准):
#   Strong : >= 0.95
#   Medium : in [0.90, 0.95)
#   Flat   : in [0.85, 0.90), 跟 v14 raw load 持平
#   Fail   : < 0.85, 或 ep0 step 100 内 NaN/loss 爆 -> 回退方案见 cfg docstring
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (跑 1 ep + 启动日志验证, 不正式 ship)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v17 sanity] running 1-ep DDP smoke test..."
    python train.py \
      configs/data/llvip_trainval.py \
      configs/loftr/eloftr_full_v17_llvip_singleh_aggressive_ddp.py \
      --exp_name=llvip_v17_singleh_aggressive_ddp_sanity \
      --ckpt_path=logs/tb_logs/msyn_v14_pose_ddp/version_0/checkpoints/epoch=12-auc@5=0.151-auc@10=0.273-auc@20=0.428.ckpt \
      --gpus=4 \
      --num_nodes=1 \
      --batch_size=4 \
      --num_workers=12 \
      --pin_memory=true \
      --check_val_every_n_epoch=1 \
      --log_every_n_steps=500 \
      --limit_train_batches=1.0 \
      --limit_val_batches=0.5 \
      --num_sanity_val_steps=0 \
      --max_epochs=1 \
      --disable_mp \
      --thr 0.1
    echo "[v17 sanity] 1-ep done; check val precision@3px ~ baseline (v14 raw load)"
    exit 0
fi

# Prerequisite checks
for p in data/LLVIP/infrared/train \
         data/LLVIP/infrared/test \
         data/LLVIP/visible/train \
         data/LLVIP/visible/test \
         data/LLVIP/index/train_pairs.txt \
         data/LLVIP/index/test_pairs.txt \
         logs/tb_logs/msyn_v14_pose_ddp/version_0/checkpoints/epoch=12-auc@5=0.151-auc@10=0.273-auc@20=0.428.ckpt; do
    [ -e "$p" ] || { echo "[v17 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/llvip_trainval.py \
  configs/loftr/eloftr_full_v17_llvip_singleh_aggressive_ddp.py \
  --exp_name=llvip_v17_singleh_aggressive_ddp \
  --ckpt_path=logs/tb_logs/msyn_v14_pose_ddp/version_0/checkpoints/epoch=12-auc@5=0.151-auc@10=0.273-auc@20=0.428.ckpt \
  --gpus=4 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=12 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=500 \
  --limit_train_batches=1.0 \
  --limit_val_batches=0.5 \
  --num_sanity_val_steps=0 \
  --max_epochs=12 \
  --disable_mp \
  --thr 0.1
