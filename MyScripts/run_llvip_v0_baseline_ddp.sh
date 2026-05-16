#!/bin/bash
# ============================================================================
# run_llvip_v0_baseline_ddp.sh
#
# v0 (baseline) entry: pure cold-start finetune from eloftr_outdoor.ckpt on
# LLVIP with aggressive single-side H aug (rot=25 / scale=0.75-1.25 / trans=0.12 /
# persp=0.08, image1=VIS only, eval_roadscene_all_singleh_finetuned.bat 协议一致).
#
# v0 是 cross-modal IR-VIS finetune 起点:
#   起点 ckpt: eloftr_outdoor.ckpt (RGB-RGB MegaDepth, 未见 IR)
#   监督: H 监督 (precision@3px, IR-VIS path)
#   数据: LLVIP 12025 train / 3463 test 像素对齐 IR-VIS
#   schedule: cold-start 黄金标准 (v10/v14 沿用)
#     TRUE_LR=1.25e-4, actual WARMUP=1800 step, MSLR=[4,8,12], ES patience=5,
#     max_ep=18
#   wall-clock: ES 大概率 ~2.5 h, max_ep 跑满 ~3.6 h
#
# Diff vs run_llvip_v17_singleh_aggressive_ddp.sh (5 处):
#   1. main_cfg : configs/loftr/eloftr_full_v17_llvip_singleh_aggressive_ddp.py
#                 -> configs/loftr/eloftr_full_v0_baseline.py
#   2. --exp_name : llvip_v17_singleh_aggressive_ddp -> llvip_v0_baseline_ddp
#   3. --ckpt_path : v14 best (msyn_v14_pose_ddp ep12) -> weights/eloftr_outdoor.ckpt
#   4. --max_epochs : 12 -> 18 (cold-start 需要更多 ep, MSLR=[4,8,12] 后留 6 ep 微调)
#   5. prereq check : v14 ckpt 路径 -> weights/eloftr_outdoor.ckpt
#
# Usage:
#   bash MyScripts/run_llvip_v0_baseline_ddp.sh
#   bash MyScripts/run_llvip_v0_baseline_ddp.sh --sanity-only
#
# Prerequisites (本地 git push 之后, 服务器 git pull 完成才能跑):
#   1. data/LLVIP/{infrared,visible}/{train,test}/*.jpg 存在
#      (服务器: 子目录软链到 /data/xyjiang/Datasets/Infrared_image_datasets/LLVIP/,
#       本地: 已有真实数据)
#   2. data/LLVIP/index/train_pairs.txt (12025 行, 'train/<id>.jpg' 形式)
#      data/LLVIP/index/test_pairs.txt  (3463 行,  'test/<id>.jpg'  形式)
#      生成命令: python MyScripts/make_llvip_splits.py
#   3. weights/eloftr_outdoor.ckpt 存在 (官方 RGB-RGB MegaDepth 预训练 ckpt)
#
# DDP schedule (cold-start golden, bs=4 反向缩放 TRUE_LR=1.25e-4):
#   max_epochs=18, ES patience=5, MSLR=[4,8,12], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4
#   WARMUP_STEP=450 -> actual 1800 step ~2.4 ep (warmup 起点 1.25e-5)
#   N_SAMPLES_PER_SUBSET=3006 (= 12025 / world_size=4, IR-VIS short-circuit
#                              image-level pre-shard 后 per-rank size)
#   SB_SUBSET_SAMPLE_REPLACEMENT=False (randperm 全覆盖)
#   --limit_val_batches=0.5 (val 3463 pair * 0.5 = 1731 effective)
#   --log_every_n_steps=500 (跟 v14/v17 一致)
#
# Wall-clock estimate (服务器 4×3090 DDP, v14 实测 step time 1.0 s 推算 0.83s):
#   per ep: train 10.4 min + val 1.1 min + ckpt save 0.4 min = ~12 min
#   max_ep=18 跑满: ~3.6 h (上限)
#   ES @ best_ep+5 大概率 (best_ep est 6-9): ~2.2-2.8 h, center ~2.5 h
#
# Acceptance gate (val precision@3px, sanity-only mode 1 ep 后校准):
#   ep0 末 val: estimate wide range 0.30-0.70 (待校准)
#   全 ship best ep:
#     Strong : >= 0.95
#     Medium : in [0.90, 0.95)
#     Flat   : in [0.85, 0.90), 跟旧 v0 RoadScene baseline 类似量级
#     Fail   : < 0.85, 或 ep0 step 100 内 NaN/loss > 5.0
#
# Dispatch 正确性硬性 gate (启动日志, 非 val 数字):
#   [rank 0]: building RoadSceneDataset (dataset_name=LLVIP) from data/LLVIP/index/{train,test}_pairs.txt
#   [rank 0]: aligned IR-VIS train pre-shard: 3006/12025 samples (disjoint across 4 ranks; seed=...)
#   EarlyStopping enabled (monitor=precision@3px, mode=max, patience=5)
#
# v17 SKILL doc 警告: v13/v14/v17 cfg 都没显式 cfg.TRAINER.EARLY_STOPPING=True,
#   default 是 False, 所以那些实验 ES 实际从未启用. v0 cfg 必须显式设 True
#   (configs/loftr/eloftr_full_v0_baseline.py 第 ~190 行已设).
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (跑 1 ep + 启动日志验证, 不正式 ship)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v0 sanity] running 1-ep DDP smoke test..."
    python train.py \
      configs/data/llvip_trainval.py \
      configs/loftr/eloftr_full_v0_baseline.py \
      --exp_name=llvip_v0_baseline_ddp_sanity \
      --ckpt_path=weights/eloftr_outdoor.ckpt \
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
    echo "[v0 sanity] 1-ep done; check val precision@3px (cold-start outdoor.ckpt raw + 0-ep 学习)"
    exit 0
fi

# Prerequisite checks
for p in data/LLVIP/infrared/train \
         data/LLVIP/infrared/test \
         data/LLVIP/visible/train \
         data/LLVIP/visible/test \
         data/LLVIP/index/train_pairs.txt \
         data/LLVIP/index/test_pairs.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v0 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/llvip_trainval.py \
  configs/loftr/eloftr_full_v0_baseline.py \
  --exp_name=llvip_v0_baseline_ddp \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
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
  --max_epochs=18 \
  --disable_mp \
  --thr 0.1
