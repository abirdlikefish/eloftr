#!/bin/bash
# ============================================================================
# run_msyn_v16_pose_ddp.sh
# v16 entry: v14 + v2 cross-modal learnable modality embedding (modemb).
#
# v16 vs v14 是 1-2 项改动 ablation:
#   cfg.LOFTR.USE_MODALITY_EMB False -> True (核心)
#   cfg.LOFTR.MODALITY_EMB_INIT 'zeros' (显式声明保 step 0 byte-identical v14)
# 见 configs/loftr/eloftr_full_v16_pose_msyn_ddp.py docstring 完整 rationale.
#
# v16 训练 fingerprint 仅由 USE_MODALITY_EMB 1 个 cfg 区分, 其他全部继承 v14:
#   IMG_RESIZE=640, bs=4, CANONICAL_LR 5e-4, WARMUP 450, MSLR [6,10,14],
#   ES patience 5, N_SAMPLES=100, max_ep=18, EVAL_TIMES=1,
#   limit_val_batches=0.2, ENABLE_PLOTTING=False, log_every_n_steps=500.
#
# NOTE: v14/v15/v16 cfg 均不显式设 cfg.LOFTR.COARSE.NPE (跟 v0-v13 一致, 走
# train.py:130 fallback [832,832,832,832], ratio=1.0 不 stretch). v14 初版误设
# NPE=[832,832,640,640] 让 RoPE position stretch 1.3x 破坏 attention pattern,
# ep0 起步 0.214 < outdoor.ckpt baseline (~0.30+), 已修复. 详见 v14 cfg
# "NPE bug post-mortem".
#
# 跟 v15 (contrastive) 正交: v16 测 modemb 单独效果, v15 测 contrastive 单独效果.
# 未来 v17 可组合两者验证 stack 效应.
#
# Diff vs run_msyn_v14_pose_ddp.sh (3 处, 其他全部一致):
#   1. main_cfg : configs/loftr/eloftr_full_v14_pose_msyn_ddp.py
#                 -> configs/loftr/eloftr_full_v16_pose_msyn_ddp.py
#   2. --exp_name : msyn_v14_pose_ddp -> msyn_v16_pose_ddp
#   3. (注释更新; data_cfg 复用 v14 的 640.py)
# data_cfg : configs/data/megadepth_syn_pose_640.py (v14 已建, 直接复用)
#
# Usage:
#   bash MyScripts/run_msyn_v16_pose_ddp.sh
#   bash MyScripts/run_msyn_v16_pose_ddp.sh --sanity-only   # pre-flight only
#
# Prerequisites (跟 v14 完全相同, 不用重做):
#   1. Symlinks: data/Megadepth_Syn/index/scene_info_pose +
#                data/Megadepth_Syn/index/trainvaltest_list_src
#   2. val_list_loftr_small.txt (v13 已 commit)
#   3. weights/eloftr_outdoor.ckpt
#   4. Numeric sanity gate: python MyScripts/sanity_megadepth_syn_pose.py
#
# DDP schedule (跟 v14 完全一致, ~9-10h ship):
#   max_epochs=18, ES patience=5, MSLR=[6,10,14], MSLR_GAMMA=0.5
#   bs=4 x 4 GPU = effective_bs=16 -> _scaling=0.25
#   CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4
#   WARMUP_STEP=450  -> actual 1800 step
#   N_SAMPLES_PER_SUBSET=100, SB_SUBSET_SAMPLE_REPLACEMENT=False
#   USE_MODALITY_EMB=True (MODALITY_EMB_INIT='zeros', v2 default)
#   --limit_val_batches=0.2 (val 4493 pair * 0.2 = 899 effective)
#   --log_every_n_steps=500 (TB scalar 10x 稀疏)
#
# v16 vs v14 实际差异 (期望训练时观察):
#   - 显存 +2 KB / 卡 (2 个 nn.Parameter(256) fp32), 可忽略
#   - step time 微秒级 (forward 多 1 次 broadcast add), 0% 影响
#   - TB 多 2 个 scalar:
#     * train/mod_emb_ir_norm 应单调上涨 (从 0 涨到 ~0.5-1.0)
#     * train/mod_emb_vis_norm 同上
#   - ir/vis norm 长期接近 0 -> modemb dead, 下次 v16b 改 INIT='normal_0.02'
#   - ir/vis norm 持续 > 5.0 -> modemb runaway, 加 weight_decay 或减 LR
#
# Acceptance (METU all auc@20, eval_metu_vistir_finetuned.bat 跑 v16 ckpt):
#   Strong   : METU auc@20 >= v14 + 1.0pp -> "modemb in pose works"
#   Medium-up: v14 + [0.3, 1.0)pp -> ablation 表小贡献
#   Flat     : v14 +- 0.3pp -> modemb 边际, 论文叙事可省
#   Negative : v14 - [0.3, 1.0)pp -> 罕见 (zeros init 起点等价 v14)
#   Fail     : < v14 - 1.0pp -> 严重负效应, 检查 mod_emb norm trajectory
#   Crash    : NaN -> zeros init 下几乎不可能
#
# 显存预算 (v14 实测 16-17 GB / 卡 + v16 modemb +2 KB):
#   v16 ~17 GB / 卡 (vs 24 GB), 留 7 GB 余量, 安全.
#
# GPU 散热 (继承 v14 risk; v14 ship 实测 GPU 2 接近 88 C):
#   v16 跑前可选 sudo nvidia-smi -i 2 -pl 280 限 GPU 2 power.
# ============================================================================
set -euo pipefail

source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional sanity-only mode (复用 v13/v14/v15 sanity 脚本)
if [ "${1:-}" = "--sanity-only" ]; then
    echo "[v16 sanity] running numeric + visual pre-flight checks..."
    python MyScripts/sanity_megadepth_syn_pose.py
    python MyScripts/visualize_megadepth_syn_pose_pairs.py --split train --n_pairs 10 --seed 42
    echo "[v16 sanity] done; check dump/v13_visualize_pairs_train_seed42/ eyeball before ship"
    exit 0
fi

# Prerequisite checks (跟 v14 完全相同)
for p in data/Megadepth_Syn/train/infrared \
         data/Megadepth_Syn/train/phoenix \
         data/Megadepth_Syn/index/scene_info_pose \
         data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt \
         data/Megadepth_Syn/index/val_list_loftr_small.txt \
         weights/eloftr_outdoor.ckpt; do
    [ -e "$p" ] || { echo "[v16 prereq] missing: $p"; exit 1; }
done

python train.py \
  configs/data/megadepth_syn_pose_640.py \
  configs/loftr/eloftr_full_v16_pose_msyn_ddp.py \
  --exp_name=msyn_v16_pose_ddp \
  --ckpt_path=weights/eloftr_outdoor.ckpt \
  --gpus=4 \
  --num_nodes=1 \
  --batch_size=4 \
  --num_workers=12 \
  --pin_memory=true \
  --check_val_every_n_epoch=1 \
  --log_every_n_steps=500 \
  --limit_train_batches=1.0 \
  --limit_val_batches=0.2 \
  --num_sanity_val_steps=0 \
  --max_epochs=18 \
  --disable_mp \
  --thr 0.1
