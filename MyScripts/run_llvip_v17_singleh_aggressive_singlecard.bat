@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================================
REM v17 LLVIP single-card sanity (本地 Windows + RTX 5070 Ti, NOT for ship)
REM
REM 用途: 本地快速验证 v17 cfg + data path 不爆 (启动日志 + 1 ep + ckpt format).
REM 正式 ship 用服务器 4×3090 DDP: bash MyScripts/run_llvip_v17_singleh_aggressive_ddp.sh
REM
REM 单卡 vs DDP 4 卡的关键差异:
REM   1. dataset.__len__() = 12025 (no image-level pre-shard, 单卡看全集)
REM   2. v17 ddp cfg N_SAMPLES_PER_SUBSET=3006 (= 12025/4) 在单卡场景会让 sampler
REM      只取 25% sample, 仅够 sanity 验证 (启动日志 + shape + loss 不爆), 不能做
REM      论文实验. 如果要单卡跑完整 12025, 改 cfg 临时 override:
REM        set N_SAMPLES_PER_SUBSET=12025 in this .bat (但 plan 里没这个 CLI flag,
REM        需要单独建 v17_singlecard cfg). Sanity 不需要.
REM   3. effective_bs=1*4=4 vs DDP 4*4=16, _scaling=1.0 vs 0.25
REM      TRUE_LR = CANONICAL_LR * 1.0 = 4e-4 (vs DDP 1e-4). 单卡 LR 偏高, 但 sanity
REM      只 1 ep 1-2 step 就停, LR 影响小, 不修.
REM   4. 单卡无 sync_bn (DDP 自动开), 单卡 step time 更快.
REM
REM 启动后第 30 行内必须出现:
REM   - "[rank 0]: building RoadSceneDataset (dataset_name=LLVIP) from data/LLVIP/index/train_pairs.txt"
REM     (NOT 'aligned IR-VIS train pre-shard:' 那行 - 单卡没 pre-shard)
REM   - ckpt 文件名格式应是 epoch=*-precision@1px=*-precision@3px=*-precision@5px=*.ckpt
REM     (NOT epoch=*-auc@5=*-auc@10=*-auc@20=* 那种 megadepth 格式)
REM   - 第一个 train_loss 不爆 NaN, 量级跟 v14 ep12 train_loss 接近 (finetune 起点)
REM
REM Prerequisites:
REM   1. data/LLVIP/{infrared,visible}/{train,test}/*.jpg 已有 (本地真实数据)
REM   2. data/LLVIP/index/{train,test}_pairs.txt 已生成
REM      (运行: python MyScripts\make_llvip_splits.py)
REM   3. v14 ckpt 文件存在
REM ============================================================================
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py ^
  configs\data\llvip_trainval.py ^
  configs\loftr\eloftr_full_v17_llvip_singleh_aggressive_ddp.py ^
  --exp_name=llvip_v17_singleh_aggressive_singlecard_sanity ^
  --ckpt_path=logs\tb_logs\msyn_v14_pose_ddp\version_0\checkpoints\epoch=12-auc@5=0.151-auc@10=0.273-auc@20=0.428.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=4 ^
  --num_workers=6 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=50 ^
  --limit_train_batches=0.05 ^
  --limit_val_batches=0.05 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=1 ^
  --disable_mp ^
  --thr 0.1

endlocal
pause
