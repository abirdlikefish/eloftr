@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v0 baseline: pure finetune from official eloftr_outdoor.ckpt on RoadScene.
REM 模型零修改 (USE_CONTRASTIVE/USE_MODALITY_EMB/USE_EDGE_INPUT/USE_CLAHE_IR/
REM USE_MSBN/BACKBONE_IN_CHANNELS 全 default), 启用 v12 同款 aggressive 强度
REM 的单侧 H aug (DUAL=False, PROB=1.0, KWARGS: rot 25 / scale 0.75-1.25 /
REM trans 0.12 / persp 0.08). 作为 v1-v15 ablation 链的共同对照点.
REM
REM Schedule 沿用 v4 REVISION 2 实测验证的 RoadScene 单卡 finetune schedule
REM (CANONICAL_LR=4e-3 / WARMUP=2 / MSLR=[10,15,20] / ES PATIENCE=8 / max_ep=30,
REM 全部在 cfg 里, 不要在 CLI 改). bs=2 反向缩放后 TRUE_LR=1.25e-4 /
REM actual WARMUP=64 step (~0.6 ep), 跟 v4/v9/v10/v13 实测的 finetune LR 一致.
REM
REM Watch for in TensorBoard:
REM   - train/loss_c 不能爆掉 (aggressive aug + outdoor.ckpt cold start);
REM     ep0 loss < 2.0 安全, > 3.0 持续需要 fallback weak preset
REM   - val precision@3px 应从 ep1 起单调上涨 (warmup 在 step 64 即 ep 0.6 完成)
REM   - ep 10/15/20 LR x= 0.5 后 loss curve 应有可见的 "再次下降" 信号
python train.py ^
  configs\data\roadscene_trainval.py ^
  configs\loftr\eloftr_full_v0_baseline.py ^
  --exp_name=roadscene_v0_baseline ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=4 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=50 ^
  --limit_train_batches=1.0 ^
  --limit_val_batches=1.0 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=30 ^
  --disable_mp ^
  --thr 0.1

pause
