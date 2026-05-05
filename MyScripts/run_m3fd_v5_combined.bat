@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v5 M3FD FULL FINETUNE: first cross-dataset experiment.
REM
REM ============================================================================
REM Inherits v4 REVISION 2 architecture (FREEZE_BACKBONE_BN=True / InfoNCE /
REM modality embedding). Only the data source + schedule change. M3FD has
REM 3780 train pairs vs RoadScene 177, so 1 epoch is ~21x longer in steps.
REM
REM Diff vs v4 REVISION 2 (run_roadscene_v4_combined.bat):
REM   - data:               RoadScene 177    -> M3FD 3780     (+training data)
REM   - main_cfg:           v4_combined.py   -> v5_m3fd.py    (schedule retuned)
REM   - exp_name:           roadscene_v4_*   -> m3fd_v5_*
REM   - N_SAMPLES_PER_SUBSET (in v5 config):
REM                         200 (default)    -> 3780          (per-epoch quota = full M3FD)
REM   - SB_SUBSET_SAMPLE_REPLACEMENT (in v5 config):
REM                         True (default)   -> False         (randperm not randint; every pair seen once)
REM       Note: WITHOUT both overrides above, M3FD epochs collapse to 50 step/epoch
REM       (LoFTR's RandomConcatSampler treats RoadScene/M3FD as a 1-subset ConcatDataset
REM       and the default 200/scene quota silently caps the dataset at 200 samples/epoch).
REM       With them, the 945 step/epoch / MSLR=[3,5,7] / WARMUP=320abs numbers below are real.
REM   - WARMUP_STEP:        2                -> 20            (in v5 config; 320 abs step ~3.4% total)
REM   - MSLR_MILESTONES:    [10,15,20]       -> [3,5,7]       (in v5 config)
REM   - ES_PATIENCE:        8                -> 3             (in v5 config)
REM   - max_epochs:         30               -> 10            (set here)
REM   - num_workers:        4                -> 6             (set here; M3FD PNG I/O heavier)
REM   - batch_size:         4 (kept; 16GB VRAM forces this; bs=8 would need ~48GB)
REM   - everything else (FREEZE_BACKBONE_BN, LR, contrastive, modemb) inherited
REM
REM Sanity check on first launch: PL progress bar should show "Epoch 0: ... 945/1155 train"
REM (945 = 3780/4 train batches + 210 val batches at val_bs=1). If you see "50/260" instead,
REM the sampler overrides did NOT take effect and you're training on 5% of M3FD per epoch.
REM
REM Effective TRUE LR trajectory at bs=4 (~945 train batches/epoch):
REM   step    0       LR ramps from 0  (WARMUP_STEP=20 -> 320 steps after _scaling)
REM   step  ~320      LR = 1.25e-4 (full; ~3.4% of total training spent in warmup)
REM   epoch 3 end     LR = 6.25e-5 (MSLR step #1, ~2800 grad steps in)
REM   epoch 5 end     LR = 3.13e-5 (MSLR step #2, ~4700 steps)
REM   epoch 7 end     LR = 1.56e-5 (MSLR step #3, ~6600 steps)
REM   epoch 7-10      LR stays 1.56e-5 (refinement, ~9450 steps total)
REM   EarlyStopping (patience=3) likely fires between epoch 5-9.
REM
REM Step1 (InfoNCE) and Step2 (modemb) inherited via v4 -> v3 -> v2 -> v1
REM chain. Step3 (FREEZE_BACKBONE_BN=True freeze backbone BN only) inherited
REM via v4 REVISION 2.
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v5_m3fd.py ^
  --exp_name=m3fd_v5_combined ^
  --ckpt_path=weights\eloftr_outdoor.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=4 ^
  --num_workers=6 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=50 ^
  --limit_train_batches=1.0 ^
  --limit_val_batches=1.0 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=10 ^
  --disable_mp ^
  --thr 0.1

pause
