@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================================
REM v7 prerequisite check: PC cache must exist before launching training.
REM Same logic as run_m3fd_v7_pcclahe_debug.bat. RoadScene PC cache is also
REM needed eventually (for v7 OOD eval via eval_roadscene_finetuned.bat 7),
REM so the recommended one-liner double-clicks precompute_pc_edges.bat which
REM defaults to BOTH datasets in one ~35-40 min go.
REM ============================================================================
if not exist "data\M3FD_Detection\Ir_pc" (
    echo [v7 prerequisite] PC cache missing: data\M3FD_Detection\Ir_pc
    echo                   Run: MyScripts\precompute_pc_edges.bat
    pause
    exit /b 1
)
if not exist "data\M3FD_Detection\Vis_pc" (
    echo [v7 prerequisite] PC cache missing: data\M3FD_Detection\Vis_pc
    echo                   Run: MyScripts\precompute_pc_edges.bat
    pause
    exit /b 1
)

REM ============================================================================
REM Spillover cure inherited from v6.1: switch PyTorch CUDA allocator to
REM expandable_segments. v6's slab allocator fragmented after ~6 epochs of mixed
REM alloc patterns (train batch + val batch + ckpt save + matplotlib figure),
REM triggering shared-GPU-memory fallback (PCIe ~32 GB/s vs VRAM ~700 GB/s
REM = 22x slower). expandable_segments uses CUDA Virtual Memory API to map
REM non-contiguous physical pages into contiguous virtual segments via the
REM GPU MMU, eliminating fragmentation. Cost: ~2-3 ms extra on first allocation
REM of each segment, zero ongoing cost. MUST be set BEFORE python invocation
REM (env var inheritance) and AFTER conda activate (avoid pollution of other envs).
REM ============================================================================
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM ============================================================================
REM v7 M3FD INPUT-SIDE FINETUNE: A1 (PC edge channel) + A2 (CLAHE on IR)
REM ----------------------------------------------------------------------------
REM Inherits v6.1's full schedule (TRUE_LR=2.5e-5 / WARMUP_STEP=50 /
REM MSLR=[15,25,35] / ES patience=12), and v6.1's (= v6 = v5 = ...) full
REM architecture stack (FREEZE_BACKBONE_BN / InfoNCE / modality embedding).
REM The ONLY differences from v6.1 are the three v7 cfg flags:
REM   USE_EDGE_INPUT       = True
REM   BACKBONE_IN_CHANNELS = 2
REM   USE_CLAHE_IR         = True
REM
REM Diff vs v6.1 (run_m3fd_v6_1_finetune.bat):
REM   - main_cfg:        eloftr_full_v6_1_finetune.py -> eloftr_full_v7_pcclahe.py
REM   - exp_name:        m3fd_v6_1_finetune -> m3fd_v7_pcclahe
REM   - ckpt_path:       v6 ep6 ckpt -> v6.1 ep4 ckpt (the latest validated SOTA)
REM   - max_epochs:      50 -> 10  (v7 only adds 640 new scalars; 10 epochs
REM                                  matches v5's schedule depth and is more
REM                                  than enough to converge the new params,
REM                                  while still leaving margin within the
REM                                  v6.1-inherited ES patience=12)
REM   - everything else (bs=4, num_workers=6, --disable_mp, etc.) inherits v6.1.
REM
REM Sanity gates to watch in the first ~30 lines of the startup log:
REM   - "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
REM       (Gate 12: shows the v6.1 ckpt was successfully bridged to v7's
REM        in_ch=2 backbone via zero-extension)
REM   - "RoadSceneDataset: CLAHE enabled (clipLimit=2.0, tile=(8, 8),
REM       ir=True, vis=False)" for both train and val datasets
REM   - PL model summary line for matcher.backbone.layer0.rbr_dense.conv:
REM       1.2K params (was 0.6K in v6.1)
REM   - epoch 0 val end p@1 ~ 0.443 (= v6.1 ep4); deviation > 0.01 means
REM       inflated init is broken
REM
REM Expected per-epoch wall time: ~10 min on RTX 5070 Ti (matches v6.1 since
REM the only added compute is one cv2.resize+warpPerspective per channel pair
REM and one cv2.imread per PC cache file, all CPU-bound and absorbed by 6 workers).
REM Total: ~100 min for 10 epochs.
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v7_pcclahe.py ^
  --exp_name=m3fd_v7_pcclahe ^
  --ckpt_path=logs\tb_logs\m3fd_v6_1_finetune\version_0\checkpoints\epoch=4-precision@1px=0.443-precision@3px=0.814-precision@5px=0.879.ckpt ^
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

endlocal
pause
