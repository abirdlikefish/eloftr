@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================================
REM v7 prerequisite: PC cache must exist (v0-v6.1 had no equivalent prerequisite)
REM ----------------------------------------------------------------------------
REM   v7's RoadSceneDataset reads pre-computed Phase Congruency edge maps when
REM   USE_EDGE_INPUT=True. The dataset's __init__ raises FileNotFoundError with
REM   a clear hint if these are missing, so the abort below is just a faster /
REM   friendlier early exit (avoids waiting for python startup just to crash).
REM ============================================================================
if not exist "data\M3FD_Detection\Ir_pc" (
    echo [v7 prerequisite] PC cache missing: data\M3FD_Detection\Ir_pc
    echo                   Run: MyScripts\precompute_pc_edges.bat
    echo                   ^(default = M3FD + RoadScene, ~35-40 min^)
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
REM v7 DEBUG: smoke test of "v6.1 architecture + A1 PC channel + A2 CLAHE on IR"
REM ----------------------------------------------------------------------------
REM v0-v6.1 (a)-(g) acceptance checklist (inherited via v_x cumulative chain):
REM (a) [rank 0]: building RoadSceneDataset (dataset_name=M3FD) from
REM     data/M3FD_Detection/index/train_pairs.txt
REM (b) Same line for val_pairs.txt
REM (c) NO 'Froze backbone: ...' line                         (FREEZE_BACKBONE=False, inherited v4 REV2)
REM (d) NO 'Froze all BatchNorm2d layers' line                (FREEZE_BN=False, inherited v4 REV2)
REM (e) Froze backbone BatchNorm2d layers (eval-mode + no-grad);
REM     fine_preprocess BN remains trainable                  (FREEZE_BACKBONE_BN=True, inherited v4 REV2)
REM (f) Trainable params: ~16.00M / Total: ~16.03M           (KEY signal vs v3 ~5.7M)
REM     v7 adds only ~640 scalars in stage0 (PC channel) -> still rounds to 16.00M
REM (g) NO 'modality_emb_*' in missing_keys (v6.1 ckpt already has them)
REM
REM v7 NEW gates:
REM (h) [Gate 10] No FileNotFoundError when reading PC cache
REM     -> implies precompute_pc_edges.bat already ran successfully
REM (i) [Gate 11] "RoadSceneDataset: CLAHE enabled (clipLimit=2.0, tile=(8, 8),
REM     ir=True, vis=False)" appears for both train AND val datasets
REM (j) [Gate 12] "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
REM     appears once at startup (v6.1 ckpt in_ch=1 -> v7 model in_ch=2)
REM (k) [Gate 13] PL model summary shows
REM     matcher.backbone.layer0.rbr_dense.conv with 1.2K params (= 64*2*3*3),
REM     up from v6.1's 0.6K (= 64*1*3*3); rbr_1x1.conv with 128 params (= 64*2),
REM     up from v6.1's 64
REM (l) [Gate 12 follow-up] First val end p@1 should be very close to v6.1
REM     ep4's 0.443 (any deviation > 0.01 means inflated init is broken)
REM
REM Why limit_train_batches=50 instead of v5_debug's 10:
REM   M3FD has 6.5% non-1024x768 images. v5 debug at bs=2, 10 batches has ~23%
REM   chance of hitting at least one small image. v7 adds new code paths
REM   (PC channel resize + CLAHE before resize + 2-channel stack); we want
REM   ~100% coverage of small-size images in debug to flush out any subtle
REM   shape-handling bugs in those paths. 50 batches at bs=2 = 100 images,
REM   covering small-size images with ~99.7% probability.
REM
REM Total runtime expectation: ~2-3 min on RTX 5070 Ti (50 train + 2 val).
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v7_pcclahe.py ^
  --exp_name=m3fd_v7_pcclahe_debug ^
  --ckpt_path=logs\tb_logs\m3fd_v6_1_finetune\version_0\checkpoints\epoch=4-precision@1px=0.443-precision@3px=0.814-precision@5px=0.879.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=0 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=1 ^
  --limit_train_batches=50 ^
  --limit_val_batches=2 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=1 ^
  --disable_ckpt ^
  --disable_mp ^
  --thr 0.1

endlocal
pause
