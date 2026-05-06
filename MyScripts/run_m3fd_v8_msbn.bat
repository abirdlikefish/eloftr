@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================================
REM v8 prerequisite check: PC cache must exist (inherited from v7)
REM ============================================================================
if not exist "data\M3FD_Detection\Ir_pc" (
    echo [v8 prerequisite] PC cache missing: data\M3FD_Detection\Ir_pc
    echo                   Run: MyScripts\precompute_pc_edges.bat
    pause
    exit /b 1
)
if not exist "data\M3FD_Detection\Vis_pc" (
    echo [v8 prerequisite] PC cache missing: data\M3FD_Detection\Vis_pc
    echo                   Run: MyScripts\precompute_pc_edges.bat
    pause
    exit /b 1
)

REM ============================================================================
REM Spillover cure inherited from v6.1: switch PyTorch CUDA allocator to
REM expandable_segments. v8 keeps v7's compute pattern + adds 4 new BN modules
REM (negligible memory increase ~1.5K params) so the same fragmentation profile
REM and same cure apply. MUST be set BEFORE python invocation, AFTER conda activate.
REM ============================================================================
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM ============================================================================
REM v8 M3FD MSBN FINETUNE: fine_preprocess BN dual-branch (bn_ir / bn_vis)
REM ----------------------------------------------------------------------------
REM Inherits v7's full input pipeline (PC + CLAHE) and v7's full architecture
REM stack (FREEZE_BACKBONE_BN / InfoNCE / modality embedding / BACKBONE_IN_CH=2).
REM The ONLY architectural difference from v7 is USE_MSBN=True; the only
REM schedule differences are the three v7-dead-code fixes (WARMUP/MSLR/ES).
REM
REM Diff vs v7 (run_m3fd_v7_pcclahe.bat):
REM   - main_cfg:        eloftr_full_v7_pcclahe.py -> eloftr_full_v8_msbn.py
REM   - exp_name:        m3fd_v7_pcclahe -> m3fd_v8_msbn
REM   - ckpt_path:       v6.1 ep4 ckpt -> v7 ep6 ckpt (the new SOTA starting point)
REM   - max_epochs:      10 -> 20  (give MSBN BN branches time to diverge from
REM                                  zero-shift start; also activates v7's
REM                                  previously-dead MSLR=[6,11,15] decay points)
REM   - everything else (bs=4, num_workers=6, --disable_mp, etc.) inherits v7.
REM
REM Sanity gates to watch in the first ~30 lines of startup log:
REM   - "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
REM     (v7 inherited stage0 inflate; v8 ckpt has stage0 in_ch=2 already so
REM      this line should NOT appear -- _maybe_inflate_stage0 is no-op for
REM      v7 ckpts. If it DOES appear, ckpt is wrong.)
REM   - "MSBN inflated layer2_outconv2.1 (256ch) -> bn_ir + bn_vis (zero-shift, 5 keys -> 10 keys)"
REM   - "MSBN inflated layer1_outconv2.1 (128ch) -> bn_ir + bn_vis (zero-shift, 5 keys -> 10 keys)"
REM   - PL model summary line for fine_preprocess.layer2_outconv2_bn_ir.weight:
REM     should show 256 params (similar lines for _bn_vis and layer1_*)
REM   - epoch 0 val end p@1 ~ 0.475 (= v7 ep6); deviation > 0.005 means
REM     MSBN inflate hook produced different weights from v7 (BUG)
REM
REM Expected per-epoch wall time: ~10 min on RTX 5070 Ti (matches v7; the
REM extra forward cost is one extra inter_fpn call per batch since IR/VIS
REM no longer cat -- offset by smaller per-call activation memory).
REM Total budget: ~3.3h for max 20 epochs; ES patience=8 likely stops early
REM around ep 14-16 if v8 sees the typical "ep 6 peak + microregress" pattern.
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v8_msbn.py ^
  --exp_name=m3fd_v8_msbn ^
  --ckpt_path=logs\tb_logs\m3fd_v7_pcclahe\version_0\checkpoints\epoch=6-precision@1px=0.475-precision@3px=0.855-precision@5px=0.913.ckpt ^
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
  --max_epochs=20 ^
  --disable_mp ^
  --thr 0.1

endlocal
pause
