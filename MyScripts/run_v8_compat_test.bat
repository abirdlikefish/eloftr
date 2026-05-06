@echo off
setlocal

REM ============================================================================
REM v8 R8 COMPATIBILITY SANITY TEST (mandatory gate before v8 training!)
REM ----------------------------------------------------------------------------
REM Purpose: validate that v8's lightning_loftr.py / fine_preprocess.py /
REM default.py changes are byte-identical to v0-v7 behaviour when running
REM a v0-v7 cfg (USE_MSBN=False default).
REM
REM Setup: load v7 cfg + v7 ep6 ckpt, run 1 epoch with --limit_train_batches=2
REM and --disable_ckpt. Should take ~4 min on RTX 5070 Ti.
REM
REM 4 sub-gates that MUST ALL PASS (else FIX before any v8 training):
REM
REM   R8a (no MSBN log):
REM     grep startup log for "MSBN inflated"
REM     -> 0 matches expected (USE_MSBN=False, _maybe_inflate_msbn silent no-op)
REM
REM   R8b (no key warnings):
REM     grep startup log for "missing_keys" and "unexpected_keys"
REM     -> 0 matches expected (state_dict key set unchanged from v7)
REM
REM   R8c (param count unchanged):
REM     check PL model summary "Trainable params:" line
REM     -> 16.00M / Total: 16.03M expected (= v7 SKILL S6 14d实测)
REM     -> if 16.00M / 16.04M means Identity placeholder created _bn_ir/_bn_vis
REM        -> means USE_MSBN=False branch broken
REM
REM   R8d (forward math byte-identical):
REM     epoch 0 val end p@3 score
REM     -> 0.855 expected (= v7 ep6 known value from v7 SKILL S11.2)
REM     -> tolerance < 0.001 (val sampling noise should be negligible at
REM        limit_val_batches=2)
REM     -> if differs >> 0.001, fine_preprocess forward math is not byte-
REM        identical
REM ============================================================================

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM v7 PC cache prerequisite (v7 cfg loads PC channel)
if not exist "data\M3FD_Detection\Ir_pc" (
    echo [v8 compat test] PC cache missing: data\M3FD_Detection\Ir_pc
    echo                  Run: MyScripts\precompute_pc_edges.bat
    pause
    exit /b 1
)
if not exist "data\M3FD_Detection\Vis_pc" (
    echo [v8 compat test] PC cache missing: data\M3FD_Detection\Vis_pc
    pause
    exit /b 1
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo.
echo =====================================================================
echo  v8 R8 compatibility sanity test
echo  Loading: v7 cfg (USE_MSBN=False default) + v7 ep6 ckpt
echo  Expected: byte-identical to v7 ep6 (4 sub-gates listed in script)
echo =====================================================================
echo.

python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v7_pcclahe.py ^
  --exp_name=v8_compat_test ^
  --ckpt_path=logs\tb_logs\m3fd_v7_pcclahe\version_0\checkpoints\epoch=6-precision@1px=0.475-precision@3px=0.855-precision@5px=0.913.ckpt ^
  --gpus=1 ^
  --num_nodes=1 ^
  --batch_size=2 ^
  --num_workers=0 ^
  --pin_memory=false ^
  --check_val_every_n_epoch=1 ^
  --log_every_n_steps=1 ^
  --limit_train_batches=2 ^
  --limit_val_batches=1.0 ^
  --num_sanity_val_steps=0 ^
  --max_epochs=1 ^
  --disable_ckpt ^
  --disable_mp ^
  --thr 0.1

echo.
echo =====================================================================
echo  v8 R8 compatibility test complete
echo  Manual check 4 sub-gates above (R8a/R8b/R8c/R8d).
echo  All must pass before launching v8 training.
echo =====================================================================
echo.

endlocal
pause
