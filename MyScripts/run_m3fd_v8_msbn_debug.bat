@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================================
REM v8 prerequisite: PC cache must exist (inherited from v7, which v8 resumes)
REM ----------------------------------------------------------------------------
REM v8 inherits v7's input pipeline (PC + CLAHE), so the M3FD PC cache must
REM exist before launch. RoadScene PC cache is also needed for v8 OOD eval
REM (eval_roadscene_finetuned.bat <v8>) but not for training.
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
REM Spillover cure (inherited from v6.1): switch PyTorch CUDA allocator to
REM expandable_segments. v8 reuses v7's compute pattern (train+val+ckpt+
REM matplotlib), so the same fragmentation risk exists. MUST be set BEFORE
REM python invocation and AFTER conda activate.
REM ============================================================================
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM ============================================================================
REM v8 DEBUG: smoke test of "v7 architecture + USE_MSBN=True"
REM ----------------------------------------------------------------------------
REM Inherited acceptance gates from v0-v7 chain (a)-(l) -- see v7_debug.bat
REM for the full list. Below are v8-specific NEW gates (gate 14-17 in plan §5.1):
REM
REM (m) [Gate 14] Two log lines appear during ckpt load:
REM     "MSBN inflated layer2_outconv2.1 (256ch) -> bn_ir + bn_vis (zero-shift, 5 keys -> 10 keys)"
REM     "MSBN inflated layer1_outconv2.1 (128ch) -> bn_ir + bn_vis (zero-shift, 5 keys -> 10 keys)"
REM     -> the v7 BN keys were successfully copied to both _bn_ir and _bn_vis
REM
REM (n) [Gate 15] First val end p@1 >= 0.470 (= v7 ep6 0.475 +/- val noise)
REM     -> the zero-shift initialisation worked: v8 ep0 forward is byte-
REM        equivalent to v7 ep6 (both branches start identical to v7's BN).
REM     If p@1 << 0.470, the inflate hook is broken -- check that .clone()
REM     was actually done and that key prefixes matched.
REM
REM (o) [Gate 16] After a few hundred steps, train/bn_drift_ratio_layer2 in TB
REM     should start growing from ~0 (zero-shift starts at exactly 0). It
REM     reaches >0.05 by ep5+ if MSBN is doing something useful. In debug
REM     we just want to see it >0 to confirm the monitor block fires.
REM
REM (p) [Gate 17] PL model summary shows:
REM     matcher.fine_preprocess.layer2_outconv2_bn_ir.weight    [256]
REM     matcher.fine_preprocess.layer2_outconv2_bn_vis.weight   [256]
REM     matcher.fine_preprocess.layer1_outconv2_bn_ir.weight    [128]
REM     matcher.fine_preprocess.layer1_outconv2_bn_vis.weight   [128]
REM     and layer{1,2}_outconv2.1.* should be ABSENT (replaced by Identity).
REM
REM Plus inherited v7 gates 10-13 (PC cache / CLAHE init / inflated stage0 / channel shape).
REM
REM Total runtime expectation: ~2-3 min on RTX 5070 Ti (50 train + 2 val).
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v8_msbn.py ^
  --exp_name=m3fd_v8_msbn_debug ^
  --ckpt_path=logs\tb_logs\m3fd_v7_pcclahe\version_0\checkpoints\epoch=6-precision@1px=0.475-precision@3px=0.855-precision@5px=0.913.ckpt ^
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
