@echo off
setlocal

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" eff_loftr

cd /d "%~dp0.."
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ============================================================================
REM Spillover cure: switch PyTorch CUDA allocator to expandable_segments
REM
REM   Default slab allocator fragments after ~6 epoch of v6's mixed alloc
REM   pattern (train batch + val batch + ckpt save + matplotlib figures),
REM   triggering shared GPU memory fallback (PCIe 4.0 ~32 GB/s vs VRAM
REM   ~700 GB/s = 22x slower). expandable_segments uses CUDA Virtual Memory
REM   API to map non-contiguous physical pages into contiguous virtual
REM   segments via the GPU MMU, eliminating fragmentation. Cost: ~2-3 ms
REM   extra on first allocation of each segment, zero ongoing cost.
REM
REM   This MUST be set BEFORE python invocation (env var inheritance) and
REM   AFTER conda activate (to avoid pollution of other envs).
REM ============================================================================
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM v6.1 M3FD SLOW-LR FINETUNE: spillover-fixed continuation of v6 from ep6 ckpt.
REM
REM ============================================================================
REM v6 actual run summary (m3fd_v6_finetune/version_0)
REM
REM   ep0 (val end):  p@1=0.400  p@3=0.788  p@5=0.865   ~25min (ckpt load + warmup)
REM   ep1:            p@1=0.409  p@3=0.773  p@5=0.849   06:56  fast (~7 min/epoch)
REM   ep2:            p@1=0.423  p@3=0.790  p@5=0.863   06:47
REM   ep3:            p@1=0.417  p@3=0.787  p@5=0.861   06:46
REM   ep4:            p@1=0.408  p@3=0.794  p@5=0.869   07:01
REM   ep5:            p@1=0.421  p@3=0.796  p@5=0.866   06:46
REM   ep6:            p@1=0.425  p@3=0.801  p@5=0.873   46:39  ← spillover starts
REM   ep7:            >80min projected at 4.36 s/step  → user aborted
REM
REM   ep6 ckpt is v6's best on every metric AND beats v5 ep9 in-domain on
REM   every metric (v5 was 0.418 / 0.789 / 0.864). Slow-LR strategy works.
REM
REM ============================================================================
REM Diff vs v6 (run_m3fd_v6_finetune.bat)
REM
REM   1. set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (NEW; cures spillover)
REM   2. main_cfg:   eloftr_full_v6_finetune.py  →  eloftr_full_v6_1_finetune.py
REM      (zero-override pure inheritance; v6.1 schedule == v6 schedule by design)
REM   3. exp_name:   m3fd_v6_finetune  →  m3fd_v6_1_finetune
REM      (separate TB log dir; ckpt traceability)
REM   4. --ckpt_path:  v5 ep9 ckpt  →  v6 ep6 ckpt (current best)
REM
REM   Everything else identical to v6 (LR / WARMUP / MSLR / ES / sampler /
REM   freeze / contrastive / modemb / persistent_workers / N_VAL_PAIRS_TO_PLOT
REM   all inherited via v6 → v5 → v4 REV2 → v3 → v2 → v1 → base chain).
REM
REM   See configs/loftr/eloftr_full_v6_1_finetune.py docstring for full
REM   diagnosis (PyTorch caching allocator fragmentation) and design rationale
REM   (Strategy A: full v6 schedule replay, one-variable-at-a-time discipline).
REM
REM ============================================================================
REM Operator pre-launch checklist (CRITICAL -- close before launch)
REM
REM   1. Close non-training GPU consumers via Task Manager:
REM        - Cursor / Edge WebView2 (often 1-2 GB combined)
REM        - Epic Games Launcher / Steam (200-400 MB each)
REM        - NVIDIA App / asus_framework / TranslucentTB
REM        - Any browser tabs with WebGL or hardware-accel video
REM
REM   2. Run THIS bat from an INDEPENDENT WindowsTerminal, NOT Cursor's
REM      integrated terminal. Closing Cursor mid-run would kill training.
REM
REM   3. Verify VRAM headroom: nvidia-smi should show < 1 GB used by
REM      non-training processes BEFORE launching this bat. If still > 2 GB
REM      consumed by other apps, the spillover risk persists.
REM
REM ============================================================================
REM Validation gates (read launch log + monitor first 2 epoch carefully)
REM
REM   Gate 1-6 (inherited from v6, see eloftr_full_v6_finetune.py docstring):
REM     - "Load '...epoch=6-precision@1px=0.425...ckpt' as pretrained checkpoint"
REM     - missing_keys / unexpected_keys both EMPTY (v6.1 graph == v6 graph)
REM     - PL progress bar Epoch 0: must show "... 945/1155" (sampler inheritance)
REM     - persistent_workers verified (worker PIDs persist across epoch boundaries)
REM
REM   Gate 7 (NEW, allocator sanity):
REM     nvidia-smi after epoch 1 should show python.exe VRAM ~12-13 GB and
REM     STABLE across epoch 2-10. If grows past 14 GB by epoch 5,
REM     expandable_segments did not propagate -- re-check the `set` command
REM     order (must be BEFORE `python train.py`, AFTER `call conda activate`).
REM
REM   Gate 8 (NEW, no second spillover):
REM     per-epoch wall time should stay ~7 min ALL THROUGH epoch 50. If epoch
REM     N suddenly jumps to >15 min, an external GPU app started consuming
REM     VRAM (browser tab loaded heavy WebGL etc.) -- not a v6.1 regression,
REM     just close the app and continue.
REM
REM   Gate 9 (NEW, ckpt resume sanity):
REM     epoch 0 val end p@1 MUST be >= 0.420 (v6 ep6 was 0.425, slight
REM     regression OK due to AdamW second moment reset). If < 0.40, ckpt
REM     loaded but graph mismatch silently re-init some weights -- check
REM     missing_keys list closely.
REM
REM ============================================================================
REM Effective TRUE LR trajectory (identical to v6, repeated for convenience)
REM
REM   step      0 - 800     LR ramps 0 → 2.5e-5 (WARMUP=50 → 800 abs after _scaling)
REM   step    800 - 14175   LR = 2.5e-5         (full LR, ep 0.85 → 15)
REM   ep   15 - 25          LR = 1.25e-5        (MSLR step #1)
REM   ep   25 - 35          LR = 6.25e-6        (MSLR step #2)
REM   ep   35 - 50          LR = 3.13e-6        (MSLR step #3)
REM   EarlyStopping (patience=12) likely fires between epoch 25-45.
REM
REM   Total training time estimate: ~7 min/epoch × 50 = ~6 hours (vs v6's
REM   actual ep 1-5 fast pace; the ~25min ep0 startup tax still applies).
REM
REM ============================================================================
REM Expected outcomes
REM
REM   Strong success (M3FD val p@1 >= 0.55 + RoadScene OOD p@1 >= 0.25):
REM     Ship v6.1 as final cross-modal finetune. Optional v7 MSBN to push
REM     p@1 toward 0.65+.
REM
REM   Weak success (M3FD val p@1 in 0.45-0.55, OOD didn't move):
REM     Slow LR helped but fine path's modality-blindness is the structural
REM     ceiling. Move to v7 MSBN per cross-modal-experiments §6 path E1.
REM
REM   Spillover regression (epoch N suddenly slow even with expandable_segments):
REM     Check gate 8 root cause (external GPU app). If not external, v6.2
REM     fallback: also disable PERSISTENT_WORKERS + drop prefetch_factor
REM     to 1 (would need src/lightning/data.py edit).
REM
REM ============================================================================
python train.py ^
  configs\data\m3fd_trainval.py ^
  configs\loftr\eloftr_full_v6_1_finetune.py ^
  --exp_name=m3fd_v6_1_finetune ^
  --ckpt_path=logs\tb_logs\m3fd_v6_finetune\version_0\checkpoints\epoch=6-precision@1px=0.425-precision@3px=0.801-precision@5px=0.873.ckpt ^
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
  --max_epochs=50 ^
  --disable_mp ^
  --thr 0.1

pause
