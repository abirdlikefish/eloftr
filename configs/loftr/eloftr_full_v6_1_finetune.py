"""v6.1: spillover-fixed continuation of v6 (PyTorch allocator fragmentation cure).

==============================================================================
Origin (v6 actual run on m3fd_v6_finetune/version_0)
------------------------------------------------------------------------------
v6 训练实测（2026-05-05，RTX 5070 Ti 16GB）：

  ep0 (val end):  p@1=0.400  p@3=0.788  p@5=0.865   ~25min (含 ckpt load + warmup spawn)
  ep1:            p@1=0.409  p@3=0.773  p@5=0.849   06:56  fast (~7 min/epoch)
  ep2:            p@1=0.423  p@3=0.790  p@5=0.863   06:47
  ep3:            p@1=0.417  p@3=0.787  p@5=0.861   06:46
  ep4:            p@1=0.408  p@3=0.794  p@5=0.869   07:01
  ep5:            p@1=0.421  p@3=0.796  p@5=0.866   06:46
  ep6:            p@1=0.425  p@3=0.801  p@5=0.873   46:39  ← spillover starts (16.6GB VRAM)
  ep7:            >80min projected at 4.36 s/step  → user aborted

ep6 ckpt is v6's best on every metric AND beats v5 ep9 in-domain on
every metric (v5 was 0.418 / 0.789 / 0.864). Slow-LR strategy works.

==============================================================================
Diagnosis: PyTorch caching allocator fragmentation
------------------------------------------------------------------------------
nvidia-smi mid ep7:
  VRAM:  15876 / 16303 MiB  (97.4%, only 427 MiB free)
  Power: 85W / 300W         (28%; GPU underutilised but kernels at 100%)

Symptom = "GPU-Util 100% + power far below TDP" = textbook
PCIe-bottlenecked GPU = VRAM is full, PyTorch is using shared GPU memory
(system RAM via PCIe 4.0 ~32 GB/s vs VRAM ~700 GB/s = 22x slower).

Root cause: default PyTorch caching allocator is slab-based (fixed-size
blocks). After ~6 epochs of v6's mixed allocation pattern (train batch +
val batch + matplotlib figures + ckpt saves), the free pool fragments.
Even though >1GB technically free, PyTorch can't find a single
contiguous block large enough for new gradient tensors -> falls back to
shared GPU memory.

Contributing factors:
  - 21 non-training GPU processes shared the same 16GB (Cursor, Edge
    WebView2, Epic Games, Steam, NVIDIA App, asus_framework etc.),
    silently consuming 2-4GB
  - v6's PERSISTENT_WORKERS=True kept 6 workers + their pinned memory
    buffers alive across 50 epochs, increasing total memory pressure

==============================================================================
Cure (deployed in run_m3fd_v6_1_finetune.bat, NOT in this config)
------------------------------------------------------------------------------
  set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

This switches PyTorch allocator from slab-based to CUDA Virtual Memory
API. Physical pages can be non-contiguous but get mapped into virtually
contiguous segments via the GPU's MMU. Result: fragmentation cannot
occur because virtual addresses are always contiguous from the
allocator's view. Cost: ~2-3 ms extra on first allocation of each
segment, zero ongoing cost.

Plus operator discipline (close non-training GPU apps before launch):
  - Cursor / Edge WebView2 / Epic Games / Steam / NVIDIA App
  - asus_framework / TranslucentTB / Notepad
  - Run v6.1 from independent WindowsTerminal, NOT Cursor's integrated
    terminal (closing Cursor would kill the training)

==============================================================================
Why this config has ZERO override vs v6
------------------------------------------------------------------------------
User decision: Strategy A (full v6 schedule replay) over B (simulated
resume with shrunk max_epochs). Rationale:

  - LR math stays clean: TRUE_LR=2.5e-5, MSLR at ep15/25/35
  - --ckpt_path semantics reset AdamW second moments to 0, so a proper
    warmup is needed anyway (50 step = 800 abs)
  - Wasting 1 epoch of warmup on a converged ckpt is 2% of total
    50-epoch budget, negligible
  - One-variable-at-a-time discipline: only expandable_segments changes
    between v6 and v6.1, so any speed/quality diff is attributable
    purely to the cure (not a confounded LR change)
  - prefetch_factor=1 / batch_size=2 / mixed precision were considered
    and explicitly rejected (system RAM not VRAM / LR math change /
    fp16 numerical risk respectively)

Inherited from v6 (verified -- do NOT re-set, would invalidate v0-v5
inheritance discipline):
  CANONICAL_LR=4e-4 / WARMUP_STEP=50 / MSLR_MILESTONES=[15,25,35]
  EARLY_STOPPING_PATIENCE=12
  PERSISTENT_WORKERS=True / N_VAL_PAIRS_TO_PLOT=1
  (and entire v5 -> v4 -> v3 -> v2 -> v1 chain)

==============================================================================
Validation gates (additive to v6's gates 1-6 in eloftr_full_v6_finetune.py)
------------------------------------------------------------------------------
Gate 7 (NEW, allocator sanity): nvidia-smi after epoch 1 should show
       python.exe VRAM ~12-13 GB and STABLE across epoch 2-10. If it
       grows past 14 GB by epoch 5, expandable_segments did not
       propagate -- check the bat's `set` command order (must be BEFORE
       `python train.py`, AFTER `call conda activate`).

Gate 8 (NEW, no second spillover): per-epoch wall time should stay
       ~7 min ALL THROUGH epoch 50. If epoch N suddenly jumps to
       >15 min, an external GPU app started consuming VRAM (browser
       tab loaded heavy WebGL etc.) -- not a v6.1 regression, just
       close the app and continue.

Gate 9 (NEW, ckpt resume sanity): epoch 0 val end p@1 MUST be
       >= 0.420 (v6 ep6 was 0.425, slight regression OK due to AdamW
       second moment reset). If < 0.40, ckpt loaded but graph
       mismatch silently re-init some weights -- check v6.1 launch
       log for "missing_keys" / "unexpected_keys" output.

==============================================================================
Expected outcomes
------------------------------------------------------------------------------
Strong success (M3FD val p@1 >= 0.55 + RoadScene OOD p@1 >= 0.25):
  Slow-LR refinement closes the gap. Ship v6.1 as final cross-modal
  finetune. Optional: v7 MSBN (eloftr-cross-modal-experiments §6 path
  E1) to push p@1 toward 0.65+.

Weak success (M3FD val p@1 in 0.45-0.55 + OOD didn't improve):
  Slow LR helped but fine path's modality-blindness is the structural
  ceiling. Move to v7 MSBN per §6 path E1.

Failure (p@1 stuck at 0.42 from epoch 0 onward):
  Either ckpt didn't load (gate 9 failed) or the full v6 schedule
  truly has zero learning effect. Try v6.2 with CANONICAL_LR raised
  to 6e-4 ~ 8e-4.

Spillover regression (epoch N suddenly slow even with
expandable_segments):
  Check gate 8 root cause first (external GPU app). If not external,
  v6.2 fallback: also disable PERSISTENT_WORKERS + drop prefetch_factor
  to 1 (would need src/lightning/data.py edit).
"""
from configs.loftr.eloftr_full_v6_finetune import cfg
# (no override -- v6.1 schedule == v6 schedule by design; the only diff
# from v6 is the .bat-level expandable_segments env var + --ckpt_path)
