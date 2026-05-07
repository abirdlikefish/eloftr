"""v10 mode B: v9 stack on Megadepth_Syn, 4-GPU DDP ship run.

Inherits the single-GPU cfg and reverse-scales LR / WARMUP so 4-card
effective TRUE_LR matches v9 single-card 1.25e-4 (avoiding the 4x LR
amplification trap documented in eloftr-server-multigpu SS4.1).

LR / WARMUP timeline (4-GPU DDP bs=4, effective_batch=16):
  _scaling = (4 * 4) / 64 = 0.25
  TRUE_LR = CANONICAL_LR * _scaling = 5e-4 * 0.25 = 1.25e-4 (==v9 single-card)
  actual WARMUP step = WARMUP_STEP / _scaling = 0 / 0.25 = 0 (skip warmup,
  Megadepth_Syn is large enough + outdoor.ckpt is a good init)

Per epoch step count (~115K pair / 16 effective_batch = ~7187 step), so:
  warmup    : (skipped, 0 step)
  full LR   : ep 0-2           (acclimate from outdoor.ckpt to Megadepth_Syn)
  MSLR #1   : ep 2-4           (LR /= 10)
  MSLR #2   : ep 4-6           (LR /= 10)
  MSLR #3   : ep 6-8/10        (LR /= 10)

DDP traps to monitor in mode B debug (see eloftr-server-multigpu SS4):
  - SS4.1 LR scaling: handled (CANONICAL_LR=5e-4 reverse-scaled)
  - SS4.2 sync_bn x MSBN: kept enabled (4x sample size for MSBN dual BN
    convergence stability); accept v9 byte-mismatch
  - SS4.3 RandomConcatSampler not sharded: monitor [rank N] log lines;
    if scenario 1 (same-seed -> 4x duplicated), N_SAMPLES=29K saturates train;
    if scenario 2 (diff-seed -> 4x distinct), 4-rank * 29K = 115K = full set
  - SS4.4 non-determinism: accept +/- 0.005
  - SS4.5 large-batch gen gap: bs=16 still in safe zone

This is the v10 ship cfg. Produces logs/tb_logs/m3fd_v10_msyn/ ckpts.
"""
from configs.loftr.eloftr_full_v10_msyn_singlecard import cfg

# 4-card reverse scaling: TRUE_LR = CANONICAL_LR * (16/64) = CANONICAL_LR * 0.25
# Want TRUE_LR = 1.25e-4 (match v9 single-card) -> CANONICAL_LR = 5e-4
cfg.TRAINER.CANONICAL_LR = 5e-4
cfg.TRAINER.WARMUP_STEP = 0

# Sampler quota for 4-rank DDP: 4 * 29000 = 116K ~= train full set 115K.
# Robust to both scenario 1 (same-seed dup) and scenario 2 (diff-seed split).
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 29000

# bat / sh: --max_epochs=10, --batch_size=4, --gpus=4, --disable_mp, --thr 0.1
