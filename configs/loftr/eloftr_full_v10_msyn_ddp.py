"""v10 mode B: v9 stack on Megadepth_Syn, 4-GPU DDP ship run.

Inherits the single-GPU cfg and reverse-scales LR / WARMUP so 4-card
effective TRUE_LR matches v9 single-card 1.25e-4 (avoiding the 4x LR
amplification trap documented in eloftr-server-multigpu SS4.1).

LR / WARMUP timeline (4-GPU DDP bs=4, effective_batch=16):
  _scaling = (4 * 4) / 64 = 0.25
  TRUE_LR = CANONICAL_LR * _scaling = 5e-4 * 0.25 = 1.25e-4 (==v9 single-card)
  actual WARMUP step = WARMUP_STEP / _scaling = 450 / 0.25 = 1800 step
                     ~ 0.25 ep per rank, ~1800 grad updates with effective_bs=16
                       = same total sample passes (28800) as mode A 1-ep warmup

Per epoch step count (28750 pair per rank shard / bs=4 = 7187 step per rank), so:
  warmup    : ep 0-0.25        (1800 step linear ramp 0 -> 1.25e-4)
  full LR   : ep 0.25-3        (acclimate from outdoor.ckpt to Megadepth_Syn)
  MSLR #1   : ep 3-5           (LR x= MSLR_GAMMA=0.5 -> 6.25e-5)
  MSLR #2   : ep 5-7           (LR x= 0.5 -> 3.13e-5)
  MSLR #3   : ep 7-12          (LR x= 0.5 -> 1.56e-5, final stabilisation)

NOTE on MSLR: src/config/default.py:280 sets MSLR_GAMMA=0.5 (NOT 0.1). v9
docstring is correct (its 6.25e-5 / 3.13e-5 / 1.56e-5 numbers). Earlier v10
docstring drafts incorrectly said "LR /= 10" -- that was a writing mistake,
the cfg always inherited GAMMA=0.5.

DDP traps to monitor in mode B debug (see eloftr-server-multigpu SS4):
  - SS4.1 LR scaling: handled (CANONICAL_LR=5e-4 reverse-scaled)
  - SS4.2 sync_bn x MSBN: kept enabled (4x sample size for MSBN dual BN
    convergence stability); accept v9 byte-mismatch
  - SS4.3 RandomConcatSampler not sharded: FIXED via image-level pre-sharding
    in src/lightning/data.py IR-VIS short-circuit branch (mirrors
    ScanNet/MegaDepth get_local_split pattern). Each rank now gets a
    disjoint 28750-sample shard; sampler operates locally as designed.
  - SS4.4 non-determinism: accept +/- 0.005
  - SS4.5 large-batch gen gap: bs=16 still in safe zone

This is the v10 ship cfg. Produces logs/tb_logs/msyn_v10_ddp/ ckpts.
"""
from configs.loftr.eloftr_full_v10_msyn_singlecard import cfg

# 4-card reverse scaling: TRUE_LR = CANONICAL_LR * (16/64) = CANONICAL_LR * 0.25
# Want TRUE_LR = 1.25e-4 (match v9 single-card) -> CANONICAL_LR = 5e-4
cfg.TRAINER.CANONICAL_LR = 5e-4

# WARMUP: defensive ~0.25 ep ramp (1800 step) instead of 0. With effective_bs=16,
# 1800 grad updates = 28800 sample passes = same total data exposure as mode A
# 1-ep warmup. cold start with TRUE_LR=1.25e-4 first step is risky even with
# zero-shift inflate hooks; cheap insurance.
cfg.TRAINER.WARMUP_STEP = 450

# Sampler quota = local shard size (= 115000 / 4 ranks = 28750), so each rank's
# RandomConcatSampler does randperm(28750) and takes all 28750 = full local
# shard. 4 ranks * 28750 unique = 115K = full train set per epoch (no overlap,
# no padding-with-replacement, no duplicates). The image-level pre-sharding
# happens in src/lightning/data.py IR-VIS branch via get_local_split, mirroring
# the ScanNet/MegaDepth scene-level sharding pattern (data.py L283).
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 28750

# bat / sh: --max_epochs=12, --batch_size=4, --gpus=4, --disable_mp, --thr 0.1
