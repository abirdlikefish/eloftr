"""v10 mode A: v9 stack on Megadepth_Syn, single GPU validation run.

Style A inheritance from v9_e2e: keep the entire v1-v8 stack opt-in flags
(USE_CONTRASTIVE / USE_MODALITY_EMB / USE_EDGE_INPUT / USE_CLAHE_IR /
USE_MSBN / BACKBONE_IN_CHANNELS=2) byte-identically inherited. Schedule +
sampler + freeze fields are overridden because the dataset went from v9
M3FD ~3.78K pair to Megadepth_Syn ~115K pair (31x larger, same Homography
supervision contract). Per-epoch step density is ~30x higher so training
saturates much earlier; the abundant data also makes the v9 backbone-BN
freeze unnecessary (see freeze section below).

LR / WARMUP timeline (single GPU bs=4):
  _scaling = (4 * 1) / 64 = 0.0625
  TRUE_LR = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4 (== v9)
  actual WARMUP step = WARMUP_STEP / _scaling = 1800 / 0.0625 = 28800 step
                                            ~ 1 epoch (matches v9's 1-ep ratio)

Per epoch step count (115K pair / bs=4 = 28750 step), so:
  warmup    : ep 0-1.0         (28800 step linear ramp 0 -> 1.25e-4)
  full LR   : ep 1-3           (acclimate from outdoor.ckpt to Megadepth_Syn)
  MSLR #1   : ep 3-5           (LR x= MSLR_GAMMA=0.5 -> 6.25e-5)
  MSLR #2   : ep 5-7           (LR x= 0.5 -> 3.13e-5)
  MSLR #3   : ep 7-12          (LR x= 0.5 -> 1.56e-5, final stabilisation)

NOTE on MSLR: src/config/default.py:280 sets MSLR_GAMMA=0.5 (NOT 0.1). v9
docstring is correct (its 6.25e-5 / 3.13e-5 / 1.56e-5 numbers). Earlier v10
docstring drafts incorrectly said "LR /= 10" -- that was a writing mistake,
the cfg always inherited GAMMA=0.5.

ES patience=3 + max_ep=12 -> earliest stop ep ~3+3=6, latest ep 12.
Estimated peak ep 1.7 (extrapolating v9's peak/max_ep ratio 65% by sample
passes), so MSLR_#1 at ep 3 should land just past peak.

This single-card cfg is mainly for the 5-min `--max_epochs=1
--limit_train_batches=10 --disable_ckpt` debug + as a fallback if mode B
DDP misbehaves and we need a strict baseline.
"""
from configs.loftr.eloftr_full_v9_e2e import cfg

# v10 unfreeze-all override: v9_e2e set FREEZE_BACKBONE_BN=True (v4 REVISION 2
# design: pin backbone BN running stats to MegaDepth pretrained values to
# prevent small-data BN drift to extreme stats on M3FD's ~3780 pair). On
# Megadepth_Syn ~115K pair (31x larger and ~80x more sample passes per ep)
# the small-data drift concern doesn't apply -- letting backbone BN running
# stats adapt to the new IR-VIS distribution should help, not hurt.
# All five freeze fields explicitly set False here so behaviour is auditable
# at a glance; this matches src/config/default.py defaults except for the
# overridden BACKBONE_BN flag.
cfg.LOFTR.FREEZE_BACKBONE = False
cfg.LOFTR.FREEZE_BN = False
cfg.LOFTR.FREEZE_BACKBONE_BN = False     # v9 was True; v10 abundant data -> unfreeze
cfg.LOFTR.FREEZE_FINE_BN_IR = False
cfg.LOFTR.FREEZE_FINE_BN_VIS = False

# v10 schedule: 31x more data per epoch -> shrink ep 80 -> ~12, keep MSLR
# milestones at the same 25%/40%/60% positions of max_ep.
cfg.TRAINER.MSLR_MILESTONES = [3, 5, 7]
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3

# WARMUP: keep "1 epoch warmup" ratio from v9. v9 had WARMUP_STEP=60 cfg ->
# 960 actual step ~= 1 ep on M3FD (945 step/ep). v10 has 28750 step/ep so
# 1-ep warmup needs WARMUP_STEP cfg = 1800 -> actual 28800 step. Inheriting
# v9's WARMUP_STEP=60 unchanged would give only 960 actual = 3.3% of ep1,
# too short for cold start with PC-channel + MSBN inflated weights.
cfg.TRAINER.WARMUP_STEP = 1800

# v5 sampler: MUST opt-in N_SAMPLES_PER_SUBSET to full set, otherwise default
# 200 per-scene quota collapses to per-dataset 200/epoch (see
# eloftr-cross-modal-experiments SKILL.md SS3). Single-card uses train full set.
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 115000
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False

# bat / sh: --max_epochs=12, --batch_size=4, --gpus=1, --disable_mp, --thr 0.1
