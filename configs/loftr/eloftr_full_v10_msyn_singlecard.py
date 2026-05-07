"""v10 mode A: v9 stack on Megadepth_Syn, single GPU validation run.

Style A inheritance from v9_e2e: keep the entire v1-v8 stack opt-in flags
(USE_CONTRASTIVE / USE_MODALITY_EMB / FREEZE_BACKBONE_BN / USE_EDGE_INPUT /
USE_CLAHE_IR / USE_MSBN / BACKBONE_IN_CHANNELS=2) byte-identically inherited.
Only schedule fields are overridden because the dataset went from v9 M3FD
~3.78K pair to Megadepth_Syn ~115K pair (31x larger, same Homography
supervision contract). Per-epoch step density is ~30x higher so training
saturates much earlier.

LR / WARMUP timeline (single GPU bs=4):
  _scaling = (4 * 1) / 64 = 0.0625
  TRUE_LR = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4 (== v9)
  actual WARMUP step = WARMUP_STEP / _scaling = 2 / 0.0625 = 32 step

Per epoch step count (~115K pair / bs=4 = ~28750 step), so:
  warmup    : step 0-32        (<1% of one epoch, basically instant)
  full LR   : ep 0-2           (acclimate from outdoor.ckpt to Megadepth_Syn)
  MSLR #1   : ep 2-4           (LR /= 10 -> 1.25e-5, primary refinement)
  MSLR #2   : ep 4-6           (LR /= 10 -> 1.25e-6, late polish)
  MSLR #3   : ep 6-8/10        (LR /= 10 -> 1.25e-7, final stabilisation)

ES patience=3 means earliest stop ep ~2+3=5, latest ep 8-10. v10 is mostly
to validate v9 stack works on the new data; mode B 4-card DDP is the
production run.

This single-card cfg is mainly for the 5-min `--max_epochs=1
--limit_train_batches=10 --disable_ckpt` debug + as a fallback if mode B
DDP misbehaves and we need a strict baseline.
"""
from configs.loftr.eloftr_full_v9_e2e import cfg

# v10 schedule: 31x data scale-down means schedule shrinks from 80 ep to ~10
cfg.TRAINER.MSLR_MILESTONES = [2, 4, 6]
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3

# v5 sampler: MUST opt-in N_SAMPLES_PER_SUBSET to full set, otherwise default
# 200 per-scene quota collapses to per-dataset 200/epoch (see
# eloftr-cross-modal-experiments SKILL.md SS3). Single-card uses train full set.
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 115000
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False

# bat / sh: --max_epochs=10, --batch_size=4, --gpus=1, --disable_mp, --thr 0.1
