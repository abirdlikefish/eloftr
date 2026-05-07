"""v9_e2e: comprehensive cold start from outdoor.ckpt with v1-v8 full stack.

Not a v8 reproduction (v8 is from v7 ep6 resume, ~50 ep cumulative finetune chain
on M3FD). v9_e2e collapses v0-v8 into one ~80-epoch run from the MegaDepth
paper baseline (weights/eloftr_outdoor.ckpt), evaluating whether the v0-v7
intermediate finetune steps can be skipped without losing the v8 SOTA score.

==============================================================================
Style A inheritance (per .cursor/plans/20260506-2155-v9-e2e-cold-start.md)
------------------------------------------------------------------------------
Inherit from `eloftr_full` (v0 paper baseline) directly, NOT from any v_x
config in the v1->v2->...->v8 chain. This ensures:
  - cfg-level changes between v_x versions cannot accidentally leak into v9
  - all v1-v8 features are explicitly opt-in here, easy to audit
  - schedule fields are fully owned by v9 (no inherited dead code)

The src/ code paths for v1-v8 are already landed and gated by cfg flags
(default False); enabling them here triggers the full stack:
  - v1 contrast loss path (loftr_loss.py)
  - v2 modality embedding (loftr.py module)
  - v3/v4 freeze logic (lightning_loftr.py train() override)
  - v5 M3FD sampler opt-in (data.py / sampler.py)
  - v7 PC channel + CLAHE (roadscene.py + repvgg.py + _maybe_inflate_stage0)
  - v8 MSBN (fine_preprocess.py + _maybe_inflate_msbn)

==============================================================================
Hooks behaviour with outdoor.ckpt (strict=False load + 2 inflate hooks)
------------------------------------------------------------------------------
load order in src/lightning/lightning_loftr.py:230-243:
  1. _maybe_inflate_stage0:
     stage0 conv [64, 1, 3, 3] -> [64, 2, 3, 3], alpha=0 zero-extend
     (PC channel starts as no-op, learns from scratch)
  2. _maybe_inflate_msbn:
     fine_preprocess.layer{1,2}_outconv2.1 (single BN) ->
       _bn_ir + _bn_vis (zero-shift copy)
     (NOTE: copies MegaDepth-domain BN, not v7 ep6 M3FD-finetuned BN -- the
      MSBN dual branches start more "raw" than in v8 resume. Compensated by
      the longer 80-ep schedule.)
  3. matcher.load_state_dict(state_dict, strict=False):
     Missing keys: matcher.modality_embedding_ir/vis (v2 introduced),
     auto-initialised via MODALITY_EMB_INIT='zeros' (safe: step 0 == baseline).

==============================================================================
Schedule design (cold start, bs=4 single GPU)
------------------------------------------------------------------------------
train.py:124-130 auto-scales LR / WARMUP based on TRUE_BATCH_SIZE / CANONICAL_BS:
  _scaling = (4 * 1) / 64 = 0.0625
  TRUE_LR = CANONICAL_LR * _scaling = 2e-3 * 0.0625 = 1.25e-4 (v5 level fast LR)
  actual WARMUP step = WARMUP_STEP / _scaling = 60 / 0.0625 = 960 step ~ 1 ep

LR timeline (M3FD ~945 step/epoch @ bs=4, 80 ep total = 75,600 step):
  warmup    ep 0-1.02   ->  0 - 960 step    | 0 -> 1.25e-4 | linear ramp
  full LR   ep 1-35     ->  960 - 33,075    | 1.25e-4      | M3FD adaptation + backbone training
  MSLR #1   ep 35-55    ->  33,075 - 51,975 | 6.25e-5      | first decay near peak (peak est ep 40-50)
  MSLR #2   ep 55-70    ->  51,975 - 66,150 | 3.13e-5      | refinement (= v8 mid-late LR)
  MSLR #3   ep 70-80    ->  66,150 - 75,600 | 1.56e-5      | final stabilization (ES likely triggers here)

Peak position estimate ep 40-55 (50-70% of max_ep), based on:
  - v7 peaked ep 6/10 = 60%
  - v8 peaked ep 16/20 = 80%
  - cold start has no good init (favors later peak) but uses 5x LR vs v8 (favors earlier peak)

ES patience=20 means earliest stop at ep 35+20=55 (if peak at ep 35 and val
monotonically degrades after), latest at ep 80. Compatible with max_ep=80.

==============================================================================
Acceptance grading (cold start, NOT measuring against v8 0.5494)
------------------------------------------------------------------------------
strong  : peak ep 35-55, val p@1 >= 0.51, OOD p@1 not regress past 0.21
medium  : peak ep 35-65, val p@1 in [0.45, 0.51], OOD not regress past 0.21
weak    : peak ep 35-78, val p@1 in [0.40, 0.45]
fail    : val p@1 < 0.40 across all epochs
"""
from configs.loftr.eloftr_full import cfg

# v1: symmetric InfoNCE (cross-modal contrastive loss)
cfg.LOFTR.LOSS.USE_CONTRASTIVE = True
cfg.LOFTR.LOSS.CONTRASTIVE_WEIGHT = 0.01
cfg.LOFTR.LOSS.CONTRASTIVE_TEMP = 0.1

# v2: learnable modality embedding (added to coarse tokens)
cfg.LOFTR.USE_MODALITY_EMB = True
cfg.LOFTR.MODALITY_EMB_INIT = 'zeros'      # safest: step 0 == baseline (no perturbation)

# v3/v4 freeze: cold start unfreezes backbone (need 9.5M params to adapt to IR-VIS),
# but pins backbone BN running stats (v4 REVISION 2 design: prevent small-data
# BN drift to extreme stats while letting fine BN adapt)
cfg.LOFTR.FREEZE_BACKBONE = False
cfg.LOFTR.FREEZE_BN = False
cfg.LOFTR.FREEZE_BACKBONE_BN = True
cfg.LOFTR.FREEZE_FINE_BN_IR = False
cfg.LOFTR.FREEZE_FINE_BN_VIS = False

# v7: input-side optimization
#   A1: Phase Congruency edge map as 2nd input channel (modality-invariant)
#   A2: CLAHE histogram equalization on raw IR (narrow-histogram boost)
cfg.LOFTR.BACKBONE_IN_CHANNELS = 2          # stage0 conv now [64, 2, 3, 3]
cfg.LOFTR.USE_EDGE_INPUT = True             # dataset stacks PC alongside gray
cfg.LOFTR.USE_CLAHE_IR = True
cfg.LOFTR.USE_CLAHE_VIS = False
cfg.LOFTR.CLAHE_CLIP_LIMIT = 2.0
cfg.LOFTR.CLAHE_TILE_SIZE = [8, 8]

# v8: Modality-Specific BatchNorm on fine_preprocess
# fine_preprocess.layer{1,2}_outconv2.1 split into bn_ir / bn_vis dual branches
cfg.LOFTR.USE_MSBN = True

# v9 cold start schedule (revised 2026-05-06 per LR-timeline analysis above).
# bs=4 single GPU: _scaling = 0.0625, so cfg field values are auto-scaled by
# train.py:127-130. See module docstring for the full LR timeline.
cfg.TRAINER.CANONICAL_LR = 2e-3              # -> TRUE_LR = 1.25e-4 (v5-level fast LR)
cfg.TRAINER.WARMUP_STEP = 60                 # -> actual 960 step ~ 1.02 ep warmup
cfg.TRAINER.MSLR_MILESTONES = [35, 55, 70]   # epoch values; first decay near peak
cfg.TRAINER.EARLY_STOPPING = True
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 20
cfg.TRAINER.PERSISTENT_WORKERS = True        # v6.1 inheritance, save ~30s/epoch worker spin-up

# v5: M3FD sampler opt-in (default N_SAMPLES_PER_SUBSET=200 = per-scene quota,
# undersamples M3FD's flat 3780-pair list at 5.3% per epoch). See
# .cursor/skills/eloftr-cross-modal-experiments SS3.
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 3780
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False

# bat / sh: --max_epochs=80, --batch_size=4, --disable_mp, --thr 0.1
