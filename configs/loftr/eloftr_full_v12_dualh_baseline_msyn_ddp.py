"""v12 = baseline + v11 dual-side aggressive Homography aug + Megadepth_Syn 4-GPU DDP.

==============================================================================
Ablation goal
------------------------------------------------------------------------------
Isolate the contribution of v11's dual-side aggressive Homography aug, with
all v1-v8 USE_* (contrastive / modemb / PC / CLAHE / MSBN) explicitly OFF
(default values). Direct inheritance from eloftr_full.py (NOT v9_e2e -> v10
chain) so the diff against the EfficientLoFTR paper baseline is exactly:

  (a) v11 input-side dual H aug (7 fields below)
  (b) Megadepth_Syn 4-GPU DDP infra (6 fields below)

This answers the question: "How much of v11's SOTA jump comes from the dual H
aug alone, when stripped of the v1-v8 cumulative model/loss/freeze stack?"

==============================================================================
Inheritance choice: short-circuit from eloftr_full (v0 baseline)
------------------------------------------------------------------------------
v11 chain: eloftr_full -> v9_e2e -> v10_msyn_singlecard -> v10_msyn_ddp -> v11
v12 chain: eloftr_full -> v12     (THIS file)

Going through v9_e2e would silently re-enable USE_CONTRASTIVE / USE_MODALITY_EMB
/ USE_EDGE_INPUT / USE_CLAHE_IR / USE_MSBN / BACKBONE_IN_CHANNELS=2; explicit
overrides back to False would clutter v12 cfg with ~10 more lines and break the
"audit cfg = read this file alone" property. Short-circuit inheritance keeps
v12's diff against baseline minimal and auditable.

What v12 INHERITS from baseline (eloftr_full.py + src/config/default.py):
  USE_CONTRASTIVE     = False  (default)   v11 had True (v1)
  USE_MODALITY_EMB    = False  (default)   v11 had True (v2)
  BACKBONE_IN_CHANNELS= 1      (default)   v11 had 2    (v7)
  USE_EDGE_INPUT      = False  (default)   v11 had True (v7)
  USE_CLAHE_IR        = False  (default)   v11 had True (v7)
  USE_MSBN            = False  (default)   v11 had True (v8)
  FREEZE_BACKBONE     = False  (default)   v11 had False (=== same)
  FREEZE_BN           = False  (default)   v11 had False (=== same)
  FREEZE_BACKBONE_BN  = False  (default)   v11 had False (=== same; v9 had True
                                                          but v10 reverted it)
  FREEZE_FINE_BN_*    = False  (default)   v11 had False (=== same)

Net effect on the model graph: v12 is the EfficientLoFTR paper architecture
unchanged (single-channel raw image, single BN per layer, no contrastive loss,
no modality embedding, no freeze). The ONLY difference vs running the paper
baseline on Megadepth_Syn is the v11 input-side dual H aug.

==============================================================================
LR / WARMUP timeline (4-GPU DDP bs=4, effective_bs=16)
------------------------------------------------------------------------------
train.py:124-130 auto-scales LR / WARMUP based on TRUE_BATCH_SIZE / CANONICAL_BS:
  _scaling = (4 * 4) / 64 = 0.25
  TRUE_LR  = CANONICAL_LR * _scaling = 5e-4 * 0.25 = 1.25e-4   (== v11)
  actual WARMUP step = WARMUP_STEP / _scaling = 900 / 0.25 = 3600 step
                     ~ 0.5 ep per rank (== v11 cold-start aug-shock ramp)

Per epoch step count (28750 pair per rank shard / bs=4 = 7187 step per rank):
  warmup    : ep 0-0.5         (3600 step linear ramp 0 -> 1.25e-4)
  full LR   : ep 0.5-3         (acclimate from outdoor.ckpt under aug shock)
  MSLR #1   : ep 3-5           (LR x= MSLR_GAMMA=0.5 -> 6.25e-5)
  MSLR #2   : ep 5-7           (LR x= 0.5 -> 3.13e-5)
  MSLR #3   : ep 7-12          (LR x= 0.5 -> 1.56e-5, final stabilisation)

==============================================================================
Cold start: weights/eloftr_outdoor.ckpt strict=False load
------------------------------------------------------------------------------
load order in src/lightning/lightning_loftr.py:230-243:
  1. _maybe_inflate_stage0:
       NO-OP because in_channels=1 (default) matches ckpt's [64, 1, 3, 3].
  2. _maybe_inflate_msbn:
       NO-OP because USE_MSBN=False (default) -- model has single BN per layer
       which matches ckpt structure exactly.
  3. matcher.load_state_dict(state_dict, strict=False):
       missing_keys:    []  (no modality_emb_ir/vis to insert because
                              USE_MODALITY_EMB=False)
       unexpected_keys: []  (ckpt is the same arch as the model)

Same logical "cold start" as v11 (same outdoor.ckpt, same strict=False),
but the ckpt -> model load is byte-clean rather than relying on inflate
fallbacks. This makes v12 a maximally clean baseline.

==============================================================================
Acceptance grading (vs v11 strong: M3FD-OOD P@1 >= 0.50, RoadScene-OOD >= 0.45)
------------------------------------------------------------------------------
H_isolated_strong : M3FD-OOD P@1 >= 0.30, RoadScene-OOD P@1 >= 0.25
                    (= dual H aug alone closes ~50% of the gap from
                     v0 outdoor.ckpt baseline ~0.21 to v11 0.45+;
                     would be a big result -- means most of v11 SOTA is
                     from the aug, not from the v1-v8 stack)
H_isolated_medium : M3FD-OOD P@1 in [0.20, 0.30], RoadScene-OOD in [0.18, 0.25]
                    (dual H aug helps modestly without v9 stack support;
                     means v1-v8 stack and dual H aug are roughly additive)
H_isolated_weak   : M3FD-OOD P@1 in [0.15, 0.20]
                    (dual H aug needs v1-v8 stack to be effective;
                     means v11's dual H aug is multiplicative on top of v9)
H_isolated_fail   : M3FD-OOD P@1 < 0.15 OR ep0 train_loss > 3.0 sustained
                    (cold-start aug-shock too severe without modemb/MSBN
                     to absorb modality variance; retry with PROB=0.5 +
                     WARMUP_STEP=1800)

==============================================================================
Risk note
------------------------------------------------------------------------------
v12 has a higher cold-start crash risk than v11 because it lacks:
  - modemb to absorb IR-vs-VIS distribution shift at coarse-token level
  - MSBN dual BN to absorb IR-vs-VIS BN running-stat divergence
  - PC edge channel to provide modality-invariant geometric prior
So the same dual-aggressive aug that v11 tolerated at PROB=0.7 might be
too harsh on v12. Monitoring guidance is in the run script docstring.
If ep0 train_loss > 3.0 sustained, kill and retry with WARMUP_STEP=1800
(actual 7200 step ~1 ep) and PROB=0.5 (per the same fail-mode recovery
as v11).
"""
from configs.loftr.eloftr_full import cfg

cfg.DATASET.ROAD_HOMOGRAPHY_DUAL = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 0.7
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08

cfg.TRAINER.WARMUP_STEP = 900   # v11 cold-start aug-shock ramp; auto-scales to 3600 step under bs=16

cfg.TRAINER.CANONICAL_LR = 5e-4
cfg.TRAINER.MSLR_MILESTONES = [3, 5, 7]
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 28750
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False
cfg.TRAINER.PERSISTENT_WORKERS = True

# bat / sh: --max_epochs=12, --batch_size=4, --gpus=4, --disable_mp, --thr 0.1
# NOTE: exp_name passed via CLI --exp_name in MyScripts/run_msyn_v12_dualh_baseline_ddp.sh,
# NOT set on cfg. TRAINER.EXP_NAME is not a registered field in src/config/default.py
# (matches v10/v11 cfg pattern); setting it here would raise KeyError("Non-existent
# config key: TRAINER.EXP_NAME") in YACS strict mode at config.merge_from_file() time.
