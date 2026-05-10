"""v11 = v10_msyn_ddp + dual-side aggressive Homography augmentation.

Inherits v10_msyn_ddp byte-identical except for the 5 fields below:

  DATASET.ROAD_HOMOGRAPHY_DUAL    : False -> True   (warp BOTH IR and VIS)
  DATASET.ROAD_HOMOGRAPHY_PROB    : 1.0   -> 0.7    (30% pair stays identity)
  DATASET.ROAD_HOMOGRAPHY_KWARGS  : {}    -> aggressive preset
                                     rot_deg=25.0, scale_range=(0.75,1.25),
                                     trans_ratio=0.12, persp_ratio=0.08
  TRAINER.WARMUP_STEP             : 450   -> 900    (actual 1800 -> 3600 step,
                                                     ~0.5 ep, defensive ramp
                                                     under aug-shocked cold start)
  TRAINER.EXP_NAME                : msyn_v10_ddp -> msyn_v11_dualh_aggressive_ddp

What stays IDENTICAL to v10_msyn_ddp (do not override here):

  CANONICAL_LR=5e-4 -> TRUE_LR=1.25e-4 (LR independent of aug strength;
    loss internally normalises by valid GT count)
  MSLR_MILESTONES=[3,5,7], MSLR_GAMMA=0.5
  EARLY_STOPPING_PATIENCE=3 (val homography_aug=False, P@3 saturation pattern
    unchanged from v10)
  N_SAMPLES_PER_SUBSET=28750 per rank (full 115K per epoch, 4-rank disjoint
    via image-level pre-shard in src/lightning/data.py)
  FREEZE_BACKBONE=False, FREEZE_BACKBONE_BN=False (v10 unfreeze sustained)
  bs=4 x 4 GPU = effective_bs=16, 7187 step/ep/rank
  ckpt_path=weights/eloftr_outdoor.ckpt (cold start = fair v10 ablation)
  sync_batchnorm=True (auto-enabled by train.py L237 when WORLD_SIZE>1)
  v9 stack: USE_CONTRASTIVE / USE_MODALITY_EMB / USE_EDGE_INPUT /
    USE_CLAHE_IR / USE_MSBN / BACKBONE_IN_CHANNELS=2 (all True via v9_e2e
    inheritance through v10_msyn_singlecard)

Wall-clock estimate: ~17h for 12 ep on 4-card 3090 sync_bn DDP (same as v10
because dual aug ~negligible CPU cost vs network forward/backward).

Aug strength rationale (vs v0-v10 single-side and SuperPoint COCO baseline):

  axis        v0-v10    v11 (this)   SuperPoint COCO
  rot_deg     +/-10     +/-25         +/-90
  scale       0.9-1.1   0.75-1.25    0.8-1.2
  trans       +/-5%     +/-12%       ~+/-15%
  persp       +/-3%     +/-8%        +/-20%
  dual                  yes          no (single-side)
  prob         1.0      0.7           1.0

Per-pair covisibility (intersect(IR_valid, VIS_valid) / total):
  v0-v10 single weak  : ~66%
  v11 single equiv    : ~32%   (=hypothetical aggressive single-side)
  v11 dual            : ~32% x ~32% = ~10%  (per-pair when prob fires)
  v11 dual + prob=0.7 : 0.7 x 10% + 0.3 x 100% = ~37% average

The 0.7 prob keeps average covisibility ~37%, in the same ballpark as v10
(~66%), so per-batch GT count stays in a regime where the loss has
sufficient signal (see plan SS "共视坍塌" analysis).

Acceptance grading (v11 ship; matches v10 thresholds because LR/MSLR/data
identical, only aug differs):

  strong : peak ep 2-7, M3FD-OOD test P@1 >= 0.50, RoadScene OOD P@1 >= 0.45
  medium : peak ep 2-9, M3FD-OOD P@1 in [0.40, 0.50], RoadScene OOD P@1 >= 0.40
  weak   : peak ep 2-11, M3FD-OOD P@1 in [0.30, 0.40]
  fail   : ep0 train_loss > 3.0 sustained, OR M3FD-OOD P@1 < 0.30

If fail mode hits because of cold-start crash (ep0 loss > 3.0), bump
WARMUP_STEP to 1800 (actual 7200 step ~1 ep) and PROB to 0.5, retry.
"""
from configs.loftr.eloftr_full_v10_msyn_ddp import cfg


cfg.DATASET.ROAD_HOMOGRAPHY_DUAL = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 0.7
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08

cfg.TRAINER.WARMUP_STEP = 900   # v10 was 450 (~0.25 ep); doubled for cold start under aggressive aug shock

cfg.TRAINER.EXP_NAME = "msyn_v11_dualh_aggressive_ddp"
