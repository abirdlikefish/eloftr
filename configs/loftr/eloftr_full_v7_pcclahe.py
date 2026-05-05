"""v7_pcclahe: input-side cross-modal optimization (A1 PC edge channel + A2 CLAHE on IR).

Continues v6.1's all-M3FD slow-LR finetune from the v6.1 ep4 ckpt, but adds
two input-side improvements:

  A1 (PC edge map as 2nd input channel) -- modality-invariant geometric prior.
      RepVGG stage0 in_channels: 1 -> 2. The new (gray, PC) tensor is stacked
      in the dataset; lightning_loftr._maybe_inflate_stage0 zero-extends the
      v6.1 ckpt's stage0 conv weights so the PC channel starts contributing
      nothing (alpha=0) and the model is mathematically equivalent to v6.1
      at epoch 0. Gradient pushes the PC weight away from 0 from epoch 1.
      Pre-compute the PC cache via MyScripts/precompute_pc_edges.bat (defaults
      to both M3FD + RoadScene -- needed for in-domain training AND OOD eval).

  A2 (CLAHE on raw uint8 IR) -- per-tile contrast-limited histogram equalisation.
      Boosts contrast in IR's typically narrow 80-120 histogram peak; brings
      the IR pixel distribution closer to VIS's near-uniform spread, easing
      the burden on stage0 conv to "compensate the IR/VIS distribution gap".
      VIS CLAHE is left disabled by default (V7.3 ablation reserves it).
      CLAHE is NOT applied to the PC channel (PC is sparse-edge and CLAHE
      would destroy that sparsity).

Inherits everything else from v6.1 (which inherits from v6, v5, v4 REVISION 2,
v3, v2, v1):
  - schedule:    TRUE_LR=2.5e-5 / WARMUP_STEP=50 / MSLR=[15,25,35] / ES patience=12
  - architecture: contrastive (v1) + modemb (v2) + FREEZE_BACKBONE_BN=True (v4)
  - data:        M3FD 3780 train + 210 val (v5)
  - resume:      v6.1 ep4 ckpt via --ckpt_path in run_m3fd_v7_pcclahe.bat
  - operator:    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (v6.1)

==============================================================================
Multi-variable change vs the v0-v6.1 one-variable-at-a-time discipline
------------------------------------------------------------------------------
v7 ships A1 + A2 together rather than one at a time. This is intentional --
the input-side optimisation theme is most credibly demonstrated as a package,
and the additional epochs to split A1/A2/A1+A2 cleanly are reserved for
v7.1 / v7.2 / v7.3 ablation runs after v7's outcome is known. See plan
v7_pcclahe trade-off discussion + SKILL.md S6 path F.

==============================================================================
v7-specific overrides (3 cfg flags only -- everything else inherits v6.1)
------------------------------------------------------------------------------
  USE_EDGE_INPUT       = True    enables A1 (dataset stacks PC, model in_ch=2)
  BACKBONE_IN_CHANNELS = 2       must match the dataset's stacked output
  USE_CLAHE_IR         = True    enables A2 on raw IR (only)

==============================================================================
Validation gates (additive to v6.1 gates 1-9)
------------------------------------------------------------------------------
Gate 10 (PC cache 完整性):
  Before launching, MyScripts/precompute_pc_edges.bat MUST have been run.
  M3FD 4200 imgs -> data/M3FD_Detection/Ir_pc/ + Vis_pc/ ; RoadScene 222 imgs
  -> data/RoadScene/cropinfrared_pc/ + crop_LR_visible_pc/. Missing -> the
  RoadSceneDataset __init__ raises FileNotFoundError with a clear hint.

Gate 11 (CLAHE lazy init):
  Startup log shows
    "RoadSceneDataset: CLAHE enabled (clipLimit=2.0, tile=(8, 8), ir=True, vis=False)"
  for both train and val datasets. Workers each lazy-init their own cv2.CLAHE
  instance on the first __getitem__ call (avoids pickling the C++ object
  across DataLoader spawn).

Gate 12 (inflated init mathematical equivalence):
  Startup log shows
    "Inflated stage0 conv weights: 1ch -> 2ch (alpha=0.0, ...)"
  AND epoch 0 val end p@1 >= 0.443  (= v6.1 ep4's value).
  If p@1 << 0.443, the inflated init alpha is wrong (not zero) or the PC
  channel weight got initialised non-zero somehow. Halt and debug.

Gate 13 (channel shape):
  In the model summary printed by PL, line for `matcher.backbone.layer0.rbr_dense.conv`
  should show 1.2K params (= 64*2*3*3) instead of v6.1's 0.6K (64*1*3*3).
  Equivalent: Trainable params goes from v6.1's 16.00M to 16.00M + 0.0006M
  = barely moved (PC channel adds only ~640 scalars).

==============================================================================
Expected outcomes (relative to v6.1 ep4 = M3FD val p@1 0.443 / p@3 0.814)
------------------------------------------------------------------------------
Strong success:
  M3FD val p@1 >= 0.46 (>=+3% rel) AND OOD (RoadScene test 22) p@1 >= 0.18
  (recovers some of v5's 0.186). Ship v7 as the final cross-modal finetune.
  Then run v7.1 / v7.2 ablation to attribute the gain to A1 vs A2.

Weak success:
  M3FD val p@1 in [0.443, 0.46], OOD trade-off does not regress past v6.1's
  0.176. Ship v7; ablation reveals which of A1/A2 carries the signal.

Failure:
  M3FD val p@1 < 0.443 (i.e. v7 is worse than v6.1 at the same epoch). Means
  PC/CLAHE adds noise on top of v6.1's already converged input statistics.
  Roll back to v6.1 ep4 as final ckpt; write a negative-result paragraph in
  the thesis explaining input-side optimisation hits a ceiling once the
  backbone has fully adapted to the IR distribution.

==============================================================================
"""
from configs.loftr.eloftr_full_v6_1_finetune import cfg

cfg.LOFTR.USE_EDGE_INPUT = True
cfg.LOFTR.BACKBONE_IN_CHANNELS = 2
cfg.LOFTR.USE_CLAHE_IR = True
# CLAHE_CLIP_LIMIT=2.0, CLAHE_TILE_SIZE=[8,8], USE_CLAHE_VIS=False, and the
# entire v6.1 schedule (TRUE_LR / WARMUP_STEP / MSLR / ES / FREEZE_BACKBONE_BN
# / PERSISTENT_WORKERS / N_VAL_PAIRS_TO_PLOT) all inherit from v6.1 untouched.
