"""v18 = v14 + single-side rotation-only Homography augmentation on image1
(VIS, MegaDepth-style cross-view pose supervision). image0 (IR) untouched,
image1 warped by H_vis sampled in resized pixel space; rot_deg=50, scale=1,
trans=0, persp=0, prob=1.0.

==============================================================================
 v18 vs v14: 1 ablation variable + supporting GT/inference plumbing
==============================================================================
v18 inherits v14 ddp byte-identical except for these 7 cfg overrides:

  cfg.DATASET.MGDPT_HOMOGRAPHY_AUG          False -> True
  cfg.DATASET.MGDPT_HOMOGRAPHY_PROB         (default 1.0, declared explicitly)
  cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.rot_deg          -> 50.0
  cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.scale_range      -> [1.0, 1.0]
  cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.trans_ratio      -> 0.0
  cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.persp_ratio      -> 0.0
  cfg.LOFTR.MATCH_FINE.ABSOLUTE_FINE_IDX    False -> True

The first 6 configure the H sampling (rot_deg=50 only; other axes off so the
ablation isolates "rotation in pose-supervised cross-modal training" with no
confound from scale/translation/perspective). The 7th is REQUIRED because
H_vis rotation breaks the v0 paper "-3.5 trick"'s implicit local-linearity
assumption (~4 fine-pixel residual at rot=50 deg). With ABSOLUTE_FINE_IDX
both supervision (spvs_fine) and inference (fine_matching.get_fine_ds_match)
switch to absolute fine pixel idx convention (cell top-left anchor); this
preserves train/test consistency (same anchor in both paths) and eliminates
the systematic offset.

Wall-clock + memory budget:
  H aug overhead is dominated by 1 extra cv2.warpPerspective per image1
  (~15 ms / batch at 640+bs=4) + 2 _warp_pts_homography calls in spvs_coarse
  (~5 ms / batch). Total ~+0.5% per step, negligible vs v14 baseline.
  Memory: H_vis [bs, 3, 3] fp32 = 144 B / batch + spvs_coarse intermediates
  scale1*pull/push tensors ~50 KB / batch. Negligible.
  17 ep wall-clock: ~9 h (same as v14).

==============================================================================
 ABSOLUTE_FINE_IDX rationale (Option H-A, post 6-round audit)
==============================================================================
The "-3.5 trick" (grid_value = idx - W//2 + 0.5, supervision +W//2-0.5
cancellation) is EfficientLoFTR paper v0 design (NOT a v14 introduction).
git blame: src/loftr/utils/fine_matching.py:138 @ commit 9cb9ca0
"Add EfficientLoFTR" + src/loftr/utils/supervision.py:311+339 @ commit
ac619fb "Add training". The trick's purpose: anchor fine head outputs at
cell centre so the [-3.5, +3.5] range maps cleanly to the 3x3 dsnt second
stage's [-1, +1] range (paper code aesthetics, not geometric necessity).

Geometrically the trick relies on warp_kpts being approximately linear
inside an 8x8 fine window. v0-v17 cross-view pose paths satisfy this
(parallax over an 8-pixel patch is ~sub-pixel in 90% of cases). v18 H_vis
rot=50 deg adds an extra rotation J = R(50 deg) on the image1 side, so the
input-end -3.5 offset becomes (-3.5*cos(50), -3.5*sin(50)) on the image1
side, no longer cancelled by the fixed +3.5 cancellation -> residual
|I - R(50)|*3.5 ~= 4.18 fine pixel = 4*scale1 ~= 13 orig pixel error in GT.

Two fixes were considered:
  - H-B: keep -3.5 trick, add Jacobian-aware compensation in spvs_fine.
    rejected (audit round 6) because the residual systematic offset is
    invisible to the fine head (no H_vis input), so it cannot learn to
    absorb it within 18 ep.
  - H-A (this cfg): drop the -3.5 trick entirely, change BOTH supervision
    and inference to absolute idx. Train/test stay consistent because the
    cfg flag is read by both paths. fine head must re-learn its anchor
    distribution from outdoor.ckpt (which was trained with the trick),
    expected to converge in 1-2 ep based on fine head's small capacity.

==============================================================================
 cold-start risk (outdoor.ckpt fine head shock)
==============================================================================
outdoor.ckpt's fine head was trained with the -3.5 trick anchor. v18 ep0
inference uses absolute idx -> stage-1 mkpts1_f systematically offset by
~3.5*scale1 ~= 1 orig pixel from where outdoor.ckpt expects to "score
high". RANSAC tolerates this (epipolar constraint relaxation), val auc@10
cold-start is expected to drop 1-3 pp vs v14 ep0 = 0.34. By ep1-2 the
fine head should re-anchor (conf_matrix_f_gt is anchor-invariant so the
gradient signal is preserved).

Acceptance gate (METU all auc@20 vs v14 baseline):
  Strong   : v18 >= v14 + 1.0 pp -> rot=50 augmentation in pose supervision
                                    is a clean win
  Medium   : v18 in [v14 + 0.3, v14 + 1.0) pp -> small benefit, ablation row
  Flat     : v18 in [v14 - 0.3, v14 + 0.3] pp -> rot=50 too aggressive or
                                                  too marginal; try rot=25
  Negative : v18 in [v14 - 1.0, v14 - 0.3) pp -> single-side aug interferes
                                                  with pose supervision;
                                                  drop prob to 0.5
  Fail     : v18 < v14 - 1.0 pp -> regression, check sanity tests pass

Sanity gates (run before 9h ship; reuse sanity_megadepth_syn_pose.py):
  [B] v14 byte-identical: re-run v14 cfg 1 ep, train_loss step 100 deviates
      from historical v14 logs by < 1e-5
  [C] v18 H_vis=I degenerate: temporarily set rot_deg=0, dataset short-
      circuits cv2.warpPerspective (Bug #14 fix), supervision falls into
      H_vis=I trivial path; coarse b_ids/i_ids/j_ids set-equal v14 output
  [D] v18 rot=50 numerical health: assert H_vis pure-rotation
      (H_vis[:, 2, :2].abs().max() < 1e-6), inv numerically stable
      ((H_vis @ inv(H_vis) - I).abs().max() < 1e-5), no IndexError after
      100 step (clamp works), b_ids count >= 30
  [E] v0-v12 RoadScene path not affected: re-run v11 cfg 1 ep, byte-
      identical to historical v11 logs (RoadScene goes spvs_coarse_roadscene,
      doesn't read H_vis, doesn't read ABSOLUTE_FINE_IDX flag)
  [G] collate keyset stability: assert v18 train batch contains 'H_vis',
      v18 val batch does NOT (data.py train-only gate), v14 batch does NOT
"""
from configs.loftr.eloftr_full_v14_pose_msyn_ddp import cfg

# ============ v18: enable single-side rotation-only H aug on image1 ============
# image0 (IR) is NEVER warped. image1 (VIS) gets a per-pair random Homography
# sampled in resized pixel space; only the rotation axis is enabled.
# prob=1.0 means every pair sees a random rotation in [-50 deg, +50 deg].
cfg.DATASET.MGDPT_HOMOGRAPHY_AUG = True
cfg.DATASET.MGDPT_HOMOGRAPHY_PROB = 1.0

cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.rot_deg = 50.0
cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.scale_range = [1.0, 1.0]
cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.trans_ratio = 0.0
cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.persp_ratio = 0.0

# ============ v18: switch fine-pixel-idx anchor convention ============
# Required by H_vis rot=50 (see docstring "ABSOLUTE_FINE_IDX rationale").
# Both supervision.spvs_fine and inference.fine_matching.get_fine_ds_match
# read this flag so train/test stay consistent.
cfg.LOFTR.MATCH_FINE.ABSOLUTE_FINE_IDX = True
