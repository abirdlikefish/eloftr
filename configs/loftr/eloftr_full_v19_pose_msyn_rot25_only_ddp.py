"""v19 = v18 with rot_deg lowered 50.0 -> 25.0.

This is THE retry path explicitly recommended by v18 cfg's acceptance gate
"Flat" branch (see eloftr_full_v18_pose_msyn_rot50_only_ddp.py):

    Flat : v18 in [v14 - 0.3, v14 + 0.3] pp -> rot=50 too aggressive or
                                                too marginal; try rot=25

==============================================================================
 v19 vs v18: 1 cfg knob only
==============================================================================
v19 inherits v18 ddp byte-identical except for ONE override:

  cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.rot_deg          50.0 -> 25.0

Everything else preserved from v18 (which itself adds 7 overrides on v14):
  - cfg.DATASET.MGDPT_HOMOGRAPHY_AUG                = True
  - cfg.DATASET.MGDPT_HOMOGRAPHY_PROB               = 1.0
  - cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.scale_range = [1.0, 1.0]
  - cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.trans_ratio = 0.0
  - cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.persp_ratio = 0.0
  - cfg.LOFTR.MATCH_FINE.ABSOLUTE_FINE_IDX          = True

The H aug therefore stays "single-side rotation-only on image1 (VIS)":
image0 (IR) untouched, image1 warped by H_vis sampled in resized pixel
space; only the rotation axis is enabled, prob=1.0.

==============================================================================
 Why ABSOLUTE_FINE_IDX = True still required at rot=25
==============================================================================
The "-3.5 trick" residual under H_vis rotation is ~|I - R(theta)| * 3.5
fine pixels, which scales as ~7 * sin(theta/2). Concretely:
  rot = 50 deg -> residual ~ 4.18 fine pixel ~ 13 orig pixel error  (v18)
  rot = 25 deg -> residual ~ 1.48 fine pixel ~  4.4 orig pixel error (v19)
A 4.4-orig-pixel systematic GT offset is still well outside the fine
head's sub-pixel target accuracy, so the absolute-idx convention must
stay on. Train/test consistency is preserved (both spvs_fine and
fine_matching.get_fine_ds_match read the same flag).

==============================================================================
 Cold-start risk vs v18 (smaller fine-head shock)
==============================================================================
outdoor.ckpt's fine head was trained with the v0 -3.5 trick anchor.
v19 inference uses absolute idx convention -> stage-1 mkpts1_f
systematically offset by ~3.5 * scale1 ~ 1 orig pixel from where
outdoor.ckpt expects to "score high" at ep 0.

The shock magnitude is the SAME as v18 (it only depends on the anchor-
convention switch, not on rot_deg), so ep0 val auc@10 is still expected
to drop 1-3 pp vs v14 ep0 = 0.34. Fine head re-anchoring takes 1-2 ep
in v18 and should take the same in v19.

==============================================================================
 Acceptance gate (METU all auc@20 vs v14 baseline)
==============================================================================
  Strong   : v19 >= v14 + 1.0 pp -> rot=25 augmentation in pose
                                    supervision is a clean win
  Medium   : v19 in [v14 + 0.3, v14 + 1.0) pp -> small benefit, ablation row
  Flat     : v19 in [v14 - 0.3, v14 + 0.3] pp -> rotation H aug 在 pose
                                                  监督下 24/7 都不工作;
                                                  转向 v15 (contrast) /
                                                  v16 (modemb) 路线
  Negative : v19 in [v14 - 1.0, v14 - 0.3) pp -> 即使 25 度也干扰 pose
                                                  supervision; 试 prob=0.5
  Fail     : v19 < v14 - 1.0 pp -> 几何 plumbing 退化, 重跑 sanity gates
                                    [B][C][D][E][G]

Decision matrix v18 vs v19 (after both ship + METU eval):
  v18 Strong + v19 Strong -> rotation works in pose path; pick larger
                              of the two as canonical aug strength
  v18 Strong + v19 Medium -> rot=50 sweet spot, document v19 as ablation
  v18 Flat   + v19 Strong -> rot=50 too aggressive (predicted by v18 cfg)
  v18 Flat   + v19 Medium -> small but real gain at lower rot
  v18 Flat   + v19 Flat   -> rotation H aug in pose path doesn't help,
                              switch to v15/v16 alternative routes
  v18 Strong + v19 Flat   -> rot=50 needed to overcome cold-start; rare
                              but documents minimum aug strength

Sanity gates: 复用 v18 的 [B][C][D][E][G]. H_vis 数值健康检测
(H_vis pure-rotation test, inv stability) 不依赖 rot_deg 大小,
25 度跟 50 度都是 pure-rotation matrix.

Wall-clock + memory budget: identical to v18 / v14 (~9 h for 17 ep on
4-GPU DDP). H aug overhead is ~+0.5% per step regardless of rot_deg.
"""
from configs.loftr.eloftr_full_v18_pose_msyn_rot50_only_ddp import cfg

# ============ v19: only knob change vs v18, rot_deg 50 -> 25 ============
# All other 6 v18 overrides (HOMOGRAPHY_AUG/PROB, scale/trans/persp,
# ABSOLUTE_FINE_IDX) inherited byte-identical from v18.
cfg.DATASET.MGDPT_HOMOGRAPHY_KWARGS.rot_deg = 25.0
