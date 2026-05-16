"""Eval-only cfg: 1-channel baseline architecture + v11/v12 aggressive
single-side Homography augmentation kwargs, with rotation amplitude
DOUBLED (rot_deg=50 vs the rot25 baseline cfg).

Used by MyScripts/eval_roadscene_all_singleh_finetuned_rot50.bat to
evaluate any 1-channel ckpt (USE_EDGE_INPUT=False,
BACKBONE_IN_CHANNELS=1) on a RoadScene index file under deterministic
single-side H aug (warp VIS only) at a stronger rotation regime.

Architecture: byte-identical to eloftr_full.py (no USE_*).

Aug knobs consumed by eval_roadscene.py:228 via
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS:
  rot_deg=50 / scale_range=[0.75, 1.25] / trans_ratio=0.12 / persp_ratio=0.08

Only rot_deg differs from configs/loftr/eloftr_eval_aggressive_singleh.py
(rot_deg=25 there); scale / trans / persp are kept identical so this cfg
isolates the rotation-amplitude axis for a clean ablation.

Schedule fields are NOT relevant: this cfg never feeds train.py.
"""
from configs.loftr.eloftr_full import cfg

cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 50.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08
