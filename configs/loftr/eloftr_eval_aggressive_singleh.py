"""Eval-only cfg: 1-channel baseline architecture + v11/v12 aggressive
single-side Homography augmentation kwargs.

Used by MyScripts/eval_roadscene_all_singleh_{official,finetuned}.bat
to evaluate any 1-channel ckpt (USE_EDGE_INPUT=False,
BACKBONE_IN_CHANNELS=1) on a RoadScene index file under deterministic
single-side H aug (warp VIS only).

Architecture: byte-identical to eloftr_full.py (no USE_*).

Aug knobs consumed by eval_roadscene.py:228 via
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS:
  rot_deg=25 / scale_range=[0.75, 1.25] / trans_ratio=0.12 / persp_ratio=0.08

These match the v11 / v12 / v0_baseline aggressive preset (see
configs/loftr/eloftr_full_v0_baseline.py L57-L60). Only the H aug fields
are overridden -- TRAINER.* / LOFTR.* / DATASET.TRAIN_LIST_PATH etc. all
inherit from eloftr_full.py + the data cfg passed via --data_cfg.

Schedule fields are NOT relevant: this cfg never feeds train.py.
"""
from configs.loftr.eloftr_full import cfg

cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.rot_deg = 25.0
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.scale_range = [0.75, 1.25]
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.trans_ratio = 0.12
cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.persp_ratio = 0.08
