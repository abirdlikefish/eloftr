"""Data config for the M3FD_Detection IR-VIS cross-modal training experiment.

Reuses the aligned IR-VIS short-circuit branch in ``src/lightning/data.py`` --
because ``'M3FD'`` is registered in
``src.utils.data_source.ALIGNED_IRVIS_SOURCES``, ``RoadSceneDataset`` is
built directly with the M3FD root + sub-directories below, and all
downstream dispatch (supervision, val/test metric, plotting) automatically
follows the same IR-VIS path RoadScene already uses. No code change is
needed beyond this file + ``make_m3fd_splits.py`` once the data is on disk.

Loaded by ``train.py`` / ``test.py`` via ``config.merge_from_file(...)``.
Every key set here must already exist somewhere in ``src/config/default.py``
(yacs raises ``KeyError`` for unknown keys).

Notes:
- M3FD originals are 1024x768 BMP/PNG. ROAD_IMG_RESIZE=480 long-edge brings
  them to 480x360 to share canvas / memory budget with RoadScene runs;
  raise to 640 once a baseline is confirmed.
- M3FD_Detection ships pixel-aligned IR-VIS pairs (calibrated + homography-
  warped), so Homography augmentation at train time is still beneficial as
  in RoadScene -- defaults match RoadScene config.
"""
from configs.data.base import cfg


M3FD_PATH = "data/M3FD_Detection"

# 1. data source -- registered in ALIGNED_IRVIS_SOURCES so dispatch goes to
#    the same IR-VIS path as RoadScene.
cfg.DATASET.TRAINVAL_DATA_SOURCE = "M3FD"
cfg.DATASET.TEST_DATA_SOURCE = "M3FD"

# 2. train / val / test list and roots
cfg.DATASET.TRAIN_DATA_ROOT = M3FD_PATH
cfg.DATASET.TRAIN_LIST_PATH = f"{M3FD_PATH}/index/train_pairs.txt"
cfg.DATASET.TRAIN_NPZ_ROOT = None
cfg.DATASET.TRAIN_INTRINSIC_PATH = None
cfg.DATASET.TRAIN_POSE_ROOT = None

cfg.DATASET.VAL_DATA_ROOT = M3FD_PATH
cfg.DATASET.VAL_LIST_PATH = f"{M3FD_PATH}/index/val_pairs.txt"
cfg.DATASET.VAL_NPZ_ROOT = None
cfg.DATASET.VAL_INTRINSIC_PATH = None
cfg.DATASET.VAL_POSE_ROOT = None

cfg.DATASET.TEST_DATA_ROOT = M3FD_PATH
cfg.DATASET.TEST_LIST_PATH = f"{M3FD_PATH}/index/test_pairs.txt"
cfg.DATASET.TEST_NPZ_ROOT = None
cfg.DATASET.TEST_INTRINSIC_PATH = None
cfg.DATASET.TEST_POSE_ROOT = None

# 3. M3FD has no overlap_score either (no per-pair geometric overlap concept).
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 4. Aligned IR-VIS knobs (consumed by RoadSceneDataset via the short-circuit
#    branch in src/lightning/data.py). Field names are deliberately reused
#    from the RoadScene config -- they describe the aligned IR-VIS
#    pipeline, not RoadScene specifically. Sub-dir casing matches the
#    actual M3FD_Detection download (Ir / Vis are capitalised).
cfg.DATASET.ROAD_IR_SUBDIR = "Ir"
cfg.DATASET.ROAD_VIS_SUBDIR = "Vis"
# v7_pcclahe Phase Congruency edge cache sub-dirs. Naming follows the source
# folders (Ir / Vis -> Ir_pc / Vis_pc, peer-level under M3FD_PATH). Pre-compute
# via MyScripts/precompute_pc_edges.bat. These fields being set does NOT enable
# PC loading on its own -- the dataset only reads them when
# LOFTR.USE_EDGE_INPUT=True (v7 cfg opts in; v5/v6/v6_1 leave it default False
# so retraining stays byte-identical even though the field is now set).
cfg.DATASET.ROAD_IR_PC_SUBDIR = "Ir_pc"
cfg.DATASET.ROAD_VIS_PC_SUBDIR = "Vis_pc"
cfg.DATASET.ROAD_IMG_RESIZE = 480     # longer-edge target before df-rounding
cfg.DATASET.ROAD_DF = 32              # final H, W are multiples of df
cfg.DATASET.ROAD_PAD_SIZE = 480       # square zero-padded canvas (matches RoadScene)
cfg.DATASET.ROAD_HOMOGRAPHY_AUG = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0

# 5. Disable training-step plotting because the default plotting branch calls
#    compute_symmetrical_epipolar_errors(), which needs camera intrinsics
#    that aligned IR-VIS datasets do not have. Validation-time plotting is
#    handled separately by the IR-VIS branch in PL_LoFTR.
cfg.TRAINER.ENABLE_PLOTTING = False
