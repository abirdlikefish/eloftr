"""Data config for the RoadScene IR-VIS cross-modal training experiment.

Loaded by ``train.py`` / ``test.py`` via ``config.merge_from_file(...)``. Every
key set here must already exist somewhere in ``src/config/default.py`` because
yacs raises ``KeyError`` for unknown keys.
"""
from configs.data.base import cfg


ROADSCENE_PATH = "data/RoadScene"

# 1. data source
cfg.DATASET.TRAINVAL_DATA_SOURCE = "RoadScene"
cfg.DATASET.TEST_DATA_SOURCE = "RoadScene"

# 2. train / val / test list and roots
cfg.DATASET.TRAIN_DATA_ROOT = ROADSCENE_PATH
cfg.DATASET.TRAIN_LIST_PATH = f"{ROADSCENE_PATH}/index/train_pairs.txt"
cfg.DATASET.TRAIN_NPZ_ROOT = None
cfg.DATASET.TRAIN_INTRINSIC_PATH = None
cfg.DATASET.TRAIN_POSE_ROOT = None

cfg.DATASET.VAL_DATA_ROOT = ROADSCENE_PATH
cfg.DATASET.VAL_LIST_PATH = f"{ROADSCENE_PATH}/index/val_pairs.txt"
cfg.DATASET.VAL_NPZ_ROOT = None
cfg.DATASET.VAL_INTRINSIC_PATH = None
cfg.DATASET.VAL_POSE_ROOT = None

cfg.DATASET.TEST_DATA_ROOT = ROADSCENE_PATH
cfg.DATASET.TEST_LIST_PATH = f"{ROADSCENE_PATH}/index/test_pairs.txt"
cfg.DATASET.TEST_NPZ_ROOT = None
cfg.DATASET.TEST_INTRINSIC_PATH = None
cfg.DATASET.TEST_POSE_ROOT = None

# 3. RoadScene has no overlap_score; disable the threshold filter.
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 4. RoadScene-specific knobs (all consulted in src/lightning/data.py).
cfg.DATASET.ROAD_IR_SUBDIR = "cropinfrared"
cfg.DATASET.ROAD_VIS_SUBDIR = "crop_LR_visible"  # IR-aligned (same FoV / aspect ratio as cropinfrared); crop_HR_visible is a wider-FoV high-res companion and is NOT pixel-aligned.
# v7_pcclahe Phase Congruency edge cache sub-dirs. Naming mirrors the source
# folders (cropinfrared / crop_LR_visible -> *_pc) and lives at the dataset
# root, peer-level. Pre-compute via MyScripts/precompute_pc_edges.bat. These
# fields being set does NOT enable PC loading on its own -- the dataset only
# reads them when LOFTR.USE_EDGE_INPUT=True (v7 cfg opts in; v0-v4 leave it
# default False so retraining/eval stays byte-identical).
cfg.DATASET.ROAD_IR_PC_SUBDIR = "cropinfrared_pc"
cfg.DATASET.ROAD_VIS_PC_SUBDIR = "crop_LR_visible_pc"
cfg.DATASET.ROAD_IMG_RESIZE = 480     # longer-edge target before df-rounding
cfg.DATASET.ROAD_DF = 32              # final H, W are multiples of df
# Square zero-padded canvas size (Solution A3). Every sample becomes
# (PAD_SIZE, PAD_SIZE) so DataLoader collate can stack a real batch.
# Must be >= ROAD_IMG_RESIZE and divisible by ROAD_DF.
cfg.DATASET.ROAD_PAD_SIZE = 480
cfg.DATASET.ROAD_HOMOGRAPHY_AUG = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0

# 5. Disable training-step plotting because the default plotting branch calls
#    compute_symmetrical_epipolar_errors(), which needs camera intrinsics that
#    RoadScene does not have. Validation-time plotting is handled separately
#    by the RoadScene branch in PL_LoFTR.
cfg.TRAINER.ENABLE_PLOTTING = False
