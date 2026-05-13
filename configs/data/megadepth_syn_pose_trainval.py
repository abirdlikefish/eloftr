"""Data config for v13 Megadepth_Syn pose-supervised training.

Loaded by train.py / test.py via ``cfg.merge_from_file(...)`` AFTER the
LOFTR cfg (eloftr_full_v13_pose_msyn_*.py) so DATASET fields here take
precedence over anything the LOFTR cfg sets.

Pairs scene_info_0.1_0.7/*.npz (K + W2C pose + depth_paths + pair_infos)
with the Megadepth_Syn image assets (style-transferred IR + phoenix VIS).

Train list is LoFTR official ``train_list.txt`` (153 scene / 368 npz / 8.86M
pair, MegaDepth upstream baseline). Val list is a 2-line truncation of LoFTR
``val_list.txt`` (Trevi ``0015_0.3_0.5`` + Pantheon ``0022_0.5_0.7``, 4493
pair total) to keep DDP val cost in check; ``--limit_val_batches=0.5`` in
the run script further halves to 2247 pair (still > LoFTR megadepth_val_1500
business baseline of 1500 pair). Test is unused: OOD eval on M3FD / RoadScene
/ METU drives final results, no v13-internal test pool needed.

scene-disjoint: LoFTR train 153 scene n {0015, 0022} = empty (LoFTR upstream
design guarantees); the legacy 145/4/4 split files in
``trainvaltest_list_pose/`` are kept on disk as archive only.

NOTE: every key set here must already exist in src/config/default.py
or it's added there (see MSYN_POSE_CROSS_MODAL_MODE just below METU
options); yacs raises KeyError on merge for unknown keys.
"""
from configs.data.base import cfg


SYN_PATH = "data/Megadepth_Syn"
INDEX_PATH = f"{SYN_PATH}/index"

# 1. data source -- routed by src/loftr/utils/supervision.py + src/lightning/data.py
#    to spvs_coarse (pose-based), NOT spvs_coarse_roadscene (H-based).
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Megadepth_Syn_Pose"
cfg.DATASET.TEST_DATA_SOURCE = "Megadepth_Syn_Pose"

# 2. train / val / test list + per-npz scene_info root
cfg.DATASET.TRAIN_DATA_ROOT = SYN_PATH
cfg.DATASET.TRAIN_NPZ_ROOT = f"{INDEX_PATH}/scene_info_pose"
cfg.DATASET.TRAIN_LIST_PATH = f"{INDEX_PATH}/trainvaltest_list_src/train_list.txt"
cfg.DATASET.TRAIN_INTRINSIC_PATH = None
cfg.DATASET.TRAIN_POSE_ROOT = None

cfg.DATASET.VAL_DATA_ROOT = SYN_PATH
cfg.DATASET.VAL_NPZ_ROOT = f"{INDEX_PATH}/scene_info_pose"
cfg.DATASET.VAL_LIST_PATH = f"{INDEX_PATH}/val_list_loftr_small.txt"
cfg.DATASET.VAL_INTRINSIC_PATH = None
cfg.DATASET.VAL_POSE_ROOT = None

cfg.DATASET.TEST_DATA_ROOT = SYN_PATH
cfg.DATASET.TEST_NPZ_ROOT = f"{INDEX_PATH}/scene_info_pose"
cfg.DATASET.TEST_LIST_PATH = f"{INDEX_PATH}/val_list_loftr_small.txt"  # placeholder; OOD eval on M3FD/RoadScene/METU
cfg.DATASET.TEST_INTRINSIC_PATH = None
cfg.DATASET.TEST_POSE_ROOT = None

# 3. overlap threshold
# LoFTR scene_info_0.1_0.7 buckets pair_infos by overlap [0.1, 0.3), [0.3, 0.5),
# [0.5, 0.7) at the source level (filename suffix). Setting 0.0 here means
# accept every pair the npz holds; LoFTR convention for MegaDepth training.
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 4. MegaDepth-style image / depth knobs (matches outdoor.ckpt training domain).
# DEFAULT in src/config/default.py is (640, 8), MUST override here or
# scale0/scale1 will be wrong and pose AUC will collapse.
cfg.DATASET.MGDPT_IMG_RESIZE = 832
cfg.DATASET.MGDPT_DF = 32
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DEPTH_PAD = True

# 5. Cross-view cross-modal direction. 'ir2vis' = image0=IR(idx0),
# image1=VIS(idx1), matches v0..v11 + METU_VISTIR training-side
# convention (image0=thermal). v13 main run only uses 'ir2vis'.
cfg.DATASET.MSYN_POSE_CROSS_MODAL_MODE = "ir2vis"

# 6. Per-dataset name used by src/loftr/utils/position_encoding.py for the
# NPE (NeighbourPositionalEncoding) presets. We borrow 'megadepth' because
# v13 image shapes / scale conventions are identical (MegaDepth-style).
cfg.DATASET.NPE_NAME = "megadepth"
