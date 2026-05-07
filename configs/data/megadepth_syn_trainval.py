"""Data config for the Megadepth_Syn IR-VIS cross-modal training experiment (v10).

Style: ports m3fd_trainval.py 1-for-1; the only structural differences are:

  - Megadepth_Syn has nested source layout (train/infrared/phoenix/S6/zl548/
    MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.jpg vs M3FD's flat Ir/<id>.png).
    ROAD_IR_SUBDIR absorbs the deep prefix, index files store nested
    relative paths (<scene>/<denseN>/imgs/<stem>.jpg) -- identical contract,
    just longer subdir + longer index lines.

  - M3FD-style source / _pc peer-level layout: PC cache lives at
    train/{infrared_pc, phoenix_pc}/S6/zl548/MegaDepth_v1/... mirroring
    source structure. precompute_pc_edges.py with --recursive writes there.

  - v10 only uses train/ (195 scenes). test/Undistorted_SfM (806 pairs,
    MegaDepth1500 layout) is NOT used here -- left for future v11/v12 with
    real epipolar supervision (route B). 195 scenes are split scene-disjoint
    175/10/10 by build_megadepth_syn_index.py with seed=42.

Reuses the aligned IR-VIS short-circuit branch in src/lightning/data.py --
because 'Megadepth_Syn' is registered in src.utils.data_source.ALIGNED_IRVIS_SOURCES,
RoadSceneDataset is built directly with the deep root + sub-directories below,
and all downstream dispatch (supervision, val/test metric, plotting) follows
the same IR-VIS path RoadScene/M3FD already use.

Loaded by train.py / test.py via config.merge_from_file(...).
Every key set here must already exist somewhere in src/config/default.py
(yacs raises KeyError for unknown keys).

Notes:
- MegaDepth_v1 source images range 1280-1600 long edge; ROAD_IMG_RESIZE=480
  matches v9 M3FD setup for apples-to-apples comparison. Bump to 640/832
  in a v10.x ablation if memory allows.
- Megadepth_Syn IR is style-transferred from VIS so (IR_i, VIS_i) is
  pixel-aligned by construction; the Homography aug supplies the geometric
  variability that makes modemb / MSBN / PC learn real signal (same role
  as in v9 M3FD where IR-VIS has ~1px calibration residual + Homography aug).
"""
from configs.data.base import cfg


MSYN_PATH = "data/Megadepth_Syn"

# 1. data source -- registered in ALIGNED_IRVIS_SOURCES so dispatch goes to
#    the same IR-VIS path as RoadScene/M3FD.
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Megadepth_Syn"
cfg.DATASET.TEST_DATA_SOURCE = "Megadepth_Syn"

# 2. train / val / test list and roots (m3fd_trainval.py mirror; train/val/test
#    all live under train/ physical dir thanks to scene-disjoint split).
cfg.DATASET.TRAIN_DATA_ROOT = MSYN_PATH
cfg.DATASET.TRAIN_LIST_PATH = f"{MSYN_PATH}/index/train_pairs.txt"
cfg.DATASET.TRAIN_NPZ_ROOT = None
cfg.DATASET.TRAIN_INTRINSIC_PATH = None
cfg.DATASET.TRAIN_POSE_ROOT = None

cfg.DATASET.VAL_DATA_ROOT = MSYN_PATH
cfg.DATASET.VAL_LIST_PATH = f"{MSYN_PATH}/index/val_pairs.txt"
cfg.DATASET.VAL_NPZ_ROOT = None
cfg.DATASET.VAL_INTRINSIC_PATH = None
cfg.DATASET.VAL_POSE_ROOT = None

cfg.DATASET.TEST_DATA_ROOT = MSYN_PATH
cfg.DATASET.TEST_LIST_PATH = f"{MSYN_PATH}/index/test_pairs.txt"
cfg.DATASET.TEST_NPZ_ROOT = None
cfg.DATASET.TEST_INTRINSIC_PATH = None
cfg.DATASET.TEST_POSE_ROOT = None

# 3. No overlap_score (MegaDepth_Syn aligned IR-VIS has no per-pair geometric
#    overlap concept -- pairs are by-construction co-located images warped via
#    Homography aug at train time).
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 4. Aligned IR-VIS knobs (consumed by RoadSceneDataset via the short-circuit
#    branch in src/lightning/data.py). Subdir names absorb the deep MegaDepth_v1
#    nesting so per-image index lines stay short (<scene>/<denseN>/imgs/<stem>.jpg).
cfg.DATASET.ROAD_IR_SUBDIR = "train/infrared/phoenix/S6/zl548/MegaDepth_v1"
cfg.DATASET.ROAD_VIS_SUBDIR = "train/phoenix/S6/zl548/MegaDepth_v1"
# v7+ Phase Congruency edge cache subdirs. M3FD-style source/_pc peer-level
# layout: cache lives at train/{infrared_pc, phoenix_pc}/S6/zl548/MegaDepth_v1/...
# (mirror source nesting). Pre-compute via:
#   bash MyScripts/precompute_pc_edges.sh --dataset Megadepth_Syn \
#       --recursive --max_long_edge 640 --pc_nscale 3 --workers 24
# (~46 min vs ~14h baseline thanks to L1 pyfftw + L2 nscale=3 + L3 pre-resize 640).
# These fields being set does NOT enable PC loading on its own -- the dataset
# only reads them when LOFTR.USE_EDGE_INPUT=True (v10 cfg opts in via inheritance
# from v9_e2e).
cfg.DATASET.ROAD_IR_PC_SUBDIR = "train/infrared_pc/S6/zl548/MegaDepth_v1"
cfg.DATASET.ROAD_VIS_PC_SUBDIR = "train/phoenix_pc/S6/zl548/MegaDepth_v1"
cfg.DATASET.ROAD_IMG_RESIZE = 480     # longer-edge target before df-rounding
cfg.DATASET.ROAD_DF = 32              # final H, W are multiples of df
cfg.DATASET.ROAD_PAD_SIZE = 480       # square zero-padded canvas (matches v9 M3FD)
cfg.DATASET.ROAD_HOMOGRAPHY_AUG = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0

# 5. Disable training-step plotting because the default plotting branch calls
#    compute_symmetrical_epipolar_errors(), which needs camera intrinsics
#    that aligned IR-VIS datasets do not have. Validation-time plotting is
#    handled separately by the IR-VIS branch in PL_LoFTR.
cfg.TRAINER.ENABLE_PLOTTING = False
