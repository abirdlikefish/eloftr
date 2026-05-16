"""Data config for LLVIP IR-VIS cross-modal training experiment (v17 entry).

Reuses the aligned IR-VIS short-circuit branch in ``src/lightning/data.py`` --
because ``'LLVIP'`` is registered in ``src.utils.data_source.ALIGNED_IRVIS_SOURCES``,
``RoadSceneDataset`` is built directly with the LLVIP root + sub-directories
below, and all downstream dispatch (supervision via spvs_*_roadscene,
val/test metric via _compute_roadscene_metrics, plotting) automatically
follows the same IR-VIS path RoadScene/M3FD/Megadepth_Syn already uses.
**No code change is needed beyond this file + make_llvip_splits.py + 1-line
whitelist edit in src/utils/data_source.py.**

Loaded by ``train.py`` / ``test.py`` via ``config.merge_from_file(...)``.
Every key set here must already exist somewhere in ``src/config/default.py``
(yacs raises ``KeyError`` for unknown keys).

------------------------------------------------------------------------------
Key design point: split index file format with subdirectory prefix
------------------------------------------------------------------------------
LLVIP has **official train/test split** at the source level:
    data/LLVIP/{infrared,visible}/{train,test}/<id>.jpg

To let a single cfg ``ROAD_IR_SUBDIR='infrared'`` (NOT 'infrared/train' or
'infrared/test') drive both train and val, the split files emit lines with
a subdirectory prefix:
    data/LLVIP/index/train_pairs.txt:  "train/010001.jpg" .. (12025 lines)
    data/LLVIP/index/test_pairs.txt :  "test/190001.jpg"  .. (3463 lines)

Then RoadSceneDataset's ``osp.join(ir_dir, name)`` becomes
``osp.join('data/LLVIP/infrared', 'train/010001.jpg')`` =
``data/LLVIP/infrared/train/010001.jpg``, automatically resolving the
correct path for both train (split='train') and val (split='test').

If we instead used cfg ``ROAD_IR_SUBDIR='infrared/train'``, train would
work but val would fail (val_list points to test/ but cfg says train/),
forcing two separate cfg files. The prefixed-line approach lets one cfg
serve both modes.

------------------------------------------------------------------------------
Validation strategy (v17 specific): LLVIP test as val pool
------------------------------------------------------------------------------
v17 does not split LLVIP train into train/val. Instead:
- TRAIN_LIST_PATH = train_pairs.txt = 12025 (full LLVIP official train)
- VAL_LIST_PATH   = test_pairs.txt  = 3463  (full LLVIP official test as val pool)
- --limit_val_batches=0.5 -> effective 1731 val pair (LoFTR megadepth_val_1500
  business standard)
- TEST_LIST_PATH  = test_pairs.txt  (placeholder; v17 OOD eval runs on
  M3FD/RoadScene/METU instead, see eval_*.bat in MyScripts/)

Future: if independent test holdout is needed, re-shuffle test_pairs.txt
into val_pairs.txt + holdout_test_pairs.txt and re-emit make_llvip_splits.py.

------------------------------------------------------------------------------
Image specs
------------------------------------------------------------------------------
LLVIP originals are 1280x1024 .jpg (street scenes, low-light + thermal).
ROAD_IMG_RESIZE=640 long-edge brings them to 640x512 -> df-rounded to
640x512 (df=32). This matches v14 train resolution (640) for clean
finetune-from-v14 inheritance (no resolution mismatch shock).

ROAD_HOMOGRAPHY_AUG=True with ROAD_HOMOGRAPHY_PROB=1.0 enables aggressive
single-side H aug at every train batch (image1 / VIS only); H_KWARGS are
overridden in the v17 LOFTR cfg, NOT here, because aug strength is a v17
ablation knob (cfg.DATASET.ROAD_HOMOGRAPHY_KWARGS.{rot_deg, scale_range,
trans_ratio, persp_ratio}).
"""
from configs.data.base import cfg


LLVIP_PATH = "data/LLVIP"

# 1. data source -- registered in ALIGNED_IRVIS_SOURCES so dispatch goes to
#    the same IR-VIS path as RoadScene / M3FD / Megadepth_Syn.
cfg.DATASET.TRAINVAL_DATA_SOURCE = "LLVIP"
cfg.DATASET.TEST_DATA_SOURCE = "LLVIP"

# 2. train list = LLVIP official train (12025), val list = LLVIP official test
#    (3463 as val pool, --limit_val_batches=0.5 -> 1731 effective).
cfg.DATASET.TRAIN_DATA_ROOT = LLVIP_PATH
cfg.DATASET.TRAIN_LIST_PATH = f"{LLVIP_PATH}/index/train_pairs.txt"
cfg.DATASET.TRAIN_NPZ_ROOT = None
cfg.DATASET.TRAIN_INTRINSIC_PATH = None
cfg.DATASET.TRAIN_POSE_ROOT = None

cfg.DATASET.VAL_DATA_ROOT = LLVIP_PATH
cfg.DATASET.VAL_LIST_PATH = f"{LLVIP_PATH}/index/test_pairs.txt"
cfg.DATASET.VAL_NPZ_ROOT = None
cfg.DATASET.VAL_INTRINSIC_PATH = None
cfg.DATASET.VAL_POSE_ROOT = None

# TEST_* placeholder pointing to test_pairs.txt; v17 does not run
# independent test on LLVIP (OOD eval lives in M3FD/RoadScene/METU).
cfg.DATASET.TEST_DATA_ROOT = LLVIP_PATH
cfg.DATASET.TEST_LIST_PATH = f"{LLVIP_PATH}/index/test_pairs.txt"
cfg.DATASET.TEST_NPZ_ROOT = None
cfg.DATASET.TEST_INTRINSIC_PATH = None
cfg.DATASET.TEST_POSE_ROOT = None

# 3. LLVIP has no overlap_score (no per-pair geometric overlap concept,
#    pixel-aligned by construction).
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 4. Aligned IR-VIS knobs (consumed by RoadSceneDataset via the short-circuit
#    branch in src/lightning/data.py). ROAD_*_SUBDIR is the **parent**
#    directory only (NOT 'infrared/train' or 'infrared/test'); the
#    train/test sub-split is encoded in each line of the split files
#    (e.g. "train/010001.jpg" or "test/190001.jpg") so a single cfg file
#    drives both train (TRAIN_LIST_PATH) and val (VAL_LIST_PATH).
cfg.DATASET.ROAD_IR_SUBDIR = "infrared"
cfg.DATASET.ROAD_VIS_SUBDIR = "visible"
# v7_pcclahe Phase Congruency edge cache sub-dirs. Set as cfg-hygiene
# (must exist for cfg-key consistency); NOT read at train time because
# v17 inherits v14 which has USE_EDGE_INPUT=False (default). v17 does
# NOT enable PC channel input.
cfg.DATASET.ROAD_IR_PC_SUBDIR = "infrared_pc"
cfg.DATASET.ROAD_VIS_PC_SUBDIR = "visible_pc"

# 5. Image preprocessing (v14-aligned for clean finetune inheritance).
cfg.DATASET.ROAD_IMG_RESIZE = 640     # long-edge, matches v14 (avoid resolution shock)
cfg.DATASET.ROAD_DF = 32              # final H, W are multiples of df
cfg.DATASET.ROAD_PAD_SIZE = 640       # square zero-padded canvas

# 6. Homography augmentation (single-side, VIS warped). Aug strength
#    (rot_deg / scale_range / trans_ratio / persp_ratio) is v17 ablation
#    knob -> set in eloftr_full_v17_llvip_singleh_aggressive_ddp.py, NOT here.
#    ROAD_HOMOGRAPHY_DUAL stays default False -> single-side warp VIS only,
#    matching eval_roadscene_all_singleh_finetuned.bat protocol.
cfg.DATASET.ROAD_HOMOGRAPHY_AUG = True
cfg.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0

# 7. Disable training-step plotting because the default plotting branch
#    calls compute_symmetrical_epipolar_errors() which needs camera
#    intrinsics that aligned IR-VIS datasets do not have. Validation-time
#    plotting is handled separately by the IR-VIS branch in PL_LoFTR.
cfg.TRAINER.ENABLE_PLOTTING = False
