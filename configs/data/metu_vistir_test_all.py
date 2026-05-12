"""METU_VISTIR pose-based test cfg (full set: 10 npz / 2590 pair).

Drives test.py via:
    python test.py configs/data/metu_vistir_test_all.py <main_cfg> \\
        --ckpt_path <ckpt> --ransac LO-RANSAC --pixel_thr 2.0 --ransac_times 5

The cfg dispatches to METUVisTIRDataset via the new
``elif data_source.lower() == 'metu_vistir':`` branch in
src/lightning/data.py:_build_concat_dataset.

RANSAC parameters (POSE_ESTIMATION_METHOD / RANSAC_PIXEL_THR / EVAL_TIMES)
are deliberately NOT set here -- pass them via test.py command-line
flags (--ransac / --pixel_thr / --ransac_times) so the same data cfg can
be reused with different ransac protocols.

Group-report variants (subset list files):
    metu_vistir_test_cloudy_cloudy.py   6 npz / 1382 pair  (same-illumination)
    metu_vistir_test_cloudy_sunny.py    4 npz / 1208 pair  (cross-illumination)
"""
from configs.data.base import cfg


METU_PATH = "data/METU_VISTIR"

cfg.DATASET.TEST_DATA_SOURCE = "METU_VISTIR"
cfg.DATASET.TEST_DATA_ROOT   = METU_PATH
cfg.DATASET.TEST_NPZ_ROOT    = f"{METU_PATH}/index/scene_info_test"
cfg.DATASET.TEST_LIST_PATH   = "assets/metu_vistir_test_lists/all.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
cfg.DATASET.MGDPT_IMG_RESIZE = 832
cfg.DATASET.NPE_NAME = 'megadepth'

# METU-specific knobs (must be pre-registered in src/config/default.py
# under _CN.DATASET; otherwise yacs strict-mode merge_from_file raises
# KeyError: Non-existent config key).
cfg.DATASET.METU_UNDISTORT = True
# image0=thermal / image1=vis aligns byte-for-byte with v0..v11 training-side
# convention (see src/datasets/roadscene.py:376-378 + src/loftr/loftr.py:117).
# 30-pair sweep on v10 ckpt (tmp_metu_side_sweep.py 2026-05-11): thermal-as-
# image0 R_med=17.55 deg, auc@20=0.87% vs vis-as-image0 R_med=20.38 deg,
# auc@20=0.58%. Earlier 5-pair smoke had picked 'vis' off raw num_matches
# alone (13 vs 267) -- that judgment was wrong: 30-pair shows thermal-side
# median jumps to 226 matches, comfortably above the 5-match RANSAC floor.
cfg.DATASET.METU_SIDE0     = 'thermal'
cfg.DATASET.METU_SIDE1     = 'vis'
