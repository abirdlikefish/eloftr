"""Data config for v14 Megadepth_Syn pose-supervised training at IMG_RESIZE=640.

v14 vs v13: only ``MGDPT_IMG_RESIZE`` differs (832 -> 640). All other DATASET
fields are byte-identical to ``megadepth_syn_pose_trainval.py`` so v14 vs v13
is a clean training-resolution ablation (data / split / val_list / overlap /
df / pad / cross_modal_mode all the same).

Why a separate data cfg (not just override in main cfg):
    train.py:127 merges data_cfg AFTER main_cfg, so any cfg.DATASET.* set in
    the v14 main cfg would be silently overwritten by the v13 data cfg's 832.
    Hence v14 needs its own data cfg with IMG_RESIZE=640.

Why 640 (not other values, see plan §0/§7 for full rationale):
    - METU eval (eval_metu_vistir_finetuned.bat L304) hard-codes --megasize 640
      per MINIMA / XoFTR paper protocol. v14 train at 640 -> train/eval domain
      match (vs v13 832 train + 640 eval mismatch).
    - LWIR thermal native long-edge is 640. v13's 832 path bilinear-upsampled
      thermal to 832 introduces fake high-freq detail that doesn't exist in the
      sensor signal. 640 path matches thermal native res.
    - Memory: sim_matrix at coarse_matching.py:122 is (bs, hw0_c, hw1_c) fp32.
      640 path hw0_c = (640/8)^2 = 6400; 832 path = 10816. sim_matrix size
      ratio = 6400^2 / 10816^2 = 0.35 (-65%). With bs=4 (vs v13 bs=2) net
      sim_matrix peak: 0.66 GB / batch (vs v13 0.93 GB), still smaller. Lets
      v14 safely use bs=4 instead of v13's bs=2 OOM workaround.

NPE consideration:
    v13 cfg leaves cfg.LOFTR.COARSE.NPE unset, so train.py:130 falls back to
    [832, 832, 832, 832]. For v14 at IMG_RESIZE=640 we MUST set NPE explicitly
    to [832, 832, 640, 640] (outdoor.ckpt train res, then current data long
    edge) so RoPE position scaler = 832/640 = 1.3 stretches the eval-time grid
    to match outdoor.ckpt's training-time RoPE frequency. NPE is set in the
    LOFTR cfg (eloftr_full_v14_pose_msyn_ddp.py), NOT here, because NPE is a
    LOFTR-side architecture knob, not a DATASET field.

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

# 2. train / val / test list + per-npz scene_info root (identical to v13)
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

# 3. overlap threshold (identical to v13)
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# 4. MegaDepth-style image / depth knobs.
#    *** v14 ONLY DIFFERENCE vs v13 ***: MGDPT_IMG_RESIZE 832 -> 640.
#    df=32 + 640 = 20 -> hw0_c = 20^2 = 400 (vs v13 hw0_c = 26^2 = 676).
#    sim_matrix shape (bs=4, 6400, 6400) fp32 = 0.66 GB / batch
#    (vs v13 bs=2 (2, 10816, 10816) = 0.93 GB / batch).
cfg.DATASET.MGDPT_IMG_RESIZE = 640
cfg.DATASET.MGDPT_DF = 32
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DEPTH_PAD = True

# 5. Cross-view cross-modal direction (identical to v13).
cfg.DATASET.MSYN_POSE_CROSS_MODAL_MODE = "ir2vis"

# 6. NPE preset name (identical to v13).
cfg.DATASET.NPE_NAME = "megadepth"
