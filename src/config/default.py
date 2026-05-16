from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE_TYPE = 'RepVGG'
_CN.LOFTR.ALIGN_CORNER = False
_CN.LOFTR.RESOLUTION = (8, 1)
_CN.LOFTR.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be even
_CN.LOFTR.MP = False
_CN.LOFTR.REPLACE_NAN = False
_CN.LOFTR.EVAL_TIMES = 1
_CN.LOFTR.HALF = False

# -- # cross-modal modality embedding (RoadScene IR-VIS)
# When enabled, LoFTR adds two learnable C-dim vectors `modality_emb_ir` and
# `modality_emb_vis` to feat_c0 / feat_c1 just before the coarse transformer
# (broadcast over H, W). Because residual connections preserve them across all
# subsequent self/cross-attention layers, a single injection at the input is
# enough to make every downstream attention "modality-aware".
#
# Init choices:
#   'zeros'       : safest. step 0 == baseline (adding 0 changes nothing) so
#                   finetuning the official ckpt cannot make things worse.
#                   The norms `mod_emb_ir_norm` / `mod_emb_vis_norm` should
#                   grow from 0 over training.
#   'normal_0.02' : small random init. Use only if 'zeros' fails to learn
#                   (norms stuck near 0 after several epochs).
_CN.LOFTR.USE_MODALITY_EMB = False
_CN.LOFTR.MODALITY_EMB_INIT = 'zeros'

# -- # parameter freezing for finetune (small-dataset overfit control)
# When training on a small dataset (e.g. RoadScene, ~hundreds of pairs) the
# 16M ELoFTR easily overfits within ~3 epochs. Freezing the 9.5M backbone
# caps the effective parameter count at ~5.7M (transformer + fine + modemb)
# so the model has less capacity to memorise the training set.
#
# FREEZE_BACKBONE:    hard-disable gradients on `matcher.backbone.*`.
# FREEZE_BN:          additionally lock `BatchNorm2d.eval()` (running_mean /
#                     running_var stop updating) AND freeze BN affine params,
#                     across ALL BN in the model (backbone + fine_preprocess).
#                     Implemented in PL_LoFTR with a `train()` override so PL's
#                     per-epoch model.train() does not silently undo it.
# FREEZE_BACKBONE_BN: only freeze BN layers inside `matcher.backbone.*`
#                     (running stats stay pinned to MegaDepth pretrained
#                     values), while `matcher.fine_preprocess.*` BN remains
#                     trainable + train-mode. Use this when the dataset is
#                     small but you still want fine-level BN to adapt to the
#                     new task distribution.
#                     NOTE: FREEZE_BN=True takes precedence over this flag and
#                     freezes ALL BN; FREEZE_BACKBONE_BN is then a no-op.
_CN.LOFTR.FREEZE_BACKBONE = False
_CN.LOFTR.FREEZE_BN = False
_CN.LOFTR.FREEZE_BACKBONE_BN = False

# -- # v8: Modality-Specific BatchNorm on fine_preprocess (cross-modal §4 E1)
# When USE_MSBN=True, layer{1,2}_outconv2.1 (the 2 BNs in fine_preprocess)
# are replaced by nn.Identity() and a pair of (bn_ir / bn_vis) is added
# beside each. fine_preprocess.forward routes IR features through bn_ir and
# VIS through bn_vis. Default False keeps v0-v7 ckpt loadable byte-identically
# (no key changes, no architectural changes, no log changes).
#
# FREEZE_FINE_BN_IR / VIS: ablation knobs to freeze only one MSBN branch.
# Use case v8.1 (freeze_ir): tests if VIS-only adaptation suffices given v7
# has already trained IR/PC features well.
#
# Freeze field interaction (5 fields, OR semantics, no priority chain):
#   - All freeze fields (FREEZE_BACKBONE / FREEZE_BN / FREEZE_BACKBONE_BN /
#     FREEZE_FINE_BN_IR / FREEZE_FINE_BN_VIS) are INDEPENDENT and ADDITIVE.
#   - For any param p: p is frozen iff ANY freeze field whose scope includes
#     p is set to True. Repeated freeze (e.g. FREEZE_BN=T and
#     FREEZE_BACKBONE_BN=T together cover backbone BN twice) is IDEMPOTENT
#     -- no side effect (m.eval() and p.requires_grad=False are safe to
#     call multiple times).
#   - This replaces v0-v7's if/elif "priority chain" with 5 independent if's
#     in lightning_loftr.py. Behavior is byte-identical to v0-v7 because no
#     v0-v7 cfg sets multiple BN-freeze fields simultaneously.
#   - FREEZE_FINE_BN_IR/VIS are meaningful only when USE_MSBN=True (otherwise
#     fine_preprocess has no _bn_ir/_bn_vis attrs; helpers silent no-op via
#     getattr defense + a debug warning is emitted).
#
# Three-layer default-value discipline (mirrors v7 R3):
#   default.py (yacs CN):                   _CN.LOFTR.USE_MSBN = False
#   lightning_loftr.py (yacs CN access):    config.LOFTR.get('USE_MSBN', False)
#   loftr.py / fine_preprocess.py (dict):   config.get('use_msbn', False)
#                                           (key lower-cased by lower_config())
# All three layers must give "do nothing" so v0-v7 cfg merge yields
# byte-identical behaviour to before this field landed.
#
# Considered but DEFERRED to future v8.x ablations (not added now to keep
# v8 cfg surface minimal, per cross-modal §6 "incremental field addition"):
#   MSBN_ALPHA  (vis = ir x alpha): currently always 1.0, no flexibility need
#   MSBN_INIT   ('zero-shift'/'random'): for v8.2 random-init ablation
#   USE_MSBN_LAYER1/LAYER2: per-layer MSBN toggle, over-engineered
#   MSBN_REGULARIZATION_WEIGHT: zero-shift + shared conv already constrains
_CN.LOFTR.USE_MSBN              = False
_CN.LOFTR.FREEZE_FINE_BN_IR     = False
_CN.LOFTR.FREEZE_FINE_BN_VIS    = False

# -- # v7_pcclahe: input-side cross-modal optimisation (A1 PC edge channel + A2 CLAHE on IR)
# All flags default to disabled so v0-v6.1 cfg merge yields byte-identical
# behaviour. Each v7+ config opts in explicitly. See plan v7_pcclahe and the
# "v0-v6.1 兼容性纪律" section for the three-layer default-value-equality
# discipline (default.py / data.py getattr fallback / RoadSceneDataset.__init__).
#
# BACKBONE_IN_CHANNELS:  stage0 RepVGGBlock in_channels. Default 1 keeps
#                        v0-v6.1 ckpt loadable without inflation.
# USE_EDGE_INPUT:        feed Phase Congruency edge map as a 2nd input channel
#                        alongside the raw image. Requires BACKBONE_IN_CHANNELS=2
#                        and pre-computed PC cache (run MyScripts/precompute_pc_edges.bat).
# USE_CLAHE_IR / VIS:    apply CLAHE histogram equalisation on the raw uint8
#                        IR / VIS image before resize/warp/pad. Boosts contrast
#                        on narrow-histogram IR. CLAHE is NOT applied to the
#                        PC channel even when USE_EDGE_INPUT is on.
# CLAHE_CLIP_LIMIT / CLAHE_TILE_SIZE: standard CLAHE knobs. Tile size is a
#                        list (yacs override-friendly), converted to tuple
#                        inside the dataset for cv2.createCLAHE().
_CN.LOFTR.BACKBONE_IN_CHANNELS = 1
_CN.LOFTR.USE_EDGE_INPUT = False
_CN.LOFTR.USE_CLAHE_IR = False
_CN.LOFTR.USE_CLAHE_VIS = False
_CN.LOFTR.CLAHE_CLIP_LIMIT = 2.0
_CN.LOFTR.CLAHE_TILE_SIZE = [8, 8]

# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR.BACKBONE = CN()
_CN.LOFTR.BACKBONE.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 256
_CN.LOFTR.COARSE.D_FFN = 256
_CN.LOFTR.COARSE.NHEAD = 8
_CN.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.LOFTR.COARSE.AGG_SIZE0 = 4
_CN.LOFTR.COARSE.AGG_SIZE1 = 4
_CN.LOFTR.COARSE.NO_FLASH = False
_CN.LOFTR.COARSE.ROPE = True
_CN.LOFTR.COARSE.NPE = None # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832

# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.2 # recommend 0.2 for full model and 25 for optimized model
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 2
_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_COARSE.SKIP_SOFTMAX = False
_CN.LOFTR.MATCH_COARSE.FP16MATMUL = False

# 4. Fine-Matching config
_CN.LOFTR.MATCH_FINE = CN()
_CN.LOFTR.MATCH_FINE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 1.0
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8
# v18: fine-pixel-idx anchor convention. False = v0 paper "-3.5 trick" (cell
# centre anchor; grid value range [-W/2+0.5, W/2-0.5]). True = absolute idx
# (cell top-left anchor; grid value range [0, W-1]). Used jointly by:
#   - supervision.spvs_fine: grid_pt0_f creation + delta_w_pt0_f offset
#   - inference.fine_matching.get_fine_ds_match: stage-1 grid creation
# so train/test stay consistent. Default False keeps v0-v17 byte-identical;
# v18 cfg sets True because H_vis rotation breaks the -3.5 trick's implicit
# local-linearity assumption (~4 fine-pixel residual at rot=50 deg).
_CN.LOFTR.MATCH_FINE.ABSOLUTE_FINE_IDX = False

# 5. LoFTR Losses
# -- # coarse-level
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
_CN.LOFTR.LOSS.COARSE_SIGMOID_WEIGHT = 1.0
_CN.LOFTR.LOSS.LOCAL_WEIGHT = 0.5
_CN.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = False
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = False
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT2 = False
# -- - -- # focal loss (coarse)
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0

# -- # fine-level
_CN.LOFTR.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

# -- # cross-modal contrastive loss (RoadScene IR-VIS, symmetric InfoNCE)
# When enabled, LoFTR.forward stores the post-transformer coarse tokens into
# `data` and LoFTRLoss adds a CLIP-style symmetric InfoNCE term that pulls
# IR/VIS features at GT-matched positions together while pushing all other
# in-batch tokens apart. Negatives span the whole batch, so batch_size >= 2
# is required to gain cross-scene negatives (bs=1 degenerates to within-image
# competition, equivalent to dual_softmax pool).
_CN.LOFTR.LOSS.USE_CONTRASTIVE = False
_CN.LOFTR.LOSS.CONTRASTIVE_WEIGHT = 0.01      # weight on (loss_i2v + loss_v2i) / 2
_CN.LOFTR.LOSS.CONTRASTIVE_TEMP = 0.1         # InfoNCE temperature, matches DSMAX_TEMPERATURE


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
_CN.DATASET.FP16 = False
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# scanNet options
_CN.DATASET.SCAN_IMG_RESIZEX = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.SCAN_IMG_RESIZEY = 480  # resize the shorter side, zero-pad bottom-right to square.

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# v18: Single-side rotation Homography augmentation for Megadepth-style cross-view
# pose datasets (only consulted by MegadepthSynPoseDataset, ignored by RoadScene/
# M3FD/MegadepthSyn-H paths). Image1 (VIS) is warped by H_vis sampled in resized
# pixel space; image0 (IR) is untouched. supervision.spvs_coarse / spvs_fine pick
# up data['H_vis'] only when AUG=True so v0-v16 ckpt training stays byte-identical
# (no H_vis key in batch -> 'H_vis' in data guards skip new branches).
#
# Defaults preserve v0-v17 byte-identical; v18 cfg sets AUG=True with
# KWARGS.rot_deg=50, scale_range=[1.0,1.0], trans_ratio=0.0, persp_ratio=0.0.
_CN.DATASET.MGDPT_HOMOGRAPHY_AUG = False                # default off, v0-v16 byte-identical
_CN.DATASET.MGDPT_HOMOGRAPHY_PROB = 1.0                 # 1.0 = always trigger when AUG=True
_CN.DATASET.MGDPT_HOMOGRAPHY_KWARGS = CN(new_allowed=True)  # rot_deg / scale_range / trans_ratio / persp_ratio

# RoadScene options (only consulted when TRAINVAL_DATA_SOURCE == 'RoadScene')
_CN.DATASET.ROAD_IR_SUBDIR = 'cropinfrared'
_CN.DATASET.ROAD_VIS_SUBDIR = 'crop_LR_visible'  # IR-aligned. crop_HR_visible has a wider FoV and is NOT pixel-aligned with cropinfrared.
_CN.DATASET.ROAD_IMG_RESIZE = 480     # longer-edge target before df-rounding
_CN.DATASET.ROAD_DF = 32              # final H, W are multiples of df
# Square canvas size after zero-padding (must be >= ROAD_IMG_RESIZE and divisible
# by ROAD_DF). All samples are padded to (ROAD_PAD_SIZE, ROAD_PAD_SIZE) so the
# DataLoader can stack tensors of identical shape into a batch (batch_size > 1).
# When None, defaults to ROAD_IMG_RESIZE rounded up to ROAD_DF at runtime.
_CN.DATASET.ROAD_PAD_SIZE = None
_CN.DATASET.ROAD_HOMOGRAPHY_AUG = True  # train-only random Homography on VIS
_CN.DATASET.ROAD_HOMOGRAPHY_PROB = 1.0
# v11 dual-side Homography: when True, warp BOTH IR (image0) and VIS (image1)
# by independent random Homographies; homography_0to1 = H_vis @ inv(H_ir).
# Default False = v0-v10 byte-identical (only VIS warped, IR untouched).
_CN.DATASET.ROAD_HOMOGRAPHY_DUAL = False
# Per-axis kwargs forwarded to _random_homography (rot_deg, scale_range,
# trans_ratio, persp_ratio). new_allowed=True so v11 cfg can declare new
# sub-fields without YACS strict-mode KeyError. When the cfg leaves this
# CN empty, data.py forwards an empty dict and _random_homography uses
# its function defaults (rot_deg=10, scale (0.9,1.1), trans 0.05, persp
# 0.03 = v0-v10 strength). v11 cfg overrides with the aggressive preset.
_CN.DATASET.ROAD_HOMOGRAPHY_KWARGS = CN(new_allowed=True)

# -- # v7_pcclahe PC-cache sub-directories (per-dataset because RoadScene uses
#    snake_case sub-dirs and M3FD uses CapitalCase; default '' = unused).
#    Each data config (configs/data/<dataset>_trainval.py) sets these
#    explicitly when it wants to enable PC, e.g.:
#      m3fd_trainval.py    -> 'Ir_pc'             / 'Vis_pc'
#      roadscene_trainval.py -> 'cropinfrared_pc' / 'crop_LR_visible_pc'
#    The dataset class only attempts to load PC when LOFTR.USE_EDGE_INPUT=True;
#    these sub-dir fields being non-empty alone does NOT trigger PC loading.
_CN.DATASET.ROAD_IR_PC_SUBDIR = ''
_CN.DATASET.ROAD_VIS_PC_SUBDIR = ''

# METU_VISTIR options (only consulted when TEST_DATA_SOURCE == 'METU_VISTIR').
# Pose-based test only; METU has no train_list. Defaults match the v10/v11
# training-time IR-VIS convention (image0 = visible) so a v0..v11 ckpt eval
# with this dataset class behaves identically without needing per-cfg overrides.
_CN.DATASET.METU_UNDISTORT = True   # cv2.undistort with 8-coef OpenCV model
# image0 = thermal / image1 = vis: aligns with the v0..v11 training-side
# convention (RoadSceneDataset.__getitem__:376-378 puts ir_dir into image0,
# vis_dir into image1; modemb_ir/modemb_vis are bound to image0/image1
# accordingly in src/loftr/loftr.py:117-118).
#
# Empirical sweep on v10 ckpt over 30 METU pairs (tmp_metu_side_sweep.py
# 2026-05-11) reports thermal-as-image0 yields auc@20=0.87% vs 0.58%
# (vis-as-image0) -- i.e. the training-aligned direction is ~50% better
# despite ~3x fewer raw matches (training-side feature alignment >
# raw match count). Keep this default unless a future ckpt's training cfg
# explicitly puts visible into image0.
_CN.DATASET.METU_SIDE0 = 'thermal'
_CN.DATASET.METU_SIDE1 = 'vis'

# Pad-to-square switch for METU_VISTIR test loop. MINIMA's reference protocol
# (data_io_loftr.py L34-71 + test_relative_pose_infrared.py) explicitly runs
# matcher with padding=False at bs=1: each pair forwards through ELoFTR at its
# own undistorted+resized H/W. Our previous dataset always pad-zeroed to a
# square canvas (to keep bs>1 collate well-defined for hypothetical batched
# eval), which is a different code path from MINIMA and a small numerical
# perturbation (border-padded zeros leak into RepVGG/coarse attention near
# the seam). Default False = MINIMA-exact; set True only if you run METU
# eval with bs>1 (PyTorch default_collate then needs uniform shapes).
_CN.DATASET.METU_PAD_TO_SQUARE = False

# Megadepth_Syn_Pose (v13) options. Only consulted when TRAINVAL_DATA_SOURCE
# == 'Megadepth_Syn_Pose' (cross-view + cross-modal pose-supervised training
# on LoFTR scene_info_0.1_0.7 + Megadepth_Syn IR/VIS assets). 'ir2vis' is
# the only mode used by v13a main run; 'vis2vis' is reserved for future
# RGB-RGB cross-view in-domain ceiling ablation (same dispatch + supervision
# pipeline, only the modality of image0 changes).
_CN.DATASET.MSYN_POSE_CROSS_MODAL_MODE = 'ir2vis'

_CN.DATASET.NPE_NAME = None

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, LO-RANSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# Opt-in DataLoader performance flag. When True, workers are kept alive
# across epochs (saves spin-up overhead, ~30s/epoch with num_workers=6).
# Default False to preserve v0-v5's bytes-identical augmentation RNG
# sequence on retrain: RoadScene/M3FD use np.random.rand() in __getitem__
# (src/datasets/roadscene.py:249) without any worker_init_fn, so persistent
# workers would let numpy state accumulate across epochs -- statistically
# equivalent but not bytes-identical. Each v_x config opts in explicitly
# (currently only configs/loftr/eloftr_full_v6_finetune.py).
_CN.TRAINER.PERSISTENT_WORKERS = False

# EarlyStopping callback (added by train.py when ENABLED). Reuses the same
# monitor / mode as ModelCheckpoint, so for RoadScene it watches
# `precision@3px` (mode=max) and for ScanNet/MegaDepth it watches `auc@10`.
# When val metric does not improve for PATIENCE consecutive validation
# epochs, training is stopped. Set ENABLED = False to keep legacy behaviour.
_CN.TRAINER.EARLY_STOPPING = False
_CN.TRAINER.EARLY_STOPPING_PATIENCE = 2

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
