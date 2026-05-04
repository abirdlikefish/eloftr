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
