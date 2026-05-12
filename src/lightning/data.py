import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
from src.utils.data_source import is_aligned_irvis
from src.datasets.megadepth import MegaDepthDataset
from src.datasets.scannet import ScanNetDataset
from src.datasets.roadscene import RoadSceneDataset
from src.datasets.metu_vistir import METUVisTIRDataset
from src.datasets.sampler import RandomConcatSampler


class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # ScanNet options
        self.scan_img_resizeX = config.DATASET.SCAN_IMG_RESIZEX  # 640
        self.scan_img_resizeY = config.DATASET.SCAN_IMG_RESIZEY  # 480


        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 832
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD   # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD   # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125. for training loftr.

        # RoadScene options (only consulted when DATA_SOURCE == 'RoadScene')
        self.road_ir_subdir = getattr(config.DATASET, 'ROAD_IR_SUBDIR', 'cropinfrared')
        self.road_vis_subdir = getattr(config.DATASET, 'ROAD_VIS_SUBDIR', 'crop_LR_visible')
        self.road_img_resize = getattr(config.DATASET, 'ROAD_IMG_RESIZE', 480)
        self.road_df = getattr(config.DATASET, 'ROAD_DF', 32)
        # ROAD_PAD_SIZE may be None (resolved to img_resize-rounded-to-df inside the dataset)
        self.road_pad_size = getattr(config.DATASET, 'ROAD_PAD_SIZE', None)
        self.road_homography_aug = getattr(config.DATASET, 'ROAD_HOMOGRAPHY_AUG', True)
        self.road_homography_prob = getattr(config.DATASET, 'ROAD_HOMOGRAPHY_PROB', 1.0)
        self.road_homography_kwargs = dict(getattr(config.DATASET,
                                                  'ROAD_HOMOGRAPHY_KWARGS', {}))
        # v11 dual-side Homography: warp BOTH IR and VIS by independent H.
        # Default False = v0-v10 byte-identical (only VIS warped).
        self.road_homography_dual = getattr(config.DATASET, 'ROAD_HOMOGRAPHY_DUAL', False)

        # v7_pcclahe input-side options. R3 three-layer default-value equality:
        # the fallback literals here MUST match src/config/default.py and
        # RoadSceneDataset.__init__ defaults so a missing yacs field still
        # yields the v0-v6.1 byte-identical "everything off" behaviour.
        self.use_edge_input = getattr(config.LOFTR, 'USE_EDGE_INPUT', False)
        self.use_clahe_ir = getattr(config.LOFTR, 'USE_CLAHE_IR', False)
        self.use_clahe_vis = getattr(config.LOFTR, 'USE_CLAHE_VIS', False)
        self.clahe_clip_limit = getattr(config.LOFTR, 'CLAHE_CLIP_LIMIT', 2.0)
        self.clahe_tile_size = getattr(config.LOFTR, 'CLAHE_TILE_SIZE', [8, 8])
        self.road_ir_pc_subdir = getattr(config.DATASET, 'ROAD_IR_PC_SUBDIR', '')
        self.road_vis_pc_subdir = getattr(config.DATASET, 'ROAD_VIS_PC_SUBDIR', '')

        # METU_VISTIR test-only options. Defaults match src/config/default.py
        # so v0..v11 cfgs that don't set them stay byte-identical.
        self.metu_undistort = getattr(config.DATASET, 'METU_UNDISTORT', True)
        self.metu_side0 = getattr(config.DATASET, 'METU_SIDE0', 'vis')
        self.metu_side1 = getattr(config.DATASET, 'METU_SIDE1', 'thermal')

        self.fp16 = config.DATASET.FP16

        # 3.loader parameters
        # ``persistent_workers`` is gated by cfg.TRAINER.PERSISTENT_WORKERS
        # (default False, set in src/config/default.py). v0-v5 configs do
        # not opt in, so loader behaviour stays bytes-identical to history;
        # only v6_finetune.py opts in explicitly. The ``args.num_workers > 0``
        # guard avoids PyTorch's "persistent_workers requires num_workers > 0"
        # error in debug bats that pass --num_workers=0.
        _persistent = bool(getattr(config.TRAINER, 'PERSISTENT_WORKERS', False)) and args.num_workers > 0
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'persistent_workers': _persistent,
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'persistent_workers': _persistent,
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        
        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        
        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'validate', 'test'], "stage must be either fit or test"

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0
            logger.warning("Distributed process group is not initialized. Set world_size=1 and rank=0.")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode='train',
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root)
            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        elif stage == 'validate':
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       intri_path,
                       mode='train',
                       min_overlap_score=0.,
                       pose_dir=None):
        """ Setup train / val / test set"""
        # Aligned IR-VIS datasets (RoadScene, M3FD, ...) use a single flat txt
        # list of image filenames rather than per-scene .npz files, so we
        # short-circuit the per-scene iteration used by ScanNet/MegaDepth and
        # build one RoadSceneDataset directly. Membership is decided by
        # ``src.utils.data_source.is_aligned_irvis`` so adding a new dataset
        # is a one-line change to ``ALIGNED_IRVIS_SOURCES``.
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        if is_aligned_irvis(data_source):
            # DDP image-level pre-sharding (mirrors the scene-level
            # get_local_split call below for ScanNet/MegaDepth, see L283).
            # Without this, every rank loads the full index file and
            # RandomConcatSampler -- which intentionally is NOT DDP-aware
            # (sampler.py L16-17: "It assumes the dataset is splitted across
            # ranks instead of replicated") -- gives all ranks identical
            # batches because they all torch.manual_seed(cfg.TRAINER.SEED) on
            # the same global generator. v0-v9 ran single-card so this design
            # gap never surfaced; v10 mode B 4-card DDP is the first time it
            # matters.
            #
            # Pre-shard at TRAIN time only; val/test stay full because their
            # dataloaders wrap with PyTorch's standard DistributedSampler
            # (val_dataloader L412-413, test_dataloader L429-430).
            with open(scene_list_path, 'r', encoding='utf-8') as f:
                all_names = [line.strip() for line in f.readlines() if line.strip()]
            if mode == 'train' and self.world_size > 1:
                local_names = list(get_local_split(
                    all_names, self.world_size, self.rank, self.seed))
                logger.info(
                    f'[rank {self.rank}]: aligned IR-VIS train pre-shard: '
                    f'{len(local_names)}/{len(all_names)} samples '
                    f'(disjoint across {self.world_size} ranks; seed={self.seed})')
            else:
                local_names = all_names
            logger.info(f'[rank {self.rank}]: building RoadSceneDataset (dataset_name={data_source}) from {scene_list_path}')
            ds = RoadSceneDataset(
                root_dir=data_root,
                list_path=scene_list_path,
                mode=mode,
                dataset_name=str(data_source),
                ir_subdir=self.road_ir_subdir,
                vis_subdir=self.road_vis_subdir,
                img_resize=self.road_img_resize,
                pad_size=self.road_pad_size,
                df=self.road_df,
                coarse_scale=self.coarse_scale,
                homography_aug=(self.road_homography_aug and mode == 'train'),
                homography_prob=self.road_homography_prob,
                homography_kwargs=self.road_homography_kwargs,
                homography_dual=self.road_homography_dual,
                augment_fn=(self.augment_fn if mode == 'train' else None),
                fp16=self.fp16,
                names=local_names,  # DDP pre-shard hand-off (None-equivalent in single-card)
                # v7_pcclahe (defaults to all-off so v0-v6.1 cfg is byte-identical)
                use_edge_input=self.use_edge_input,
                ir_pc_subdir=self.road_ir_pc_subdir,
                vis_pc_subdir=self.road_vis_pc_subdir,
                use_clahe_ir=self.use_clahe_ir,
                use_clahe_vis=self.use_clahe_vis,
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_tile_size=self.clahe_tile_size,
            )
            return ConcatDataset([ds])

        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
        else:
            local_npz_names = npz_names
        logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')
        
        dataset_builder = self._build_concat_dataset_parallel \
                            if self.parallel_load_data \
                            else self._build_concat_dataset
        return dataset_builder(data_root, local_npz_names, split_npz_root, intri_path,
                                mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir)

    def _build_concat_dataset(
        self,
        data_root,
        npz_names,
        npz_dir,
        intrinsic_path,
        mode,
        min_overlap_score=0.,
        pose_dir=None
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        for npz_name in tqdm(npz_names,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
            npz_path = osp.join(npz_dir, npz_name)
            if data_source == 'ScanNet':
                datasets.append(
                    ScanNetDataset(data_root,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode,
                                   min_overlap_score=min_overlap_score,
                                   augment_fn=augment_fn,
                                   pose_dir=pose_dir,
                                   img_resize=(self.scan_img_resizeX, self.scan_img_resizeY),
                                   fp16 = self.fp16,
                                   ))
            elif data_source == 'MegaDepth':
                datasets.append(
                    MegaDepthDataset(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale,
                                     fp16 = self.fp16,
                                     ))
            elif str(data_source).lower() == 'metu_vistir':
                # METU_VISTIR cross-modal pose-based eval (test only). One npz
                # per scene; ConcatDataset stitches all listed npz together.
                # PC channel comes from runtime fallback (compute_pc_v11_runtime)
                # because METU has no pre-computed cache.
                datasets.append(
                    METUVisTIRDataset(
                        root_dir=data_root,
                        npz_path=npz_path,
                        mode=mode,
                        img_resize=self.mgdpt_img_resize,
                        df=self.mgdpt_df,
                        coarse_scale=self.coarse_scale,
                        undistort=self.metu_undistort,
                        side0=self.metu_side0,
                        side1=self.metu_side1,
                        use_edge_input=self.use_edge_input,
                        use_clahe_ir=self.use_clahe_ir,
                        use_clahe_vis=self.use_clahe_vis,
                        clahe_clip_limit=self.clahe_clip_limit,
                        clahe_tile_size=self.clahe_tile_size,
                        fp16=self.fp16,
                    ))
            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)
    
    def _build_concat_dataset_parallel(
        self,
        data_root,
        npz_names,
        npz_dir,
        intrinsic_path,
        mode,
        min_overlap_score=0.,
        pose_dir=None,
    ):
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        with tqdm_joblib(tqdm(desc=f'[rank:{self.rank}] loading {mode} datasets',
                              total=len(npz_names), disable=int(self.rank) != 0)):
            if data_source == 'ScanNet':
                datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / comm.get_local_size()))(
                    delayed(lambda x: _build_dataset(
                        ScanNetDataset,
                        data_root,
                        osp.join(npz_dir, x),
                        intrinsic_path,
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        augment_fn=augment_fn,
                        pose_dir=pose_dir))(name)
                    for name in npz_names)
            elif data_source == 'MegaDepth':
                # TODO: _pickle.PicklingError: Could not pickle the task to send it to the workers.
                raise NotImplementedError()
                datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / comm.get_local_size()))(
                    delayed(lambda x: _build_dataset(
                        MegaDepthDataset,
                        data_root,
                        osp.join(npz_dir, x),
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        img_resize=self.mgdpt_img_resize,
                        df=self.mgdpt_df,
                        img_padding=self.mgdpt_img_pad,
                        depth_padding=self.mgdpt_depth_pad,
                        augment_fn=augment_fn,
                        coarse_scale=self.coarse_scale))(name)
                    for name in npz_names)
            else:
                raise ValueError(f'Unknown dataset: {data_source}')
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        assert self.data_sampler in ['scene_balance']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.subset_replacement,
                                          self.shuffle, self.repeat, self.seed)
        else:
            sampler = None
        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        use_distributed = dist.is_available() and dist.is_initialized() and self.world_size > 1
        if not isinstance(self.val_dataset, abc.Sequence):
            if use_distributed:
                sampler = DistributedSampler(self.val_dataset, shuffle=False)
                return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
            return DataLoader(self.val_dataset, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                if use_distributed:
                    sampler = DistributedSampler(dataset, shuffle=False)
                    dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
                else:
                    dataloaders.append(DataLoader(dataset, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        use_distributed = dist.is_available() and dist.is_initialized() and self.world_size > 1
        if use_distributed:
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
            return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)
        return DataLoader(self.test_dataset, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
