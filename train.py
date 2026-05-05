import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin, NativeMixedPrecisionPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.utils.data_source import is_aligned_irvis
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR
import torch

# PyTorch 2.6 changed torch.load's default weights_only from False to True.
# Our own torch.load call in src/lightning/lightning_loftr.py already passes
# weights_only=False, but PyTorch Lightning 1.3.5's internal pl_load
# (pytorch_lightning/utilities/cloud_io.py) does not -- so any code path that
# goes through PL's checkpoint connector (e.g. --resume_from_checkpoint, PL
# auto-resume, ModelCheckpoint internal verification) crashes with
#   _pickle.UnpicklingError: Weights only load failed ...
#   Unsupported global: pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
# because PL 1.x pickles its own callback / scheduler instances into the ckpt.
# Monkey-patching torch.load here makes every torch.load call (including PL's
# internal pl_load) default to weights_only=False, while still respecting any
# explicit weights_only=True passed by the caller. Only safe because every ckpt
# we load in this project comes from a trusted source (official ELoFTR weights
# or our own training runs).
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

import numpy as np
np.Inf = np.inf

loguru_logger = get_rank_zero_only_logger(loguru_logger)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--thr', type=float, default=0.1)
    parser.add_argument(
        '--train_coarse_percent', type=float, default=0.1, help='training tricks: save GPU memory')
    parser.add_argument(
        '--disable_mp', action='store_true', help='disable mixed-precision training')
    parser.add_argument(
        '--deter', action='store_true', help='use deterministic mode for training')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    get_cfg_default = get_cfg_defaults

    config = get_cfg_default()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    
    if config.LOFTR.COARSE.NPE is None:
        config.LOFTR.COARSE.NPE = [832, 832, 832, 832]  # training at 832 resolution on MegaDepth datasets
    
    if args.deter:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation
    
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr
    if args.disable_mp:
        config.LOFTR.MP = False

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LoFTR LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"LoFTR DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]

    # Pick monitored metric + filename template based on the dataset.
    # ScanNet/MegaDepth use the geometric AUC@10; aligned IR-VIS sources
    # (RoadScene / M3FD / any future entry in ALIGNED_IRVIS_SOURCES) have no
    # camera geometry, so we monitor the pixel precision@3px logged by
    # PL_LoFTR's _compute_roadscene_metrics. Dispatch must go through
    # is_aligned_irvis() -- never compare the source name as a literal,
    # otherwise adding a new IR-VIS dataset silently falls into the
    # auc@10 branch and crashes EarlyStopping at the first val epoch
    # (RuntimeError: Early stopping conditioned on metric `auc@10` which is
    # not available). See eloftr-m3fd-data SKILL.md SS7 for the rule.
    # Computed outside the ckpt/EarlyStopping branches so both callbacks can
    # share it (and EarlyStopping can be enabled even when --disable_ckpt
    # is set).
    if is_aligned_irvis(config.DATASET.TRAINVAL_DATA_SOURCE):
        monitor_metric = 'precision@3px'
        monitor_mode = 'max'
        filename_tpl = '{epoch}-{precision@1px:.3f}-{precision@3px:.3f}-{precision@5px:.3f}'
    else:
        monitor_metric = 'auc@10'
        monitor_mode = 'max'
        filename_tpl = '{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}'

    if not args.disable_ckpt:
        ckpt_callback = ModelCheckpoint(monitor=monitor_metric, verbose=True, save_top_k=5, mode=monitor_mode,
                                        save_last=True,
                                        dirpath=str(ckpt_dir),
                                        filename=filename_tpl)
        callbacks.append(ckpt_callback)

    # Optional EarlyStopping (independent of --disable_ckpt). Reuses the same
    # monitor metric/mode as ModelCheckpoint so the same target drives both
    # "save best" and "stop when stops improving". Useful for small datasets
    # where the val curve peaks early and then degrades (RoadScene baseline
    # runs peaked at epoch <4 and dropped through epoch 20).
    if config.TRAINER.EARLY_STOPPING:
        callbacks.append(EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=config.TRAINER.EARLY_STOPPING_PATIENCE,
            verbose=True,
            strict=True,
        ))
        loguru_logger.info(
            f"EarlyStopping enabled "
            f"(monitor={monitor_metric}, mode={monitor_mode}, "
            f"patience={config.TRAINER.EARLY_STOPPING_PATIENCE})"
        )

    plugins = [NativeMixedPrecisionPlugin()]
    if config.TRAINER.WORLD_SIZE > 1:
        plugins.insert(0, DDPPlugin(find_unused_parameters=False,
                                    num_nodes=args.num_nodes,
                                    sync_batchnorm=True))

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        # Original DDP-only setup fails on Windows single-GPU environments because NCCL is unavailable:
        # plugins=[DDPPlugin(find_unused_parameters=False,
        #                   num_nodes=args.num_nodes,
        #                   sync_batchnorm=config.TRAINER.WORLD_SIZE > 0), NativeMixedPrecisionPlugin()],
        plugins=plugins,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 1,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()