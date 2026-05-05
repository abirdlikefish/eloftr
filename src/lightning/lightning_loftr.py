
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.data_source import is_aligned_irvis

from torch.profiler import profile

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher


def _maybe_inflate_stage0(state_dict, backbone, alpha=0.0):
    """Inflate the backbone stage0 (a.k.a. layer0 in ``RepVGG_8_1_align``) conv
    weights from ``in_ch=1`` to ``in_ch=2`` when the model expects 2 channels
    but the ckpt was saved with 1.

    Why this exists
    ---------------
    v7_pcclahe stacks a Phase Congruency edge map alongside the raw image and
    sets ``cfg.LOFTR.BACKBONE_IN_CHANNELS=2`` so stage0's first conv accepts
    ``(64, 2, 3, 3)`` weights. v6.1 ckpt was trained with in_ch=1, giving
    ``(64, 1, 3, 3)`` weights. ``load_state_dict(strict=False)`` does NOT
    handle shape mismatches -- it would raise ``RuntimeError: size mismatch``
    on every key. ``strict`` only relaxes the *key set*, not shapes.

    This helper re-shapes the two affected weights in-place inside the ckpt's
    state_dict before ``load_state_dict`` is called:

      ``(64, 1, 3, 3)`` -> ``(64, 2, 3, 3)`` via ``cat([w_old, w_old * alpha])``

    With ``alpha=0`` (the default for v7) the PC channel weight is exactly 0,
    so stage0's forward output is mathematically identical to v6.1 at epoch 0.
    Training pushes it away from 0 from epoch 1 onward.

    v0-v6.1 backwards compatibility
    -------------------------------
    When the model itself is built with ``in_channels=1`` (which is the
    default for every v0-v6.1 cfg because ``BACKBONE_IN_CHANNELS`` defaults to
    1 in ``src/config/default.py``), ``target_in_ch == 1`` matches the ckpt
    and this function is a no-op (no inflate, no log line). The v0-v6.1 retrain
    / eval log output is therefore byte-identical to before this helper landed.

    Both ckpt key conventions are handled
    -------------------------------------
    PL-saved ckpts (everything trained in this repo) prefix every key with
    ``matcher.`` because ``PL_LoFTR.matcher = LoFTR(...)``. The official
    ``eloftr_outdoor.ckpt`` is a bare LoFTR state_dict with no prefix.
    ``LoFTR.load_state_dict`` (see ``src/loftr/loftr.py``) strips the
    ``matcher.`` prefix on the way in, but we run BEFORE that, so we look for
    both forms.
    """
    target_in_ch = backbone.layer0.rbr_dense.conv.in_channels
    keys_to_check = [
        # PL-saved: keys still carry the ``matcher.`` prefix at this point.
        'matcher.backbone.layer0.rbr_dense.conv.weight',
        'matcher.backbone.layer0.rbr_1x1.conv.weight',
        # Official / bare LoFTR ckpt: no prefix.
        'backbone.layer0.rbr_dense.conv.weight',
        'backbone.layer0.rbr_1x1.conv.weight',
    ]
    inflated_any = False
    for k in keys_to_check:
        if k not in state_dict:
            continue
        w_old = state_dict[k]
        if w_old.shape[1] == target_in_ch:
            continue                                            # v0-v6.1 path: skip silently
        if w_old.shape[1] == 1 and target_in_ch == 2:
            w_pc = w_old * alpha                                # alpha=0 -> all zeros
            state_dict[k] = torch.cat([w_old, w_pc], dim=1)     # (out, 1, K, K) -> (out, 2, K, K)
            inflated_any = True
        else:
            raise RuntimeError(
                f"Cannot inflate {k}: ckpt in_ch={w_old.shape[1]}, "
                f"model in_ch={target_in_ch}; only 1->2 inflate is supported")
    if inflated_any:
        logger.info(
            f"Inflated stage0 conv weights: 1ch -> 2ch (alpha={alpha}, "
            f"PC channel zero-initialized; gate 12 should pass)")
    return state_dict


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr'], profiler=self.profiler)
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)['state_dict']
            # v7_pcclahe: when the v7 model is built with in_channels=2 but
            # we resume from a v6.1 (or earlier) ckpt that was trained with
            # in_channels=1, inflate the two stage0 conv weights so
            # load_state_dict shape-matches. No-op when in_channels match
            # (which is every v0-v6.1 cfg) -- see _maybe_inflate_stage0.
            state_dict = _maybe_inflate_stage0(state_dict, self.matcher.backbone, alpha=0.0)
            msg=self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
            # Explicitly surface mismatched keys so we can verify that any
            # newly-added params (e.g. modality_emb_ir / modality_emb_vis in
            # Step 2) really are the only "missing" keys -- if anything else
            # shows up here it likely means the ckpt is the wrong arch.
            if msg.missing_keys:
                logger.info(f"  missing_keys (kept at init value): {msg.missing_keys}")
            if msg.unexpected_keys:
                logger.warning(f"  unexpected_keys (ignored): {msg.unexpected_keys}")

        # Optional freezing for small-dataset finetune (must run AFTER the
        # pretrained ckpt is loaded so load_state_dict still happens on the
        # full trainable graph). Both flags default to False, so behaviour is
        # unchanged for v0/v1/v2 runs.
        self._freeze_bn          = bool(config.LOFTR.get('FREEZE_BN', False))
        self._freeze_backbone_bn = bool(config.LOFTR.get('FREEZE_BACKBONE_BN', False))
        if config.LOFTR.get('FREEZE_BACKBONE', False):
            for p in self.matcher.backbone.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.matcher.backbone.parameters())
            logger.info(f"Froze backbone: {n_frozen/1e6:.2f}M params")
        if self._freeze_bn:
            self._apply_freeze_bn()
            logger.info("Froze all BatchNorm2d layers (eval-mode + no-grad)")
        # FREEZE_BN takes precedence: when all BN already frozen above, skip the
        # backbone-only branch to avoid redundant work and a misleading log line.
        if self._freeze_backbone_bn and not self._freeze_bn:
            self._apply_freeze_backbone_bn()
            logger.info("Froze backbone BatchNorm2d layers (eval-mode + no-grad); fine_preprocess BN remains trainable")

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable params: {n_trainable/1e6:.2f}M / Total: {n_total/1e6:.2f}M")

        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0

    def _apply_freeze_bn(self):
        """Set every BatchNorm2d to eval-mode and disable grads on its affine
        params. Called once in __init__ AND from `train()` because PL's per-
        epoch model.train() would otherwise re-enable training mode on BN.
        """
        for m in self.matcher.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _apply_freeze_backbone_bn(self):
        """Freeze BN only inside `matcher.backbone.*`. Does NOT touch BN inside
        fine_preprocess (those 2 layers stay trainable + train-mode). Called
        once in __init__ AND from `train()` to survive PL's per-epoch
        model.train() that would otherwise re-enable BN train-mode.
        """
        for m in self.matcher.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        """Override to keep frozen BN layers in eval-mode across PL's per-epoch
        model.train() calls. Without this, freezing BN in __init__ would only
        last until the first epoch boundary. The if/elif ensures FREEZE_BN
        (all BN) takes precedence over FREEZE_BACKBONE_BN (backbone BN only)
        so v3 (FREEZE_BN=True) is never double-frozen even if both flags are
        set."""
        super().train(mode)
        if getattr(self, '_freeze_bn', False):
            self._apply_freeze_bn()
        elif getattr(self, '_freeze_backbone_bn', False):
            self._apply_freeze_backbone_bn()
        return self

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=False, device_type='cuda'):
                compute_supervision_fine(batch, self.config, self.logger)
            
        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.loss(batch)
    
    def _compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [(batch['epi_errs'].reshape(-1,1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'num_matches': [batch['mconf'].shape[0]], # batch size = 1 only
            }
        ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    @torch.no_grad()
    def _compute_roadscene_metrics(self, batch):
        """Pixel-error metrics for RoadScene IR-VIS validation/testing.

        Uses ``batch['homography_0to1']`` to project predicted IR keypoints
        (``mkpts0_f``) into VIS coordinates and compares against the predicted
        VIS keypoints (``mkpts1_f``). When no Homography augmentation is
        applied this falls back to direct same-coordinate comparison because
        ``H = I``.
        """
        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        m_bids = batch['m_bids']
        mkpts0 = batch['mkpts0_f']
        mkpts1 = batch['mkpts1_f']
        mconf = batch['mconf']

        # Apply per-batch Homography to mkpts0 to obtain predicted VIS coords.
        if mkpts0.shape[0] > 0 and 'homography_0to1' in batch:
            H = batch['homography_0to1'].to(mkpts0.dtype)
            if H.dim() == 2:
                H = H[None].expand(bs, -1, -1)
            H_per_match = H[m_bids]                            # [M, 3, 3]
            ones = torch.ones(mkpts0.size(0), 1, dtype=mkpts0.dtype, device=mkpts0.device)
            pts_h = torch.cat([mkpts0, ones], dim=-1)[..., None]  # [M, 3, 1]
            warped_h = torch.bmm(H_per_match, pts_h).squeeze(-1)  # [M, 3]
            warped = warped_h[..., :2] / warped_h[..., 2:3].clamp(min=1e-8)
            pixel_errs = torch.linalg.norm(warped - mkpts1, dim=-1)
        else:
            pixel_errs = torch.linalg.norm(mkpts0 - mkpts1, dim=-1)

        # Stash pixel_errs on the batch so plotting can colour matches by error.
        batch['pixel_errs'] = pixel_errs

        per_pair_pixel_errs = []
        per_pair_num_matches = []
        per_pair_mean_conf = []
        for b in range(bs):
            mask = m_bids == b
            errs = pixel_errs[mask].detach().cpu().numpy()
            confs = mconf[mask].detach().cpu().numpy() if mconf.numel() > 0 else np.zeros(0)
            per_pair_pixel_errs.append(errs)
            per_pair_num_matches.append(int(mask.sum().item()))
            per_pair_mean_conf.append(float(confs.mean()) if len(confs) else 0.0)

        metrics = {
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'pixel_errs': per_pair_pixel_errs,
            'num_matches': per_pair_num_matches,
            'mean_conf': per_pair_mean_conf,
        }
        return {'metrics': metrics}
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # Log L2 norm of the modality embeddings so we can confirm they
            # actually grow from 0 (zeros init) rather than staying dead. Stuck
            # near 0 over multiple epochs indicates the gradient signal is
            # too weak and we should switch MODALITY_EMB_INIT to 'normal_0.02'.
            if getattr(self.matcher, 'use_modality_emb', False):
                self.logger.experiment.add_scalar(
                    'train/mod_emb_ir_norm',
                    self.matcher.modality_emb_ir.detach().norm().item(),
                    self.global_step)
                self.logger.experiment.add_scalar(
                    'train/mod_emb_vis_norm',
                    self.matcher.modality_emb_vis.detach().norm().item(),
                    self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)

    def on_validation_epoch_start(self):
        self.matcher.fine_matching.validate = True

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        is_roadscene = is_aligned_irvis(batch['dataset_name'][0])
        if is_roadscene:
            ret_dict = self._compute_roadscene_metrics(batch)
        else:
            ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        self.matcher.fine_matching.validate = False
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        is_roadscene = is_aligned_irvis(self.config.DATASET.TRAINVAL_DATA_SOURCE)
        roadscene_thresholds = (1.0, 3.0, 5.0)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

            if is_roadscene:
                val_metrics_4tb = self._aggregate_roadscene_metrics(metrics, roadscene_thresholds)
                for k, v in val_metrics_4tb.items():
                    multi_val_metrics[k].append(v)
            else:
                # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
                for thr in [5, 10, 20]:
                    multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        if is_roadscene:
            keys_to_log = [f'precision@{int(t)}px' for t in roadscene_thresholds]
            keys_to_log += ['mean_pixel_error', 'num_matches']
            for k in keys_to_log:
                vals = multi_val_metrics.get(k, [])
                if not vals:
                    continue
                self.log(k, torch.tensor(float(np.mean(vals))))
        else:
            for thr in [5, 10, 20]:
                # log on all ranks for ModelCheckpoint callback to work properly
                self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    @staticmethod
    def _aggregate_roadscene_metrics(metrics, thresholds=(1.0, 3.0, 5.0)):
        """Aggregate per-pair pixel errors into precision@Npx + summary stats."""
        from collections import OrderedDict

        # filter duplicates (DistributedSampler can repeat pairs)
        unq_ids = OrderedDict((iden, idx) for idx, iden in enumerate(metrics['identifiers']))
        unq_ids = list(unq_ids.values())

        per_pair_errs = [metrics['pixel_errs'][i] for i in unq_ids]
        per_pair_num = [metrics['num_matches'][i] for i in unq_ids]
        per_pair_conf = [metrics['mean_conf'][i] for i in unq_ids]

        all_errs = np.concatenate(per_pair_errs) if per_pair_errs else np.zeros(0)
        results = {}
        for t in thresholds:
            if len(all_errs):
                results[f'precision@{int(t)}px'] = float((all_errs < t).mean())
            else:
                results[f'precision@{int(t)}px'] = 0.0
        results['mean_pixel_error'] = float(all_errs.mean()) if len(all_errs) else 0.0
        results['num_matches'] = float(np.mean(per_pair_num)) if per_pair_num else 0.0
        results['mean_conf'] = float(np.mean(per_pair_conf)) if per_pair_conf else 0.0
        return results

    def test_step(self, batch, batch_idx):
        if (self.config.LOFTR.BACKBONE_TYPE == 'RepVGG') and not self.reparameter:
            self.matcher = reparameter(self.matcher)
            if self.config.LOFTR.HALF:
                self.matcher = self.matcher.eval().half()
            self.reparameter = True

        if not self.warmup:
            if self.config.LOFTR.HALF:
                for i in range(50):
                    self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                    for i in range(50):
                        self.matcher(batch)
            self.warmup = True
            torch.cuda.synchronize()

        if self.config.LOFTR.HALF:
            self.start_event.record()
            self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.LOFTR.MP, device_type='cuda'):
                self.start_event.record()
                self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.trainer.global_rank == 0:
            print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))