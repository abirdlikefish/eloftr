
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


def _maybe_inflate_msbn(state_dict, model):
    """v8: Bridge a v0-v7 ckpt's single ``layer{1,2}_outconv2.1.*`` (the
    BatchNorm weights/buffers inside the original Sequential) to v8's
    Modality-Specific BN layout where each layer has been replaced by an
    ``nn.Identity()`` placeholder + two named BN attrs ``_bn_ir`` / ``_bn_vis``.

    Why this exists
    ---------------
    v8 sets ``USE_MSBN=True`` and ``fine_preprocess.layer{1,2}_outconv2.1`` is
    ``nn.Identity()`` -- it has NO state_dict keys. The MSBN branches live at
    ``layer{1,2}_outconv2_bn_ir`` and ``layer{1,2}_outconv2_bn_vis``. The v7
    ckpt has 5 keys per layer at the original ``.1.*`` location. Without
    this hook:
      - 10 ``unexpected_keys`` warnings (v7 ckpt has ``.1.*`` but v8 model
        does not register them via Identity)
      - 20 ``missing_keys`` warnings (v8 model registers ``_bn_ir/_bn_vis.*``
        but v7 ckpt has neither)
      - The new BN branches stay at PyTorch defaults (gamma=1, beta=0,
        running_mean=0, running_var=1) instead of inheriting v7's well-tuned
        statistics -> v8 epoch 0 would NOT be byte-identical to v7 ep 6.

    Operation per layer (5 BN keys each)
    ------------------------------------
    For each ``k_orig`` in
    ``{weight, bias, running_mean, running_var, num_batches_tracked}``:
      1. Copy contents to ``..._bn_ir.<suffix>``  (new key)
      2. Copy contents to ``..._bn_vis.<suffix>`` (new key, == _bn_ir; this
         is "zero-shift" init: the two MSBN branches start identical to v7
         and only diverge as training accumulates per-modality EMA stats)
      3. Pop original ``..._outconv2.1.<suffix>`` (no longer in v8 state_dict)
    Net key delta: -5 + 10 = +5 per layer * 2 layers = +10 keys total.

    Triggering
    ----------
    Only runs when:
      (a) ``model.fine_preprocess.use_msbn == True`` AND
      (b) the ckpt does NOT already contain MSBN keys (so v8 ckpt -> v8
          model is a no-op without log noise).
    Otherwise silent no-op + no log line. This preserves byte-identical
    behaviour for every v0-v7 cfg (USE_MSBN defaults to False).

    Both ckpt key conventions are handled
    -------------------------------------
    Same as ``_maybe_inflate_stage0``: PL-saved ckpts carry the
    ``matcher.`` prefix; bare official ckpts do not. We detect both.
    """
    use_msbn = bool(getattr(model.fine_preprocess, 'use_msbn', False))
    if not use_msbn:
        return state_dict

    # If ckpt already has MSBN keys, this is a v8->v8 resume; don't re-inflate.
    has_msbn_keys = any(
        '_outconv2_bn_ir.' in k or '_outconv2_bn_vis.' in k
        for k in state_dict
    )
    if has_msbn_keys:
        return state_dict

    bn_suffixes = ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')
    bn_channels = {  # (block_dims default; only used for log message)
        'layer2_outconv2': model.fine_preprocess.layer2_outconv2_bn_ir.weight.shape[0],
        'layer1_outconv2': model.fine_preprocess.layer1_outconv2_bn_ir.weight.shape[0],
    }
    inflated_any = False
    for layer_name in ('layer2_outconv2', 'layer1_outconv2'):
        # Try both prefixes (PL-saved with `matcher.` and bare).
        for prefix in ('matcher.fine_preprocess.', 'fine_preprocess.'):
            base_orig = f'{prefix}{layer_name}.1.'
            # Quick check: is this prefix-orig key actually present?
            if not any(k.startswith(base_orig) for k in state_dict):
                continue
            base_ir  = f'{prefix}{layer_name}_bn_ir.'
            base_vis = f'{prefix}{layer_name}_bn_vis.'
            for suffix in bn_suffixes:
                k_orig = base_orig + suffix
                if k_orig not in state_dict:
                    continue
                tensor = state_dict[k_orig]
                state_dict[base_ir  + suffix] = tensor.clone()
                state_dict[base_vis + suffix] = tensor.clone()
                state_dict.pop(k_orig)
            inflated_any = True
            logger.info(
                f"MSBN inflated {layer_name}.1 ({bn_channels[layer_name]}ch) "
                f"-> bn_ir + bn_vis (zero-shift, 5 keys -> 10 keys)")
            break   # found the prefix that matched, don't try the other
    if not inflated_any:
        logger.warning(
            "MSBN inflate hook found no matching layer{1,2}_outconv2.1.* keys "
            "in the ckpt; v8 _bn_ir/_bn_vis stay at PyTorch defaults "
            "(gamma=1, beta=0). This means epoch 0 will NOT be byte-identical "
            "to the v7 ckpt -- check ckpt path / cfg.")
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
            # v8: bridge v0-v7 single-BN ckpt to v8 MSBN dual-BN model. No-op
            # when use_msbn=False (every v0-v7 cfg) or when ckpt already has
            # MSBN keys (v8 -> v8 resume). See _maybe_inflate_msbn for spec.
            state_dict = _maybe_inflate_msbn(state_dict, self.matcher)
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
        # full trainable graph). All flags default to False, so behaviour is
        # unchanged for v0/v1/v2 runs.
        #
        # v8 OR-semantics refactor: 5 freeze fields are INDEPENDENT and ADDITIVE.
        # For any param p: p is frozen iff ANY freeze field whose scope includes
        # p is set to True. Repeated freeze (e.g. FREEZE_BN=T also covers
        # backbone BN already covered by FREEZE_BACKBONE_BN=T) is IDEMPOTENT --
        # m.eval() and p.requires_grad=False are safe to call multiple times.
        # Behaviour is byte-identical to the v0-v7 if/elif scheme because no
        # v0-v7 cfg sets multiple BN-freeze fields simultaneously (each cfg
        # used at most one of FREEZE_BN / FREEZE_BACKBONE_BN, never both).
        # The v8-new fields FREEZE_FINE_BN_IR / FREEZE_FINE_BN_VIS only kick in
        # when USE_MSBN=True and fine_preprocess has _bn_ir/_bn_vis attrs;
        # otherwise the helpers silent no-op via getattr defense.
        self._freeze_bn          = bool(config.LOFTR.get('FREEZE_BN', False))
        self._freeze_backbone_bn = bool(config.LOFTR.get('FREEZE_BACKBONE_BN', False))
        # v8-new: FREEZE_FINE_BN_IR / VIS. Use cfg.LOFTR.get(..., False) so that
        # when default.py has not yet defined the fields (during incremental
        # implementation), the call falls back to False and behaviour stays
        # byte-identical to v0-v7.
        self._freeze_fine_bn_ir  = bool(config.LOFTR.get('FREEZE_FINE_BN_IR',  False))
        self._freeze_fine_bn_vis = bool(config.LOFTR.get('FREEZE_FINE_BN_VIS', False))
        # Debug-aid warning when MSBN-specific freeze flags are set without
        # USE_MSBN=True (the fine_preprocess will not have _bn_ir/_bn_vis
        # attrs so the helpers silent no-op anyway, but the user probably
        # made a cfg mistake).
        if (self._freeze_fine_bn_ir or self._freeze_fine_bn_vis) and \
                not bool(config.LOFTR.get('USE_MSBN', False)):
            logger.warning(
                "FREEZE_FINE_BN_IR/VIS set but USE_MSBN=False -> "
                "fine_preprocess has no _bn_ir/_bn_vis attrs; helpers will silent no-op")

        if config.LOFTR.get('FREEZE_BACKBONE', False):
            for p in self.matcher.backbone.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.matcher.backbone.parameters())
            logger.info(f"Froze backbone: {n_frozen/1e6:.2f}M params")

        # 5 BN-related freeze fields, OR semantics, idempotent. No priority
        # chain: each field independently freezes its own scope. Repeated
        # m.eval() / p.requires_grad=False are no-op when already applied.
        if self._freeze_bn:
            self._apply_freeze_bn()
            logger.info("Froze all BatchNorm2d layers (eval-mode + no-grad)")
        if self._freeze_backbone_bn:
            self._apply_freeze_backbone_bn()
            logger.info("Froze backbone BatchNorm2d layers (eval-mode + no-grad); fine_preprocess BN remains trainable")
        if self._freeze_fine_bn_ir:
            self._apply_freeze_fine_bn_ir()
            logger.info("Froze fine_preprocess _bn_ir layers (eval-mode + no-grad)")
        if self._freeze_fine_bn_vis:
            self._apply_freeze_fine_bn_vis()
            logger.info("Froze fine_preprocess _bn_vis layers (eval-mode + no-grad)")

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

    def _apply_freeze_fine_bn_ir(self):
        """v8 MSBN: freeze fine_preprocess.layer{1,2}_outconv2_bn_ir (the IR
        branch of the modality-specific BN pair). Uses getattr defense so that
        when USE_MSBN=False (the v0-v7 path where fine_preprocess has no
        _bn_ir/_bn_vis attrs) this helper silently no-ops -- no error, no log.
        Called once in __init__ AND from `train()` to survive PL's per-epoch
        model.train() that would otherwise re-enable BN train-mode.
        """
        fp = self.matcher.fine_preprocess
        for name in ('layer2_outconv2_bn_ir', 'layer1_outconv2_bn_ir'):
            m = getattr(fp, name, None)
            if m is None:
                continue                                            # USE_MSBN=False: silent no-op
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def _apply_freeze_fine_bn_vis(self):
        """v8 MSBN: same as _apply_freeze_fine_bn_ir but for the VIS branch."""
        fp = self.matcher.fine_preprocess
        for name in ('layer2_outconv2_bn_vis', 'layer1_outconv2_bn_vis'):
            m = getattr(fp, name, None)
            if m is None:
                continue                                            # USE_MSBN=False: silent no-op
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        """Override to keep frozen BN layers in eval-mode across PL's per-epoch
        model.train() calls. Without this, freezing BN in __init__ would only
        last until the first epoch boundary.

        v8 OR-semantics refactor: 5 freeze fields are applied independently
        (each is an `if`, not `if/elif`). Repeated m.eval() / requires_grad=
        False are idempotent so no harm if multiple flags overlap. The v0-v7
        if/elif scheme is byte-identical to this new structure for any v0-v7
        cfg (which only sets at most one BN-freeze field at a time).
        """
        super().train(mode)
        if getattr(self, '_freeze_bn', False):
            self._apply_freeze_bn()
        if getattr(self, '_freeze_backbone_bn', False):
            self._apply_freeze_backbone_bn()
        if getattr(self, '_freeze_fine_bn_ir', False):
            self._apply_freeze_fine_bn_ir()
        if getattr(self, '_freeze_fine_bn_vis', False):
            self._apply_freeze_fine_bn_vis()
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

            # v8 MSBN diagnostics: track how far the IR/VIS BN branches have
            # diverged from the zero-shift starting point. bn_drift_ratio is
            # the key gate-16 metric: should grow from ~0 at ep0 to >0.05 by
            # ep5 if MSBN is actually doing something. If it stays near 0,
            # the IR/VIS distributions are too similar (likely PC+CLAHE has
            # already aligned them in earlier layers) and MSBN adds nothing.
            # gate(getattr) so v0-v7 (USE_MSBN=False) writes zero new scalars
            # -> TB log byte-identical to before this block landed.
            if getattr(self.matcher.fine_preprocess, 'use_msbn', False):
                fp = self.matcher.fine_preprocess
                eps = 1e-8
                for layer_name in ('layer2', 'layer1'):
                    bn_ir  = getattr(fp, f'{layer_name}_outconv2_bn_ir')
                    bn_vis = getattr(fp, f'{layer_name}_outconv2_bn_vis')
                    m_ir  = bn_ir.running_mean.detach()
                    m_vis = bn_vis.running_mean.detach()
                    # Drift ratio: |mean_ir - mean_vis| / (|mean_ir| + |mean_vis| + eps)
                    drift = ((m_ir - m_vis).norm() /
                             (m_ir.norm() + m_vis.norm() + eps)).item()
                    self.logger.experiment.add_scalar(
                        f'train/bn_drift_ratio_{layer_name}', drift, self.global_step)
                    # Gamma L2 norms: visualise per-branch affine learning
                    self.logger.experiment.add_scalar(
                        f'train/bn_ir_gamma_l2_{layer_name}',
                        bn_ir.weight.detach().norm().item(), self.global_step)
                    self.logger.experiment.add_scalar(
                        f'train/bn_vis_gamma_l2_{layer_name}',
                        bn_vis.weight.detach().norm().item(), self.global_step)

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