import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from loguru import logger

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        block_dims = config['backbone']['block_dims']
        self.W = self.config['fine_window_size']
        self.fine_d_model = block_dims[0]
        # v8: Modality-Specific BatchNorm (MSBN) toggle. False keeps v0-v7
        # state_dict key set + forward math byte-identical. See default.py
        # comment block on USE_MSBN for full design rationale.
        self.use_msbn = config.get('use_msbn', False)

        # 3 x 1x1 outconv: channel projection only, no BN, identical in both
        # USE_MSBN paths -> defined OUTSIDE the if/else to share the v7 weight
        # keys (`layer{3,2,1}_outconv.weight`).
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])

        if self.use_msbn:
            # v8 path (Identity placeholder): keep the Sequential outer shell
            # so conv keys (.0/.3) stay byte-identical to v7. The BN slot (.1)
            # is replaced by nn.Identity which registers no parameter or
            # buffer -- so `layer*_outconv2.1.*` keys disappear from state_dict.
            # The MSBN branches are added as separate named attrs whose keys
            # do NOT collide with any v0-v7 ckpt key.
            self.layer2_outconv2 = nn.Sequential(
                conv3x3(block_dims[2], block_dims[2]),  # .0  (= v7 weight)
                nn.Identity(),                           # .1  (replaces BN, no param)
                nn.LeakyReLU(),                          # .2  (= v7 activation)
                conv3x3(block_dims[2], block_dims[1]),   # .3  (= v7 weight)
            )
            self.layer1_outconv2 = nn.Sequential(
                conv3x3(block_dims[1], block_dims[1]),
                nn.Identity(),
                nn.LeakyReLU(),
                conv3x3(block_dims[1], block_dims[0]),
            )
            # New MSBN branches: BN(256) for layer2 and BN(128) for layer1.
            # Initialised by _reset_parameters() to PyTorch BN defaults
            # (gamma=1, beta=0, running_mean=0, running_var=1) but those will
            # be overwritten by _maybe_inflate_msbn() in lightning_loftr.py
            # before load_state_dict, so the v8 model at step 0 is byte-
            # identical to the v7 ckpt it resumed from.
            self.layer2_outconv2_bn_ir  = nn.BatchNorm2d(block_dims[2])  # BN(256)
            self.layer2_outconv2_bn_vis = nn.BatchNorm2d(block_dims[2])  # BN(256)
            self.layer1_outconv2_bn_ir  = nn.BatchNorm2d(block_dims[1])  # BN(128)
            self.layer1_outconv2_bn_vis = nn.BatchNorm2d(block_dims[1])  # BN(128)
        else:
            # v0-v7 path: original Sequential with BN at index .1, byte-
            # identical state_dict key set to v7.
            self.layer2_outconv2 = nn.Sequential(
                conv3x3(block_dims[2], block_dims[2]),
                nn.BatchNorm2d(block_dims[2]),
                nn.LeakyReLU(),
                conv3x3(block_dims[2], block_dims[1]),
            )
            self.layer1_outconv2 = nn.Sequential(
                conv3x3(block_dims[1], block_dims[1]),
                nn.BatchNorm2d(block_dims[1]),
                nn.LeakyReLU(),
                conv3x3(block_dims[1], block_dims[0]),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1, stride):
        """v0-v7 forward path (single BN, IR+VIS cat), kept byte-identical."""
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2+feat_c)
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1+x2)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        return x1

    def inter_fpn_modal(self, feat_c, x2, x1, stride, modality):
        """v8 MSBN forward path: manually unroll the outconv2 Sequential and
        route through `_bn_ir` or `_bn_vis` based on modality. Sequential
        index .1 (Identity) is intentionally skipped -- the actual BN happens
        via the MSBN branch attribute.

        modality: str, 'ir' or 'vis'.
        """
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)

        # layer2_outconv2: conv3x3 (.0) -> MSBN -> LeakyReLU (.2) -> conv3x3 (.3)
        x2 = self.layer2_outconv(x2)
        h = self.layer2_outconv2[0](x2 + feat_c)
        bn2 = self.layer2_outconv2_bn_ir if modality == 'ir' else self.layer2_outconv2_bn_vis
        h = bn2(h)                                           # MSBN, replaces Sequential[1] Identity
        h = self.layer2_outconv2[2](h)                       # LeakyReLU
        x2 = self.layer2_outconv2[3](h)                      # conv3x3
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        # layer1_outconv2: same pattern
        x1 = self.layer1_outconv(x1)
        h = self.layer1_outconv2[0](x1 + x2)
        bn1 = self.layer1_outconv2_bn_ir if modality == 'ir' else self.layer1_outconv2_bn_vis
        h = bn1(h)
        h = self.layer1_outconv2[2](h)
        x1 = self.layer1_outconv2[3](h)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        return x1
    
    def forward(self, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            feat1 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            return feat0, feat1

        if data['hw0_i'] == data['hw1_i']:
            if self.use_msbn:
                # v8 path: split IR/VIS so each goes through its own MSBN branch.
                # Costs one extra inter_fpn call (no longer the cat optimisation
                # used in v0-v7) but enables modality-specific BN statistics.
                feat_c0_4d = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data['hw0_c'][0])
                feat_c1_4d = rearrange(feat_c1, 'b (h w) c -> b c h w', h=data['hw1_c'][0])
                # data['feats_x{2,1}'] from backbone is the cat([IR, VIS], dim=0)
                # tensor (see loftr.py:81). Split it back into per-modality halves.
                bs_half = data['bs']
                x2_0, x2_1 = torch.split(data['feats_x2'], bs_half, dim=0)
                x1_0, x1_1 = torch.split(data['feats_x1'], bs_half, dim=0)
                del data['feats_x2'], data['feats_x1']

                # 1. fine feature extraction (separate IR/VIS paths)
                feat_f0 = self.inter_fpn_modal(feat_c0_4d, x2_0, x1_0, stride, 'ir')
                feat_f1 = self.inter_fpn_modal(feat_c1_4d, x2_1, x1_1, stride, 'vis')
            else:
                # v0-v7 path: cat IR+VIS through the single shared BN
                # (preserves "faster & better BN convergence" optimisation).
                feat_c = rearrange(torch.cat([feat_c0, feat_c1], 0), 'b (h w) c -> b c h w', h=data['hw0_c'][0]) # 1/8 feat
                x2 = data['feats_x2'] # 1/4 feat
                x1 = data['feats_x1'] # 1/2 feat
                del data['feats_x2'], data['feats_x1']

                # 1. fine feature extraction
                x1 = self.inter_fpn(feat_c, x2, x1, stride)
                feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

            # 2. unfold(crop) all local windows
            feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

            # 3. select only the predicted matches
            feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
            feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

            return feat_f0, feat_f1
        else:  # handle different input shapes
            feat_c0, feat_c1 = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data['hw0_c'][0]), rearrange(feat_c1, 'b (h w) c -> b c h w', h=data['hw1_c'][0]) # 1/8 feat
            x2_0, x2_1 = data['feats_x2_0'], data['feats_x2_1'] # 1/4 feat
            x1_0, x1_1 = data['feats_x1_0'], data['feats_x1_1'] # 1/2 feat
            del data['feats_x2_0'], data['feats_x1_0'], data['feats_x2_1'], data['feats_x1_1']

            # 1. fine feature extraction
            if self.use_msbn:
                # v8 path: route per-modality
                feat_f0 = self.inter_fpn_modal(feat_c0, x2_0, x1_0, stride, 'ir')
                feat_f1 = self.inter_fpn_modal(feat_c1, x2_1, x1_1, stride, 'vis')
            else:
                feat_f0, feat_f1 = self.inter_fpn(feat_c0, x2_0, x1_0, stride), self.inter_fpn(feat_c1, x2_1, x1_1, stride)

            # 2. unfold(crop) all local windows
            feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

            # 3. select only the predicted matches
            feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
            feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

            return feat_f0, feat_f1