from math import log
from loguru import logger as loguru_logger

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.utils import create_meshgrid
from src.utils.plotting import make_matching_figures
from src.utils.data_source import is_aligned_irvis

from .geometry import warp_kpts

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # v18 H aug coordinate-system bridge: H_vis is sampled in resized pixel
    # space (because megadepth_syn_pose._read_resize_warp_pad applies it to
    # the resized canvas), but warp_kpts works in original pixel space (K +
    # depth + pose are all defined at original resolution). Both forward and
    # backward warp paths need explicit 4-step scale conversions to bridge
    # the two coordinate systems. v0-v17 path (no 'H_vis' key in data) skips
    # both branches -> byte-identical to the un-augmented contract.
    if 'H_vis' in data:
        # cast to fp32 for numerical stability under autocast/fp16; warp_kpts
        # is fp32-internal so this matches its accuracy budget.
        H_vis_fp32 = data['H_vis'].to(torch.float32)
        H_vis_inv_fp32 = torch.linalg.inv(H_vis_fp32)
        inv_s1 = (1.0 / data['scale1'])[:, None, :].to(torch.float32)  # [N, 1, 2]
        s1 = data['scale1'][:, None, :].to(torch.float32)              # [N, 1, 2]
    else:
        H_vis_fp32 = H_vis_inv_fp32 = inv_s1 = s1 = None

    # v18 (C) backward inv(H_vis) pull: image1 coarse cell idx grid lives in
    # AUG image1 pixel coordinates (because dataset already warped image1).
    # warp_kpts(grid_pt1, depth1, ...) needs ORIGINAL image1 pixel idx so we
    # invert the homography first: aug_pixel -> aug_resized -> orig_resized
    # -> orig_pixel.
    if 'H_vis' in data:
        grid_pt1_resized_aug = grid_pt1_i.to(torch.float32) * inv_s1     # aug 'orig-equiv' -> aug resized
        grid_pt1_resized_orig, _ = _warp_pts_homography(grid_pt1_resized_aug, H_vis_inv_fp32)
        grid_pt1_for_warp = grid_pt1_resized_orig * s1                   # orig resized -> orig pixel
        # Bug #8 (CRASH FIX): warp_kpts internally does
        #   depth[..., y.long(), x.long()]
        # without a bounds check; rotated cells can land outside [0, P) even
        # after intersecting with mask1. Clamp to depth1.shape[-2:] (= 2000
        # for Megadepth pad_to=2000, NOT image1.shape).
        Hd, Wd = data['depth1'].shape[-2:]
        grid_pt1_for_warp[..., 0].clamp_(0, Wd - 1)
        grid_pt1_for_warp[..., 1].clamp_(0, Hd - 1)
        grid_pt1_for_warp = grid_pt1_for_warp.to(grid_pt1_i.dtype)
    else:
        grid_pt1_for_warp = grid_pt1_i

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1_i = warp_kpts(grid_pt1_for_warp, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])

    # v18 (B) forward H_vis push: w_pt0_i comes out of warp_kpts in ORIGINAL
    # image1 pixel; push it to AUG image1 'orig-equiv' pixel via
    # scale1 . H_vis . (1/scale1) so the downstream
    #   w_pt0_c = w_pt0_i / scale1 = aug coarse cell idx
    # naturally lands in aug image1 cell space (consistent with grid_pt1_c).
    if 'H_vis' in data:
        w_pt0_resized = w_pt0_i.to(torch.float32) * inv_s1               # orig pixel -> resized
        w_pt0_resized_aug, _ = _warp_pts_homography(w_pt0_resized, H_vis_fp32)
        w_pt0_i = (w_pt0_resized_aug * s1).to(w_pt0_i.dtype)             # aug resized -> aug 'orig-equiv'

    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round()
    # calculate the overlap area between warped patch and grid patch as the loss weight.
    # (larger overlap area between warped patches and grid patch with higher weight)
    # (overlap area range from [0, 1] rather than [0.25, 1] as the penalty of warped kpts fall on midpoint of two grid kpts)
    if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
        w_pt0_c_error = (1.0 - 2*torch.abs(w_pt0_c - w_pt0_c_round)).prod(-1)
    w_pt0_c_round = w_pt0_c_round[:, :, :].long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1

    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    # v18 (B-bis) forward mask1 cell post-filter: out_bound_mask only catches
    # nearest cells outside [0, w1) x [0, h1); cells INSIDE the canvas but
    # in aug-padded / vis-invalid region also need suppressing (otherwise we
    # would build a phantom GT match into a black pixel).
    # v18 (C-bis) backward mask1 cell post-filter: nearest_index0[b, j] was
    # computed from cell j in aug image1; if cell j is in aug padded region
    # the recovered image0 cell is unreliable -> suppress it so the
    # mutual-NN loop_back check at L99-101 cannot accept it as GT.
    if 'H_vis' in data and 'mask1' in data:
        mask1_flat = data['mask1'].flatten(-2)                                          # [N, h1*w1] bool
        # gather along dim=1 with idx in [0, h1*w1); always safe because
        # out_bound_mask above already set out-of-canvas idx to 0 and
        # mask1_flat[b, 0] is True (top-left corner is never padded).
        target_real = torch.gather(mask1_flat, 1, nearest_index1)                       # [N, h0*w0] bool
        nearest_index1 = torch.where(target_real, nearest_index1, torch.zeros_like(nearest_index1))
        nearest_index0 = torch.where(mask1_flat, nearest_index0, torch.zeros_like(nearest_index0))

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # use overlap area as loss weight
    if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
        conf_matrix_error_gt = w_pt0_c_error[b_ids, i_ids]  # weight range: [0.0, 1.0]
        data.update({'conf_matrix_error_gt': conf_matrix_error_gt})


    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        loguru_logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


@torch.no_grad()
def _warp_pts_homography(pts, H):
    """Apply per-batch 3x3 Homography ``H`` to 2D points.

    Args:
        pts: [N, K, 2] (x, y) in image0 pixel coordinates.
        H:   [N, 3, 3].
    Returns:
        warped: [N, K, 2] (x, y) in image1 pixel coordinates.
        valid:  [N, K] bool mask, True if the homogeneous w is finite and > 0.
    """
    N, K, _ = pts.shape
    ones = torch.ones(N, K, 1, dtype=pts.dtype, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=-1)            # [N, K, 3]
    warped_h = torch.einsum('nij,nkj->nki', H, pts_h)  # [N, K, 3]
    w = warped_h[..., 2:3]
    valid = (w.abs() > 1e-8).squeeze(-1)
    w = torch.where(w.abs() > 1e-8, w, torch.ones_like(w))
    warped = warped_h[..., :2] / w
    return warped, valid


@torch.no_grad()
def spvs_coarse_roadscene(data, config):
    """Homography-aware coarse supervision for RoadScene.

    Builds the same set of fields as :func:`spvs_coarse` but uses a per-pair
    3x3 Homography ``data['homography_0to1']`` (identity when augmentation is
    disabled) to project image0 coarse-grid centres into image1.
    """
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    H_0to1 = data['homography_0to1'].to(grid_pt0_i.dtype)
    if H_0to1.dim() == 2:
        H_0to1 = H_0to1[None].expand(N, -1, -1)

    w_pt0_i, valid_h = _warp_pts_homography(grid_pt0_i, H_0to1)
    w_pt0_c = w_pt0_i / scale1

    w_pt0_c_round = w_pt0_c.round()
    if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
        w_pt0_c_error = (1.0 - 2 * torch.abs(w_pt0_c - w_pt0_c_round)).prod(-1)
    w_pt0_c_round = w_pt0_c_round.long()

    out_of_bound = (
        (w_pt0_c_round[..., 0] < 0) | (w_pt0_c_round[..., 0] >= w1)
        | (w_pt0_c_round[..., 1] < 0) | (w_pt0_c_round[..., 1] >= h1)
    )
    valid = valid_h & (~out_of_bound)

    # Compute the index used to look up coarse cells in image1, clamping
    # out-of-bound positions to a safe value so the gather below cannot crash.
    # Cells that are still flagged invalid by ``valid`` are zeroed out
    # immediately after the gather, so the clamped lookups never affect the
    # produced supervision.
    nearest_index1 = (
        w_pt0_c_round[..., 0].clamp(0, w1 - 1)
        + w_pt0_c_round[..., 1].clamp(0, h1 - 1) * w1
    )

    # When the dataset provides padding masks (RoadScene A3 pipeline),
    # additionally require that:
    #   - the source IR coarse cell is real (mask0)
    #   - the projected VIS coarse cell is real (mask1 looked up at nearest_index1)
    # so we never supervise against zero-padded regions.
    if 'mask0' in data and 'mask1' in data:
        mask0_flat = data['mask0'].reshape(N, -1)              # [N, h0*w0]
        mask1_flat = data['mask1'].reshape(N, -1)              # [N, h1*w1]
        valid = valid & mask0_flat
        target_valid = mask1_flat.gather(1, nearest_index1)
        valid = valid & target_valid

    nearest_index1[~valid] = 0

    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(valid)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    if config.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT:
        conf_matrix_error_gt = w_pt0_c_error[b_ids, i_ids]
        data.update({'conf_matrix_error_gt': conf_matrix_error_gt})

    if len(b_ids) == 0:
        loguru_logger.warning(
            f"No groundtruth coarse match found for: {data['pair_names']}")
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids,
    })
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i,
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'megadepth_syn_pose']:
        # v13 (Megadepth_Syn_Pose): cross-view cross-modal pose supervision
        # shares the same K + W2C pose + depth GT contract as MegaDepth, so it
        # routes to spvs_coarse, NOT spvs_coarse_roadscene.
        spvs_coarse(data, config)
    elif is_aligned_irvis(data_source):
        spvs_coarse_roadscene(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@static_vars(counter = 0)
@torch.no_grad()
def spvs_fine(data, config, logger = None):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2], used as subpixel-level gt
            "conf_matrix_f_gt": [M, WW, WW], M is the number of all coarse-level gt matches
            "conf_matrix_f_error_gt": [Mp], Mp is the number of all pixel-level gt matches
            "m_ids_f": [Mp]
            "i_ids_f": [Mp]
            "j_ids_f_di": [Mp]
            "j_ids_f_dj": [Mp]
            }
    """
    # 1. misc
    pt1_i = data['spv_pt1_i']
    W = config['LOFTR']['FINE_WINDOW_SIZE']
    WW = W*W
    scale = config['LOFTR']['RESOLUTION'][1]
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    hf0, wf0, hf1, wf1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]  # h, w of fine feature
    assert not config.LOFTR.ALIGN_CORNER, 'only support training with align_corner=False for now.'

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
    scalei0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
    scalei1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale

    # 3. compute gt
    m = b_ids.shape[0]
    if m == 0:  # special case: there is no coarse gt
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device)

        data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            conf_matrix_f_error_gt = torch.zeros(1, device=device)
            data.update({'conf_matrix_f_error_gt': conf_matrix_f_error_gt})
        
        data.update({'expec_f': torch.zeros(1, 2, device=device)})
        data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
    else:
        # v18 (F-1) grid_pt0_f anchor convention selector.
        # Default (False) = v0 paper "-3.5 trick": cell-centre anchor, grid
        # value range [-W/2+0.5, W/2-0.5] = [-3.5, +3.5] for W=8. The trick
        # is paired with `+ W // 2 - 0.5` at L339 (now F-3) to yield a
        # delta_w_pt0_f range [0, W). v0-v17 go through this branch and stay
        # byte-identical.
        # True = v18 absolute-idx convention: cell-top-left anchor, grid value
        # range [0, W-1] = [0, 7] for W=8. The +3.5 cancellation is removed
        # at F-3 below. Required for v18 because H_vis rotation breaks the
        # trick's implicit local-linearity assumption (~4 fine-pixel residual
        # at rot=50 deg). Both spvs_fine and fine_matching.get_fine_ds_match
        # must read the same flag so train/test stay consistent.
        absolute_fine_idx = config.LOFTR.MATCH_FINE.ABSOLUTE_FINE_IDX
        if absolute_fine_idx:
            grid_pt0_f = create_meshgrid(hf0, wf0, False, device)            # [1, hf0, wf0, 2]; absolute fine idx
        else:
            grid_pt0_f = create_meshgrid(hf0, wf0, False, device) - W // 2 + 0.5 # [1, hf0, wf0, 2] # use fine coordinates
        grid_pt0_f = rearrange(grid_pt0_f, 'n h w c -> n c h w')
        # 1. unfold(crop) all local windows
        if config.LOFTR.ALIGN_CORNER is False: # even windows
            assert W==8
            grid_pt0_f_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W, padding=0)
        grid_pt0_f_unfold = rearrange(grid_pt0_f_unfold, 'n (c ww) l -> n l ww c', ww=W**2) # [1, hc0*wc0, W*W, 2]
        grid_pt0_f_unfold = repeat(grid_pt0_f_unfold[0], 'l ww c -> N l ww c', N=N)

        # 2. select only the predicted matches
        grid_pt0_f_unfold = grid_pt0_f_unfold[data['b_ids'], data['i_ids']]  # [m, ww, 2]
        grid_pt0_f_unfold = scalei0[:,None,:] * grid_pt0_f_unfold  # [m, ww, 2]
        
        # 3. warp grids and get covisible & depth_consistent mask
        correct_0to1_f = torch.zeros(m, WW, device=device, dtype=torch.bool)
        w_pt0_i = torch.zeros(m, WW, 2, device=device, dtype=torch.float32)
        for b in range(N):
            mask = b_ids == b  # mask of each batch
            match = int(mask.sum())
            correct_0to1_f_mask, w_pt0_i_mask = warp_kpts(grid_pt0_f_unfold[mask].reshape(1,-1,2), data['depth0'][[b],...],
                    data['depth1'][[b],...], data['T_0to1'][[b],...], 
                    data['K0'][[b],...], data['K1'][[b],...]) # [k, WW], [k, WW, 2]
            correct_0to1_f[mask] = correct_0to1_f_mask.reshape(match, WW)
            w_pt0_i[mask] = w_pt0_i_mask.reshape(match, WW, 2)

        # v18 (F-2) per-match H_vis push: w_pt0_i comes out of warp_kpts in
        # ORIGINAL image1 pixel; push it to AUG image1 'orig-equiv' pixel
        # via scale1 . H_vis . (1/scale1), per-match because b_ids selects
        # a different sample for each row of w_pt0_i.
        if 'H_vis' in data:
            H_vis_pm = data['H_vis'][b_ids].to(torch.float32)               # [m, 3, 3]
            inv_s1_pm = (1.0 / data['scale1'])[b_ids][:, None, :].to(torch.float32)  # [m, 1, 2]
            s1_pm = data['scale1'][b_ids][:, None, :].to(torch.float32)              # [m, 1, 2]
            w_pt0_resized = w_pt0_i.to(torch.float32) * inv_s1_pm                    # [m, WW, 2] orig pixel -> resized
            w_pt0_resized_aug, _ = _warp_pts_homography(w_pt0_resized, H_vis_pm)     # resized -> aug resized
            w_pt0_i = (w_pt0_resized_aug * s1_pm).to(w_pt0_i.dtype)                  # aug resized -> aug 'orig-equiv'

        # 4. calculate the gt index of pixel-level refinement
        delta_w_pt0_i = w_pt0_i - pt1_i[b_ids, j_ids][:,None,:] # [m, WW, 2]
        del b_ids, i_ids, j_ids
        # v18 (F-3) delta convention: in v0 (-3.5 trick) path the +W//2-0.5
        # cancels the -W//2+0.5 baked into grid_pt0_f at F-1, yielding
        # delta_w_pt0_f range [0, W). In v18 absolute-idx path no offset is
        # baked in, so we drop the cancellation and use delta_w_pt0_i /
        # scalei1 directly (same [0, W) range, same downstream semantics for
        # round / nearest_index1 / expec_f_gt). conf_matrix_f_gt and
        # expec_f_gt are anchor-invariant cell-local fields, so LoFTRLoss
        # numerical behaviour is unchanged across both branches.
        if absolute_fine_idx:
            delta_w_pt0_f = delta_w_pt0_i / scalei1[:,None,:]
        else:
            delta_w_pt0_f = delta_w_pt0_i / scalei1[:,None,:] + W // 2 - 0.5
        delta_w_pt0_f_round = delta_w_pt0_f[:, :, :].round()
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            # calculate the overlap area between warped patch and grid patch as the loss weight.
            w_pt0_f_error = (1.0 - 2*torch.abs(delta_w_pt0_f - delta_w_pt0_f_round)).prod(-1) # [0, 1]     
        delta_w_pt0_f_round = delta_w_pt0_f_round.long()
    
        nearest_index1 = delta_w_pt0_f_round[..., 0] + delta_w_pt0_f_round[..., 1] * W # [m, WW]
        
        # corner case: out of fine windows
        def out_bound_mask(pt, w, h):
            return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        ob_mask = out_bound_mask(delta_w_pt0_f_round, W, W)
        nearest_index1[ob_mask] = 0
        correct_0to1_f[ob_mask] = 0

        m_ids, i_ids = torch.where(correct_0to1_f != 0)
        j_ids = nearest_index1[m_ids, i_ids]  # i_ids, j_ids range from [0, WW-1]
        j_ids_di, j_ids_dj = j_ids // W, j_ids % W  # further get the (i, j) index in fine windows of image1 (right image); j_ids_di, j_ids_dj range from [0, W-1]
        m_ids, i_ids, j_ids_di, j_ids_dj = m_ids.to(torch.long), i_ids.to(torch.long), j_ids_di.to(torch.long), j_ids_dj.to(torch.long)

        # expec_f_gt will be used as the gt of subpixel-level refinement
        expec_f_gt = delta_w_pt0_f - delta_w_pt0_f_round
        
        if m_ids.numel() == 0:  # special case: there is no pixel-level gt
            loguru_logger.warning(f"No groundtruth fine match found for local regress: {data['pair_names']}")
            # this won't affect fine-level loss calculation
            data.update({'expec_f': torch.zeros(1, 2, device=device)})
            data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
        else:
            expec_f_gt = expec_f_gt[m_ids, i_ids]
            data.update({"expec_f_gt": expec_f_gt})
            data.update({"m_ids_f": m_ids,
                            "i_ids_f": i_ids,
                            "j_ids_f_di": j_ids_di,
                            "j_ids_f_dj": j_ids_dj
                            })

        # 5. construct a pixel-level gt conf_matrix
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device, dtype=torch.bool)
        conf_matrix_f_gt[m_ids, i_ids, j_ids] = 1
        data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            # calculate the overlap area between warped pixel and grid pixel as the loss weight.
            w_pt0_f_error = w_pt0_f_error[m_ids, i_ids]
            data.update({'conf_matrix_f_error_gt': w_pt0_f_error})
            
        if  conf_matrix_f_gt.sum() == 0:
            loguru_logger.info(f'no fine matches to supervise')
                
@torch.no_grad()
def spvs_fine_roadscene(data, config, logger=None):
    """Homography-aware fine-level supervision for RoadScene.

    Mirrors :func:`spvs_fine` but replaces depth/pose-based warping with the
    per-pair Homography in ``data['homography_0to1']``.
    """
    pt1_i = data['spv_pt1_i']
    W = config['LOFTR']['FINE_WINDOW_SIZE']
    WW = W * W
    scale = config['LOFTR']['RESOLUTION'][1]
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    hf0, wf0, hf1, wf1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]
    assert not config.LOFTR.ALIGN_CORNER, 'only support training with align_corner=False for now.'

    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
    scalei0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
    scalei1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale

    m = b_ids.shape[0]
    if m == 0:
        conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device)
        data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
        if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
            data.update({'conf_matrix_f_error_gt': torch.zeros(1, device=device)})
        data.update({'expec_f': torch.zeros(1, 2, device=device)})
        data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
        return

    grid_pt0_f = create_meshgrid(hf0, wf0, False, device) - W // 2 + 0.5  # [1, hf0, wf0, 2]
    grid_pt0_f = rearrange(grid_pt0_f, 'n h w c -> n c h w')
    if config.LOFTR.ALIGN_CORNER is False:
        assert W == 8
        grid_pt0_f_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W, padding=0)
    grid_pt0_f_unfold = rearrange(grid_pt0_f_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
    grid_pt0_f_unfold = repeat(grid_pt0_f_unfold[0], 'l ww c -> N l ww c', N=N)

    grid_pt0_f_unfold = grid_pt0_f_unfold[data['b_ids'], data['i_ids']]   # [m, ww, 2]
    grid_pt0_f_unfold = scalei0[:, None, :] * grid_pt0_f_unfold

    H_0to1 = data['homography_0to1'].to(grid_pt0_f_unfold.dtype)
    if H_0to1.dim() == 2:
        H_0to1 = H_0to1[None].expand(N, -1, -1)

    correct_0to1_f = torch.zeros(m, WW, device=device, dtype=torch.bool)
    w_pt0_i = torch.zeros(m, WW, 2, device=device, dtype=torch.float32)
    for b in range(N):
        mask = b_ids == b
        match = int(mask.sum())
        if match == 0:
            continue
        pts = grid_pt0_f_unfold[mask].reshape(1, -1, 2)
        warped, valid = _warp_pts_homography(pts, H_0to1[[b], ...])
        # Warped point must also fall inside the image1 area to be a usable supervision.
        in_bound = (
            (warped[..., 0] >= 0) & (warped[..., 0] < W1)
            & (warped[..., 1] >= 0) & (warped[..., 1] < H1)
        )
        valid = valid & in_bound
        correct_0to1_f[mask] = valid.reshape(match, WW)
        w_pt0_i[mask] = warped.reshape(match, WW, 2)

    delta_w_pt0_i = w_pt0_i - pt1_i[b_ids, j_ids][:, None, :]
    del b_ids, i_ids, j_ids
    delta_w_pt0_f = delta_w_pt0_i / scalei1[:, None, :] + W // 2 - 0.5
    delta_w_pt0_f_round = delta_w_pt0_f.round()
    if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
        w_pt0_f_error = (1.0 - 2 * torch.abs(delta_w_pt0_f - delta_w_pt0_f_round)).prod(-1)
    delta_w_pt0_f_round = delta_w_pt0_f_round.long()

    nearest_index1 = delta_w_pt0_f_round[..., 0] + delta_w_pt0_f_round[..., 1] * W

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    ob_mask = out_bound_mask(delta_w_pt0_f_round, W, W)
    nearest_index1[ob_mask] = 0
    correct_0to1_f[ob_mask] = 0

    m_ids, i_ids = torch.where(correct_0to1_f != 0)
    j_ids = nearest_index1[m_ids, i_ids]
    j_ids_di, j_ids_dj = j_ids // W, j_ids % W
    m_ids = m_ids.to(torch.long)
    i_ids = i_ids.to(torch.long)
    j_ids_di = j_ids_di.to(torch.long)
    j_ids_dj = j_ids_dj.to(torch.long)

    expec_f_gt = delta_w_pt0_f - delta_w_pt0_f_round.float()
    if m_ids.numel() == 0:
        loguru_logger.warning(
            f"No groundtruth fine match found for local regress: {data['pair_names']}")
        data.update({'expec_f': torch.zeros(1, 2, device=device)})
        data.update({'expec_f_gt': torch.zeros(1, 2, device=device)})
    else:
        expec_f_gt = expec_f_gt[m_ids, i_ids]
        data.update({'expec_f_gt': expec_f_gt})
        data.update({
            'm_ids_f': m_ids,
            'i_ids_f': i_ids,
            'j_ids_f_di': j_ids_di,
            'j_ids_f_dj': j_ids_dj,
        })

    conf_matrix_f_gt = torch.zeros(m, WW, WW, device=device, dtype=torch.bool)
    conf_matrix_f_gt[m_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_f_gt': conf_matrix_f_gt})
    if config.LOFTR.LOSS.FINE_OVERLAP_WEIGHT:
        w_pt0_f_error = w_pt0_f_error[m_ids, i_ids]
        data.update({'conf_matrix_f_error_gt': w_pt0_f_error})

    if conf_matrix_f_gt.sum() == 0:
        loguru_logger.info('no fine matches to supervise')


def compute_supervision_fine(data, config, logger=None):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'megadepth_syn_pose']:
        # v13: see comment in compute_supervision_coarse above.
        spvs_fine(data, config, logger)
    elif is_aligned_irvis(data_source):
        spvs_fine_roadscene(data, config, logger)
    else:
        raise NotImplementedError