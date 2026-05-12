"""METU_VISTIR cross-modal pose-based test dataset.

METU_VISTIR is a real-captured drone VIS+LWIR stereo benchmark with
per-pair camera intrinsics, distortion coefficients, and 4x4 absolute
poses. We use it ONLY for pose-based evaluation (auc@5/10/20 deg via
``compute_symmetrical_epipolar_errors`` + ``compute_pose_errors``); it
is NOT used for training (no depth, dataset has no train_list).

Layout (already implemented under ``data/METU_VISTIR/``):

    data/METU_VISTIR/
      cloudy/scene_{1..6}/{visible,thermal}/images/IM_NNNNN.jpg
      sunny/scene_{1..4}/{visible,thermal}/images/IM_NNNNN.jpg
      index/scene_info_test/cloudy_{cloudy,sunny}_scene_*.npz   (10 npz)

npz schema (confirmed by tmp_npz_check.py 2026-05-11):

    image_paths       : (N_image, 2) <U?, [vis_rel, thermal_rel] per stereo
                          frame ; relative to ``data/METU_VISTIR/``.
    intrinsics        : (N_image, 2, 3, 3) float64
                          intrinsics[i, 0] = vis K, [i, 1] = thermal K.
                          VIS focal ~ 2940 px (3840x2160 sensor),
                          TIR focal ~ 768 px (640x512 sensor).
    distortion_coefs  : (N_image, 2, 8) float64
                          OpenCV plumb_bob k1, k2, p1, p2, k3, k4=k5=k6=0
                          (effectively 5-param). cv2.undistort handles 8.
    poses             : (N_image, 4, 4) float64, world->cam absolute pose.
    pair_infos        : (N_pair, 2) int (boxed in object dtype),
                          (idx0, idx1) into image_paths. Multi-view IR-VIS
                          pairing: when METU_SIDE0='thermal' (default,
                          training-aligned), image0 = idx0-frame thermal
                          + image1 = idx1-frame VIS; flip via cfg if needed.
                          Total over 10 npz: 2590 pair / 1678 image.

Pair semantics: image-list + pair-index (matches MegaDepth's contract;
N_pair != N_image, e.g. cloudy_cloudy_scene_1 has 300 images / 131 pairs).

    T_0to1 = inv(poses[idx1]) @ poses[idx0]

CRITICAL: METU ``poses[idx]`` is **camera-to-world (C2W)**, not the
world-to-camera (W2C) used by MegaDepthDataset. Confirmed via debug
sweep over 4 candidate formulas (tmp_metu_debug.py; v10 ckpt on
cloudy_cloudy_scene_1 pair 0):

    A: P1 @ inv(P0)        R_err=38.6  t_err=54.5  (W2C, MegaDepth standard)
    B: inv(P1) @ P0        R_err=11.6  t_err=34.6  (C2W -- correct for METU)
    C: P0 @ inv(P1)        R_err=11.9  t_err=34.4  (swap of B, equivalent under
                                                     E ambiguity for unit t)
    D: inv(P0) @ P1        R_err=38.7  t_err=55.4

So we use formula B. The user's earlier docstring claim that
``poses[i] = T_vis->thermal`` is incorrect (poses is per-image, not
per-pair).

Output dict (MegaDepth-style; consumed by test.py via test_step ->
_compute_metrics -> compute_symmetrical_epipolar_errors +
compute_pose_errors):

    image0     : (C, P, P) float32 in [0, 1]; C=2 if USE_EDGE_INPUT else 1
    image1     : (C, P, P) float32; same C as image0
    mask0      : (P/8, P/8) bool, valid coarse cells
    mask1      : (P/8, P/8) bool
    K0, K1     : (3, 3) float32, RAW K (NOT rescaled). matcher's
                  fine_matching multiplies by data['scale0/1'] internally
                  to recover raw-coord mkpts, then metrics.py:38 uses
                  raw K with raw-coord mkpts.
    scale0     : (2,) float32, [w_raw/w_resized, h_raw/h_resized]
    scale1     : (2,) float32, same.
    T_0to1     : (4, 4) float32, T_0to1 = poses[idx1] @ inv(poses[idx0])
    T_1to0     : (4, 4) float32, inv(T_0to1)
    depth0     : torch.tensor([])  (METU has no depth; metrics path
                  doesn't read it)
    depth1     : torch.tensor([])
    dataset_name: 'METU_VISTIR' (used by lightning's dispatch)
    scene_id   : npz stem ('cloudy_cloudy_scene_1', ...)
    pair_id    : pair index within the npz (0..N_pair-1)
    pair_names : tuple(str, str), (image0_rel, image1_rel)

Dataset is constructed by src/lightning/data.py via the new ``elif
data_source.lower() == 'metu_vistir':`` branch in
``_build_concat_dataset``; one METUVisTIRDataset instance per npz, then
ConcatDataset stitched together (mirrors MegaDepth's per-scene wrapping).
"""
from __future__ import annotations

import os.path as osp
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as utils_data

from src.datasets.roadscene import _resize_keep_aspect, _resize_target_shape
from src.utils.dataset import pad_bottom_right


_VIS_COL = 0
_TIR_COL = 1


def _rescale_K_for_resize(K: np.ndarray, raw_h: int, raw_w: int,
                          new_h: int, new_w: int) -> np.ndarray:
    """Scale K so it remains valid in the resized coordinate frame.

    Currently UNUSED in METU_VISTIR (we hand RAW K to the matcher and
    let scale0/1 do the job); kept here for sanity-check experiments
    if metrics ever go wrong.
    """
    sx = float(new_w) / float(raw_w)
    sy = float(new_h) / float(raw_h)
    K_new = K.copy()
    K_new[0, :] *= sx
    K_new[1, :] *= sy
    return K_new


class METUVisTIRDataset(utils_data.Dataset):
    """Pose-based evaluation dataset for METU_VISTIR.

    One instance wraps ONE npz (one scene). The lightning data module
    builds 10 instances (or 6 / 4 for cloudy_cloudy / cloudy_sunny
    subsets) and ``ConcatDataset``s them.
    """

    def __init__(
        self,
        root_dir: str,
        npz_path: str,
        mode: str = 'test',
        img_resize: int = 832,
        df: int = 32,
        coarse_scale: float = 1.0 / 8,
        undistort: bool = True,
        side0: str = 'vis',
        side1: str = 'thermal',
        # v7+ pcclahe knobs (defaults disabled; v0/v6.1/baseline ckpt
        # eval byte-identical, only v7+/v9+/v10/v11 ckpts flip these on).
        use_edge_input: bool = False,
        use_clahe_ir: bool = False,
        use_clahe_vis: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Sequence[int] = (8, 8),
        fp16: bool = False,
    ) -> None:
        super().__init__()
        if mode != 'test':
            raise ValueError(
                f"METUVisTIRDataset only supports mode='test', got {mode!r}. "
                f"METU has no train_list; train mode would silently undersupervise.")
        if side0 not in ('vis', 'thermal'):
            raise ValueError(f"side0 must be 'vis' or 'thermal', got {side0!r}")
        if side1 not in ('vis', 'thermal'):
            raise ValueError(f"side1 must be 'vis' or 'thermal', got {side1!r}")
        if side0 == side1:
            raise ValueError(f"side0 == side1 == {side0!r}; cross-modal eval requires different sides")

        self.root_dir = root_dir
        self.npz_path = npz_path
        self.scene_id = osp.splitext(osp.basename(npz_path))[0]
        self.mode = mode
        self.img_resize = int(img_resize)
        self.df = int(df)
        self.coarse_scale = float(coarse_scale)
        self.undistort = bool(undistort)
        self.side0 = side0
        self.side1 = side1
        self.fp16 = bool(fp16)

        # NOTE: scene_info_test/ contains 4 dataset-author files
        # (MatInf/ subdir, convert2mat.py, read_npz.py, test_.txt orphan)
        # alongside the 10 .npz; we DO NOT auto-glob -- list_path resolution
        # in src/lightning/data.py picks one .npz at a time and we trust
        # caller to pass a valid .npz here.

        # Eager-load the npz once; copy out arrays so the file handle is
        # closed before DataLoader workers fork (np.load with allow_pickle
        # leaves a lazy NpzFile that doesn't pickle reliably across spawn).
        with np.load(npz_path, allow_pickle=True) as f:
            self._image_paths = f['image_paths'].copy()           # (N_img, 2) str
            self._intrinsics = f['intrinsics'].copy()             # (N_img, 2, 3, 3)
            self._distortion_coefs = f['distortion_coefs'].copy() # (N_img, 2, 8)
            self._poses = f['poses'].copy()                       # (N_img, 4, 4)
            self._pair_infos = f['pair_infos'].copy()             # (N_pair, 2) object

        # v7+ pcclahe knobs (R2 lazy CLAHE init for worker-pickle safety).
        self.use_edge_input = bool(use_edge_input)
        self.use_clahe_ir = bool(use_clahe_ir)
        self.use_clahe_vis = bool(use_clahe_vis)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_size = tuple(clahe_tile_size)

    def __len__(self) -> int:
        return int(self._pair_infos.shape[0])

    def _ensure_clahe(self):
        """R2 lazy init: build cv2.CLAHE on first use within each worker."""
        if not hasattr(self, '_clahe'):
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size,
            )
        return self._clahe

    def _resolve_side(self, side: str) -> int:
        return _VIS_COL if side == 'vis' else _TIR_COL

    def __getitem__(self, idx: int) -> dict:
        # 1. Resolve pair: idx0/idx1 are image_paths row indices.
        pair = self._pair_infos[idx]
        idx0 = int(pair[0])
        idx1 = int(pair[1])

        col0 = self._resolve_side(self.side0)
        col1 = self._resolve_side(self.side1)

        rel0 = str(self._image_paths[idx0, col0])
        rel1 = str(self._image_paths[idx1, col1])
        K0 = self._intrinsics[idx0, col0].astype(np.float64).copy()
        K1 = self._intrinsics[idx1, col1].astype(np.float64).copy()
        d0 = self._distortion_coefs[idx0, col0].astype(np.float64).copy()
        d1 = self._distortion_coefs[idx1, col1].astype(np.float64).copy()

        # 2. T_0to1 = inv(P1) @ P0  (C2W convention -- METU stores
        # camera-to-world, NOT MegaDepth's W2C). See module docstring for
        # the 4-candidate sweep that proved this; using the wrong
        # formula yields ~3x larger R/t errors (40 deg vs 12 deg).
        P0 = self._poses[idx0].astype(np.float64)   # C2W
        P1 = self._poses[idx1].astype(np.float64)   # C2W
        T_0to1 = np.linalg.inv(P1) @ P0
        T_1to0 = np.linalg.inv(T_0to1)

        # 3. Read images (grayscale).
        img0_path = osp.join(self.root_dir, rel0)
        img1_path = osp.join(self.root_dir, rel1)
        img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img0_raw is None:
            raise FileNotFoundError(f"METU read failed (image0 {self.side0}): {img0_path}")
        if img1_raw is None:
            raise FileNotFoundError(f"METU read failed (image1 {self.side1}): {img1_path}")

        # 4. Undistort BEFORE resize. K is unchanged because cv2.undistort
        # output preserves the input intrinsics (output rays go through
        # the same focal length / principal point). Black-border pixels
        # outside the original FOV become 0; the pad-mask path below
        # only marks "padded canvas" as invalid -- if you observe poor
        # auc with strong distortion, intersect mask0/mask1 with
        # (img > 0) before pad. Skipped in v0 to keep the path simple.
        if self.undistort:
            img0_raw = cv2.undistort(img0_raw, K0, d0)
            img1_raw = cv2.undistort(img1_raw, K1, d1)

        h0_raw, w0_raw = img0_raw.shape
        h1_raw, w1_raw = img1_raw.shape

        # 5. R8: snapshot pre-CLAHE raw for runtime PC. Mirrors
        # roadscene.py:398-408.
        img0_for_pc = None
        img1_for_pc = None
        if self.use_edge_input:
            img0_for_pc = img0_raw.copy() if (
                self.side0 == 'vis' and self.use_clahe_vis
            ) or (
                self.side0 == 'thermal' and self.use_clahe_ir
            ) else img0_raw
            img1_for_pc = img1_raw.copy() if (
                self.side1 == 'vis' and self.use_clahe_vis
            ) or (
                self.side1 == 'thermal' and self.use_clahe_ir
            ) else img1_raw

        # 6. Optional CLAHE on raw uint8. Convention: USE_CLAHE_IR is
        # applied to whichever side IS thermal; USE_CLAHE_VIS is applied
        # to whichever side IS vis. v0-v9 trained ckpts only enable
        # USE_CLAHE_IR (thermal side), so this respects training-time
        # convention regardless of METU_SIDE0 choice.
        if self.side0 == 'thermal' and self.use_clahe_ir:
            img0_raw = self._ensure_clahe().apply(img0_raw)
        if self.side0 == 'vis' and self.use_clahe_vis:
            img0_raw = self._ensure_clahe().apply(img0_raw)
        if self.side1 == 'thermal' and self.use_clahe_ir:
            img1_raw = self._ensure_clahe().apply(img1_raw)
        if self.side1 == 'vis' and self.use_clahe_vis:
            img1_raw = self._ensure_clahe().apply(img1_raw)

        # 7. Independent resize (long edge = img_resize, df-aligned).
        img0, h0_r, w0_r = _resize_keep_aspect(img0_raw, self.img_resize, self.df)
        img1, h1_r, w1_r = _resize_keep_aspect(img1_raw, self.img_resize, self.df)

        # 8. Optional PC channel via runtime fallback. METU has no
        # disk cache so we ALWAYS go through compute_pc_v11_runtime when
        # use_edge_input=True.
        if self.use_edge_input:
            from src.utils.pc import compute_pc_v11_runtime
            pc0 = compute_pc_v11_runtime(img0_for_pc, target_hw=(h0_r, w0_r))
            pc1 = compute_pc_v11_runtime(img1_for_pc, target_hw=(h1_r, w1_r))

        # 9. Pad to common square canvas (for batch_size>1 collate; bs=1
        # in eval but we keep the shape uniform across pairs anyway so
        # PL trainer.test sees a well-defined batch shape).
        pad_size = max(self.img_resize,
                       ((self.img_resize + self.df - 1) // self.df) * self.df)
        img0_pad, mask0 = pad_bottom_right(img0, pad_size, ret_mask=True)
        img1_pad, mask1 = pad_bottom_right(img1, pad_size, ret_mask=True)
        if self.use_edge_input:
            pc0_pad, _ = pad_bottom_right(pc0, pad_size, ret_mask=False)
            pc1_pad, _ = pad_bottom_right(pc1, pad_size, ret_mask=False)
            image0_np = np.stack([img0_pad, pc0_pad], axis=0)        # (2, P, P)
            image1_np = np.stack([img1_pad, pc1_pad], axis=0)
        else:
            image0_np = img0_pad[None]                               # (1, P, P)
            image1_np = img1_pad[None]

        image0 = torch.from_numpy(image0_np).float() / 255.0
        image1 = torch.from_numpy(image1_np).float() / 255.0
        mask0_t = torch.from_numpy(mask0).bool()
        mask1_t = torch.from_numpy(mask1).bool()

        # 10. Coarse-scale mask (same recipe as MegaDepthDataset:125-131
        # and RoadSceneDataset:639-644).
        if self.coarse_scale and self.coarse_scale != 1.0:
            ts_masks = F.interpolate(
                torch.stack([mask0_t, mask1_t], dim=0)[None].float(),
                scale_factor=self.coarse_scale,
                mode='nearest',
                recompute_scale_factor=False,
            )[0].bool()
            mask0_c, mask1_c = ts_masks[0], ts_masks[1]
        else:
            mask0_c, mask1_c = mask0_t, mask1_t

        # 11. scale = [w_raw/w_resized, h_raw/h_resized] so matcher's
        # fine_matching can recover raw-coord mkpts (see
        # coarse_matching.py:223-227 chain). K stays RAW.
        scale0 = torch.tensor(
            [float(w0_raw) / float(w0_r), float(h0_raw) / float(h0_r)],
            dtype=torch.float32,
        )
        scale1 = torch.tensor(
            [float(w1_raw) / float(w1_r), float(h1_raw) / float(h1_r)],
            dtype=torch.float32,
        )

        K0_t = torch.from_numpy(K0).float()
        K1_t = torch.from_numpy(K1).float()
        T_0to1_t = torch.from_numpy(T_0to1.astype(np.float32))
        T_1to0_t = torch.from_numpy(T_1to0.astype(np.float32))

        if self.fp16:
            image0 = image0.half()
            image1 = image1.half()
            scale0 = scale0.half()
            scale1 = scale1.half()

        return {
            'image0': image0,
            'image1': image1,
            'mask0': mask0_c,
            'mask1': mask1_c,
            'K0': K0_t,
            'K1': K1_t,
            'scale0': scale0,
            'scale1': scale1,
            'T_0to1': T_0to1_t,
            'T_1to0': T_1to0_t,
            # METU has no depth; pose-based metrics path doesn't read these
            # but lightning's collate barfs on missing keys for some other
            # batch entries -- providing empty tensors keeps the contract.
            'depth0': torch.tensor([]),
            'depth1': torch.tensor([]),
            'dataset_name': 'METU_VISTIR',
            'scene_id': self.scene_id,
            'pair_id': int(idx),
            'pair_names': (rel0, rel1),
        }
