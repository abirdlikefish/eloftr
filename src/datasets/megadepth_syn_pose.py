"""Megadepth_Syn cross-view + cross-modal pose-supervised training dataset (v13).

v13 supervises matching with **real camera geometry (K + W2C pose + depth)**
on cross-viewpoint cross-modal pairs:

  image0 = IR  (style-transferred infrared) at viewpoint A (stem A)
  image1 = VIS (RGB tourist photo)         at viewpoint C (stem C, A != C)

with K_A, K_C, T_AtoC, depth_A, depth_C all coming from LoFTR's official
``scene_info_0.1_0.7/*.npz`` SfM reconstruction (one npz per
``(scene, overlap_range)``).

This is fundamentally different from v10 / v11 / v12, which all train on
**same-viewpoint** IR-VIS pairs (image0=IR(stem), image1=VIS(same stem),
pixel-aligned by style transfer) supervised by 2D Homography. v13 uses
the pair_infos list -- pairs of *different* image indices that SfM
confirmed have a common 3D visibility region -- so the model is forced
to learn 3D geometry on top of modality invariance.

Path conventions reverse-engineered from depth_paths (the only reliable
root in this dataset; ``image_paths`` is the SfM ``Undistorted_SfM/...``
layout which is NOT present under Megadepth_Syn ``train/`` and must not
be used). For ``depth_rel = "phoenix/S6/zl548/MegaDepth_v1/<scene>/<dense?>
/depths/<stem>.h5"``:

  VIS img : ``<root>/train/<depth_rel with depths/->imgs/, .h5->.jpg>``
  IR  img : ``<root>/train/infrared/<above with same .jpg>``
  depth   : ``<root>/train/<depth_rel>``

The 100% physical existence of this formula across the train/ subset
is verified by ``MyScripts/sanity_megadepth_syn_pose.py``.

Inheritance / reuse:
- Does NOT subclass MegaDepthDataset because the latter's __getitem__
  uses ``scene_info['image_paths']`` which is Undistorted_SfM-style and
  doesn't resolve under Megadepth_Syn/train/.
- DOES reuse ``read_megadepth_gray`` and ``read_megadepth_depth`` for
  byte-identical image / depth I/O with the LoFTR-trained outdoor.ckpt
  data pipeline (same resize / padding / dtype contract).

Dispatch (see plan §3):
- Returns ``dataset_name='Megadepth_Syn_Pose'`` so
  ``src/loftr/utils/supervision.py``'s string-list match
  ``['scannet', 'megadepth', 'megadepth_syn_pose']`` routes to ``spvs_coarse``
  (pose-based) instead of ``spvs_coarse_roadscene`` (Homography-based).
"""
from __future__ import annotations

import os.path as osp
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import (
    read_megadepth_gray,
    read_megadepth_depth,
    imread_gray,
    get_resized_wh,
    get_divisible_wh,
    pad_bottom_right,
)
# v18: reuse roadscene._random_homography for Megadepth-style cross-view pose
# datasets so the H-sampling distribution + signature stay identical to the
# v0..v12 RoadScene H-supervision path. v13/v14/v15/v16 do NOT activate
# this import path (homography_aug=False from data.py train-only gate),
# so adding this import is byte-identical to before.
from src.datasets.roadscene import _random_homography


# Reverse-engineering image paths from depth_paths.
# depth_rel example: 'phoenix/S6/zl548/MegaDepth_v1/0000/dense0/depths/3409963756_o.h5'
def _vis_rel_from_depth(depth_rel: str) -> str:
    """phoenix/.../<scene>/<denseN>/depths/<stem>.h5 -> phoenix/.../<scene>/<denseN>/imgs/<stem>.jpg"""
    return depth_rel.replace("/depths/", "/imgs/").replace(".h5", ".jpg")


def _ir_rel_from_depth(depth_rel: str) -> str:
    """Insert 'infrared/' before 'phoenix/' and switch depths->imgs, .h5->.jpg."""
    if not depth_rel.startswith("phoenix/"):
        # be defensive: only the very first occurrence
        return depth_rel.replace("phoenix/", "infrared/phoenix/", 1).replace(
            "/depths/", "/imgs/").replace(".h5", ".jpg")
    return ("infrared/" + depth_rel).replace("/depths/", "/imgs/").replace(
        ".h5", ".jpg")


class MegadepthSynPoseDataset(Dataset):
    """One scene_info npz worth of pose-supervised cross-modal pairs.

    ``src.lightning.data.MultiSceneDataModule._build_concat_dataset`` builds
    one instance per npz listed in ``train_list_pose.txt`` /
    ``val_list_pose.txt`` and wraps them in a ConcatDataset.
    """

    def __init__(
        self,
        root_dir: str,
        npz_path: str,
        mode: str = "train",
        min_overlap_score: float = 0.0,
        img_resize: int = 832,
        df: int = 32,
        img_padding: bool = True,
        depth_padding: bool = True,
        augment_fn=None,
        fp16: bool = False,
        cross_modal_mode: str = "ir2vis",
        ir_subdir: str = "train/infrared",  # informational only; actual
        vis_subdir: str = "train",           # path is computed from depth_rel
        depth_subdir: str = "train",         # (these three are kept in __init__
                                             # so cfg overrides won't crash
                                             # YACS strict mode)
        # v18: single-side Homography aug on image1 (VIS).
        # When homography_aug=True, sample H_vis per __getitem__ call,
        # warp image1 in resized pixel space (after resize, before pad),
        # and emit data['H_vis'] so spvs_coarse / spvs_fine can pull it.
        # When homography_aug=False (default; v13/v14/v15/v16 path), this
        # entire branch is skipped and no 'H_vis' key is added -> the
        # output dict / batch keyset is byte-identical to before.
        # data.py applies a train-only gate so val/test always sees False.
        homography_aug: bool = False,
        homography_prob: float = 1.0,
        homography_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if cross_modal_mode not in ("ir2vis", "vis2vis"):
            raise ValueError(
                f"cross_modal_mode must be 'ir2vis' or 'vis2vis' "
                f"(see SKILL eloftr-v13-pose); got {cross_modal_mode!r}")

        self.root_dir = root_dir
        self.npz_path = npz_path
        self.scene_id = osp.splitext(osp.basename(npz_path))[0]
        self.mode = mode
        self.cross_modal_mode = cross_modal_mode

        # Lazy-load npz once; copy arrays out so the file handle is closed
        # before DataLoader workers fork (np.load with allow_pickle leaves
        # a lazy NpzFile that doesn't pickle reliably across spawn).
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info["pair_infos"].copy()
        # filter by overlap (pair_infos[i] = ((idx0, idx1), overlap, central))
        self.pair_infos = [
            p for p in self.pair_infos if float(p[1]) > min_overlap_score
        ]
        # cache the per-image arrays we need without keeping the NpzFile
        self._image_paths = self.scene_info["image_paths"].copy()
        self._depth_paths = self.scene_info["depth_paths"].copy()
        self._intrinsics = self.scene_info["intrinsics"].copy()
        self._poses = self.scene_info["poses"].copy()
        # drop the NpzFile so workers don't carry a file handle across forks
        del self.scene_info

        if mode == "train":
            assert img_resize is not None and img_padding and depth_padding, (
                "v13 train mode requires img_resize / img_padding / depth_padding "
                "all set (see MegaDepthDataset L55-56 for the same constraint)")
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None

        self.augment_fn = augment_fn if mode == "train" else None
        self.coarse_scale = kwargs.get("coarse_scale", 0.125)
        self.fp16 = bool(fp16)

        # v18 H aug state. Stored on self so __getitem__ can branch cheaply.
        self.homography_aug = bool(homography_aug)
        self.homography_prob = float(homography_prob)
        self.homography_kwargs = dict(homography_kwargs) if homography_kwargs else {}

    def __len__(self) -> int:
        return len(self.pair_infos)

    def _read_resize_warp_pad(self, path: str, H: np.ndarray):
        """Inline read_megadepth_gray + cv2.warpPerspective(H) on the resized
        canvas + pad_bottom_right. Returns (image[1,P,P], mask[P,P], scale[2]).

        H is sampled in *resized* pixel coordinates so we apply it AFTER
        cv2.resize and BEFORE pad_bottom_right (otherwise the rotation centre
        + scale of H_vis would not match supervision.spvs_coarse's 4-step
        scale conversion chain).

        When H is the identity, we short-circuit cv2.warpPerspective so the
        output is byte-identical to read_megadepth_gray (catches Bug #14
        from plan: cv2.warpPerspective with H=I introduces sub-pixel
        resampling noise even with INTER_LINEAR + BORDER_CONSTANT; we want
        rot_deg=0 -> H=I -> exact byte-identical v14 contract).
        """
        # mirrors read_megadepth_gray L106-126 with a warp inserted before pad.
        image_orig = imread_gray(path, augment_fn=None)
        h_orig, w_orig = image_orig.shape[0], image_orig.shape[1]
        w_new, h_new = get_resized_wh(w_orig, h_orig, self.img_resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, self.df)

        image_resized = cv2.resize(image_orig, (w_new, h_new))
        scale = torch.tensor([w_orig / w_new, h_orig / h_new], dtype=torch.float)

        # H=I short-circuit: skip warp entirely so byte-identical to baseline.
        if not np.allclose(H, np.eye(3, dtype=H.dtype), atol=1e-12):
            image_resized = cv2.warpPerspective(
                image_resized, H, (w_new, h_new),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            # Build vis_valid by warping an all-ones mask with INTER_NEAREST
            # (so border cells are exactly 0/1, no fractional bleed) and
            # borderValue=0 (anything outside the source rectangle is invalid).
            ones = np.ones((h_new, w_new), dtype=np.uint8)
            vis_valid = cv2.warpPerspective(
                ones, H, (w_new, h_new),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(bool)
        else:
            vis_valid = np.ones((h_new, w_new), dtype=bool)

        # pad to (P, P) where P == max(h_new, w_new) (read_megadepth_gray L117)
        if self.img_padding:
            pad_to = max(h_new, w_new)
            image_padded, pad_mask = pad_bottom_right(image_resized, pad_to, ret_mask=True)
            # vis_valid lives on the resized (h_new, w_new) canvas; embed it
            # into the padded canvas at the same top-left origin so the
            # intersection with pad_mask is well-defined elementwise.
            vis_valid_padded = np.zeros_like(pad_mask)
            vis_valid_padded[:h_new, :w_new] = vis_valid
            mask = pad_mask & vis_valid_padded
        else:
            image_padded = image_resized
            mask = vis_valid

        image = torch.from_numpy(image_padded).float()[None] / 255
        mask_t = torch.from_numpy(mask) if mask is not None else None
        return image, mask_t, scale

    def _resolve_pair_paths(self, idx0: int, idx1: int) -> tuple[str, str, str, str]:
        """Return absolute paths for (image0, depth0, image1, depth1)."""
        depth_rel0 = str(self._depth_paths[idx0])
        depth_rel1 = str(self._depth_paths[idx1])
        depth_path0 = osp.join(self.root_dir, "train", depth_rel0)
        depth_path1 = osp.join(self.root_dir, "train", depth_rel1)

        if self.cross_modal_mode == "ir2vis":
            # image0 = IR at stem(idx0), image1 = VIS at stem(idx1)
            img_rel0 = _ir_rel_from_depth(depth_rel0)
            img_rel1 = _vis_rel_from_depth(depth_rel1)
        else:  # "vis2vis": RGB-RGB cross-view ablation (NOT used in v13 main)
            img_rel0 = _vis_rel_from_depth(depth_rel0)
            img_rel1 = _vis_rel_from_depth(depth_rel1)

        img_path0 = osp.join(self.root_dir, "train", img_rel0)
        img_path1 = osp.join(self.root_dir, "train", img_rel1)
        return img_path0, depth_path0, img_path1, depth_path1

    def __getitem__(self, idx: int) -> dict:
        pair_info = self.pair_infos[idx]
        (idx0, idx1), overlap_score, central_matches = pair_info
        idx0 = int(idx0)
        idx1 = int(idx1)

        img_path0, depth_path0, img_path1, depth_path1 = self._resolve_pair_paths(idx0, idx1)

        # I/O contract identical to MegaDepthDataset.__getitem__ L79-91.
        # image0 (IR) is ALWAYS read by the original read_megadepth_gray so
        # this leg stays byte-identical to v13/v14/v15/v16. image1 (VIS)
        # branches on self.homography_aug (data.py train-only gate ensures
        # val/test always sees False).
        image0, mask0, scale0 = read_megadepth_gray(
            img_path0, self.img_resize, self.df, self.img_padding, None)

        H_vis = None  # will be added to data dict iff self.homography_aug
        if self.homography_aug:
            # We need the resized (h_new, w_new) BEFORE sampling H, because
            # _random_homography centres the rotation on (w_new/2, h_new/2)
            # and supervision.spvs_coarse expects H to live in the resized
            # pixel space.
            #
            # Read once just for shape; cheap because cv2.imread is the same
            # whether or not we then resize/warp. (We could cache shape from
            # depth0 etc. but that conflates depth shape vs image shape and
            # we want the contract trivially auditable.)
            tmp = imread_gray(img_path1, augment_fn=None)
            h_orig1, w_orig1 = tmp.shape[0], tmp.shape[1]
            w_new1, h_new1 = get_resized_wh(w_orig1, h_orig1, self.img_resize)
            w_new1, h_new1 = get_divisible_wh(w_new1, h_new1, self.df)

            if np.random.rand() < self.homography_prob:
                H_vis = _random_homography(h_new1, w_new1, **self.homography_kwargs)
            else:
                # rare branch when prob < 1.0; still emit H_vis = I so the
                # supervision contract (data['H_vis'] always present in the
                # H-aug branch) is uniform across batch elements -> collate
                # builds [N, 3, 3] cleanly.
                H_vis = np.eye(3, dtype=np.float32)

            image1, mask1, scale1 = self._read_resize_warp_pad(img_path1, H_vis)
        else:
            image1, mask1, scale1 = read_megadepth_gray(
                img_path1, self.img_resize, self.df, self.img_padding, None)

        if self.mode in ("train", "val"):
            depth0 = read_megadepth_depth(depth_path0, pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(depth_path1, pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        K_0 = torch.tensor(
            np.asarray(self._intrinsics[idx0]).reshape(3, 3), dtype=torch.float)
        K_1 = torch.tensor(
            np.asarray(self._intrinsics[idx1]).reshape(3, 3), dtype=torch.float)

        # W2C convention (same as MegaDepth): T_0to1 = T1 @ inv(T0)
        T0 = np.asarray(self._poses[idx0])
        T1 = np.asarray(self._poses[idx1])
        T_0to1 = torch.tensor(
            np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]
        T_1to0 = T_0to1.inverse()

        if self.fp16:
            image0, image1, depth0, depth1, scale0, scale1 = map(
                lambda x: x.half(),
                [image0, image1, depth0, depth1, scale0, scale1])

        data = {
            "image0": image0,
            "depth0": depth0,
            "image1": image1,
            "depth1": depth1,
            "T_0to1": T_0to1,
            "T_1to0": T_1to0,
            "K0": K_0,
            "K1": K_1,
            "scale0": scale0,
            "scale1": scale1,
            "dataset_name": "Megadepth_Syn_Pose",
            "scene_id": self.scene_id,
            "pair_id": idx,
            "pair_names": (
                osp.relpath(img_path0, self.root_dir),
                osp.relpath(img_path1, self.root_dir),
            ),
        }
        if mask0 is not None:
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(
                    torch.stack([mask0, mask1], dim=0)[None].float(),
                    scale_factor=self.coarse_scale,
                    mode="nearest",
                    recompute_scale_factor=False,
                )[0].bool()
                data.update({"mask0": ts_mask_0, "mask1": ts_mask_1})

        # v18: emit H_vis only when the H-aug branch fired. Keeping this gated
        # by `is not None` (rather than `self.homography_aug`) is paranoid but
        # matches the supervision-side `'H_vis' in data` guard 1:1.
        if H_vis is not None:
            data["H_vis"] = torch.from_numpy(H_vis).float()

        return data
