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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth


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

    def __len__(self) -> int:
        return len(self.pair_infos)

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

        # I/O contract identical to MegaDepthDataset.__getitem__ L79-91
        image0, mask0, scale0 = read_megadepth_gray(
            img_path0, self.img_resize, self.df, self.img_padding, None)
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

        return data
