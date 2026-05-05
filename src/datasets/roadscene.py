"""RoadScene IR-VIS dataset for EfficientLoFTR training/validation/testing.

Key design decisions (Solution A3 -- padding + mask):

- The dataset reads a flat txt index (one filename per line, e.g. ``FLIR_05164.jpg``)
  and reconstructs IR/VIS paths as ``<root>/<ir_subdir>/<name>`` and
  ``<root>/<vis_subdir>/<name>``.
- IR and VIS are each independently resized so that ``max(H, W) == img_resize``
  (preserving aspect ratio), then **zero-padded bottom-right** to a fixed square
  canvas ``(pad_size, pad_size)``. This guarantees every batch element has the
  exact same tensor shape, which is required for ``default_collate`` to work
  with ``batch_size > 1``.
- Two binary masks ``mask0`` / ``mask1`` are produced (downsampled to coarse
  scale ``1/8``) marking the valid (non-padded, non-warped-to-zero) region.
  Coarse matching, supervision and loss all consume these masks so the model
  never tries to match against zero-padded pixels.
- ``homography_aug`` controls whether a random Homography is applied to the
  visible image only (image0 = IR is left untouched, image1 = warped VIS).
  The Homography is sampled and applied in the **resized-but-not-padded VIS
  frame**, so it rotates around the centre of the real VIS content (rather
  than the centre of the padded canvas, which may be inside the padded zeros
  for narrow images). The same Homography matrix is then valid as
  ``homography_0to1`` in the padded coordinate frame because both IR and VIS
  share top-left padding alignment.
- Validation/test default to ``homography_aug=False`` so metrics stay
  reproducible.
- The dataset never returns ``depth0/depth1``, ``T_0to1/T_1to0`` or ``K0/K1``
  because RoadScene has no camera geometry.

Output ``data`` dict:

    image0          : torch.float32 (1, P, P), IR in [0, 1], P = pad_size
    image1          : torch.float32 (1, P, P), VIS (possibly warped) in [0, 1]
    mask0           : torch.bool  (P/8, P/8), True on real IR coarse cells
    mask1           : torch.bool  (P/8, P/8), True on real VIS coarse cells
    scale0          : torch.float32 (2,), always [1.0, 1.0] so the padded
                       image acts as the "original-resolution" reference for
                       supervision / matching.
    scale1          : torch.float32 (2,), same as ``scale0``.
    orig_scale0     : torch.float32 (2,), (orig_w / w_resized, orig_h / h_resized)
                       useful for back-projecting predictions to original IR pixels
                       in evaluation scripts.
    orig_scale1     : torch.float32 (2,), same for VIS.
    homography_0to1 : torch.float32 (3, 3), maps image0 px -> image1 px in
                       the padded-image coordinate frame.
    dataset_name    : str, value of the constructor's ``dataset_name`` arg
                       (default 'RoadScene'). Used by the IR-VIS dispatch
                       sites; must be a key registered in
                       ``src.utils.data_source.ALIGNED_IRVIS_SOURCES``
                       (case-insensitive).
    scene_id        : str, same as ``dataset_name``
    pair_id         : int
    pair_names      : tuple(str, str), (ir_path, vis_path)
"""
from __future__ import annotations

from os import path as osp
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as utils_data

from src.utils.dataset import pad_bottom_right


def _read_index_file(list_path: str) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names


def _resize_keep_aspect(img: np.ndarray, long_edge: int, df: int) -> Tuple[np.ndarray, int, int]:
    """Resize so that ``max(H, W) == long_edge`` while preserving aspect ratio,
    then round each side down to a multiple of ``df``.

    Returns ``(resized_img, new_h, new_w)``.
    """
    h, w = img.shape
    scale = float(long_edge) / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    new_h = max(df, (new_h // df) * df)
    new_w = max(df, (new_w // df) * df)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized, new_h, new_w


def _random_homography(h: int, w: int,
                       rot_deg: float = 10.0,
                       scale_range: Tuple[float, float] = (0.9, 1.1),
                       trans_ratio: float = 0.05,
                       persp_ratio: float = 0.03,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample a 3x3 Homography combining rotation, scale, translation and a small
    perspective perturbation. The transform is centred on ``(w/2, h/2)``.

    Returns ``H`` such that ``warped_pt = H @ original_pt`` (homogeneous coords).
    Apply via ``cv2.warpPerspective(src, H, (w, h))`` to get the warped image.
    """
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = w * 0.5, h * 0.5
    theta = float(rng.uniform(-rot_deg, rot_deg)) * np.pi / 180.0
    s = float(rng.uniform(scale_range[0], scale_range[1]))
    tx = float(rng.uniform(-trans_ratio, trans_ratio)) * w
    ty = float(rng.uniform(-trans_ratio, trans_ratio)) * h

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_scale = np.array([
        [s * cos_t, -s * sin_t, 0.0],
        [s * sin_t, s * cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    to_origin = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    back_and_translate = np.array([
        [1.0, 0.0, cx + tx],
        [0.0, 1.0, cy + ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_affine = back_and_translate @ rot_scale @ to_origin

    src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_corners = src_corners.copy()
    dst_corners[:, 0] += rng.uniform(-persp_ratio, persp_ratio, size=4) * w
    dst_corners[:, 1] += rng.uniform(-persp_ratio, persp_ratio, size=4) * h
    H_persp = cv2.getPerspectiveTransform(src_corners, dst_corners.astype(np.float32))

    H = H_persp @ H_affine
    H /= H[2, 2]
    return H.astype(np.float32)


class RoadSceneDataset(utils_data.Dataset):
    """RoadScene IR-VIS dataset with optional Homography augmentation.

    Args:
        root_dir: Root of the dataset download (RoadScene, M3FD, ...).
        list_path: Path to the txt index file produced by
            ``MyScripts/make_<dataset>_splits.py``.
        mode: ``'train'`` / ``'val'`` / ``'test'``.
        dataset_name: Value written into ``data['dataset_name']`` /
            ``data['scene_id']``. Default ``'RoadScene'`` keeps every
            existing call site identical; set to ``'M3FD'`` (or any other
            name registered in
            ``src.utils.data_source.ALIGNED_IRVIS_SOURCES``) when reusing
            this class for a different aligned IR-VIS dataset.
        ir_subdir / vis_subdir: Sub-folder names under ``root_dir``.
        img_resize: Target size for ``max(H, W)`` of each image after
            aspect-ratio-preserving resize (before padding).
        pad_size: Side of the square zero-padded canvas. Must be
            ``>= img_resize`` and divisible by ``df``. Defaults to
            ``img_resize`` rounded up to a multiple of ``df``.
        df: Both ``img_resize`` and ``pad_size`` are aligned to this divisor.
            Should match the LoFTR backbone stride (typically ``32``).
        coarse_scale: Down-sampling factor used to produce coarse-scale
            ``mask0/mask1`` (typically ``1/8`` to match LoFTR coarse
            resolution).
        homography_aug: Whether to apply a random Homography on VIS at train
            time. Disabled at val/test for reproducible metrics.
        homography_prob: Probability of applying the Homography per sample
            when ``homography_aug`` is on.
        homography_kwargs: Override kwargs forwarded to ``_random_homography``.
    """

    def __init__(self,
                 root_dir: str,
                 list_path: str,
                 mode: str = "train",
                 dataset_name: str = "RoadScene",
                 ir_subdir: str = "cropinfrared",
                 vis_subdir: str = "crop_LR_visible",
                 img_resize: Optional[int] = 480,
                 pad_size: Optional[int] = None,
                 df: int = 32,
                 coarse_scale: float = 1.0 / 8,
                 homography_aug: bool = False,
                 homography_prob: float = 1.0,
                 homography_kwargs: Optional[dict] = None,
                 augment_fn=None,
                 fp16: bool = False,
                 # v7_pcclahe knobs (all default to disabled = v0-v6.1 behaviour
                 # byte-identical). Defaults must stay literal-equal to
                 # src/config/default.py so a missing cfg field still yields
                 # the same dataset behaviour as the historical pipeline.
                 use_edge_input: bool = False,
                 ir_pc_subdir: str = "",
                 vis_pc_subdir: str = "",
                 use_clahe_ir: bool = False,
                 use_clahe_vis: bool = False,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size=(8, 8),
                 **kwargs):
        super().__init__()
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unknown mode: {mode}")
        if img_resize is None:
            raise ValueError("RoadSceneDataset requires an explicit img_resize so all "
                             "samples can be padded to a common canvas.")

        self.root_dir = root_dir
        self.list_path = list_path
        self.mode = mode
        self.dataset_name = str(dataset_name)
        self.ir_dir = osp.join(root_dir, ir_subdir)
        self.vis_dir = osp.join(root_dir, vis_subdir)
        self.img_resize = int(img_resize)
        self.df = int(df)
        if pad_size is None:
            pad_size = ((self.img_resize + self.df - 1) // self.df) * self.df
        else:
            pad_size = int(pad_size)
            if pad_size % self.df != 0:
                raise ValueError(f"pad_size ({pad_size}) must be divisible by df ({self.df}).")
            if pad_size < self.img_resize:
                raise ValueError(f"pad_size ({pad_size}) must be >= img_resize ({self.img_resize}).")
        self.pad_size = pad_size
        self.coarse_scale = coarse_scale
        self.homography_aug = bool(homography_aug)
        self.homography_prob = float(homography_prob)
        self.homography_kwargs = dict(homography_kwargs) if homography_kwargs else {}
        self.augment_fn = augment_fn if mode == "train" else None
        self.fp16 = fp16

        # ------------------------------------------------------------------
        # v7_pcclahe state (R1 dataset guard discipline + R2 CLAHE worker pickle)
        # ------------------------------------------------------------------
        # R1: only the use_edge_input flag triggers PC loading. m3fd_trainval.py
        # and roadscene_trainval.py BOTH set ROAD_IR_PC_SUBDIR by default so the
        # field is non-empty for v0-v6.1 retraining too -- but those v_x configs
        # leave USE_EDGE_INPUT default False, and *that* is what the dataset
        # checks. Never branch on the subdir field alone.
        self.use_edge_input = bool(use_edge_input)
        if self.use_edge_input:
            if not ir_pc_subdir or not vis_pc_subdir:
                raise ValueError(
                    "use_edge_input=True requires non-empty ir_pc_subdir and "
                    "vis_pc_subdir (set them in the data config; see "
                    "configs/data/m3fd_trainval.py for an example).")
            self.ir_pc_dir = osp.join(root_dir, ir_pc_subdir)
            self.vis_pc_dir = osp.join(root_dir, vis_pc_subdir)
            if not osp.isdir(self.ir_pc_dir):
                raise FileNotFoundError(
                    f"PC cache directory not found: {self.ir_pc_dir}. "
                    f"Run MyScripts/precompute_pc_edges.bat (defaults to both "
                    f"M3FD + RoadScene) before any v7_pcclahe training/eval.")
            if not osp.isdir(self.vis_pc_dir):
                raise FileNotFoundError(
                    f"PC cache directory not found: {self.vis_pc_dir}. "
                    f"Run MyScripts/precompute_pc_edges.bat first.")
        else:
            # Crucial: leave PC dirs as None so any accidental access throws
            # a clear AttributeError instead of silently using a stale path.
            self.ir_pc_dir = None
            self.vis_pc_dir = None

        # R2: store CLAHE knobs only; never instantiate cv2.createCLAHE() here.
        # cv2.CLAHE C++ objects do not pickle reliably across OpenCV 4.x
        # versions, and Windows DataLoader workers spawn (= pickle the dataset).
        # The actual cv2.createCLAHE() call is lazy in __getitem__ so each
        # worker holds its own instance after fork/spawn.
        self.use_clahe_ir = bool(use_clahe_ir)
        self.use_clahe_vis = bool(use_clahe_vis)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_size = tuple(clahe_tile_size)
        if self.use_clahe_ir or self.use_clahe_vis:
            from loguru import logger
            logger.info(
                f"RoadSceneDataset: CLAHE enabled (clipLimit={self.clahe_clip_limit}, "
                f"tile={self.clahe_tile_size}, ir={self.use_clahe_ir}, vis={self.use_clahe_vis})")

        self.names = _read_index_file(list_path)
        if not self.names:
            raise RuntimeError(f"Empty RoadScene index: {list_path}")

    def __len__(self) -> int:
        return len(self.names)

    def _ensure_clahe(self):
        """R2 lazy init: build cv2.CLAHE on first use within each worker.
        Avoids pickling the C++ object across DataLoader spawn boundaries."""
        if not hasattr(self, '_clahe'):
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size,
            )
        return self._clahe

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        ir_path = osp.join(self.ir_dir, name)
        vis_path = osp.join(self.vis_dir, name)
        ir_raw = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        vis_raw = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
        if ir_raw is None:
            raise FileNotFoundError(f"Failed to read IR: {ir_path}")
        if vis_raw is None:
            raise FileNotFoundError(f"Failed to read VIS: {vis_path}")
        h0_raw, w0_raw = ir_raw.shape
        h1_raw, w1_raw = vis_raw.shape

        # CLAHE on raw uint8 BEFORE resize/warp/pad. cv2.createCLAHE only
        # accepts uint8/uint16 inputs (running it after the /255.0 float cast
        # would error). Order: read -> CLAHE -> resize -> warp -> pad ->
        # to-float. PC cache is loaded as-is (PC is sparse-edge by design;
        # CLAHE on PC would destroy that sparsity, so we never apply it).
        if self.use_clahe_ir:
            ir_raw = self._ensure_clahe().apply(ir_raw)
        if self.use_clahe_vis:
            vis_raw = self._ensure_clahe().apply(vis_raw)

        # Optional PC channel: read once on the raw resolution then carry
        # alongside the gray channel through every following geometric op
        # (resize / warp / pad). PC stem matches the source filename (the
        # precompute script writes <stem>.png regardless of source ext).
        ir_pc_raw = None
        vis_pc_raw = None
        if self.use_edge_input:
            pc_name = osp.splitext(name)[0] + ".png"
            ir_pc_path = osp.join(self.ir_pc_dir, pc_name)
            vis_pc_path = osp.join(self.vis_pc_dir, pc_name)
            ir_pc_raw = cv2.imread(str(ir_pc_path), cv2.IMREAD_GRAYSCALE)
            vis_pc_raw = cv2.imread(str(vis_pc_path), cv2.IMREAD_GRAYSCALE)
            if ir_pc_raw is None:
                raise FileNotFoundError(
                    f"Failed to read IR PC cache: {ir_pc_path}. "
                    f"Run MyScripts/precompute_pc_edges.bat to (re)generate.")
            if vis_pc_raw is None:
                raise FileNotFoundError(f"Failed to read VIS PC cache: {vis_pc_path}.")
            # Sanity: PC cache must match the source resolution exactly,
            # otherwise the synced resize below would silently misalign.
            if ir_pc_raw.shape != ir_raw.shape:
                raise RuntimeError(
                    f"IR PC cache shape {ir_pc_raw.shape} != raw IR shape "
                    f"{ir_raw.shape} for {name}. Re-run precompute.")
            if vis_pc_raw.shape != vis_raw.shape:
                raise RuntimeError(
                    f"VIS PC cache shape {vis_pc_raw.shape} != raw VIS shape "
                    f"{vis_raw.shape} for {name}. Re-run precompute.")

        # 1. Resize each image independently (long edge = img_resize, df-aligned).
        ir, h0_r, w0_r = _resize_keep_aspect(ir_raw, self.img_resize, self.df)
        vis, h1_r, w1_r = _resize_keep_aspect(vis_raw, self.img_resize, self.df)
        if self.use_edge_input:
            # Reuse the same target shape so PC and gray stay pixel-aligned.
            ir_pc = cv2.resize(ir_pc_raw, (w0_r, h0_r))
            vis_pc = cv2.resize(vis_pc_raw, (w1_r, h1_r))

        # 2. Optionally warp VIS *before* padding so the Homography is centred on
        # the real VIS content rather than the padded canvas centre.
        H_0to1 = np.eye(3, dtype=np.float32)
        if self.homography_aug and self.mode == "train":
            if np.random.rand() < self.homography_prob:
                H_0to1 = _random_homography(h1_r, w1_r, **self.homography_kwargs)
                vis = cv2.warpPerspective(vis, H_0to1, (w1_r, h1_r),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0)
                if self.use_edge_input:
                    # Apply the SAME H to the VIS PC channel so PC and gray
                    # remain pixel-aligned in image1.
                    vis_pc = cv2.warpPerspective(vis_pc, H_0to1, (w1_r, h1_r),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=0)

        # 3. Compute the post-warp VIS validity mask in the resized frame.
        # A pixel is valid iff its source pixel (under H^-1) was inside the
        # original (h1_r, w1_r) rectangle. We compute this by warping an
        # all-ones mask with the same H using nearest-neighbour interpolation.
        if not np.array_equal(H_0to1, np.eye(3, dtype=np.float32)):
            ones = np.ones((h1_r, w1_r), dtype=np.uint8)
            vis_valid = cv2.warpPerspective(ones, H_0to1, (w1_r, h1_r),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0).astype(bool)
        else:
            vis_valid = np.ones((h1_r, w1_r), dtype=bool)

        # 4. Zero-pad both images bottom-right to (pad_size, pad_size) and build
        # the canvas-level masks. mask1 is intersected with vis_valid so cells
        # warped from outside the source image are also marked invalid.
        ir_pad, mask0 = pad_bottom_right(ir, self.pad_size, ret_mask=True)
        vis_pad, mask1 = pad_bottom_right(vis, self.pad_size, ret_mask=True)
        mask1[:h1_r, :w1_r] &= vis_valid

        if self.use_edge_input:
            # PC channels share the same gray pad (mask0/mask1 already capture
            # validity). Use ret_mask=False because we don't need a second mask.
            # NOTE: pad_bottom_right ALWAYS returns a (padded, mask) tuple --
            # ret_mask=False just sets mask=None but the tuple is still
            # returned. We must unpack so np.stack below sees a plain ndarray
            # rather than a (ndarray, None) tuple (which would trigger a
            # "setting an array element with a sequence ... inhomogeneous
            # shape" ValueError on stack).
            ir_pc_pad, _ = pad_bottom_right(ir_pc, self.pad_size, ret_mask=False)
            vis_pc_pad, _ = pad_bottom_right(vis_pc, self.pad_size, ret_mask=False)
            # Stack so channel 0 = gray (post-CLAHE), channel 1 = PC. Inflated
            # init in lightning_loftr puts v6.1 stage0 weights on channel 0
            # and zeros on channel 1, so this ordering is load-bearing.
            image0_np = np.stack([ir_pad, ir_pc_pad], axis=0)        # (2, P, P)
            image1_np = np.stack([vis_pad, vis_pc_pad], axis=0)
        else:
            # v0-v6.1 path: same shape as the historical [None] indexing used to
            # produce, byte-identical to the previous implementation.
            image0_np = ir_pad[None]                                  # (1, P, P)
            image1_np = vis_pad[None]

        image0 = torch.from_numpy(image0_np).float() / 255.0
        image1 = torch.from_numpy(image1_np).float() / 255.0
        mask0_t = torch.from_numpy(mask0).bool()
        mask1_t = torch.from_numpy(mask1).bool()

        # 5. Down-sample masks to LoFTR coarse resolution (typically 1/8) so
        # they can be consumed directly by the coarse-matching layer and the
        # supervision functions. Mirrors what MegaDepthDataset does.
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

        # See the "scale0=[1,1]" rationale in the module docstring.
        scale0_t = torch.tensor([1.0, 1.0], dtype=torch.float32)
        scale1_t = torch.tensor([1.0, 1.0], dtype=torch.float32)
        orig_scale0 = torch.tensor(
            [float(w0_raw) / float(w0_r), float(h0_raw) / float(h0_r)],
            dtype=torch.float32)
        orig_scale1 = torch.tensor(
            [float(w1_raw) / float(w1_r), float(h1_raw) / float(h1_r)],
            dtype=torch.float32)
        H_t = torch.from_numpy(H_0to1.astype(np.float32))

        if self.fp16:
            image0 = image0.half()
            image1 = image1.half()
            scale0_t = scale0_t.half()
            scale1_t = scale1_t.half()
            orig_scale0 = orig_scale0.half()
            orig_scale1 = orig_scale1.half()

        ir_rel = osp.join(osp.basename(self.ir_dir), name)
        vis_rel = osp.join(osp.basename(self.vis_dir), name)

        data = {
            "image0": image0,
            "image1": image1,
            "mask0": mask0_c,
            "mask1": mask1_c,
            "scale0": scale0_t,
            "scale1": scale1_t,
            "orig_scale0": orig_scale0,
            "orig_scale1": orig_scale1,
            "homography_0to1": H_t,
            "dataset_name": self.dataset_name,
            "scene_id": self.dataset_name,
            "pair_id": idx,
            "pair_names": (ir_rel, vis_rel),
        }
        return data
