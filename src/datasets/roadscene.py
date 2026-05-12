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
- ``homography_dual`` (v11) extends the above to warp BOTH IR (image0) and
  VIS (image1) by INDEPENDENT random Homographies ``H_ir`` and ``H_vis``.
  The dataset returns ``homography_0to1 = H_vis @ inv(H_ir)``, which still
  satisfies the contract "maps image0 px -> image1 px in the padded
  coordinate frame" because both IR and VIS share top-left padding alignment
  in their own warped frames. ``mask0`` is intersected with ``ir_valid``
  (parallel to ``mask1`` & ``vis_valid``) so coarse cells warped from
  outside the source IR rectangle are also marked invalid. Default
  ``homography_dual=False`` is byte-identical to v0-v10 (only VIS warped,
  ``ir_valid`` is all-True so the new ``mask0 &= ir_valid`` line is a no-op).
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
    new_h, new_w = _resize_target_shape(img.shape, long_edge, df)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized, new_h, new_w


def _resize_target_shape(raw_shape: Tuple[int, int], long_edge: int, df: int) -> Tuple[int, int]:
    """Pure-math version of ``_resize_keep_aspect`` -- returns the (new_h, new_w)
    that ``_resize_keep_aspect(raw, long_edge, df)`` would produce, without
    actually allocating / copying pixels.

    Used by the v10 PC cache shape check in ``RoadSceneDataset.__getitem__``
    so we can verify that cache and raw both round to the SAME df-aligned
    target after the dataset's resize step. Also re-used by
    ``MyScripts/fix_pc_cache_alignment.py`` to compute the post-resize
    shape that PC caches must be pre-aligned to.
    """
    h, w = raw_shape
    scale = float(long_edge) / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    new_h = max(df, (new_h // df) * df)
    new_w = max(df, (new_w // df) * df)
    return (new_h, new_w)


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
        homography_aug: Whether to apply a random Homography at train time
            (single-sided VIS-only by default; see ``homography_dual``).
            Disabled at val/test for reproducible metrics.
        homography_prob: Probability (one Bernoulli per sample) of firing
            the Homography aug. When the gate fails, both H_ir and H_vis are
            identity regardless of ``homography_dual`` so the no-aug code
            path stays byte-identical to v0-v10.
        homography_kwargs: Override kwargs forwarded to ``_random_homography``
            (used for BOTH H_ir and H_vis when dual=True).
        homography_dual: When True (v11), warp BOTH IR (image0) and VIS
            (image1) by independent random Homographies; ``homography_0to1``
            becomes ``H_vis @ inv(H_ir)``. When False (v0-v10 default), only
            VIS is warped, identical to the historical pipeline.
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
                 homography_dual: bool = False,
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
                 # DDP pre-shard hand-off: when set, skip _read_index_file and
                 # use the provided pre-sharded list instead. data.py L283
                 # (ScanNet/MegaDepth) shards npz scenes via get_local_split;
                 # the IR-VIS short-circuit branch in data.py mirrors that by
                 # sharding `names` here. Default None preserves v0-v9 byte-
                 # identical behaviour: read the full index from list_path.
                 names: Optional[List[str]] = None,
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
        self.homography_dual = bool(homography_dual)
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
        # R8 (runtime PC fallback, 2026-05-11): in val/test mode allow PC cache
        # to be absent and compute PC on-the-fly via src.utils.pc. Training
        # mode KEEPS the original hard FileNotFoundError to protect the
        # v7+/v10/v11 training distribution (PC cache 必须在 raw 上 precompute,
        # 训练分布锁定在 precompute --max_long_edge 640 --pc_nscale 3 +
        # fix_pc_cache_alignment 之后的最终 480 long-edge cache). Two flags:
        #   _pc_runtime_fallback : True when whole PC dir is missing in val/test
        #     mode (set once in __init__, drives every __getitem__ for this
        #     dataset instance to skip cv2.imread and call compute_pc_v11_runtime).
        #   _pc_missing_warned   : True after the first per-file cache miss WARN
        #     in __getitem__ (val/test only). Avoids spamming the log when most
        #     files have cache but a handful are missing.
        self.use_edge_input = bool(use_edge_input)
        self._pc_runtime_fallback = False
        self._pc_missing_warned = False
        if self.use_edge_input:
            if not ir_pc_subdir or not vis_pc_subdir:
                raise ValueError(
                    "use_edge_input=True requires non-empty ir_pc_subdir and "
                    "vis_pc_subdir (set them in the data config; see "
                    "configs/data/m3fd_trainval.py for an example).")
            self.ir_pc_dir = osp.join(root_dir, ir_pc_subdir)
            self.vis_pc_dir = osp.join(root_dir, vis_pc_subdir)
            pc_dirs_ok = osp.isdir(self.ir_pc_dir) and osp.isdir(self.vis_pc_dir)
            if not pc_dirs_ok:
                if self.mode == "train":
                    raise FileNotFoundError(
                        f"PC cache directory not found in train mode: "
                        f"{self.ir_pc_dir} or {self.vis_pc_dir}. "
                        f"Training requires precomputed PC cache to match the "
                        f"v7+/v10/v11 training distribution. Run "
                        f"MyScripts/precompute_pc_edges.bat (defaults to M3FD + "
                        f"RoadScene; for Megadepth_Syn use "
                        f"`--dataset Megadepth_Syn --recursive --max_long_edge 640 "
                        f"--pc_nscale 3` then `fix_pc_cache_alignment.py`).")
                # val/test: enable runtime fallback, WARN once at dataset init.
                self._pc_runtime_fallback = True
                from loguru import logger
                logger.warning(
                    f"RoadSceneDataset ({self.dataset_name}, mode={self.mode}): "
                    f"PC cache dir missing ({self.ir_pc_dir} or "
                    f"{self.vis_pc_dir}); will compute PC on-the-fly per sample "
                    f"(slower, ~200-400 ms/img on CPU). For deployment speed, "
                    f"run MyScripts/precompute_pc_edges.py + "
                    f"fix_pc_cache_alignment.py to materialise cache once.")
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

        # Index source priority:
        #   1. `names` kwarg (DDP pre-sharded sub-list from data.py) -- bypasses
        #      file IO so each rank only iterates its 1/world_size shard
        #   2. `list_path` (default v0-v9 path) -- reads the full index file
        if names is not None:
            self.names = list(names)
        else:
            self.names = _read_index_file(list_path)
        if not self.names:
            raise RuntimeError(
                f"Empty RoadScene index: list_path={list_path}, "
                f"names={None if names is None else f'<pre-sharded list, len=0>'}")

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

        # R8: snapshot CLAHE-untouched raw for the runtime PC fallback path.
        # MyScripts/precompute_pc_edges.py reads raw straight from disk
        # (cv2.imread), so the cached PC was computed BEFORE CLAHE was ever
        # applied. To match the v10/v11 cache distribution byte-for-byte when
        # running PC on-the-fly, we MUST feed compute_pc_v11_runtime the raw
        # pre-CLAHE pixels. We snapshot only when use_edge_input=True (otherwise
        # the snapshot is dead weight) and only when CLAHE is actually applied
        # (otherwise ir_raw is already pre-CLAHE so the snapshot is a useless
        # copy). The two ~1 MB .copy() per pair cost <=1 ms each on a typical
        # 1024x768 grayscale, negligible vs the dataloader pipeline.
        ir_raw_for_pc = None
        vis_raw_for_pc = None
        if self.use_edge_input:
            if self.use_clahe_ir:
                ir_raw_for_pc = ir_raw.copy()
            else:
                ir_raw_for_pc = ir_raw
            if self.use_clahe_vis:
                vis_raw_for_pc = vis_raw.copy()
            else:
                vis_raw_for_pc = vis_raw

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
        #
        # R8 (runtime fallback): three states for each (ir_pc_raw, vis_pc_raw):
        #   1. self._pc_runtime_fallback=True: whole dir was absent at
        #      __init__ time, never touch disk; ir_pc_raw / vis_pc_raw stay
        #      None and the resize block below dispatches to
        #      compute_pc_v11_runtime.
        #   2. dir exists but a specific file is missing:
        #      - mode='train' -> raise (preserve v7 R1 hard-error behaviour;
        #        training MUST see the training distribution).
        #      - mode='val'/'test' -> first miss WARN, then null-out
        #        ir_pc_raw/vis_pc_raw so the resize block falls through to
        #        runtime compute (same path as state 1, just per-file).
        #   3. dir + file both exist -> read cache normally; keep the original
        #      target-shape sanity check (covers v10 unfixed cache case).
        ir_pc_raw = None
        vis_pc_raw = None
        if self.use_edge_input and not self._pc_runtime_fallback:
            pc_name = osp.splitext(name)[0] + ".png"
            ir_pc_path = osp.join(self.ir_pc_dir, pc_name)
            vis_pc_path = osp.join(self.vis_pc_dir, pc_name)
            ir_pc_raw = cv2.imread(str(ir_pc_path), cv2.IMREAD_GRAYSCALE)
            vis_pc_raw = cv2.imread(str(vis_pc_path), cv2.IMREAD_GRAYSCALE)
            if ir_pc_raw is None or vis_pc_raw is None:
                if self.mode == "train":
                    # Match v7 R1 hard-error semantics; training cannot fall
                    # back to runtime because that would silently shift the
                    # training distribution.
                    missing = []
                    if ir_pc_raw is None:
                        missing.append(f"IR PC: {ir_pc_path}")
                    if vis_pc_raw is None:
                        missing.append(f"VIS PC: {vis_pc_path}")
                    raise FileNotFoundError(
                        f"Failed to read PC cache in train mode ({', '.join(missing)}). "
                        f"Run MyScripts/precompute_pc_edges.bat to (re)generate.")
                # val/test: log once, then drop to runtime compute below.
                if not self._pc_missing_warned:
                    from loguru import logger
                    logger.warning(
                        f"RoadSceneDataset ({self.dataset_name}, mode={self.mode}): "
                        f"per-file PC cache miss starting at {name}; computing "
                        f"on-the-fly via src.utils.pc.compute_pc_v11_runtime. "
                        f"Subsequent misses will be silent.")
                    self._pc_missing_warned = True
                ir_pc_raw = None
                vis_pc_raw = None
            else:
                # Sanity: cache and raw must produce the SAME df-aligned target
                # shape under _resize_keep_aspect(., img_resize, df). Downstream
                # L383 `cv2.resize(ir_pc_raw, (w0_r, h0_r))` will force cache to
                # raw's target; if the implied caches' own target differs, the
                # cv2.resize introduces a non-uniform stretch -- which is fine for
                # SHAPE alignment (np.stack succeeds because cache lands at raw
                # target shape regardless), but produces subtle pixel
                # mis-correspondence vs. the cleaner "cache already at training
                # resolution" path. So we require strict target-shape equality.
                #
                # Compatibility:
                # - v0-v9 (M3FD/RoadScene): cache and raw are precomputed at the
                #   same resolution -> _target_shape returns identical values, passes.
                # - v10 (Megadepth_Syn) after MyScripts/fix_pc_cache_alignment.py:
                #   cache is pre-resized to dataset target -> trivially passes.
                # - v10 raw cache (no fix script run): caches at max_long_edge=640
                #   may round to different df-aligned target than raw at
                #   img_resize=480 (depends on raw aspect, see plan SS3.1) ->
                #   raises here; user needs to run fix script first.
                # - R8 runtime fallback: compute_pc_v11_runtime emits target_hw
                #   directly, so it can never trigger this check (we only reach
                #   here when ir_pc_raw/vis_pc_raw came from disk).
                target_raw_ir = _resize_target_shape(ir_raw.shape, self.img_resize, self.df)
                target_pc_ir = _resize_target_shape(ir_pc_raw.shape, self.img_resize, self.df)
                if target_raw_ir != target_pc_ir:
                    raise RuntimeError(
                        f"IR PC cache shape {ir_pc_raw.shape} resizes to {target_pc_ir} but "
                        f"raw IR shape {ir_raw.shape} resizes to {target_raw_ir} (img_resize="
                        f"{self.img_resize}, df={self.df}) for {name}. "
                        f"Run MyScripts/fix_pc_cache_alignment.py to pre-align cache to raw target, "
                        f"OR re-run precompute_pc_edges.py with --max_long_edge={self.img_resize} "
                        f"to produce caches directly at training resolution.")
                target_raw_vis = _resize_target_shape(vis_raw.shape, self.img_resize, self.df)
                target_pc_vis = _resize_target_shape(vis_pc_raw.shape, self.img_resize, self.df)
                if target_raw_vis != target_pc_vis:
                    raise RuntimeError(
                        f"VIS PC cache shape {vis_pc_raw.shape} resizes to {target_pc_vis} but "
                        f"raw VIS shape {vis_raw.shape} resizes to {target_raw_vis} for {name}. "
                        f"Run MyScripts/fix_pc_cache_alignment.py first.")

        # 1. Resize each image independently (long edge = img_resize, df-aligned).
        ir, h0_r, w0_r = _resize_keep_aspect(ir_raw, self.img_resize, self.df)
        vis, h1_r, w1_r = _resize_keep_aspect(vis_raw, self.img_resize, self.df)
        if self.use_edge_input:
            if ir_pc_raw is not None and vis_pc_raw is not None:
                # Cache-hit path (states 3 of the trio): reuse the same target
                # shape so PC and gray stay pixel-aligned. After
                # fix_pc_cache_alignment the cache shape already equals
                # (h0_r, w0_r) and this cv2.resize is a 1.0x no-op.
                ir_pc = cv2.resize(ir_pc_raw, (w0_r, h0_r))
                vis_pc = cv2.resize(vis_pc_raw, (w1_r, h1_r))
            else:
                # R8 runtime fallback (val/test only -- train mode raises in
                # the cache-read block above before reaching here). Either the
                # whole dir is missing (_pc_runtime_fallback=True) or a
                # specific file was missing. Either way, recompute PC on-the-fly
                # via the v11-equivalent path. We feed the PRE-CLAHE raw
                # (ir_raw_for_pc / vis_raw_for_pc snapshotted earlier) so the
                # phasecong input matches what precompute_pc_edges.py read from
                # disk during cache generation.
                from src.utils.pc import compute_pc_v11_runtime
                ir_pc = compute_pc_v11_runtime(
                    ir_raw_for_pc, target_hw=(h0_r, w0_r))
                vis_pc = compute_pc_v11_runtime(
                    vis_raw_for_pc, target_hw=(h1_r, w1_r))

        # 2. Optionally warp IR and/or VIS *before* padding so the Homography
        # is centred on the real image content rather than the padded canvas
        # centre. Single-sided mode (dual=False, v0-v10 default): only VIS is
        # warped, identical to historical behaviour. Dual mode (v11): sample
        # two independent H_ir/H_vis with the SAME shared probability gate
        # (one Bernoulli per pair so identity and aug pairs stay clearly
        # separated; never one-sided-only).
        H_ir = np.eye(3, dtype=np.float32)
        H_vis = np.eye(3, dtype=np.float32)
        if self.homography_aug and self.mode == "train":
            if np.random.rand() < self.homography_prob:
                H_vis = _random_homography(h1_r, w1_r, **self.homography_kwargs)
                if self.homography_dual:
                    H_ir = _random_homography(h0_r, w0_r, **self.homography_kwargs)

        # Apply the warps. Skip cv2.warpPerspective when H is identity to
        # avoid the (small) bilinear resample noise that would otherwise
        # break v0-v10 byte-identical behaviour for the no-aug code path.
        if not np.array_equal(H_vis, np.eye(3, dtype=np.float32)):
            vis = cv2.warpPerspective(vis, H_vis, (w1_r, h1_r),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            if self.use_edge_input:
                vis_pc = cv2.warpPerspective(vis_pc, H_vis, (w1_r, h1_r),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)
        if not np.array_equal(H_ir, np.eye(3, dtype=np.float32)):
            ir = cv2.warpPerspective(ir, H_ir, (w0_r, h0_r),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            if self.use_edge_input:
                ir_pc = cv2.warpPerspective(ir_pc, H_ir, (w0_r, h0_r),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

        # 3. Per-modality post-warp validity masks. A pixel is valid iff its
        # source pixel (under H^-1) was inside the original rectangle. We
        # compute this by warping an all-ones mask with the same H using
        # nearest-neighbour interpolation. ir_valid is all-True in single
        # mode (H_ir==I), so the mask0 intersection below is a no-op for
        # v0-v10 byte-identical behaviour.
        if not np.array_equal(H_vis, np.eye(3, dtype=np.float32)):
            ones1 = np.ones((h1_r, w1_r), dtype=np.uint8)
            vis_valid = cv2.warpPerspective(ones1, H_vis, (w1_r, h1_r),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0).astype(bool)
        else:
            vis_valid = np.ones((h1_r, w1_r), dtype=bool)
        if not np.array_equal(H_ir, np.eye(3, dtype=np.float32)):
            ones0 = np.ones((h0_r, w0_r), dtype=np.uint8)
            ir_valid = cv2.warpPerspective(ones0, H_ir, (w0_r, h0_r),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0).astype(bool)
        else:
            ir_valid = np.ones((h0_r, w0_r), dtype=bool)

        # 4. Zero-pad both images bottom-right to (pad_size, pad_size) and
        # build the canvas-level masks. mask0 / mask1 are intersected with
        # ir_valid / vis_valid so cells warped from outside their source
        # rectangle are also marked invalid.
        ir_pad, mask0 = pad_bottom_right(ir, self.pad_size, ret_mask=True)
        vis_pad, mask1 = pad_bottom_right(vis, self.pad_size, ret_mask=True)
        mask0[:h0_r, :w0_r] &= ir_valid
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
        # GT mapping: image0 (warped IR) px -> image1 (warped VIS) px in
        # the padded coordinate frame. Single-sided (H_ir=I): reduces to
        # H_0to1 = H_vis (v0-v10 byte-identical). Dual: H_vis @ inv(H_ir).
        H_0to1_np = (H_vis @ np.linalg.inv(H_ir)).astype(np.float32)
        H_0to1_np /= H_0to1_np[2, 2]
        H_t = torch.from_numpy(H_0to1_np)

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
