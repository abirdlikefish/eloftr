"""Phase Congruency (PC) edge map helpers shared by the dataset's runtime
fallback path and the offline cache pipeline (``MyScripts/precompute_pc_edges.py``
+ ``MyScripts/fix_pc_cache_alignment.py``).

The v10/v11 cache generation chain is:

    raw image
      |
      v
    cv2.resize(INTER_AREA) -> long_edge=640, df=32 aligned        # precompute step 1
      |
      v
    phasecong(nscale=3, norient=6)                                # PC spectral analysis (on 640)
      |
      v
    stretch_to_uint8 (per-image max norm to [0, 255])              # precompute step 3
      |
      v
    cv2.resize(INTER_AREA) -> _resize_keep_aspect target shape    # fix_pc_cache_alignment step
                              (480 long-edge df=32, matches
                               RoadSceneDataset training resolution)

This module exposes:

* :func:`stretch_to_uint8` - mirror of ``precompute_pc_edges.py:_stretch_to_uint8``
  so dataset and cache scripts share a single implementation.
* :func:`compute_pc_v11_runtime` - one-shot replacement for the cache + fix
  chain. Given a raw CLAHE-untouched grayscale image and a target HW, it
  returns a uint8 PC map whose values are (modulo phasepack floating-point
  reproducibility noise) byte-identical to what the offline pipeline would
  have written to disk and then resized to the same target.

The runtime path is ONLY used by ``RoadSceneDataset`` in val/test mode when
PC cache is absent (either whole dir or per-file missing). Training mode hard
errors on missing cache to protect the v7+/v10/v11 training distribution; see
the dataset's R8 guard.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# v11 training distribution lock. These constants mirror the CLI flags used to
# generate the v10/v11 PC cache:
#
#     python MyScripts/precompute_pc_edges.py --dataset Megadepth_Syn \
#         --recursive --max_long_edge 640 --pc_nscale 3 --workers 24
#
# Do NOT change these unless you also update v10/v11 training cfg + retrain.
# A future v12 ablation that varies PC hyperparams should add cfg fields and
# read them at runtime rather than touching this module.
PC_V11_LONG_EDGE = 640
PC_V11_DF = 32
PC_V11_NSCALE = 3
PC_V11_NORIENT = 6


def stretch_to_uint8(m: np.ndarray) -> np.ndarray:
    """Linear-stretch a float PC map to uint8 [0, 255].

    Equivalent to ``MyScripts/precompute_pc_edges.py::_stretch_to_uint8``.
    The raw M moment from ``phasepack.phasecong`` lives in [0, 1] but in
    practice peaks at 0.3-0.7 on natural images, so a per-image max
    normalisation gives much better visual contrast than a fixed *255.
    Worst-case all-zero map (e.g. pad / black image) is handled by the eps
    clamp.
    """
    eps = 1e-8
    m_max = float(m.max())
    if m_max < eps:
        return np.zeros(m.shape, dtype=np.uint8)
    m_norm = m / m_max
    return (np.clip(m_norm, 0.0, 1.0) * 255.0).astype(np.uint8)


def compute_pc_v11_runtime(img_u8: np.ndarray,
                           target_hw: Tuple[int, int]) -> np.ndarray:
    """Compute PC edge map on-the-fly, byte-equivalent to v10/v11 disk cache.

    Inputs
    ------
    img_u8 : (H, W) uint8 grayscale, MUST be CLAHE-untouched raw read straight
        from disk via ``cv2.imread(..., IMREAD_GRAYSCALE)``. The dataset takes
        a ``.copy()`` snapshot before its CLAHE block to honour this contract;
        passing the post-CLAHE tensor instead would silently shift the runtime
        PC distribution away from the training distribution because the v10/v11
        cache was generated from raw disk reads (precompute reads raw, not
        CLAHE-applied) -- see ``MyScripts/precompute_pc_edges.py:340``.
    target_hw : (target_h, target_w) tuple. Must equal what
        ``src.datasets.roadscene._resize_target_shape(img_u8.shape,
        img_resize, df)`` would produce for the dataset's configured
        img_resize / df. The runtime path resizes the final uint8 map to
        this shape with INTER_AREA so it lands at the same resolution as
        a fix_pc_cache_alignment-fixed cache; dataset L431
        ``cv2.resize(ir_pc_raw, (w0_r, h0_r))`` then becomes a 1.0x no-op.

    Pipeline (all four steps mirror the offline cache + fix chain):

    1. INTER_AREA downscale to ``PC_V11_LONG_EDGE=640`` df-aligned. Skipped
       when ``max(H, W) <= 640`` (matches precompute's ``if scale < 1.0``
       early-exit).
    2. ``phasecong(nscale=3, norient=6)`` on the 640-long-edge float64 image.
       This is where the PC spectral analysis happens; the resolution at
       which Gabor scales are evaluated must match the training distribution,
       which is why we go through 640 even when the eventual target is 480.
    3. ``stretch_to_uint8`` (per-image max norm).
    4. INTER_AREA resize to ``target_hw`` (matches
       ``fix_pc_cache_alignment.py:_process_one`` line 217).
    """
    # ------------------------------------------------------------------ #
    # Local import: phasepack is heavyweight (numpy + scipy + optional
    # pyfftw at import time) and ~2 s cold-start per process. Importing it
    # at module level would slow down every dataloader worker spawn even
    # when use_edge_input=False or PC cache is fully present.
    # ------------------------------------------------------------------ #
    from phasepack import phasecong

    # Step 1: pre-PC downscale to df-aligned 640-long-edge.
    src = img_u8
    h, w = src.shape
    scale = float(PC_V11_LONG_EDGE) / max(h, w)
    if scale < 1.0:
        new_h = max(PC_V11_DF, int(round(h * scale) // PC_V11_DF) * PC_V11_DF)
        new_w = max(PC_V11_DF, int(round(w * scale) // PC_V11_DF) * PC_V11_DF)
        src = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 2: phasecong on float64 [0, 1].
    src_f = src.astype(np.float64) / 255.0
    m = phasecong(src_f, nscale=PC_V11_NSCALE, norient=PC_V11_NORIENT)[0]

    # Step 3: per-image max-norm to uint8.
    m_u8 = stretch_to_uint8(m)

    # Step 4: INTER_AREA resize to dataset target.
    if m_u8.shape != tuple(target_hw):
        m_u8 = cv2.resize(m_u8, (target_hw[1], target_hw[0]),
                          interpolation=cv2.INTER_AREA)
    return m_u8
