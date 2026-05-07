"""Offline pre-computation of Phase Congruency (PC) edge maps for IR/VIS images.

Designed for the v7_pcclahe cross-modal experiment (see
.cursor/skills/eloftr-cross-modal-experiments SKILL.md path F). PC edge maps
are modality-invariant geometric features (Kovesi 1999) that highlight where
multiple frequency-band sinusoidals are *in phase*, regardless of absolute
brightness. Same physical edge in IR and VIS lands at the same pixel
position in PC even though raw intensities differ wildly. This makes PC the
ideal "second input channel" for IR-VIS feature matching.

PC is computed *once* on the raw image and cached to disk. At training time
the dataloader reads the cached PC alongside the raw IR/VIS, then resize /
warp / pad both channels in lockstep. This avoids the ~2 s/image runtime
cost of phasecong inside the dataloader (would bottleneck training).

Layout (after running this script for both datasets):

    data/M3FD_Detection/
        Ir/        00000.png ..        # original (4200 frames)
        Vis/       00000.png ..        # original (4200 frames)
        Ir_pc/     00000.png ..        # PC edge map of Ir/, NEW
        Vis_pc/    00000.png ..        # PC edge map of Vis/, NEW

    data/RoadScene/
        cropinfrared/         FLIR_*.jpg
        crop_LR_visible/      FLIR_*.jpg
        cropinfrared_pc/      FLIR_*.png   # NEW
        crop_LR_visible_pc/   FLIR_*.png   # NEW

PC output is uint8 PNG (lossless, 8-bit suffices for the post-stretched M
moment). Filename stems are kept identical to the source so the dataset
class can load the PC cache by joining the same index entry with a
different sub-directory.

Usage examples:

    # M3FD (uses --dataset shortcut)
    python MyScripts/precompute_pc_edges.py --dataset M3FD

    # RoadScene
    python MyScripts/precompute_pc_edges.py --dataset RoadScene

    # Both at once
    python MyScripts/precompute_pc_edges.py --dataset M3FD RoadScene

    # Custom dataset (explicit args)
    python MyScripts/precompute_pc_edges.py \\
        --root data/MyDataset --ir_subdir IR --vis_subdir VIS

    # Multiprocessing (default: cpu_count - 2, max 8)
    python MyScripts/precompute_pc_edges.py --dataset M3FD --workers 6

    # Force re-compute (default: skip already-existing outputs)
    python MyScripts/precompute_pc_edges.py --dataset M3FD --overwrite

Approximate runtime (RTX 5070 Ti host, single-image phasecong ~2 s on
1024x768 with nscale=4, norient=6, no pyfftw acceleration):

    M3FD     8400 images,  8 workers -> ~35 min
    RoadScene 444 images,  8 workers -> ~2 min

Dependencies (already in conda env eff_loftr after `pip install phasepack`):
    phasepack >= 1.5  (https://github.com/alimuldal/phasepack)
    opencv-python
    numpy
    tqdm
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# phasepack is an optional dependency; fail loudly with a hint if missing
# rather than silently falling back to Sobel (would change the published
# v7 paper story).
try:
    from phasepack import phasecong
except ImportError as exc:  # pragma: no cover - install-time guidance
    sys.stderr.write(
        "ERROR: phasepack is not installed. Install via:\n"
        "    pip install phasepack\n"
        "Optional speed-up (does not always wire up cleanly with phasepack 1.5):\n"
        "    pip install pyfftw\n"
    )
    raise


# Suppress phasepack's "pyfftw not imported, falling back to fftpack" warning
# emitted on every worker spawn -- it is informational and not actionable.
warnings.filterwarnings("ignore", category=UserWarning, module="phasepack.tools")


# Dataset shortcuts: layout matches what configs/data/m3fd_trainval.py and
# configs/data/roadscene_trainval.py declare so this script and those
# configs stay in sync (mirroring the per-dataset PC sub-dir override
# established in the v7 plan).
DATASET_SHORTCUTS = {
    "M3FD": dict(
        root="data/M3FD_Detection",
        ir_subdir="Ir",
        vis_subdir="Vis",
        ir_pc_subdir="Ir_pc",
        vis_pc_subdir="Vis_pc",
        ext=".png",
    ),
    "RoadScene": dict(
        root="data/RoadScene",
        ir_subdir="cropinfrared",
        vis_subdir="crop_LR_visible",
        ir_pc_subdir="cropinfrared_pc",
        vis_pc_subdir="crop_LR_visible_pc",
        ext=".jpg",
    ),
    # v10 Megadepth_Syn (scene-disjoint 175/10/10 train-only). M3FD-style
    # source/_pc peer-level layout: train/{infrared, infrared_pc, phoenix,
    # phoenix_pc} all share the deep MegaDepth_v1 nested structure
    # (S6/zl548/MegaDepth_v1/<scene>/<denseN>/imgs/<stem>). Source dirs are
    # symlinks into /data/.../train/{infrared,phoenix}; _pc dirs are real
    # repo-internal directories where this script writes PC PNGs (mirroring
    # nested structure thanks to --recursive). Use --recursive +
    # --max_long_edge 640 + --pc_nscale 3 for L1+L2+L3 acceleration (~46 min
    # vs ~14h baseline).
    "Megadepth_Syn": dict(
        root="data/Megadepth_Syn",
        ir_subdir="train/infrared/phoenix/S6/zl548/MegaDepth_v1",
        vis_subdir="train/phoenix/S6/zl548/MegaDepth_v1",
        ir_pc_subdir="train/infrared_pc/S6/zl548/MegaDepth_v1",
        vis_pc_subdir="train/phoenix_pc/S6/zl548/MegaDepth_v1",
        ext=".jpg",
    ),
}


# Phase-congruency hyperparameters. These match the v7 plan defaults; bumping
# nscale to 5 increases edge sharpness slightly at ~25% extra runtime, while
# dropping to 3 nearly halves runtime at noticeable edge quality loss.
# v10 overrides via --pc_nscale 3 for L2 acceleration.
PC_NSCALE = 4
PC_NORIENT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        choices=sorted(DATASET_SHORTCUTS.keys()),
        default=None,
        help=(
            "Dataset shortcut(s); when set, --root / --*_subdir are ignored. "
            "Pass multiple values to process several datasets in one run, e.g. "
            "--dataset M3FD RoadScene."
        ),
    )
    parser.add_argument("--root", type=str, default=None,
                        help="Dataset root (used when --dataset is omitted).")
    parser.add_argument("--ir_subdir", type=str, default=None,
                        help="IR sub-directory inside --root.")
    parser.add_argument("--vis_subdir", type=str, default=None,
                        help="VIS sub-directory inside --root.")
    parser.add_argument("--ir_pc_subdir", type=str, default=None,
                        help="Output IR PC sub-directory; defaults to <ir_subdir>_pc.")
    parser.add_argument("--vis_pc_subdir", type=str, default=None,
                        help="Output VIS PC sub-directory; defaults to <vis_subdir>_pc.")
    parser.add_argument("--ext", type=str, default=None,
                        help="Source image extension (lowercase, e.g. .png / .jpg). "
                             "Output is always written as .png.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel processes. Default min(cpu_count-2, 8). "
                             "Set 1 to debug serially.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute even if the PC output already exists "
                             "(default: skip existing for idempotent resume).")
    # v10 acceleration knobs (default = v0-v9 byte-identical behaviour)
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan input directory for source images "
                             "and mirror nested directory structure into the PC "
                             "output directory. Required for datasets with nested "
                             "scene/dense/imgs layout (e.g. Megadepth_Syn). "
                             "Default False = flat single-folder scan (M3FD/RoadScene).")
    parser.add_argument("--max_long_edge", type=int, default=0,
                        help="If > 0, downscale source images so max(H, W) == "
                             "max_long_edge before phasecong. Speeds up PC by "
                             "~7x at long_edge=640 (vs original ~1600). "
                             "Must be >= the training-time ROAD_IMG_RESIZE "
                             "(typically 480) to avoid quality loss from upsampling "
                             "later. Default 0 = no downscale (v0-v9 behaviour).")
    parser.add_argument("--df", type=int, default=32,
                        help="Divisibility factor for max_long_edge resize "
                             "(matches training-time df, default 32). Output "
                             "side length is rounded to multiple of df.")
    parser.add_argument("--pc_nscale", type=int, default=PC_NSCALE,
                        help=f"phasecong 'nscale' parameter. Default {PC_NSCALE} "
                             f"(v0-v9 baseline, matches v7 paper). Drop to 3 for "
                             f"~2x speedup at slight edge sharpness loss; raise "
                             f"to 5 for sharper edges at +25% runtime.")
    return parser.parse_args()


def _resolve_jobs(args: argparse.Namespace) -> List[dict]:
    """Convert CLI args into a list of per-dataset job dicts.

    Each job has: name, root, ir_subdir, vis_subdir, ir_pc_subdir,
    vis_pc_subdir, ext.
    """
    jobs: List[dict] = []
    if args.dataset:
        for name in args.dataset:
            base = dict(DATASET_SHORTCUTS[name])
            base["name"] = name
            # Allow CLI overrides on top of the shortcut (rare).
            for k in ("ir_pc_subdir", "vis_pc_subdir"):
                v = getattr(args, k)
                if v:
                    base[k] = v
            jobs.append(base)
        return jobs

    # No --dataset: build one job from explicit args.
    if not args.root or not args.ir_subdir or not args.vis_subdir:
        raise SystemExit(
            "Either --dataset must be set, or all of "
            "--root / --ir_subdir / --vis_subdir must be provided."
        )
    ir_pc = args.ir_pc_subdir or f"{args.ir_subdir}_pc"
    vis_pc = args.vis_pc_subdir or f"{args.vis_subdir}_pc"
    ext = args.ext or ".png"
    jobs.append(dict(
        name=Path(args.root).name,
        root=args.root,
        ir_subdir=args.ir_subdir,
        vis_subdir=args.vis_subdir,
        ir_pc_subdir=ir_pc,
        vis_pc_subdir=vis_pc,
        ext=ext.lower(),
    ))
    return jobs


def _resolve_workers(arg_value: Optional[int]) -> int:
    if arg_value is not None and arg_value >= 1:
        return int(arg_value)
    cpu = os.cpu_count() or 4
    # cpu_count - 2 leaves 2 cores free for OS / Cursor / shell, capped at 8
    # because phasecong is FFT-bound and saturates ~6-8 workers; more workers
    # mainly add memory pressure (each worker holds its own FFT plan).
    return max(1, min(cpu - 2, 8))


def _stretch_to_uint8(m: np.ndarray) -> np.ndarray:
    """Linear-stretch a float PC map to uint8 [0, 255].

    The raw M moment from phasecong is in [0, 1] but typically peaks at
    0.3-0.7 on natural images, so a per-image normalisation gives much
    better visual contrast than fixed *255. Worst-case all-zero map (e.g.
    pad / black image) is handled by the eps clamp.
    """
    eps = 1e-8
    m_max = float(m.max())
    if m_max < eps:
        return np.zeros(m.shape, dtype=np.uint8)
    m_norm = m / m_max
    return (np.clip(m_norm, 0.0, 1.0) * 255.0).astype(np.uint8)


def _process_one(task: Tuple[str, str, int, int, int]) -> Tuple[str, Optional[str]]:
    """Worker: read one image, optionally pre-resize, compute PC, write to disk.

    task = (src_path, dst_path, max_long_edge, df, pc_nscale)

    max_long_edge > 0 triggers pre-PC downscale to long edge = max_long_edge
    (df-aligned), giving ~7x speedup on 1600 -> 640. Must be >= training
    ROAD_IMG_RESIZE so downstream `cv2.resize(ir_pc, (w0_r, h0_r))` never
    upscales (which would lose quality vs computing PC at higher res).

    Returns (src_path, error_message). error is None on success.
    """
    src, dst, max_long_edge, df, pc_nscale = task
    try:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (src, "cv2.imread returned None")
        # L3 pre-resize: shrink large images before phasecong (FFT cost ~O(N^2 log N))
        if max_long_edge > 0:
            h, w = img.shape
            scale = float(max_long_edge) / max(h, w)
            if scale < 1.0:  # only downscale, never upscale
                new_h = max(df, int(round(h * scale) // df) * df)
                new_w = max(df, int(round(w * scale) // df) * df)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_f = img.astype(np.float64) / 255.0
        # phasecong returns 7-tuple (M, m, ori, ft, PC, EO, T). M is the
        # max-moment-of-phase-congruency, the standard "edge strength" map.
        ret = phasecong(img_f, nscale=pc_nscale, norient=PC_NORIENT)
        m_uint8 = _stretch_to_uint8(ret[0])
        # Always write PNG regardless of source extension (PNG is lossless and
        # supports the full uint8 range without JPEG quantisation noise).
        ok = cv2.imwrite(dst, m_uint8)
        if not ok:
            return (src, f"cv2.imwrite returned False for {dst}")
        return (src, None)
    except Exception as exc:  # pragma: no cover - defensive
        return (src, f"{type(exc).__name__}: {exc}")


def _build_tasks(in_dir: Path, out_dir: Path, ext: str,
                 overwrite: bool, recursive: bool,
                 max_long_edge: int, df: int,
                 pc_nscale: int) -> Tuple[List[Tuple[str, str, int, int, int]], int]:
    """Walk ``in_dir`` and pair each image with its PC output path. Returns
    (tasks, n_skipped). n_skipped counts already-existing outputs.

    recursive=True scans nested subdirectories and mirrors the directory
    structure to out_dir (e.g. in_dir/foo/bar/baz.jpg -> out_dir/foo/bar/baz.png).
    Required for datasets like Megadepth_Syn with scene/dense/imgs nesting.
    Default False = flat scan (M3FD/RoadScene single-folder layout).

    max_long_edge / df / pc_nscale are passed through to each task tuple so
    workers know how to process the image (L2 + L3 acceleration knobs).
    """
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, int, int, int]] = []
    skipped = 0
    if recursive:
        # rglob preserves nested structure; sort for determinism
        src_files = sorted(p for p in in_dir.rglob(f"*{ext}")
                           if p.is_file() and p.suffix.lower() == ext)
    else:
        src_files = sorted(p for p in in_dir.iterdir()
                           if p.is_file() and p.suffix.lower() == ext)
    for src in src_files:
        if recursive:
            # mirror nested structure under out_dir, swapping ext to .png
            rel = src.relative_to(in_dir).with_suffix(".png")
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst = out_dir / (src.stem + ".png")
        if dst.is_file() and not overwrite:
            skipped += 1
            continue
        tasks.append((str(src), str(dst), max_long_edge, df, pc_nscale))
    return tasks, skipped


def _process_split(name: str, in_dir: Path, out_dir: Path, ext: str,
                   workers: int, overwrite: bool, recursive: bool,
                   max_long_edge: int, df: int, pc_nscale: int) -> None:
    tasks, skipped = _build_tasks(in_dir, out_dir, ext, overwrite, recursive,
                                  max_long_edge, df, pc_nscale)
    total = len(tasks) + skipped
    if total == 0:
        print(f"  [{name}] no `*{ext}` files in {in_dir}")
        return
    print(f"  [{name}] total={total}  skip={skipped}  todo={len(tasks)}  "
          f"workers={workers}")
    if not tasks:
        return

    failures: List[Tuple[str, str]] = []
    t0 = time.time()
    if workers == 1:
        # Serial debug path: easier to read tracebacks
        for task in tqdm(tasks, desc=f"{name}", unit="img"):
            src, err = _process_one(task)
            if err:
                failures.append((src, err))
    else:
        # spawn keeps Windows / forked phasepack state clean; pyfftw plan
        # cache cannot be shared between forks anyway.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for src, err in tqdm(pool.imap_unordered(_process_one, tasks, chunksize=8),
                                 total=len(tasks), desc=f"{name}", unit="img"):
                if err:
                    failures.append((src, err))

    dt = time.time() - t0
    print(f"  [{name}] done in {dt/60:.1f} min "
          f"(avg {dt/max(len(tasks), 1):.2f} s/img)")
    if failures:
        print(f"  [{name}] FAILURES ({len(failures)}):")
        for src, err in failures[:10]:
            print(f"    {src}: {err}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")


def main() -> None:
    args = parse_args()
    jobs = _resolve_jobs(args)
    workers = _resolve_workers(args.workers)

    print(f"PC pre-computation: nscale={args.pc_nscale}, norient={PC_NORIENT}")
    if args.max_long_edge > 0:
        print(f"Pre-resize: max_long_edge={args.max_long_edge} (df={args.df}) -- "
              f"L3 acceleration enabled")
    if args.recursive:
        print(f"Scan mode: recursive (nested directory structure mirrored to output)")
    print(f"Workers: {workers}  |  Overwrite: {args.overwrite}")

    grand_t0 = time.time()
    for job in jobs:
        root = Path(job["root"])
        if not root.is_dir():
            raise SystemExit(f"Dataset root not found: {root}")
        print(f"\n=== {job['name']} ({root}) ===")

        ir_in = root / job["ir_subdir"]
        vis_in = root / job["vis_subdir"]
        ir_out = root / job["ir_pc_subdir"]
        vis_out = root / job["vis_pc_subdir"]

        _process_split(f"{job['name']}/IR",
                       ir_in, ir_out, job["ext"], workers, args.overwrite,
                       args.recursive, args.max_long_edge, args.df, args.pc_nscale)
        _process_split(f"{job['name']}/VIS",
                       vis_in, vis_out, job["ext"], workers, args.overwrite,
                       args.recursive, args.max_long_edge, args.df, args.pc_nscale)

    grand_dt = time.time() - grand_t0
    print(f"\nAll done in {grand_dt/60:.1f} min total")


if __name__ == "__main__":
    main()
