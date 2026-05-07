"""One-shot patch: re-resize PC caches to dataset's _resize_keep_aspect target.

Problem (v10 plan SS3.1 + dataset error 2026-05-07):
  precompute_pc_edges.py with --max_long_edge 640 --df 32 produces caches
  whose aspect ratio differs from the source raw image because df-aligned
  resize one-sidedly truncates the shorter dim. Example:
    raw (1090, 696)  aspect 1.5661
    cache (640, 384) aspect 1.6667  (W rounded 409 -> 384 by df=32)
  The dataset's L383 ``cv2.resize(ir_pc_raw, (w0_r, h0_r))`` later forces
  cache to raw target shape (480, 288) -- but for some raw aspects, cache
  and raw round to DIFFERENT df-aligned targets (e.g. raw aspect 1.45
  -> raw target (480, 320), cache target (480, 288)), making
  np.stack([ir_pad, ir_pc_pad]) fail because shapes mismatch.

Fix:
  Pre-resize every PC cache to the EXACT shape that ``_resize_keep_aspect(
  raw, ROAD_IMG_RESIZE, ROAD_DF)`` would produce. After fix, dataset's
  L383 cv2.resize becomes a 1.0x no-op (cache is already at training
  resolution), and stack always succeeds because cache and raw both go
  through identical _resize_keep_aspect on raw to compute the target.

Why this is alignment-correct:
  Every (480, 288) tensor pixel maps to the same physical raw region
  whether the path is raw->(480,288) directly or raw->(640,384)->(480,288).
  The composite scale factors are identical: H = 1090/480 = 2.27,
  W = 696/288 = 2.42 in both paths. PC quality loses ~0-6% edge sharpness
  due to the extra resize step but spatial alignment is exact.

Why we don't re-run precompute:
  Re-running phasecong on 258K images takes ~30-46 min. This fix script
  reads raw (~5ms) + reads cache (~5ms) + cv2.resize (~5ms) + writes cache
  (~5ms) per image, ~3 min total with 24 workers. ~10x speedup.

Usage::

    # dry-run: scan and report mismatch counts without writing
    python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn --dry-run

    # actually fix (writes inplace, idempotent: re-running is a no-op)
    python MyScripts/fix_pc_cache_alignment.py --dataset Megadepth_Syn

    # custom dataset / paths / training resolution
    python MyScripts/fix_pc_cache_alignment.py \\
        --root data/Megadepth_Syn \\
        --raw_subdir train/infrared/phoenix/S6/zl548/MegaDepth_v1 \\
        --pc_subdir  train/infrared_pc/S6/zl548/MegaDepth_v1 \\
        --raw_ext .jpg --pc_ext .png \\
        --img_resize 480 --df 32 --workers 24

After fix:
  - dataset roadscene.py target-shape check passes (cache.shape == target_shape)
  - dataset L383 cv2.resize(cache, (w0_r, h0_r)) is 1.0x no-op
  - np.stack([ir_pad, ir_pc_pad]) always succeeds
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


# Mirror the dataset's DATASET_SHORTCUTS so the same --dataset name works
# the same way as in precompute_pc_edges.py. Megadepth_Syn entry mirrors
# the cfg in configs/data/megadepth_syn_trainval.py.
DATASET_SHORTCUTS = {
    "M3FD": dict(
        root="data/M3FD_Detection",
        raw_subdirs=["Ir", "Vis"],
        pc_subdirs=["Ir_pc", "Vis_pc"],
        raw_ext=".png",
        pc_ext=".png",
        recursive=False,
    ),
    "RoadScene": dict(
        root="data/RoadScene",
        raw_subdirs=["cropinfrared", "crop_LR_visible"],
        pc_subdirs=["cropinfrared_pc", "crop_LR_visible_pc"],
        raw_ext=".jpg",
        pc_ext=".png",
        recursive=False,
    ),
    "Megadepth_Syn": dict(
        root="data/Megadepth_Syn",
        raw_subdirs=["train/infrared/phoenix/S6/zl548/MegaDepth_v1",
                     "train/phoenix/S6/zl548/MegaDepth_v1"],
        pc_subdirs=["train/infrared_pc/S6/zl548/MegaDepth_v1",
                    "train/phoenix_pc/S6/zl548/MegaDepth_v1"],
        raw_ext=".jpg",
        pc_ext=".png",
        recursive=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=sorted(DATASET_SHORTCUTS.keys()),
        help="Dataset shortcut. When set, overrides --root/--*_subdir/--*_ext/--recursive.",
    )
    parser.add_argument("--root", type=str, default=None,
                        help="Dataset root (used when --dataset is omitted).")
    parser.add_argument("--raw_subdir", type=str, action="append", default=None,
                        help="Raw image subdir (relative to --root). Repeat for IR + VIS, "
                             "matching --pc_subdir order. e.g. --raw_subdir Ir --raw_subdir Vis.")
    parser.add_argument("--pc_subdir", type=str, action="append", default=None,
                        help="PC cache subdir (relative to --root). Same order as --raw_subdir.")
    parser.add_argument("--raw_ext", type=str, default=".jpg",
                        help="Raw image file extension (lowercase).")
    parser.add_argument("--pc_ext", type=str, default=".png",
                        help="PC cache file extension (lowercase, default .png).")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan raw_subdir for nested layouts (Megadepth_Syn). "
                             "If --dataset is set, this is auto-configured.")
    parser.add_argument("--img_resize", type=int, default=480,
                        help="Training-time ROAD_IMG_RESIZE used by RoadSceneDataset's "
                             "_resize_keep_aspect. Cache will be resized to the resulting "
                             "df-aligned target shape.")
    parser.add_argument("--df", type=int, default=32,
                        help="Training-time ROAD_DF (divisibility factor). Same as the "
                             "df passed to _resize_keep_aspect at training time.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel processes. Default min(cpu_count-2, 24). "
                             "Set 1 to debug serially.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report mismatches without writing back.")
    return parser.parse_args()


def _resolve_jobs(args: argparse.Namespace) -> List[dict]:
    """Build a list of (raw_dir, pc_dir, raw_ext, pc_ext, recursive) jobs."""
    if args.dataset:
        cfg = DATASET_SHORTCUTS[args.dataset]
        root = Path(cfg["root"])
        raw_subdirs = cfg["raw_subdirs"]
        pc_subdirs = cfg["pc_subdirs"]
        raw_ext = cfg["raw_ext"]
        pc_ext = cfg["pc_ext"]
        recursive = cfg["recursive"]
        if len(raw_subdirs) != len(pc_subdirs):
            raise SystemExit(
                f"DATASET_SHORTCUTS[{args.dataset}] raw_subdirs/pc_subdirs length mismatch.")
        return [dict(name=f"{args.dataset}/{rs}", raw_dir=root / rs, pc_dir=root / ps,
                     raw_ext=raw_ext, pc_ext=pc_ext, recursive=recursive)
                for rs, ps in zip(raw_subdirs, pc_subdirs)]

    if not args.root or not args.raw_subdir or not args.pc_subdir:
        raise SystemExit("Either --dataset must be set, or --root + --raw_subdir + --pc_subdir.")
    if len(args.raw_subdir) != len(args.pc_subdir):
        raise SystemExit("--raw_subdir count must equal --pc_subdir count "
                         "(use --raw_subdir twice for IR + VIS).")
    root = Path(args.root)
    return [dict(name=f"{root.name}/{rs}", raw_dir=root / rs, pc_dir=root / ps,
                 raw_ext=args.raw_ext, pc_ext=args.pc_ext, recursive=args.recursive)
            for rs, ps in zip(args.raw_subdir, args.pc_subdir)]


def _resolve_workers(arg_value: Optional[int]) -> int:
    if arg_value is not None and arg_value >= 1:
        return int(arg_value)
    cpu = os.cpu_count() or 4
    # cv2 io + resize is light on memory, can push higher than phasecong (~8)
    return max(1, min(cpu - 2, 24))


def _target_shape(raw_shape: Tuple[int, int], long_edge: int, df: int) -> Tuple[int, int]:
    """Mirror src/datasets/roadscene.py::_resize_keep_aspect target shape exactly.

    Returns (new_h, new_w) that the dataset would produce when calling
    _resize_keep_aspect(raw, long_edge, df).
    """
    h, w = raw_shape
    scale = float(long_edge) / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    new_h = max(df, (new_h // df) * df)
    new_w = max(df, (new_w // df) * df)
    return (new_h, new_w)


def _process_one(task: Tuple[str, str, str, str, int, int, bool]) -> Tuple[str, str, str]:
    """Worker: read raw, compute target shape, resize cache if needed.

    task = (raw_path, pc_path, raw_ext, pc_ext, img_resize, df, dry_run)

    Returns (raw_path, status, detail) where status in
    {"ok-noop", "fixed", "skip-no-cache", "skip-no-raw", "error"}.
    """
    raw_path, pc_path, raw_ext, pc_ext, img_resize, df, dry_run = task
    try:
        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return (raw_path, "skip-no-raw", f"cv2.imread None for raw: {raw_path}")
        cache = cv2.imread(pc_path, cv2.IMREAD_GRAYSCALE)
        if cache is None:
            return (raw_path, "skip-no-cache", f"missing PC cache: {pc_path}")

        target = _target_shape(raw.shape, img_resize, df)  # (new_h, new_w)
        if cache.shape == target:
            return (raw_path, "ok-noop", "")

        if dry_run:
            return (raw_path, "fixed",
                    f"would resize {cache.shape} -> {target} (raw={raw.shape})")

        # cv2.resize takes (W, H) order
        cache_fixed = cv2.resize(cache, (target[1], target[0]),
                                 interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(pc_path, cache_fixed)
        if not ok:
            return (raw_path, "error", f"cv2.imwrite returned False for {pc_path}")
        return (raw_path, "fixed", f"resized {cache.shape} -> {target}")
    except Exception as exc:  # pragma: no cover - defensive
        return (raw_path, "error", f"{type(exc).__name__}: {exc}")


def _build_tasks(raw_dir: Path, pc_dir: Path, raw_ext: str, pc_ext: str,
                 recursive: bool, img_resize: int, df: int,
                 dry_run: bool) -> List[Tuple[str, str, str, str, int, int, bool]]:
    """Walk raw_dir, build (raw_path, pc_path, ...) task tuples."""
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw directory not found: {raw_dir}")
    if not pc_dir.is_dir():
        raise SystemExit(f"PC cache directory not found: {pc_dir}")

    raw_ext_lc = raw_ext.lower()
    if recursive:
        raw_files = sorted(p for p in raw_dir.rglob(f"*{raw_ext_lc}")
                           if p.is_file() and p.suffix.lower() == raw_ext_lc)
    else:
        raw_files = sorted(p for p in raw_dir.iterdir()
                           if p.is_file() and p.suffix.lower() == raw_ext_lc)
    if not raw_files:
        raise SystemExit(f"No `*{raw_ext_lc}` raw files under {raw_dir}")

    tasks = []
    for raw_path in raw_files:
        rel = raw_path.relative_to(raw_dir)
        pc_path = pc_dir / rel.with_suffix(pc_ext)
        tasks.append((str(raw_path), str(pc_path), raw_ext, pc_ext,
                      img_resize, df, dry_run))
    return tasks


def _process_split(name: str, tasks: List, workers: int) -> dict:
    """Run a pool over tasks; aggregate counts."""
    counts = {"ok-noop": 0, "fixed": 0, "skip-no-cache": 0, "skip-no-raw": 0, "error": 0}
    errors: List[Tuple[str, str]] = []
    fixed_examples: List[Tuple[str, str]] = []
    no_cache_examples: List[Tuple[str, str]] = []

    print(f"\n[{name}] {len(tasks)} raw files -- workers={workers}")
    t0 = time.time()
    if workers == 1:
        results_iter = (_process_one(t) for t in tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        results_iter = pool.imap_unordered(_process_one, tasks, chunksize=64)

    try:
        for i, (raw_path, status, detail) in enumerate(results_iter, 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "fixed" and len(fixed_examples) < 5:
                fixed_examples.append((raw_path, detail))
            elif status == "skip-no-cache" and len(no_cache_examples) < 5:
                no_cache_examples.append((raw_path, detail))
            elif status == "error":
                errors.append((raw_path, detail))
            if i % 5000 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                print(f"  [{name}] {i}/{len(tasks)} processed "
                      f"({elapsed:.1f}s, {rate:.0f}/s)")
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    dt = time.time() - t0
    print(f"  [{name}] done in {dt:.1f}s ({len(tasks)/max(dt,1e-3):.0f}/s):")
    for k, v in counts.items():
        if v:
            print(f"    {k}: {v}")
    if fixed_examples:
        verb = "would resize" if "would resize" in fixed_examples[0][1] else "resized"
        print(f"    fixed examples (first 5):")
        for raw_path, detail in fixed_examples:
            print(f"      {Path(raw_path).name}: {detail}")
    if no_cache_examples:
        print(f"    skip-no-cache examples (first 5):")
        for raw_path, detail in no_cache_examples:
            print(f"      {Path(raw_path).name}: {detail}")
    if errors:
        print(f"    ERRORS ({len(errors)}, first 5):")
        for raw_path, detail in errors[:5]:
            print(f"      {raw_path}: {detail}")
    return counts


def main() -> None:
    args = parse_args()
    jobs = _resolve_jobs(args)
    workers = _resolve_workers(args.workers)

    print(f"PC cache alignment fix")
    print(f"  Target: cache.shape == _resize_keep_aspect(raw, "
          f"img_resize={args.img_resize}, df={args.df}).shape")
    print(f"  Mode:   {'DRY-RUN (no writes)' if args.dry_run else 'WRITE BACK INPLACE'}")
    print(f"  Workers: {workers}")
    print(f"  Jobs:   {len(jobs)} (split: {[j['name'] for j in jobs]})")

    grand_t0 = time.time()
    grand_counts = {"ok-noop": 0, "fixed": 0, "skip-no-cache": 0, "skip-no-raw": 0, "error": 0}
    for job in jobs:
        tasks = _build_tasks(job["raw_dir"], job["pc_dir"],
                             job["raw_ext"], job["pc_ext"],
                             job["recursive"], args.img_resize, args.df,
                             args.dry_run)
        counts = _process_split(job["name"], tasks, workers)
        for k, v in counts.items():
            grand_counts[k] = grand_counts.get(k, 0) + v

    grand_dt = time.time() - grand_t0
    print(f"\nGrand total {grand_dt:.1f}s ({grand_dt/60:.1f} min):")
    for k, v in grand_counts.items():
        if v:
            print(f"  {k}: {v}")
    if grand_counts.get("error", 0):
        print(f"NOTE: {grand_counts['error']} errors above. Investigate before retraining.")
        sys.exit(1)
    if args.dry_run and grand_counts.get("fixed", 0) > 0:
        print(f"\nDry-run complete. Re-run without --dry-run to apply "
              f"{grand_counts['fixed']} fixes.")


if __name__ == "__main__":
    main()
