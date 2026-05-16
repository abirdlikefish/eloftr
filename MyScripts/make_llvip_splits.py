"""Generate index files for LLVIP dataset (12025 train + 3463 test, .jpg).

LLVIP is a pixel-aligned IR-VIS dataset with **official train/test split**
already provided by the data publisher (BUPT-AI-CZ). We do NOT re-shuffle:
just scan the two official sub-directories and emit one line per pair.

Layout (after extracting LLVIP.zip from the official Google Drive / Baidu Yun,
see data/LLVIP/README.md):

    data/LLVIP/
        infrared/
            train/  010001.jpg .. (12025 .jpg)
            test/   190001.jpg .. (3463 .jpg)
        visible/
            train/  010001.jpg .. (12025 .jpg, same names as infrared)
            test/   190001.jpg .. (3463 .jpg, same names)
        Annotations/  *.xml  (匹配任务不读)

Output (under ``data/LLVIP/index/``):

  - train_pairs.txt   12025 lines, each ``train/<id>.jpg``
  - test_pairs.txt    3463 lines, each ``test/<id>.jpg``

Note the **subdirectory prefix in each line**: this lets a single cfg
``ROAD_IR_SUBDIR='infrared'`` (without ``train/`` or ``test/``) work for both
train (lines like ``train/010001.jpg``) and val (lines like ``test/190001.jpg``)
because RoadSceneDataset uses ``osp.join(ir_dir, name)`` and Path semantics
absorb the prefix correctly.

This is a divergence from ``make_m3fd_splits.py`` which emits bare filenames
(M3FD has no train/test sub-split in the source layout). LLVIP comes
pre-split by the data publisher so we honor that split + encode it in the
filename prefix.

There is NO independent val_pairs.txt: v17 uses LLVIP test as the val pool
(--limit_val_batches=0.5 -> 1731 effective val pair, business-equivalent to
LoFTR megadepth_val_1500 protocol). Future work (independent test holdout)
can re-shuffle test_pairs.txt later.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="data/LLVIP",
                        help="LLVIP root directory.")
    parser.add_argument("--ir_subdir", type=str, default="infrared",
                        help="Sub-directory containing infrared images "
                             "(scanned for both 'train/' and 'test/' inside).")
    parser.add_argument("--vis_subdir", type=str, default="visible",
                        help="Sub-directory containing pixel-aligned visible images.")
    parser.add_argument("--out_subdir", type=str, default="index",
                        help="Output sub-directory for index txts.")
    parser.add_argument("--ext", type=str, default=".jpg",
                        help="Image file extension to scan (lowercase).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing index files.")
    return parser.parse_args()


def _scan_split(ir_split_dir: Path, vis_split_dir: Path, ext: str,
                split_name: str) -> tuple[list[str], list[str]]:
    """Scan one split directory pair and return (paired_lines, skipped_names).

    Each paired_line has the form ``<split_name>/<file_name>`` so the cfg
    ROAD_*_SUBDIR can stay at the parent level (``infrared`` / ``visible``)
    and a single dataset config file can drive train (split='train') and
    val (split='test') without changing ROAD_*_SUBDIR.
    """
    if not ir_split_dir.is_dir():
        raise SystemExit(f"Cannot find IR split directory: {ir_split_dir}")
    if not vis_split_dir.is_dir():
        raise SystemExit(f"Cannot find VIS split directory: {vis_split_dir}")

    ir_paths = sorted(p for p in ir_split_dir.iterdir()
                      if p.is_file() and p.suffix.lower() == ext)
    if not ir_paths:
        raise SystemExit(f"No `*{ext}` files found under {ir_split_dir}.")

    paired_lines: list[str] = []
    skipped: list[str] = []
    for p in ir_paths:
        if (vis_split_dir / p.name).is_file():
            paired_lines.append(f"{split_name}/{p.name}")
        else:
            skipped.append(p.name)

    return paired_lines, skipped


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    ir_root = root / args.ir_subdir
    vis_root = root / args.vis_subdir
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lines, train_skipped = _scan_split(
        ir_root / "train", vis_root / "train", args.ext, "train")
    test_lines, test_skipped = _scan_split(
        ir_root / "test", vis_root / "test", args.ext, "test")

    splits = {
        "train_pairs.txt": train_lines,
        "test_pairs.txt": test_lines,
    }

    for fname, lines in splits.items():
        out_path = out_dir / fname
        if out_path.exists() and not args.overwrite:
            raise SystemExit(
                f"{out_path} already exists. Pass --overwrite to replace.")
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"LLVIP splits written under {out_dir}:")
    print(f"  train_pairs.txt = {len(train_lines)} lines (expected 12025)")
    print(f"  test_pairs.txt  = {len(test_lines)} lines (expected 3463)")
    if train_skipped:
        print(f"  train skipped (no VIS counterpart): {len(train_skipped)}")
    if test_skipped:
        print(f"  test  skipped (no VIS counterpart): {len(test_skipped)}")


if __name__ == "__main__":
    main()
