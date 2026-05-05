"""Generate fixed train/val/test split index files for M3FD_Detection.

Layout assumed (after extracting the M3FD_Detection archive from TarDAL's
Google Drive / Baidu Yun, see .cursor/skills/eloftr-m3fd-data/SKILL.md):

    data/M3FD_Detection/
        Ir/    *.png       # 4200 infrared frames (00000.png .. 04199.png)
        Vis/   *.png       # 4200 pixel-aligned visible frames

Note: sub-directory casing is ``Ir`` / ``Vis`` (capitalised) and the
extension is ``.png`` (NOT RoadScene's ``cropinfrared`` / ``crop_LR_visible``
+ ``.jpg``). These differences are absorbed entirely via CLI defaults below
and via ``configs/data/m3fd_trainval.py``; no Python source changes needed.

Output (under ``data/M3FD_Detection/index/``):

  - train_pairs.txt
  - val_pairs.txt
  - test_pairs.txt

Each line contains a single image filename (e.g. ``00042.png``). The
``RoadSceneDataset`` will reconstruct the actual IR / VIS paths by joining
the filename with the corresponding subdirectories (which the M3FD data
config sets to ``Ir`` / ``Vis``).

Defaults: 90 / 5 / 10 (3780 train / 210 val / 210 test) -- val/test are
already comfortably larger than RoadScene's 22-pair val, so we hand most
of M3FD to training. Override via CLI when needed.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="data/M3FD_Detection",
                        help="M3FD_Detection root directory.")
    parser.add_argument("--ir_subdir", type=str, default="Ir",
                        help="Sub-directory containing infrared images.")
    parser.add_argument("--vis_subdir", type=str, default="Vis",
                        help="Sub-directory containing pixel-aligned visible images.")
    parser.add_argument("--out_subdir", type=str, default="index",
                        help="Output sub-directory for index txts.")
    parser.add_argument("--ext", type=str, default=".png",
                        help="Image file extension to scan (lowercase).")
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for the deterministic shuffle.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing index files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise SystemExit(
            f"train+val+test ratios must sum to 1.0, got {total_ratio:.4f}.")

    root = Path(args.root)
    ir_dir = root / args.ir_subdir
    vis_dir = root / args.vis_subdir
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ir_dir.is_dir():
        raise SystemExit(f"Cannot find IR directory: {ir_dir}")
    if not vis_dir.is_dir():
        raise SystemExit(f"Cannot find VIS directory: {vis_dir}")

    ir_paths = sorted(p for p in ir_dir.iterdir()
                      if p.is_file() and p.suffix.lower() == args.ext)
    if not ir_paths:
        raise SystemExit(f"No `*{args.ext}` files found under {ir_dir}.")

    paired_names: list[str] = []
    skipped: list[str] = []
    for p in ir_paths:
        if (vis_dir / p.name).is_file():
            paired_names.append(p.name)
        else:
            skipped.append(p.name)

    if not paired_names:
        raise SystemExit("No paired IR/VIS files found.")

    rng = random.Random(args.seed)
    rng.shuffle(paired_names)

    n = len(paired_names)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_train += n_test
        n_test = 0

    train = paired_names[:n_train]
    val = paired_names[n_train:n_train + n_val]
    test = paired_names[n_train + n_val:]

    splits = {
        "train_pairs.txt": train,
        "val_pairs.txt": val,
        "test_pairs.txt": test,
    }

    for fname, names in splits.items():
        out_path = out_dir / fname
        if out_path.exists() and not args.overwrite:
            raise SystemExit(
                f"{out_path} already exists. Pass --overwrite to replace.")
        out_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")

    print(f"M3FD_Detection splits written under {out_dir}:")
    print(f"  total paired = {n}")
    print(f"  train = {len(train)}")
    print(f"  val   = {len(val)}")
    print(f"  test  = {len(test)}")
    if skipped:
        print(f"  skipped (no VIS counterpart): {len(skipped)}")


if __name__ == "__main__":
    main()
