"""Generate scene-disjoint train/val/test split index files for Megadepth_Syn.

Layout assumed (after subdirectory symlinks per v10 plan):

    data/Megadepth_Syn/
        train/
            infrared    -> /data/.../train/infrared        (symlink, read-only)
            phoenix     -> /data/.../train/phoenix         (symlink, read-only)
            infrared_pc/                                   (real, PC cache target)
            phoenix_pc/                                    (real, PC cache target)

Source images live deep inside the symlinked dirs (standard MegaDepth_v1
nesting):

    train/infrared/phoenix/S6/zl548/MegaDepth_v1/<scene>/<denseN>/imgs/<stem>.jpg
    train/phoenix /                              ^ same scene/dense/imgs subpath

Scene-disjoint policy: 195 train scenes -> shuffle with seed -> first 175 to
train, next 10 to val, last 10 to test. Every image inside a chosen scene
goes into that scene's split (so val and test have completely unseen scenes
relative to train; v10 tests "did the model learn cross-modal matching that
generalises to unseen scenes within the same MegaDepth_Syn distribution").

Output (under ``data/Megadepth_Syn/index/``):

  - train_pairs.txt   ~115K lines (175 scenes)
  - val_pairs.txt     ~6.6K lines (10 scenes)
  - test_pairs.txt    ~6.6K lines (10 scenes)

Each line contains a nested relative path WITH the .jpg extension, relative
to the IR/VIS subdir base. Example line:

    0000/dense0/imgs/1000564847_9a99654012_o.jpg

The data config sets:

    ROAD_IR_SUBDIR  = "train/infrared/phoenix/S6/zl548/MegaDepth_v1"
    ROAD_VIS_SUBDIR = "train/phoenix/S6/zl548/MegaDepth_v1"

So RoadSceneDataset.__getitem__ does

    ir_path = osp.join(root, ROAD_IR_SUBDIR, name)
            = "data/Megadepth_Syn/train/infrared/phoenix/S6/zl548/MegaDepth_v1/0000/dense0/imgs/file.jpg"

which lands the actual file under the symlinked source.

Six known-empty VIS dense subdirs are detected by per-image VIS-counterpart
existence check and warned (no entries added):

    0092/dense0, 0265/dense1, 0327/dense2, 0360/dense1, 0360/dense2, 0394/dense1

Balance check (hard ValueError, suggests seed change):

  - test/val each must contain >= --min_test_imgs (default 2000) images
  - train must contain >= --min_train_imgs (default 80000) images
  - val/test image count ratio must be >= 0.3 (avoid extreme asymmetry)

If any check fails, the script raises ValueError listing the problem and the
chosen scenes, asking the user to retry with a different --seed. This catches
the rare case where random scene picks land mostly tiny scenes (range of
scene sizes: 19 .. 3975 images) and breaks val/test statistical significance.

Usage::

    python MyScripts/build_megadepth_syn_index.py
    python MyScripts/build_megadepth_syn_index.py --seed 7
    python MyScripts/build_megadepth_syn_index.py --train_scene_count 180 --val_scene_count 8 --test_scene_count 7
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="data/Megadepth_Syn",
                        help="Megadepth_Syn root directory.")
    parser.add_argument("--ir_subdir", type=str,
                        default="train/infrared/phoenix/S6/zl548/MegaDepth_v1",
                        help="Sub-directory containing IR scenes (deep nested path "
                             "down to MegaDepth_v1 so children are <scene>/...).")
    parser.add_argument("--vis_subdir", type=str,
                        default="train/phoenix/S6/zl548/MegaDepth_v1",
                        help="Sub-directory containing VIS scenes (parallel to IR).")
    parser.add_argument("--out_subdir", type=str, default="index",
                        help="Output sub-directory for index txts.")
    parser.add_argument("--ext", type=str, default=".jpg",
                        help="Image file extension (lowercase).")
    parser.add_argument("--train_scene_count", type=int, default=175,
                        help="Number of scenes assigned to train split.")
    parser.add_argument("--val_scene_count", type=int, default=10,
                        help="Number of scenes assigned to val split.")
    parser.add_argument("--test_scene_count", type=int, default=10,
                        help="Number of scenes assigned to test split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the deterministic scene shuffle.")
    parser.add_argument("--min_test_imgs", type=int, default=2000,
                        help="Hard threshold: test/val each must contain >= this "
                             "many images. Below threshold raises ValueError "
                             "(retry with different --seed).")
    parser.add_argument("--min_train_imgs", type=int, default=80000,
                        help="Hard threshold for train. Below raises ValueError.")
    parser.add_argument("--min_val_test_ratio", type=float, default=0.3,
                        help="min(val_imgs, test_imgs) / max(...) must be >= this. "
                             "Below raises ValueError.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing index files.")
    return parser.parse_args()


def _scan_scene_images(ir_dir: Path, vis_dir: Path,
                       ext: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Walk ``ir_dir/<scene>/<denseN>/imgs/*<ext>`` and produce, per scene,
    the list of nested relative paths (e.g. ``0000/dense0/imgs/foo.jpg``)
    that have a matching VIS counterpart.

    Returns (scene_to_relpaths, skipped_relpaths). skipped_relpaths are IR
    files whose VIS counterpart is missing (the 6 known empty VIS dense
    subdirs contribute the bulk).
    """
    if not ir_dir.is_dir():
        raise SystemExit(f"Cannot find IR directory: {ir_dir}")
    if not vis_dir.is_dir():
        raise SystemExit(f"Cannot find VIS directory: {vis_dir}")

    scene_to_relpaths: Dict[str, List[str]] = defaultdict(list)
    skipped: List[str] = []
    empty_vis_dirs: List[str] = []

    scene_dirs = sorted(p for p in ir_dir.iterdir() if p.is_dir())
    if not scene_dirs:
        raise SystemExit(f"No scene subdirs under {ir_dir}")

    ext_lc = ext.lower()
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        dense_dirs = sorted(p for p in scene_dir.iterdir()
                            if p.is_dir() and p.name.startswith("dense"))
        for dense_dir in dense_dirs:
            imgs_dir = dense_dir / "imgs"
            if not imgs_dir.is_dir():
                continue
            ir_files = sorted(p for p in imgs_dir.iterdir()
                              if p.is_file() and p.suffix.lower() == ext_lc)
            # Check the corresponding VIS dense imgs subdir; if it's empty
            # or missing, every IR file here will be skipped -- record once
            # for the warning.
            vis_imgs_dir = vis_dir / scene_id / dense_dir.name / "imgs"
            if not vis_imgs_dir.is_dir() or not any(vis_imgs_dir.iterdir()):
                empty_vis_dirs.append(f"{scene_id}/{dense_dir.name}")
                # still try per-file in case some pairs exist anyway
            for ir_file in ir_files:
                rel = ir_file.relative_to(ir_dir)  # e.g. PosixPath("0000/dense0/imgs/foo.jpg")
                vis_file = vis_dir / rel
                if vis_file.is_file():
                    # store as forward-slash string for cross-platform index file
                    scene_to_relpaths[scene_id].append(rel.as_posix())
                else:
                    skipped.append(rel.as_posix())

    if empty_vis_dirs:
        print(f"WARN: empty/missing VIS dense subdirs ({len(empty_vis_dirs)}):")
        for d in empty_vis_dirs:
            print(f"  - {d}")

    return dict(scene_to_relpaths), skipped


def _split_scenes(all_scenes: List[str], seed: int,
                  n_train: int, n_val: int, n_test: int) -> Tuple[List[str], List[str], List[str]]:
    """Scene-disjoint split: shuffle scenes deterministically, slice."""
    if n_train + n_val + n_test > len(all_scenes):
        raise SystemExit(
            f"train+val+test ({n_train+n_val+n_test}) > available scenes "
            f"({len(all_scenes)}). Reduce one of the counts.")
    scenes = list(all_scenes)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    train = sorted(scenes[:n_train])
    val = sorted(scenes[n_train:n_train + n_val])
    test = sorted(scenes[n_train + n_val:n_train + n_val + n_test])
    return train, val, test


def _balance_check(args, train_scenes, val_scenes, test_scenes,
                   train_imgs, val_imgs, test_imgs,
                   smallest_in_split: Dict[str, Tuple[str, int]],
                   largest_in_split: Dict[str, Tuple[str, int]]) -> None:
    """Hard threshold check. Raises ValueError with diagnostic + retry hint."""
    other_seeds = "7, 13, 99, 2024, 1234"

    def _diag(label, scenes, imgs):
        return (f"{label} scenes ({len(scenes)}): {scenes[:5]}{'...' if len(scenes) > 5 else ''}; "
                f"smallest contributing scene: {smallest_in_split[label][0]} "
                f"({smallest_in_split[label][1]} imgs); "
                f"largest: {largest_in_split[label][0]} "
                f"({largest_in_split[label][1]} imgs)")

    if test_imgs < args.min_test_imgs:
        raise ValueError(
            f"test split has only {test_imgs} images (< --min_test_imgs={args.min_test_imgs}). "
            f"{_diag('test', test_scenes, test_imgs)}. "
            f"Try a different --seed (current: {args.seed}). "
            f"Suggested: --seed {other_seeds}.")
    if val_imgs < args.min_test_imgs:
        raise ValueError(
            f"val split has only {val_imgs} images (< --min_test_imgs={args.min_test_imgs}). "
            f"{_diag('val', val_scenes, val_imgs)}. "
            f"Try a different --seed (current: {args.seed}). "
            f"Suggested: --seed {other_seeds}.")
    if train_imgs < args.min_train_imgs:
        raise ValueError(
            f"train split has only {train_imgs} images (< --min_train_imgs={args.min_train_imgs}). "
            f"{_diag('train', train_scenes, train_imgs)}. "
            f"Try a different --seed (current: {args.seed}). "
            f"Suggested: --seed {other_seeds}.")

    ratio = min(val_imgs, test_imgs) / max(val_imgs, test_imgs)
    if ratio < args.min_val_test_ratio:
        raise ValueError(
            f"val/test imbalance: val={val_imgs}, test={test_imgs}, ratio={ratio:.2f} "
            f"< --min_val_test_ratio={args.min_val_test_ratio}. "
            f"Try a different --seed (current: {args.seed}). "
            f"Suggested: --seed {other_seeds}.")


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    ir_dir = root / args.ir_subdir
    vis_dir = root / args.vis_subdir
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning IR scenes under: {ir_dir}")
    print(f"VIS counterpart base    : {vis_dir}")
    scene_to_relpaths, skipped = _scan_scene_images(ir_dir, vis_dir, args.ext)

    all_scenes = sorted(scene_to_relpaths.keys())
    print(f"Found {len(all_scenes)} non-empty scenes "
          f"with paired IR+VIS images (skipped {len(skipped)} unpaired IR files)")

    train_scenes, val_scenes, test_scenes = _split_scenes(
        all_scenes, args.seed,
        args.train_scene_count, args.val_scene_count, args.test_scene_count)

    splits = {
        "train_pairs.txt": train_scenes,
        "val_pairs.txt":   val_scenes,
        "test_pairs.txt":  test_scenes,
    }

    # collect per-split image lists + per-scene size stats for balance check
    split_imgs: Dict[str, List[str]] = {}
    smallest_in: Dict[str, Tuple[str, int]] = {}
    largest_in: Dict[str, Tuple[str, int]] = {}
    for label, scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        imgs: List[str] = []
        scene_sizes: List[Tuple[str, int]] = []
        for s in scenes:
            paths = scene_to_relpaths[s]
            imgs.extend(paths)
            scene_sizes.append((s, len(paths)))
        scene_sizes.sort(key=lambda x: x[1])
        smallest_in[label] = scene_sizes[0] if scene_sizes else ("none", 0)
        largest_in[label] = scene_sizes[-1] if scene_sizes else ("none", 0)
        split_imgs[label] = imgs

    train_imgs_n = len(split_imgs["train"])
    val_imgs_n = len(split_imgs["val"])
    test_imgs_n = len(split_imgs["test"])

    # diagnostic before balance check (so users see numbers even on failure)
    print(f"\nMegadepth_Syn index built (scene-disjoint, seed={args.seed}):")
    print(f"  train scenes : {len(train_scenes):3d}  total imgs: {train_imgs_n}")
    print(f"  val   scenes : {len(val_scenes):3d}  total imgs: {val_imgs_n}")
    print(f"  test  scenes : {len(test_scenes):3d}  total imgs: {test_imgs_n}")
    print(f"  smallest scene contribution:")
    print(f"    train: {smallest_in['train'][0]} -> {smallest_in['train'][1]} imgs")
    print(f"    val  : {smallest_in['val'][0]} -> {smallest_in['val'][1]} imgs")
    print(f"    test : {smallest_in['test'][0]} -> {smallest_in['test'][1]} imgs")
    print(f"  largest scene contribution:")
    print(f"    train: {largest_in['train'][0]} -> {largest_in['train'][1]} imgs")
    print(f"    val  : {largest_in['val'][0]} -> {largest_in['val'][1]} imgs")
    print(f"    test : {largest_in['test'][0]} -> {largest_in['test'][1]} imgs")
    print(f"  skipped (VIS missing): {len(skipped)} IR files")

    _balance_check(args, train_scenes, val_scenes, test_scenes,
                   train_imgs_n, val_imgs_n, test_imgs_n,
                   smallest_in, largest_in)

    val_test_ratio = (min(val_imgs_n, test_imgs_n) /
                      max(val_imgs_n, test_imgs_n)) if max(val_imgs_n, test_imgs_n) else 0.0
    print(f"  balance check:")
    print(f"    train >= {args.min_train_imgs}: PASS ({train_imgs_n})")
    print(f"    val   >= {args.min_test_imgs}: PASS ({val_imgs_n})")
    print(f"    test  >= {args.min_test_imgs}: PASS ({test_imgs_n})")
    print(f"    val/test ratio >= {args.min_val_test_ratio}: PASS ({val_test_ratio:.2f})")

    for fname, scenes in splits.items():
        out_path = out_dir / fname
        if out_path.exists() and not args.overwrite:
            raise SystemExit(
                f"{out_path} already exists. Pass --overwrite to replace.")
        # imgs list per split: gathered above
        label = fname.replace("_pairs.txt", "")
        imgs = split_imgs[label]
        out_path.write_text("\n".join(imgs) + ("\n" if imgs else ""), encoding="utf-8")

    print(f"\n  -> SUCCESS, written to {out_dir}/{{train,val,test}}_pairs.txt")


if __name__ == "__main__":
    main()
