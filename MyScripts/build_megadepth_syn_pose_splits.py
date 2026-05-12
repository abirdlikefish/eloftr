"""Scene-disjoint 145/4/4 split for v13 Megadepth_Syn pose-supervised training.

读 LoFTR 官方 ``train_list.txt`` (368 行, 153 unique scene), 按 scene_id
group_by + 单 scene cap (val/test 候选只从 [3K, 50K] pair 数的 scene 里
挑) + seed=42 shuffle, 切 145/4/4. 每个 scene 的多个 overlap 区间 npz
整体进同一个 split, 不能拆开 (否则同 stem 既出现在 train 又出现在 val).

Balance check (raise ValueError if violated, prompt user to bump --seed):
  - val/test 总 pair 数 >= 5000
  - val/test 任一单 scene 占比 <= 35% (1.4x of 25% even-split for 4-scene splits;
    放宽自 0.30 因为 4-scene order-statistics 期望 max/sum ~= 0.41, 0.30 触发率太高
    实测 10 seed 全 fail。0.35 仍远低于 0.50 单 scene 主导线)
  - val/test 单 scene pair 数 in [3K, 50K] (避开 0024 类百万 scene 主导 val
    metric / 避开 530 pair 类小 scene 当 val/test 信号太弱)

Inputs (server only -- 本地 Windows 跑会因软链不存在 immediate fail):
  - data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt
      (软链 -> /data/xyjiang/Datasets/LofTR/train-data/megadepth_indices/
              trainvaltest_list/train_list.txt)
  - data/Megadepth_Syn/index/scene_info_pose/*.npz
      (软链 -> /data/xyjiang/Datasets/LofTR/train-data/megadepth_indices/
              scene_info_0.1_0.7/)

Outputs (入版本库, 跨机一致):
  - data/Megadepth_Syn/index/trainvaltest_list_pose/train_list_pose.txt
  - data/Megadepth_Syn/index/trainvaltest_list_pose/val_list_pose.txt
  - data/Megadepth_Syn/index/trainvaltest_list_pose/test_list_pose.txt
    每行一个 npz stem (eg "0000_0.1_0.3"), 不带 .npz 后缀
    (src/lightning/data.py L344 会自动加 .npz)

Usage:
  cd /home/xyjiang/Desktop/yurupeng/eloftr
  source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
  python MyScripts/build_megadepth_syn_pose_splits.py
  # or with custom seed if balance check fails:
  python MyScripts/build_megadepth_syn_pose_splits.py --seed 43
"""
from __future__ import annotations

import argparse
import os.path as osp
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_LIST = REPO_ROOT / "data/Megadepth_Syn/index/trainvaltest_list_src/train_list.txt"
NPZ_DIR = REPO_ROOT / "data/Megadepth_Syn/index/scene_info_pose"
OUT_DIR = REPO_ROOT / "data/Megadepth_Syn/index/trainvaltest_list_pose"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="scene shuffle seed")
    p.add_argument("--n_train", type=int, default=145)
    p.add_argument("--n_val", type=int, default=4)
    p.add_argument("--n_test", type=int, default=4)
    p.add_argument("--min_pair_for_valtest", type=int, default=3000,
                   help="single scene must have >= N pairs to be eligible for val/test")
    p.add_argument("--max_pair_for_valtest", type=int, default=50_000,
                   help="single scene must have <= N pairs to be eligible for val/test")
    p.add_argument("--min_total_pair_valtest", type=int, default=5000,
                   help="val/test total pair must >= N each")
    p.add_argument("--max_single_scene_ratio", type=float, default=0.35,
                   help="single scene must contribute <= R fraction of val/test total. "
                        "Loosened 0.30 -> 0.35 after empirical 10-seed sweep all failed: "
                        "with n_val=n_test=4 and pool [3K, 50K], 4-scene order-statistics "
                        "max/sum expectation ~= 0.41, so 0.30 (1.2x of 25% even-split) triggers "
                        "in ~95% of seeds; 0.35 (1.4x even-split) gives ~30% pass rate, still "
                        "safely below the 0.50 'single-scene dominates split' line.")
    return p.parse_args()


def load_npz_names(list_path: Path) -> list[str]:
    """Read LoFTR train_list.txt. Each line is a npz stem like '0000_0.1_0.3'."""
    if not list_path.exists():
        raise FileNotFoundError(
            f"Source list not found: {list_path}\n"
            f"On server, create the symlink first:\n"
            f"  cd {REPO_ROOT}/data/Megadepth_Syn && mkdir -p index\n"
            f"  ln -s /data/xyjiang/Datasets/LofTR/train-data/megadepth_indices/"
            f"trainvaltest_list  index/trainvaltest_list_src"
        )
    with open(list_path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names


def count_pairs(npz_dir: Path, npz_stem: str) -> int:
    """Return len(pair_infos) for a given npz, or 0 if file missing."""
    path = npz_dir / f"{npz_stem}.npz"
    if not path.exists():
        return 0
    # Cross-numpy compat:
    #   * newer numpy (>= 1.15): np.load returns NpzFile (supports `with` /
    #     .close()).
    #   * server eloftr_yurupeng numpy: returns a plain `dict` (no __enter__,
    #     no .close()).
    # LoFTR upstream src/datasets/megadepth.py:48-51 also assumes the latter
    # (`del self.scene_info['pair_infos']` works only on dict, not NpzFile).
    # We use hasattr to handle both.
    f = np.load(path, allow_pickle=True)
    try:
        return int(len(f["pair_infos"]))
    finally:
        if hasattr(f, "close"):
            f.close()


def group_by_scene(npz_names: list[str]) -> dict[str, list[str]]:
    """LoFTR scene_id is the first '_' split (eg '0000_0.1_0.3' -> '0000')."""
    out: dict[str, list[str]] = defaultdict(list)
    for n in npz_names:
        sid = n.split("_")[0]
        out[sid].append(n)
    return dict(out)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"[build_splits] reading {SRC_LIST}")
    npz_names = load_npz_names(SRC_LIST)
    print(f"[build_splits] {len(npz_names)} npz entries in source list")

    scene_to_npz = group_by_scene(npz_names)
    scene_ids = sorted(scene_to_npz.keys())
    print(f"[build_splits] {len(scene_ids)} unique scenes")

    n_total = args.n_train + args.n_val + args.n_test
    if len(scene_ids) < n_total:
        raise ValueError(
            f"Need {n_total} scenes ({args.n_train}+{args.n_val}+{args.n_test}) "
            f"but only {len(scene_ids)} available")

    print(f"[build_splits] counting pairs per scene ...")
    scene_pair_count: dict[str, int] = {}
    for sid in scene_ids:
        total = sum(count_pairs(NPZ_DIR, n) for n in scene_to_npz[sid])
        scene_pair_count[sid] = total

    eligible_for_valtest = [
        sid for sid, n in scene_pair_count.items()
        if args.min_pair_for_valtest <= n <= args.max_pair_for_valtest
    ]
    print(f"[build_splits] {len(eligible_for_valtest)} scenes eligible for val/test "
          f"(pair count in [{args.min_pair_for_valtest}, {args.max_pair_for_valtest}])")
    if len(eligible_for_valtest) < args.n_val + args.n_test:
        raise ValueError(
            f"Only {len(eligible_for_valtest)} eligible scenes for val/test but need "
            f"{args.n_val + args.n_test}. Relax --min_pair_for_valtest / "
            f"--max_pair_for_valtest, or run with different --seed.")

    # 1. shuffle eligible pool and pick val/test
    rng.shuffle(eligible_for_valtest)
    val_scenes = eligible_for_valtest[:args.n_val]
    test_scenes = eligible_for_valtest[args.n_val:args.n_val + args.n_test]

    # 2. train = everything else (no eligibility cap on train -- million-pair
    #    scenes contribute to training without constraint)
    valtest_set = set(val_scenes) | set(test_scenes)
    remaining = [s for s in scene_ids if s not in valtest_set]
    rng.shuffle(remaining)
    train_scenes = remaining[:args.n_train]

    # 3. balance checks
    def total_pair(scenes):
        return sum(scene_pair_count[s] for s in scenes)

    train_total = total_pair(train_scenes)
    val_total = total_pair(val_scenes)
    test_total = total_pair(test_scenes)

    print(f"[build_splits] train: {len(train_scenes)} scenes, {train_total} pairs")
    print(f"[build_splits] val  : {len(val_scenes)} scenes, {val_total} pairs")
    print(f"[build_splits] test : {len(test_scenes)} scenes, {test_total} pairs")

    issues = []
    if val_total < args.min_total_pair_valtest:
        issues.append(f"val total pair {val_total} < {args.min_total_pair_valtest}")
    if test_total < args.min_total_pair_valtest:
        issues.append(f"test total pair {test_total} < {args.min_total_pair_valtest}")
    for split_name, scenes, total in (
        ("val", val_scenes, val_total), ("test", test_scenes, test_total)
    ):
        for s in scenes:
            ratio = scene_pair_count[s] / max(total, 1)
            if ratio > args.max_single_scene_ratio:
                issues.append(
                    f"{split_name} scene {s} contributes "
                    f"{ratio:.1%} > {args.max_single_scene_ratio:.0%} of split")
    if issues:
        msg = "balance check failed; retry with different --seed.\n" + "\n".join(
            f"  - {x}" for x in issues)
        raise ValueError(msg)

    # 4. expand scene -> npz list (sort npz within scene for stable output)
    train_npz = sorted(n for s in train_scenes for n in scene_to_npz[s])
    val_npz = sorted(n for s in val_scenes for n in scene_to_npz[s])
    test_npz = sorted(n for s in test_scenes for n in scene_to_npz[s])

    # 5. cross-check three way mutual exclusion (sanity)
    assert not (set(train_npz) & set(val_npz)), "train ∩ val not empty"
    assert not (set(train_npz) & set(test_npz)), "train ∩ test not empty"
    assert not (set(val_npz) & set(test_npz)), "val ∩ test not empty"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, npz_list in (
        ("train", train_npz), ("val", val_npz), ("test", test_npz)
    ):
        out_path = OUT_DIR / f"{split_name}_list_pose.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(npz_list) + "\n")
        print(f"[build_splits] wrote {out_path}: {len(npz_list)} npz entries")

    # 6. summary
    print()
    print(f"[build_splits] seed={args.seed} OK")
    print(f"  train scenes: {sorted(train_scenes)[:5]}...({len(train_scenes)})")
    print(f"  val   scenes: {sorted(val_scenes)}")
    print(f"  test  scenes: {sorted(test_scenes)}")
    print(f"  val pair shares: " + ", ".join(
        f"{s}={scene_pair_count[s]}({scene_pair_count[s] / val_total:.0%})"
        for s in val_scenes))
    print(f"  test pair shares: " + ", ".join(
        f"{s}={scene_pair_count[s]}({scene_pair_count[s] / test_total:.0%})"
        for s in test_scenes))


if __name__ == "__main__":
    main()
