"""Automated sanity gate for v13 Megadepth_Syn pose-supervised training.

NUMERIC (CI-style) gate -- pairs nicely with the eyes-on visual tool in
``visualize_megadepth_syn_pose_pairs.py``. Both should pass before going
to single-card smoke / 4-GPU DDP ship.

Five checks (raise ValueError on any failure):

  1. SOURCE  : trainvaltest_list_pose/{train,val,test}_list_pose.txt exist
               and each list non-empty.
  2. ISECT   : train / val / test scene sets are mutually disjoint
               (no scene id repeats across splits).
  3. PATHS   : sample 1000 random (npz, pair) records across train/val/test,
               for each pair resolve depth/IR/VIS via the §3 reverse-path
               formula and verify all three files physically exist.
               Also check K, W2C pose are non-None for the two indices.
  4. CROSSVIEW: stem(idx0) != stem(idx1) rate >= 0.99 over the 1000-pair
                sample (catches the "same-stem same-viewpoint" implementation
                bug that would silently degrade v13 to v10 identity matching).
  5. SUPERVISION: simulate spvs_coarse-style ground-truth coverage on 1 pair
                  from train: project image0 coarse grid via K0, T_0to1, depth0
                  to image1, then check `len(b_ids)` (valid covisible+depth-
                  consistent matches) >= 50. Catches K/T unit / W2C-vs-C2W /
                  depth-stem-mismatch bugs.

Run:
  cd /home/xyjiang/Desktop/yurupeng/eloftr
  source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
  export PYTHONPATH=$PWD
  python MyScripts/sanity_megadepth_syn_pose.py
"""
from __future__ import annotations

import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NPZ_DIR = REPO_ROOT / "data/Megadepth_Syn/index/scene_info_pose"
LIST_DIR = REPO_ROOT / "data/Megadepth_Syn/index/trainvaltest_list_pose"
DATA_ROOT = REPO_ROOT / "data/Megadepth_Syn"


def _read_list(name: str) -> list[str]:
    path = LIST_DIR / f"{name}_list_pose.txt"
    if not path.exists():
        raise ValueError(
            f"[CHECK 1 SOURCE] missing: {path}\n"
            f"  run `python MyScripts/build_megadepth_syn_pose_splits.py` first")
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise ValueError(f"[CHECK 1 SOURCE] empty: {path}")
    return names


def _scene_id(npz_stem: str) -> str:
    return npz_stem.split("_")[0]


def _resolve_paths(depth_rel_ir: str | None, depth_rel_vis: str) -> tuple[str, str, str]:
    """Return (depth_path, vis_path, ir_path) absolute, relative to repo root.

    depth_rel from npz looks like
    'phoenix/S6/zl548/MegaDepth_v1/0000/dense0/depths/<stem>.h5'.
    Image is 'imgs/<stem>.jpg' instead of 'depths/<stem>.h5'.
    IR adds 'infrared/' before 'phoenix/'.
    """
    depth = DATA_ROOT / "train" / depth_rel_vis
    vis = DATA_ROOT / "train" / depth_rel_vis.replace(
        "/depths/", "/imgs/").replace(".h5", ".jpg")
    ir_rel = depth_rel_ir if depth_rel_ir is not None else depth_rel_vis
    ir = DATA_ROOT / "train" / ir_rel.replace(
        "phoenix/", "infrared/phoenix/", 1).replace(
        "/depths/", "/imgs/").replace(".h5", ".jpg")
    return str(depth), str(vis), str(ir)


def check_1_source():
    print("[CHECK 1 SOURCE] reading split lists ...")
    train = _read_list("train")
    val = _read_list("val")
    test = _read_list("test")
    print(f"  train={len(train)} val={len(val)} test={len(test)} npz entries OK")
    return train, val, test


def check_2_isect(train: list[str], val: list[str], test: list[str]):
    print("[CHECK 2 ISECT] scene-disjoint ...")
    train_s = {_scene_id(n) for n in train}
    val_s = {_scene_id(n) for n in val}
    test_s = {_scene_id(n) for n in test}
    inter_tv = train_s & val_s
    inter_tt = train_s & test_s
    inter_vt = val_s & test_s
    if inter_tv or inter_tt or inter_vt:
        raise ValueError(
            f"[CHECK 2 ISECT] scene leak detected:\n"
            f"  train ∩ val  = {inter_tv}\n"
            f"  train ∩ test = {inter_tt}\n"
            f"  val ∩ test   = {inter_vt}")
    print(f"  train scenes={len(train_s)} val scenes={len(val_s)} test scenes={len(test_s)} OK")


def check_3_paths_and_4_crossview(train: list[str], val: list[str], test: list[str],
                                  n_sample: int = 1000, seed: int = 42):
    print(f"[CHECK 3 PATHS / 4 CROSSVIEW] sampling {n_sample} pairs ...")
    rng = random.Random(seed)
    all_lists = [("train", train), ("val", val), ("test", test)]

    miss_depth = miss_vis = miss_ir = 0
    miss_K = miss_T = 0
    same_stem = 0
    sampled = 0
    while sampled < n_sample:
        split_name, lst = rng.choice(all_lists)
        npz_name = rng.choice(lst)
        npz_path = NPZ_DIR / f"{npz_name}.npz"
        if not npz_path.exists():
            raise ValueError(
                f"[CHECK 3 PATHS] missing npz: {npz_path}\n"
                f"  check the scene_info_pose symlink in data/Megadepth_Syn/index/")
        # NpzFile.__enter__ requires numpy >= 1.15; use try/finally for
        # cross-numpy compat (server PyTorch 1.12.1 ships an older numpy).
        f = np.load(npz_path, allow_pickle=True)
        try:
            pair_infos = f["pair_infos"]
            depth_paths = f["depth_paths"]
            intrinsics = f["intrinsics"]
            poses = f["poses"]
        finally:
            f.close()
        if len(pair_infos) == 0:
            continue
        pi = rng.choice(pair_infos)
        idx0, idx1 = int(pi[0][0]), int(pi[0][1])
        depth_rel0 = depth_paths[idx0]
        depth_rel1 = depth_paths[idx1]
        if depth_rel0 is None or depth_rel1 is None:
            continue  # SfM hole: pair_infos shouldn't reach these but be defensive

        K0 = intrinsics[idx0]
        K1 = intrinsics[idx1]
        T0 = poses[idx0]
        T1 = poses[idx1]
        if K0 is None or K1 is None:
            miss_K += 1
        if T0 is None or T1 is None:
            miss_T += 1

        depth, vis, ir = _resolve_paths(str(depth_rel0), str(depth_rel1))
        if not osp.exists(depth):
            miss_depth += 1
        if not osp.exists(vis):
            miss_vis += 1
        if not osp.exists(ir):
            miss_ir += 1

        stem0 = osp.splitext(osp.basename(str(depth_rel0)))[0]
        stem1 = osp.splitext(osp.basename(str(depth_rel1)))[0]
        if stem0 == stem1:
            same_stem += 1

        sampled += 1

    print(f"  sampled {sampled} pairs")
    print(f"  missing files: depth={miss_depth}  vis={miss_vis}  ir={miss_ir}")
    print(f"  missing K={miss_K}  missing T={miss_T}")
    print(f"  same-stem pairs (degenerate same-viewpoint): {same_stem}/{sampled} "
          f"({same_stem / sampled:.2%})")

    if miss_depth or miss_vis or miss_ir:
        raise ValueError(
            f"[CHECK 3 PATHS] sample-existence failed; some files missing on disk. "
            f"Verify data/Megadepth_Syn/train/{{phoenix,infrared/phoenix}}/ symlinks "
            f"and integrity of source under /data/xyjiang/image_style_transfer/"
            f"4090_data/Megadepth_Syn/train/.")
    if miss_K or miss_T:
        raise ValueError(
            f"[CHECK 3 PATHS] some intrinsics/poses are None for indices that "
            f"appear in pair_infos; LoFTR scene_info filtering should have skipped "
            f"these. Investigate npz integrity.")

    crossview_rate = 1.0 - same_stem / sampled
    threshold = 0.99
    if crossview_rate < threshold:
        raise ValueError(
            f"[CHECK 4 CROSSVIEW] cross-view rate {crossview_rate:.2%} < {threshold:.0%}; "
            f"this means pair_infos has many same-stem pairs OR the reverse-path "
            f"formula collapsed stem0/stem1 -- v13 would degrade to v10 same-view "
            f"training. Check src/datasets/megadepth_syn_pose.py __getitem__ "
            f"stem extraction.")
    print(f"  cross-view rate {crossview_rate:.2%} >= {threshold:.0%} OK")


def check_5_supervision(train: list[str], seed: int = 42):
    """Simulate spvs_coarse on 1 pair to verify K/T/depth correctness."""
    print("[CHECK 5 SUPERVISION] simulating spvs_coarse on 1 train pair ...")
    import torch

    from src.utils.dataset import read_megadepth_depth
    from src.loftr.utils.geometry import warp_kpts

    rng = random.Random(seed)

    # iterate up to 20 random pairs until we find one with enough valid matches
    # (small overlap_score pairs can have <50 valid grid points even when K/T are
    # correct; we accept the gate if ANY of 20 attempts >= 50)
    best_b_ids = 0
    best_diag = ""
    for attempt in range(20):
        npz_name = rng.choice(train)
        f = np.load(NPZ_DIR / f"{npz_name}.npz", allow_pickle=True)
        try:
            pair_infos = f["pair_infos"]
            depth_paths = f["depth_paths"]
            intrinsics = f["intrinsics"]
            poses = f["poses"]
        finally:
            f.close()
        if len(pair_infos) == 0:
            continue
        pi = rng.choice(pair_infos)
        idx0, idx1 = int(pi[0][0]), int(pi[0][1])
        overlap = float(pi[1])

        depth_rel0 = str(depth_paths[idx0])
        depth_rel1 = str(depth_paths[idx1])
        depth0_path = DATA_ROOT / "train" / depth_rel0
        depth1_path = DATA_ROOT / "train" / depth_rel1
        if not depth0_path.exists() or not depth1_path.exists():
            continue

        d0 = read_megadepth_depth(str(depth0_path), pad_to=2000)[None]  # [1, 2000, 2000]
        d1 = read_megadepth_depth(str(depth1_path), pad_to=2000)[None]
        K0 = torch.from_numpy(np.asarray(intrinsics[idx0]).reshape(3, 3)).float()[None]
        K1 = torch.from_numpy(np.asarray(intrinsics[idx1]).reshape(3, 3)).float()[None]
        T0 = np.asarray(poses[idx0])
        T1 = np.asarray(poses[idx1])
        T_0to1 = torch.from_numpy(T1 @ np.linalg.inv(T0)).float()[None]  # [1, 4, 4]

        # build a coarse grid roughly matching IMG_RESIZE=832, df=32, scale 0.125
        # 832/8 = 104 coarse cells per dim. K is RAW (matches raw-image coords)
        # so we sample at raw resolution using the actual depth0 spatial extent.
        h0, w0 = d0.shape[1], d0.shape[2]
        h_used, w_used = min(h0, 832), min(w0, 832)
        H_c, W_c = h_used // 8, w_used // 8
        ys = torch.linspace(0, h_used - 1, H_c)
        xs = torch.linspace(0, w_used - 1, W_c)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(1, -1, 2)

        valid, _ = warp_kpts(grid, d0, d1, T_0to1, K0, K1)
        n_valid = int(valid[0].sum())
        if n_valid > best_b_ids:
            best_b_ids = n_valid
            best_diag = (f"npz={npz_name} pair (idx0={idx0}, idx1={idx1}) "
                         f"overlap={overlap:.3f} valid_grid={n_valid}/{H_c * W_c}")
        if n_valid >= 50:
            print(f"  best attempt: {best_diag}")
            print(f"  valid coarse matches {n_valid} >= 50 OK")
            return

    raise ValueError(
        f"[CHECK 5 SUPERVISION] best attempt only {best_b_ids} valid matches "
        f"over 20 random train pairs (need >= 50). "
        f"Likely cause: K/T units or W2C-vs-C2W convention wrong, or depth/K "
        f"stem mismatch. diag: {best_diag}")


def main():
    train, val, test = check_1_source()
    check_2_isect(train, val, test)
    check_3_paths_and_4_crossview(train, val, test)
    check_5_supervision(train)
    print()
    print("[sanity_megadepth_syn_pose] ALL 5 CHECKS PASSED")


if __name__ == "__main__":
    main()
