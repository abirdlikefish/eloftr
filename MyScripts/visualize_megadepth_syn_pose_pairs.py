"""Eyes-on visual sanity tool for v13 Megadepth_Syn_Pose dataset.

Picks N random pairs from a chosen split (default train), runs them
through ``MegadepthSynPoseDataset.__getitem__`` to get the actual
training-time (image0, image1, K0, K1, T_0to1, depth0, depth1, scale0,
scale1), then for each pair:

  1. Samples K random valid (mask0=True) pixels from image0.
  2. Warps them to image1 via warp_kpts (K + T + depth).
  3. Draws coloured correspondence lines on a side-by-side figure.
  4. Saves the figure with informative title + structured filename.

Diagnostic value (orthogonal to ``sanity_megadepth_syn_pose.py``'s
numeric gate):

  - same-stem same-viewpoint bug -> 10 lines all horizontal-parallel
    (x,y nearly identical between image0 and image1)
  - K/T unit / W2C-vs-C2W bug    -> lines criss-cross randomly
  - depth-stem-mismatch bug      -> valid_lines extremely low (< 3)
  - v13 correct                  -> lines show a smooth viewpoint-change
                                    pattern (slight diagonal spread)

Run on server (depends on data/Megadepth_Syn/ symlinks):

  cd /home/xyjiang/Desktop/yurupeng/eloftr
  source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
  export PYTHONPATH=$PWD
  python MyScripts/visualize_megadepth_syn_pose_pairs.py \\
      --split train --n_pairs 10 --n_lines 10 --seed 42 \\
      --cross_modal_mode ir2vis

Outputs:
  dump/v13_visualize_pairs_<split>_seed<N>/
    pair_0001_<scene>_idx<idx0>_idx<idx1>.png
    ...
    meta.txt  (tab-separated per-pair metadata + valid_lines stats)

Then `scp -P 8708 ...:dump/v13_visualize_pairs_train_seed42 .` to local
and eyeball the PNGs.
"""
from __future__ import annotations

import argparse
import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--n_pairs", type=int, default=10)
    p.add_argument("--n_lines", type=int, default=10,
                   help="number of GT correspondence lines to draw per pair")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cross_modal_mode", choices=["ir2vis", "vis2vis"],
                   default="ir2vis")
    p.add_argument("--img_resize", type=int, default=832)
    p.add_argument("--df", type=int, default=32)
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    # Heavy deps imported lazily so --help / arg parse errors stay fast.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib import cm

    from src.datasets.megadepth_syn_pose import MegadepthSynPoseDataset
    from src.loftr.utils.geometry import warp_kpts

    list_path = REPO_ROOT / f"data/Megadepth_Syn/index/trainvaltest_list_pose/{args.split}_list_pose.txt"
    if not list_path.exists():
        raise FileNotFoundError(
            f"split list missing: {list_path}\n"
            f"  run `python MyScripts/build_megadepth_syn_pose_splits.py` first")
    with open(list_path, "r", encoding="utf-8") as f:
        npz_names = [ln.strip() for ln in f if ln.strip()]

    npz_dir = REPO_ROOT / "data/Megadepth_Syn/index/scene_info_pose"
    data_root = REPO_ROOT / "data/Megadepth_Syn"
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / f"dump/v13_visualize_pairs_{args.split}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    meta_lines = []
    ok_count = 0

    for i in range(args.n_pairs):
        npz_name = rng.choice(npz_names)
        npz_path = npz_dir / f"{npz_name}.npz"
        if not npz_path.exists():
            print(f"[skip {i + 1}] npz missing: {npz_path}")
            continue

        ds = MegadepthSynPoseDataset(
            root_dir=str(data_root),
            npz_path=str(npz_path),
            mode="val",  # use val mode to skip RandomConcatSampler etc. but still load depth
            min_overlap_score=0.0,
            img_resize=args.img_resize,
            df=args.df,
            img_padding=True,
            depth_padding=True,
            cross_modal_mode=args.cross_modal_mode,
        )
        if len(ds) == 0:
            print(f"[skip {i + 1}] {npz_name} has 0 pairs after overlap filter")
            continue
        pair_idx = rng.randrange(len(ds))
        data = ds[pair_idx]

        image0 = data["image0"]  # [1, P, P]
        image1 = data["image1"]
        K0 = data["K0"][None]
        K1 = data["K1"][None]
        T_0to1 = data["T_0to1"][None]
        depth0 = data["depth0"][None]  # [1, 2000, 2000]
        depth1 = data["depth1"][None]
        scale0 = data["scale0"]  # [w_raw/w_resized, h_raw/h_resized]
        scale1 = data["scale1"]
        rel0, rel1 = data["pair_names"]
        stem0 = osp.splitext(osp.basename(rel0))[0]
        stem1 = osp.splitext(osp.basename(rel1))[0]

        # 1. pick n_lines random pixels from the valid (non-padded) region of image0
        # mask is at coarse resolution (P/8, P/8); upsample to full image size
        P = image0.shape[-1]
        mask0_full = torch.nn.functional.interpolate(
            data["mask0"][None, None].float(), size=(P, P), mode="nearest"
        )[0, 0].bool().cpu().numpy()
        ys, xs = np.where(mask0_full)
        if len(ys) < args.n_lines:
            print(f"[skip {i + 1}] {npz_name} pair {pair_idx} has too few valid pixels")
            continue
        choice = rng.sample(range(len(ys)), args.n_lines)
        # P_resized in (x, y) order, shape [n_lines, 2]
        pts_resized = np.stack([xs[choice], ys[choice]], axis=-1).astype(np.float32)

        # 2. warp via raw K. P_raw = P_resized * scale0 (scale = w_raw/w_resized, h_raw/h_resized)
        pts_raw = pts_resized * scale0.cpu().numpy()[None, :]
        pts_raw_t = torch.from_numpy(pts_raw).float()[None]  # [1, n_lines, 2]
        valid_mask, w_pts_raw = warp_kpts(pts_raw_t, depth0, depth1, T_0to1, K0, K1)
        # warped back to image1 resized coords
        w_pts_resized = (w_pts_raw[0] / scale1).cpu().numpy()
        valid = valid_mask[0].cpu().numpy()
        n_valid = int(valid.sum())

        # 3. plot
        img0_np = (image0[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img1_np = (image1[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=80)
        axes[0].imshow(img0_np, cmap="gray")
        axes[1].imshow(img1_np, cmap="gray")
        for ax in axes:
            ax.axis("off")

        tag0 = "IR" if args.cross_modal_mode == "ir2vis" else "VIS"
        tag1 = "VIS"
        axes[0].set_title(
            f"image0 = {tag0}  idx={data['pair_id']}  stem={stem0[:24]}",
            fontsize=10)
        axes[1].set_title(
            f"image1 = {tag1}  idx={data['pair_id']}  stem={stem1[:24]}",
            fontsize=10)

        # rainbow colours for lines
        colours = cm.rainbow(np.linspace(0, 1, args.n_lines))
        fig.canvas.draw()
        trans_fig = fig.transFigure.inverted()
        for k in range(args.n_lines):
            x0, y0 = pts_resized[k]
            x1, y1 = w_pts_resized[k]
            colour = colours[k]
            axes[0].scatter([x0], [y0], c=[colour], s=20, edgecolors="white", linewidths=0.6)
            if valid[k]:
                axes[1].scatter([x1], [y1], c=[colour], s=20, edgecolors="white", linewidths=0.6)
                fkpts0 = trans_fig.transform(
                    axes[0].transData.transform((x0, y0)))
                fkpts1 = trans_fig.transform(
                    axes[1].transData.transform((x1, y1)))
                fig.add_artist(mlines.Line2D(
                    [fkpts0[0], fkpts1[0]], [fkpts0[1], fkpts1[1]],
                    color=colour, linewidth=1.2, alpha=0.85,
                    transform=fig.transFigure))

        same_stem = stem0 == stem1
        fig.suptitle(
            f"{npz_name} pair {pair_idx}  "
            f"same_stem={same_stem}  valid_lines={n_valid}/{args.n_lines}",
            fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"pair_{i + 1:04d}_{npz_name}_idx{data['pair_id']}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        meta_lines.append(
            f"{i + 1:04d}\t{npz_name}\tpair_idx={pair_idx}\t"
            f"stem0={stem0}\tstem1={stem1}\tsame_stem={same_stem}\t"
            f"valid_lines={n_valid}/{args.n_lines}")
        print(f"[{i + 1}/{args.n_pairs}] {npz_name} pair {pair_idx} "
              f"same_stem={same_stem} valid_lines={n_valid}/{args.n_lines}")
        ok_count += 1

    if not ok_count:
        raise RuntimeError("No pairs visualized. Check data paths / splits.")

    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")

    print()
    print(f"[visualize_megadepth_syn_pose_pairs] {ok_count}/{args.n_pairs} pairs saved")
    print(f"  out: {out_dir}")


if __name__ == "__main__":
    main()
