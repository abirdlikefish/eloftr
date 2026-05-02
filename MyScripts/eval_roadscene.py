"""Independent RoadScene evaluation script.

Used to compare an EfficientLoFTR checkpoint (official or finetuned) on the
RoadScene IR-VIS test split using a pixel-error metric. Designed to be the
post-training counterpart of ``infer_roadscene_official.py`` and to NOT depend
on ``test.py`` (which assumes ScanNet/MegaDepth-style camera geometry).

For each pair in ``test_pairs.txt``:

  1. Load IR + VIS, resize to a common (H, W) divisible by 32.
  2. Optionally apply a deterministic random Homography to VIS.
  3. Run the matcher in eval mode.
  4. Compute pixel error: when no Homography, ``||mkpts0_f - mkpts1_f||``;
     otherwise ``||H @ mkpts0_f - mkpts1_f||``.

Outputs a per-pair CSV (``summary.csv``) and visualisation images.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

from src.datasets.roadscene import _random_homography
from src.loftr import LoFTR, full_default_cfg, reparameter
from src.utils.plotting import make_matching_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                        help="Path to checkpoint (.ckpt). Official or finetuned.")
    parser.add_argument("--root", default="data/RoadScene",
                        help="RoadScene root directory.")
    parser.add_argument("--ir_subdir", default="cropinfrared")
    parser.add_argument("--vis_subdir", default="crop_HR_visible")
    parser.add_argument("--list_path", default="data/RoadScene/index/test_pairs.txt",
                        help="Plain-text list of paired filenames (one per line).")
    parser.add_argument("--out_dir", required=True,
                        help="Directory for visualisations and summary.csv.")
    parser.add_argument("--img_resize", type=int, default=480,
                        help="Longer-edge target before df-rounding.")
    parser.add_argument("--df", type=int, default=32,
                        help="Final H, W are multiples of df.")
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="Limit to N pairs (0 = all).")
    parser.add_argument("--apply_homography", action="store_true",
                        help="Apply a random Homography to VIS at evaluation time.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed for the deterministic Homography generator.")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[1.0, 3.0, 5.0],
                        help="Pixel thresholds for precision@Npx.")
    parser.add_argument("--save_figures", action="store_true", default=True,
                        help="Save per-pair match visualisations.")
    parser.add_argument("--no_save_figures", dest="save_figures",
                        action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thr", type=float, default=0.1,
                        help="Coarse-matching confidence threshold.")
    return parser.parse_args()


def _read_index(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def _resize_pair(ir: np.ndarray, vis: np.ndarray, img_resize: int, df: int):
    h0, w0 = ir.shape
    h1, w1 = vis.shape
    scale = float(img_resize) / max(min(h0, h1), min(w0, w1))
    h_new = int(round(min(h0, h1) * scale))
    w_new = int(round(min(w0, w1) * scale))
    h_new = max(df, (h_new // df) * df)
    w_new = max(df, (w_new // df) * df)
    ir_r = cv2.resize(ir, (w_new, h_new))
    vis_r = cv2.resize(vis, (w_new, h_new))
    return ir_r, vis_r, (h_new, w_new)


def _apply_h_to_pts(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    warped = pts_h @ H.T
    warped = warped[:, :2] / np.clip(warped[:, 2:3], 1e-8, None)
    return warped


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_dir = Path(args.root) / args.ir_subdir
    vis_dir = Path(args.root) / args.vis_subdir
    ckpt_path = Path(args.ckpt)
    if not ir_dir.is_dir():
        raise SystemExit(f"Cannot find IR directory: {ir_dir}")
    if not vis_dir.is_dir():
        raise SystemExit(f"Cannot find VIS directory: {vis_dir}")
    if not ckpt_path.is_file():
        raise SystemExit(f"Cannot find checkpoint: {ckpt_path}")

    cfg = deepcopy(full_default_cfg)
    cfg["match_coarse"]["thr"] = args.thr
    matcher = LoFTR(config=cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    matcher.load_state_dict(state_dict, strict=False)
    matcher = reparameter(matcher).eval().to(args.device)

    names = _read_index(args.list_path)
    if args.max_pairs and args.max_pairs > 0:
        names = names[: args.max_pairs]
    print(f"Evaluating {len(names)} pairs with ckpt={ckpt_path}")

    rng = np.random.default_rng(args.seed)
    rows = []
    all_errs: list[np.ndarray] = []
    for idx, name in enumerate(names, start=1):
        ir_path = ir_dir / name
        vis_path = vis_dir / name
        if not ir_path.exists() or not vis_path.exists():
            print(f"[Skip] missing pair: {name}")
            continue
        ir_raw = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        vis_raw = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
        if ir_raw is None or vis_raw is None:
            print(f"[Skip] failed to read: {name}")
            continue
        ir, vis, (h, w) = _resize_pair(ir_raw, vis_raw, args.img_resize, args.df)

        H = np.eye(3, dtype=np.float32)
        if args.apply_homography:
            # Pick a deterministic-but-per-pair seed so different pairs see
            # different Homographies but the run as a whole is reproducible.
            local_rng = np.random.default_rng(args.seed + idx)
            H = _random_homography(h, w, rng=local_rng)
            vis = cv2.warpPerspective(vis, H, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

        img0 = torch.from_numpy(ir).float()[None][None].to(args.device) / 255.0
        img1 = torch.from_numpy(vis).float()[None][None].to(args.device) / 255.0
        batch = {"image0": img0, "image1": img1}
        with torch.no_grad():
            matcher(batch)

        mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
        mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()
        mconf = batch["mconf"].detach().cpu().numpy()

        warped = _apply_h_to_pts(mkpts0, H)
        pixel_errs = np.linalg.norm(warped - mkpts1, axis=-1) if len(mkpts0) else np.zeros(0)

        per_pair = {
            "name": name,
            "num_matches": int(len(mkpts0)),
            "mean_conf": float(mconf.mean()) if len(mconf) else 0.0,
            "median_conf": float(np.median(mconf)) if len(mconf) else 0.0,
            "max_conf": float(mconf.max()) if len(mconf) else 0.0,
            "mean_pixel_error": float(pixel_errs.mean()) if len(pixel_errs) else 0.0,
        }
        for t in args.thresholds:
            per_pair[f"precision@{int(t)}px"] = (
                float((pixel_errs < t).mean()) if len(pixel_errs) else 0.0
            )
        rows.append(per_pair)
        all_errs.append(pixel_errs)

        if args.save_figures:
            color = cm.jet(np.clip(1 - pixel_errs / max(args.thresholds), 0, 1)) \
                    if len(pixel_errs) else np.zeros((0, 4))
            text = [
                f"ckpt: {ckpt_path.name}",
                f"#Matches {len(mkpts0)}",
                f"Mean px err: {per_pair['mean_pixel_error']:.2f}",
                f"P@3px: {100 * per_pair.get('precision@3px', 0):.1f}%",
                f"H aug: {'on' if args.apply_homography else 'off'}",
                name,
            ]
            fig = make_matching_figure(ir, vis, mkpts0, mkpts1, color, text=text)
            fig.savefig(str(out_dir / f"{ir_path.stem}_match.png"),
                        dpi=150, bbox_inches="tight")
        print(f"[{idx}/{len(names)}] {name}: matches={len(mkpts0)} "
              f"mpe={per_pair['mean_pixel_error']:.2f} "
              f"p@3px={100 * per_pair.get('precision@3px', 0):.1f}%")

    # write per-pair CSV
    csv_path = out_dir / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # global summary
    if all_errs:
        flat = np.concatenate(all_errs) if all_errs else np.zeros(0)
        summary_path = out_dir / "overall.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"ckpt: {ckpt_path}\n")
            f.write(f"pairs: {len(rows)}\n")
            f.write(f"apply_homography: {args.apply_homography}\n")
            f.write(f"total_matches: {len(flat)}\n")
            f.write(f"mean_pixel_error: {float(flat.mean()) if len(flat) else 0.0:.4f}\n")
            for t in args.thresholds:
                v = float((flat < t).mean()) if len(flat) else 0.0
                f.write(f"precision@{int(t)}px: {v:.4f}\n")
        print(f"\nOverall summary written to {summary_path}")
    print(f"Per-pair CSV written to {csv_path}")


if __name__ == "__main__":
    main()
