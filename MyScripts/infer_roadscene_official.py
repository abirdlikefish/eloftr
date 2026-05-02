import csv
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.cm as cm
import torch

from src.loftr import LoFTR, full_default_cfg, reparameter
from src.utils.plotting import make_matching_figure


MAX_PAIRS = 5


def resize_to_divisible_by_32(img0, img1):
    h0, w0 = img0.shape
    h1, w1 = img1.shape
    h = min(h0, h1) // 32 * 32
    w = min(w0, w1) // 32 * 32
    return cv2.resize(img0, (w, h)), cv2.resize(img1, (w, h))


def main():
    ir_dir = Path("data/RoadScene/cropinfrared")
    vis_dir = Path("data/RoadScene/crop_HR_visible")
    ckpt_path = Path("weights/eloftr_outdoor.ckpt")
    out_dir = Path("dump/roadscene_official")
    out_dir.mkdir(parents=True, exist_ok=True)

    assert ir_dir.exists(), f"Cannot find infrared folder: {ir_dir}"
    assert vis_dir.exists(), f"Cannot find visible folder: {vis_dir}"
    assert ckpt_path.exists(), f"Cannot find checkpoint: {ckpt_path}"

    cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    matcher.load_state_dict(ckpt["state_dict"], strict=False)
    matcher = reparameter(matcher).eval().cuda()

    rows = []
    ir_paths = sorted(ir_dir.glob("*.jpg"))[:MAX_PAIRS]
    print(f"Found {len(list(ir_dir.glob('*.jpg')))} infrared images. Running at most {MAX_PAIRS} pairs.")

    for idx, ir_path in enumerate(ir_paths, start=1):
        vis_path = vis_dir / ir_path.name
        if not vis_path.exists():
            print(f"[Skip] visible image not found: {vis_path}")
            continue

        img0_raw = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
        if img0_raw is None or img1_raw is None:
            print(f"[Skip] failed to read pair: {ir_path.name}")
            continue

        img0_raw, img1_raw = resize_to_divisible_by_32(img0_raw, img1_raw)
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.0
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.0
        batch = {"image0": img0, "image1": img1}

        with torch.no_grad():
            matcher(batch)

        mkpts0 = batch["mkpts0_f"].cpu().numpy()
        mkpts1 = batch["mkpts1_f"].cpu().numpy()
        mconf = batch["mconf"].cpu().numpy()

        mean_conf = float(mconf.mean()) if len(mconf) else 0.0
        max_conf = float(mconf.max()) if len(mconf) else 0.0
        median_conf = float(torch.from_numpy(mconf).median().item()) if len(mconf) else 0.0

        color = cm.jet(mconf)
        text = [
            "EfficientLoFTR official",
            f"IR-VIS Matches: {len(mkpts0)}",
            f"Mean conf: {mean_conf:.3f}",
            ir_path.name,
        ]
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
        save_path = out_dir / f"{ir_path.stem}_match.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

        rows.append({
            "name": ir_path.name,
            "num_matches": len(mkpts0),
            "mean_conf": f"{mean_conf:.6f}",
            "median_conf": f"{median_conf:.6f}",
            "max_conf": f"{max_conf:.6f}",
            "figure": str(save_path),
        })
        print(
            f"[{idx}/{MAX_PAIRS}] {ir_path.name}: matches={len(mkpts0)}, "
            f"mean_conf={mean_conf:.3f}, max_conf={max_conf:.3f} -> {save_path}"
        )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "num_matches", "mean_conf", "median_conf", "max_conf", "figure"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
