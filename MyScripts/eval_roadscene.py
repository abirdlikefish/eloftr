"""Independent RoadScene evaluation script (training-val parity).

Replaces the older "manual preprocess" path with one that *exactly* mirrors
the training-time validation pipeline. Concretely, this script:

  1. Loads the same yacs config as training (default
     ``configs/loftr/eloftr_full.py`` + ``configs/data/roadscene_trainval.py``,
     overridable via ``--main_cfg`` / ``--data_cfg``). This means LoFTR is
     built with the SAME architecture as training, including any optional
     extras the trained ckpt expects (``USE_MODALITY_EMB``,
     ``USE_CONTRASTIVE``, ``MP``, ``THR``, ``NPE``, ...).

  2. Builds a real ``RoadSceneDataset`` + ``DataLoader(bs=1, shuffle=False)``
     so each batch goes through the same independent-IR/VIS resize, square
     zero-padding, and mask0/mask1 generation as training validation.

  3. Does **not** call ``reparameter()``. Training validation runs the
     RepVGG backbone in its multi-branch ("training") form, so eval here
     does the same to match numerics.

  4. Computes pixel error using ``batch['homography_0to1']`` exactly as
     ``PL_LoFTR._compute_roadscene_metrics`` does.

  5. Aggregates ``precision@Npx`` per-match across all pairs (matching
     ``PL_LoFTR._aggregate_roadscene_metrics``). Per-pair numbers are also
     written to ``summary.csv`` for inspection.

The CLI is backwards-compatible with the previous version (same flags,
same output layout). Two new flags:

    --main_cfg   path to the LoFTR yacs config used at training
    --data_cfg   path to the RoadScene data yacs config

Use the same configs you trained with so the matcher class matches the
ckpt -- mismatch only shows up as ``missing_keys`` warnings on load.

The legacy ``--apply_homography`` flag still works: it switches the
dataset to ``mode='train'`` so the train-time Homography augmentation
fires. By default eval now mirrors v11 training: BOTH IR and VIS are
warped (``--homography_dual``, can be force-off with
``--no_homography_dual``; default ``None`` lets the cfg's
``ROAD_HOMOGRAPHY_DUAL`` decide). For full reproducibility the dataset
draws H from ``np.random.default_rng(seed + pair_id)`` (threaded via
``homography_seed=args.seed``) so identical commands produce identical
``overall.txt``. We still force ``num_workers=0`` whenever Homography
aug is requested to keep dataloader ordering deterministic.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.default import get_cfg_defaults
from src.datasets.roadscene import RoadSceneDataset
from src.loftr import LoFTR
from src.utils.misc import lower_config
from src.utils.plotting import make_matching_figure, error_colormap, dynamic_alpha


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint (.ckpt). Official or PL-trained.")
    p.add_argument("--main_cfg", default="configs/loftr/eloftr_full.py",
                   help="LoFTR yacs config used at training.")
    p.add_argument("--data_cfg", default="configs/data/roadscene_trainval.py",
                   help="Data yacs config used at training.")
    p.add_argument("--out_dir", required=True,
                   help="Directory for visualisations, summary.csv and overall.txt.")
    # Optional dataset overrides; if not set, taken from data_cfg.
    p.add_argument("--list_path", default=None,
                   help="Override the index file (default: TEST_LIST_PATH from data_cfg).")
    p.add_argument("--root", default=None, help="Override root_dir.")
    p.add_argument("--ir_subdir", default=None, help="Override IR sub-dir.")
    p.add_argument("--vis_subdir", default=None, help="Override VIS sub-dir.")
    # Behaviour knobs.
    p.add_argument("--max_pairs", type=int, default=0,
                   help="Limit to N pairs (0 = all).")
    p.add_argument("--apply_homography", action="store_true",
                   help="Run dataset in train-mode so the random Homography "
                        "augmentation fires. Default: off (matches training val).")
    # Tri-state --homography_dual: True / False / None (=cfg default).
    # argparse can't natively do that with a single --flag, so we use a pair of
    # store_const flags that share `dest='homography_dual'`. Default stays
    # None so omitting both flags hands the decision back to cfg.
    p.add_argument("--homography_dual", dest="homography_dual", action="store_const",
                   const=True, default=None,
                   help="Force dual-side H aug (warp both IR and VIS, v11-style). "
                        "Overrides cfg.DATASET.ROAD_HOMOGRAPHY_DUAL.")
    p.add_argument("--no_homography_dual", dest="homography_dual", action="store_const",
                   const=False,
                   help="Force single-side H aug (warp VIS only, v0-v10 legacy). "
                        "Overrides cfg.DATASET.ROAD_HOMOGRAPHY_DUAL.")
    p.add_argument("--seed", type=int, default=123,
                   help="Seed for per-pair Homography RNG (used when "
                        "--apply_homography is set). H matrices are drawn from "
                        "np.random.default_rng(seed + pair_id) so identical "
                        "commands produce identical overall.txt.")
    p.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--save_figures", action="store_true", default=True)
    p.add_argument("--no_save_figures", dest="save_figures", action="store_false")
    p.add_argument("--max_save_figures", type=int, default=0,
                   help="Cap visualisation figures to the first N pairs "
                        "(1-indexed by loader iteration). 0 = save all "
                        "(default). Useful for full-set eval where "
                        "overall.txt should reflect every pair but you "
                        "don't want hundreds of PNGs in out_dir.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--thr", type=float, default=None,
                   help="Override LOFTR.MATCH_COARSE.THR.")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Config + model
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace):
    """Reproduce the relevant slice of train.py's config-building."""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg)
    cfg.merge_from_file(args.data_cfg)

    if cfg.LOFTR.COARSE.NPE is None:
        cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    # Match the training command line: --disable_mp was always passed for
    # RoadScene runs, so MP=False. Forcing it here keeps the autocast/fp32
    # path identical to validation_step.
    cfg.LOFTR.MP = False
    cfg.LOFTR.HALF = False

    if args.thr is not None:
        cfg.LOFTR.MATCH_COARSE.THR = float(args.thr)
    return cfg


def _strip_matcher_prefix(state_dict: Dict[str, torch.Tensor]):
    """PL ckpts save the full LightningModule state_dict, where every LoFTR
    parameter is prefixed with ``matcher.`` (and the loss buffers live under
    ``loss.``). The official released ``eloftr_outdoor.ckpt`` is already a
    bare LoFTR state_dict with no prefix. Handle both."""
    if not any(k.startswith("matcher.") for k in state_dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("matcher."):
            cleaned[k[len("matcher."):]] = v
        # Drop loss.*, etc. They don't belong to the LoFTR matcher.
    return cleaned


def build_matcher(cfg, ckpt_path: Path, device: str) -> LoFTR:
    _cfg = lower_config(cfg)
    matcher = LoFTR(config=_cfg["loftr"])

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = _strip_matcher_prefix(state_dict)
    msg = matcher.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"  [load] missing_keys ({len(msg.missing_keys)}): "
              f"{msg.missing_keys[:6]}{' ...' if len(msg.missing_keys) > 6 else ''}")
    if msg.unexpected_keys:
        print(f"  [load] unexpected_keys ({len(msg.unexpected_keys)}): "
              f"{msg.unexpected_keys[:6]}{' ...' if len(msg.unexpected_keys) > 6 else ''}")

    matcher = matcher.eval().to(device)
    return matcher


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_dataset(cfg, args: argparse.Namespace) -> RoadSceneDataset:
    ds_cfg = cfg.DATASET
    root = args.root or ds_cfg.TEST_DATA_ROOT or ds_cfg.VAL_DATA_ROOT
    list_path = args.list_path or ds_cfg.TEST_LIST_PATH or ds_cfg.VAL_LIST_PATH
    ir_subdir = args.ir_subdir or getattr(ds_cfg, "ROAD_IR_SUBDIR", "cropinfrared")
    vis_subdir = args.vis_subdir or getattr(ds_cfg, "ROAD_VIS_SUBDIR", "crop_LR_visible")

    img_resize = getattr(ds_cfg, "ROAD_IMG_RESIZE", 480)
    df = getattr(ds_cfg, "ROAD_DF", 32)
    pad_size = getattr(ds_cfg, "ROAD_PAD_SIZE", None)
    coarse_scale = 1.0 / cfg.LOFTR.RESOLUTION[0]

    if args.apply_homography:
        # Engage the dataset's train-time augmentation path. mode='train'
        # is required so the augment_fn / homography_aug branches are taken.
        # Seeding is delegated to the dataset's per-pair RNG (homography_seed=
        # args.seed below) rather than np.random.seed(); the global seed used
        # to be a no-op for H matrices because _random_homography(rng=None)
        # calls np.random.default_rng() which ignores np.random globals.
        mode = "train"
        homography_aug = True
        homography_seed = int(args.seed)
    else:
        mode = "val"
        homography_aug = False
        homography_seed = None

    # Tri-state --homography_dual: True / False / None (=cfg). When the user
    # didn't pass either of the dual flags, fall back to cfg's
    # ROAD_HOMOGRAPHY_DUAL (which is False for v0-v10 cfgs and True for
    # v11/v12 cfgs). Explicit CLI overrides cfg.
    if args.homography_dual is None:
        homography_dual = bool(getattr(ds_cfg, "ROAD_HOMOGRAPHY_DUAL", False))
    else:
        homography_dual = bool(args.homography_dual)

    return RoadSceneDataset(
        root_dir=root,
        list_path=list_path,
        mode=mode,
        ir_subdir=ir_subdir,
        vis_subdir=vis_subdir,
        img_resize=img_resize,
        pad_size=pad_size,
        df=df,
        coarse_scale=coarse_scale,
        homography_aug=homography_aug,
        homography_prob=getattr(ds_cfg, "ROAD_HOMOGRAPHY_PROB", 1.0),
        homography_kwargs=dict(getattr(ds_cfg, "ROAD_HOMOGRAPHY_KWARGS", {})),
        homography_dual=homography_dual,
        homography_seed=homography_seed,
        # v7_pcclahe knobs. Mirror src/lightning/data.py's pass-through so
        # eval_*_finetuned.bat 7 (v7 ckpt eval) builds a dataset with the
        # same 2-channel (gray + PC) + CLAHE pre-processing as training.
        # Three-layer default-value equality (R3): the fallback literals
        # here MUST match src/config/default.py and RoadSceneDataset.__init__
        # so historical ckpt eval (v0-v6.1) stays byte-identical to before.
        use_edge_input=getattr(cfg.LOFTR, "USE_EDGE_INPUT", False),
        ir_pc_subdir=getattr(ds_cfg, "ROAD_IR_PC_SUBDIR", ""),
        vis_pc_subdir=getattr(ds_cfg, "ROAD_VIS_PC_SUBDIR", ""),
        use_clahe_ir=getattr(cfg.LOFTR, "USE_CLAHE_IR", False),
        use_clahe_vis=getattr(cfg.LOFTR, "USE_CLAHE_VIS", False),
        clahe_clip_limit=getattr(cfg.LOFTR, "CLAHE_CLIP_LIMIT", 2.0),
        clahe_tile_size=getattr(cfg.LOFTR, "CLAHE_TILE_SIZE", [8, 8]),
    )


# --------------------------------------------------------------------------- #
# Pixel error (mirror of PL_LoFTR._compute_roadscene_metrics)
# --------------------------------------------------------------------------- #
def _compute_pair_pixel_errs(batch: dict) -> np.ndarray:
    mkpts0 = batch["mkpts0_f"]
    mkpts1 = batch["mkpts1_f"]
    if mkpts0.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    if "homography_0to1" in batch:
        H = batch["homography_0to1"].to(mkpts0.dtype)
        if H.dim() == 3:           # [B, 3, 3] -> assume B=1 here
            H = H[0]
        ones = torch.ones(mkpts0.size(0), 1, dtype=mkpts0.dtype, device=mkpts0.device)
        pts_h = torch.cat([mkpts0, ones], dim=-1)[..., None]   # [M, 3, 1]
        warped_h = (H[None] @ pts_h).squeeze(-1)               # [M, 3]
        warped = warped_h[..., :2] / warped_h[..., 2:3].clamp(min=1e-8)
        errs = torch.linalg.norm(warped - mkpts1, dim=-1)
    else:
        errs = torch.linalg.norm(mkpts0 - mkpts1, dim=-1)
    return errs.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def _crop_to_valid(image: torch.Tensor, mask: torch.Tensor):
    """``image`` is the padded canvas tensor (1, P, P) in [0, 1]. ``mask`` is
    the COARSE-scale mask (Hc, Wc) returned by RoadSceneDataset. We upsample
    the mask back to canvas size, find the bounding box of valid cells and
    return the cropped image (H_r, W_r) as a uint8 numpy array."""
    img = (image[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    P = img.shape[-1]
    m = mask.detach().cpu().numpy().astype(bool)
    # Upsample mask to canvas resolution by Kronecker product.
    scale = P // m.shape[-1] if m.shape[-1] else 1
    if scale > 1:
        m_full = np.kron(m, np.ones((scale, scale), dtype=bool))[:P, :P]
    else:
        m_full = m
    if not m_full.any():
        return img, (0, P, 0, P)
    rows = np.where(m_full.any(axis=1))[0]
    cols = np.where(m_full.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return img[r0:r1, c0:c1], (r0, r1, c0, c1)


def _save_pair_figure(out_path: Path,
                      batch: dict,
                      pixel_errs: np.ndarray,
                      thresholds: List[float],
                      ckpt_name: str,
                      pair_name: str,
                      apply_homography: bool,
                      homography_dual: bool = False):
    img0_crop, (r0_0, r1_0, c0_0, c1_0) = _crop_to_valid(batch["image0"][0], batch["mask0"][0])
    img1_crop, (r0_1, r1_1, c0_1, c1_1) = _crop_to_valid(batch["image1"][0], batch["mask1"][0])
    mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
    mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()

    # Filter matches whose endpoints fall inside the cropped (valid) region
    # AND offset coords so they line up with the cropped image.
    keep = (
        (mkpts0[:, 0] >= c0_0) & (mkpts0[:, 0] < c1_0) &
        (mkpts0[:, 1] >= r0_0) & (mkpts0[:, 1] < r1_0) &
        (mkpts1[:, 0] >= c0_1) & (mkpts1[:, 0] < c1_1) &
        (mkpts1[:, 1] >= r0_1) & (mkpts1[:, 1] < r1_1)
    ) if len(mkpts0) else np.zeros(0, dtype=bool)
    mkpts0_v = mkpts0[keep] - np.array([c0_0, r0_0], dtype=mkpts0.dtype) if keep.any() else np.zeros((0, 2))
    mkpts1_v = mkpts1[keep] - np.array([c0_1, r0_1], dtype=mkpts1.dtype) if keep.any() else np.zeros((0, 2))
    errs_v = pixel_errs[keep] if keep.any() else np.zeros(0)

    # Colour & alpha: align with training-side TB figure
    # (src/utils/plotting.py:_make_evaluation_figure_roadscene). Previously
    # this dump used cm.jet(1 - err/5), which is REVERSED vs the TB rule:
    # err=0 painted as deep red and err>=5 as deep blue, so the most accurate
    # matches looked like errors and vice versa. error_colormap puts
    # err=0 -> green, err=thr -> yellow, err>=2*thr -> red, matching the
    # P@thr correctness boundary and intuitive "green=good / red=bad".
    # dynamic_alpha makes dense-match figures readable rather than a wall of
    # opaque coloured lines.
    conf_thr_vis = 3.0
    if len(errs_v):
        color = error_colormap(errs_v, conf_thr_vis,
                               alpha=dynamic_alpha(len(errs_v)))
    else:
        color = np.zeros((0, 4))
    p_at_3 = float((pixel_errs < 3.0).mean()) if len(pixel_errs) else 0.0
    if not apply_homography:
        h_aug_tag = "off"
    elif homography_dual:
        h_aug_tag = "dual"
    else:
        h_aug_tag = "single"
    text = [
        f"ckpt: {ckpt_name}",
        f"#Matches {len(mkpts0)} (in-canvas: {int(keep.sum())})",
        f"Mean px err: {float(pixel_errs.mean()) if len(pixel_errs) else 0.0:.2f}",
        f"P@3px: {100 * p_at_3:.1f}%",
        f"H aug: {h_aug_tag}",
        pair_name,
    ]
    fig = make_matching_figure(img0_crop, img1_crop, mkpts0_v, mkpts1_v, color, text=text)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Cannot find checkpoint: {ckpt_path}")

    cfg = build_config(args)
    matcher = build_matcher(cfg, ckpt_path, args.device)
    dataset = build_dataset(cfg, args)

    n_total = len(dataset)
    if args.max_pairs and args.max_pairs > 0:
        n_total = min(n_total, args.max_pairs)
    print(f"Evaluating {n_total} pairs with ckpt={ckpt_path}")
    print(f"  main_cfg : {args.main_cfg}")
    print(f"  data_cfg : {args.data_cfg}")
    print(f"  ir_dir   : {dataset.ir_dir}")
    print(f"  vis_dir  : {dataset.vis_dir}")
    print(f"  list     : {dataset.list_path}")
    print(f"  H aug    : {bool(args.apply_homography)}")
    if args.apply_homography:
        print(f"  H dual   : {bool(dataset.homography_dual)}")
        print(f"  H seed   : {dataset.homography_seed}")
        print(f"  H prob   : {dataset.homography_prob}")
        print(f"  H kwargs : {dict(dataset.homography_kwargs)}")
    print(f"  THR      : {cfg.LOFTR.MATCH_COARSE.THR}")

    num_workers = 0 if args.apply_homography else int(args.num_workers)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    rows: List[dict] = []
    all_errs: List[np.ndarray] = []
    for idx, batch in enumerate(loader, start=1):
        if args.max_pairs and idx > args.max_pairs:
            break

        # Move tensors to device.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device, non_blocking=True)

        with torch.no_grad():
            matcher(batch)

        pixel_errs = _compute_pair_pixel_errs(batch)
        all_errs.append(pixel_errs)

        # pair_names is a tuple-of-tuples (one per side, each side is a tuple
        # of length B=1) because of DataLoader collation.
        ir_name = batch["pair_names"][0][0] if isinstance(batch["pair_names"][0], (list, tuple)) else batch["pair_names"][0]
        pair_short = Path(str(ir_name)).name

        mconf = batch["mconf"].detach().cpu().numpy() if "mconf" in batch else np.zeros(0)
        per_pair = {
            "name": pair_short,
            "num_matches": int(len(pixel_errs)),
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

        if args.save_figures and (args.max_save_figures <= 0
                                  or idx <= args.max_save_figures):
            fig_path = out_dir / f"{Path(pair_short).stem}_match.png"
            _save_pair_figure(fig_path, batch, pixel_errs, args.thresholds,
                              ckpt_path.name, pair_short, args.apply_homography,
                              homography_dual=bool(dataset.homography_dual))

        print(f"[{idx}/{n_total}] {pair_short}: matches={per_pair['num_matches']} "
              f"mpe={per_pair['mean_pixel_error']:.2f} "
              f"p@3px={100 * per_pair.get('precision@3px', 0):.1f}%")

    # ---- per-pair CSV
    csv_path = out_dir / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ---- per-match aggregation (matches PL_LoFTR._aggregate_roadscene_metrics)
    flat = np.concatenate(all_errs) if all_errs else np.zeros(0)
    summary_path = out_dir / "overall.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"ckpt: {ckpt_path}\n")
        f.write(f"main_cfg: {args.main_cfg}\n")
        f.write(f"data_cfg: {args.data_cfg}\n")
        f.write(f"pairs: {len(rows)}\n")
        f.write(f"apply_homography: {args.apply_homography}\n")
        f.write(f"homography_dual: {bool(dataset.homography_dual)}\n")
        f.write(f"homography_prob: {dataset.homography_prob}\n")
        f.write(f"homography_seed: {dataset.homography_seed}\n")
        f.write(f"homography_kwargs: {dict(dataset.homography_kwargs)}\n")
        f.write(f"total_matches: {len(flat)}\n")
        f.write(f"mean_pixel_error: {float(flat.mean()) if len(flat) else 0.0:.4f}\n")
        for t in args.thresholds:
            v = float((flat < t).mean()) if len(flat) else 0.0
            f.write(f"precision@{int(t)}px: {v:.4f}\n")
    print(f"\nOverall summary written to {summary_path}")
    print(f"Per-pair CSV written to {csv_path}")


if __name__ == "__main__":
    main()
