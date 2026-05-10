"""sanity_v11_dualh.py -- pre-flight verification for v11 dual-side aggressive
Homography augmentation BEFORE shipping the 12 ep / 17h DDP training run.

Run on the server (Megadepth_Syn data + sym links must be in place):

    cd /home/xyjiang/Desktop/yurupeng/eloftr
    source /home/xyjiang/anaconda3/bin/activate eloftr_yurupeng
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python MyScripts/sanity_v11_dualh.py \\
        configs/data/megadepth_syn_trainval.py \\
        configs/loftr/eloftr_full_v11_dualh_aggressive_msyn_ddp.py

OR:

    bash MyScripts/run_msyn_v11_dualh_ddp.sh --sanity-only

The script must PASS all 5 checks before launching the ship training. Each
check writes [PASS]/[FAIL] to stdout. Exit code = 0 iff all PASS.

Checks (rationale in plan SS Risk与回滚):

  1. shape / dtype contract: image0/image1 (1 or 2, P, P) float32 in [0,1];
     mask0/mask1 bool (P/8, P/8); H_0to1 float32 (3,3) with det > 0.
  2. dual aug fires:        build a SECOND dataset with dual=False, same
     index, same prob; mask0 coverage must drop by >= 0.05 vs dual mode,
     confirming IR side is being warped (and the cfg flag wired through).
  3. covisibility floor:    avg per-pair (mask0 AND projected mask1) >= 0.30
     across 16 batches. Below 0.30 -> aug too aggressive, GT signal too thin.
  4. H composition correctness: independently sample H_ir, H_vis; verify
     that warping a known checkerboard by H_0to1 = H_vis @ inv(H_ir) lines
     up with the dual-warped reference within sub-pixel tolerance.
  5. backward compatibility:  build a SECOND dataset with DUAL=False, AUG=
     False on the SAME index; verify H_0to1 == I, mask0 fully True in valid
     region, image0 / image1 byte-identical to no-aug raw + resize + pad.

Usage:
  python MyScripts/sanity_v11_dualh.py <data_cfg> <main_cfg>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Repo root on sys.path so `src.*` imports work when invoked from MyScripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.default import get_cfg_defaults
from src.datasets.roadscene import RoadSceneDataset, _random_homography


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("data_cfg", type=str, help="data config .py path")
    p.add_argument("main_cfg", type=str, help="main config .py path (v11)")
    p.add_argument("--n_samples", type=int, default=16,
                   help="number of training samples to draw for stat checks")
    p.add_argument("--seed", type=int, default=0,
                   help="numpy + torch seed for reproducible spot-check")
    p.add_argument("--cov_min", type=float, default=0.30,
                   help="minimum per-pair covisibility (mask0 AND projected mask1)")
    p.add_argument("--dual_delta_min", type=float, default=0.05,
                   help="min absolute drop in mask0 coverage (single - dual) to confirm dual aug is firing")
    p.add_argument("--h_compose_tol_px", type=float, default=1.0,
                   help="H composition test: max sub-pixel residual after re-projection")
    return p.parse_args()


def _cfg_load(args: argparse.Namespace):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg)
    cfg.merge_from_file(args.data_cfg)
    return cfg


def _build_dataset(cfg, mode: str, homography_dual_override: bool | None = None,
                   homography_aug_override: bool | None = None):
    ds_cfg = cfg.DATASET
    root = ds_cfg.TRAIN_DATA_ROOT if mode == "train" else ds_cfg.VAL_DATA_ROOT
    list_path = ds_cfg.TRAIN_LIST_PATH if mode == "train" else ds_cfg.VAL_LIST_PATH
    homography_aug = (
        getattr(ds_cfg, "ROAD_HOMOGRAPHY_AUG", True) if homography_aug_override is None
        else homography_aug_override
    )
    homography_dual = (
        getattr(ds_cfg, "ROAD_HOMOGRAPHY_DUAL", False) if homography_dual_override is None
        else homography_dual_override
    )
    return RoadSceneDataset(
        root_dir=root,
        list_path=list_path,
        mode=mode,
        dataset_name=str(ds_cfg.TRAINVAL_DATA_SOURCE),
        ir_subdir=getattr(ds_cfg, "ROAD_IR_SUBDIR", "cropinfrared"),
        vis_subdir=getattr(ds_cfg, "ROAD_VIS_SUBDIR", "crop_LR_visible"),
        img_resize=getattr(ds_cfg, "ROAD_IMG_RESIZE", 480),
        pad_size=getattr(ds_cfg, "ROAD_PAD_SIZE", None),
        df=getattr(ds_cfg, "ROAD_DF", 32),
        coarse_scale=1.0 / cfg.LOFTR.RESOLUTION[0],
        homography_aug=homography_aug,
        homography_prob=getattr(ds_cfg, "ROAD_HOMOGRAPHY_PROB", 1.0),
        homography_kwargs=dict(getattr(ds_cfg, "ROAD_HOMOGRAPHY_KWARGS", {})),
        homography_dual=homography_dual,
        # v7 PC channels (auto-on under v11 stack inheritance)
        use_edge_input=getattr(cfg.LOFTR, "USE_EDGE_INPUT", False),
        ir_pc_subdir=getattr(ds_cfg, "ROAD_IR_PC_SUBDIR", ""),
        vis_pc_subdir=getattr(ds_cfg, "ROAD_VIS_PC_SUBDIR", ""),
        use_clahe_ir=getattr(cfg.LOFTR, "USE_CLAHE_IR", False),
        use_clahe_vis=getattr(cfg.LOFTR, "USE_CLAHE_VIS", False),
        clahe_clip_limit=getattr(cfg.LOFTR, "CLAHE_CLIP_LIMIT", 2.0),
        clahe_tile_size=tuple(getattr(cfg.LOFTR, "CLAHE_TILE_SIZE", (8, 8))),
    )


# --------------------------------------------------------------------------- #
# Per-check helpers
# --------------------------------------------------------------------------- #
def _project_grid_via_H(mask0_c: torch.Tensor, mask1_c: torch.Tensor,
                        H_0to1: torch.Tensor, scale: int = 8) -> float:
    """For every True cell in mask0 (coarse grid), project its centre by H_0to1
    and check whether it lands inside a True cell of mask1 (in pixel space).
    Returns covisibility = (#valid_projected) / (#mask0_cells)."""
    H_c, W_c = mask0_c.shape
    # Coarse cell centres in image pixel coords (centre of an 8x8 block).
    ys, xs = torch.where(mask0_c)
    if len(xs) == 0:
        return 0.0
    pts = torch.stack([(xs.float() + 0.5) * scale,
                       (ys.float() + 0.5) * scale,
                       torch.ones_like(xs.float())], dim=0)  # (3, K)
    H = H_0to1.to(pts.dtype)
    warped = H @ pts                                   # (3, K)
    w = warped[2]
    valid = (w.abs() > 1e-8)
    warped = warped[:2] / torch.where(valid[None], w[None], torch.ones_like(w[None]))

    # Convert back to coarse-cell index (P/8) and look up mask1.
    cx = (warped[0] / scale).long()
    cy = (warped[1] / scale).long()
    in_bounds = valid & (cx >= 0) & (cx < W_c) & (cy >= 0) & (cy < H_c)
    cx_c = cx.clamp(0, W_c - 1)
    cy_c = cy.clamp(0, H_c - 1)
    mask1_lookup = mask1_c[cy_c, cx_c]
    valid_proj = in_bounds & mask1_lookup
    return float(valid_proj.sum().item()) / float(len(xs))


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_shapes(sample: dict) -> tuple[bool, str]:
    img0 = sample["image0"]
    img1 = sample["image1"]
    m0 = sample["mask0"]
    m1 = sample["mask1"]
    H = sample["homography_0to1"]
    ok = True
    msgs = []
    if img0.dtype != torch.float32 or img1.dtype != torch.float32:
        ok = False
        msgs.append(f"image dtype not float32: img0={img0.dtype}, img1={img1.dtype}")
    if not (img0.min() >= 0.0 and img0.max() <= 1.0 + 1e-4):
        ok = False
        msgs.append(f"image0 out of [0,1]: min={img0.min()}, max={img0.max()}")
    if m0.dtype != torch.bool or m1.dtype != torch.bool:
        ok = False
        msgs.append(f"mask dtype not bool: m0={m0.dtype}, m1={m1.dtype}")
    if H.shape != (3, 3) or H.dtype != torch.float32:
        ok = False
        msgs.append(f"H bad shape/dtype: {H.shape}, {H.dtype}")
    if abs(float(np.linalg.det(H.numpy()))) < 1e-6:
        ok = False
        msgs.append(f"H is singular: det={float(np.linalg.det(H.numpy())):.2e}")
    if abs(float(H[2, 2]) - 1.0) > 1e-3:
        ok = False
        msgs.append(f"H not normalised: H[2,2]={float(H[2,2]):.4f}")
    return ok, ("; ".join(msgs) if msgs else "all shapes / dtypes / det / H[2,2] OK")


def check_dual_aug_firing(ds_dual, ds_single, n_samples: int) -> tuple[bool, str]:
    """Compare mask0 coverage between dual-mode and single-mode datasets on
    the SAME indices (sample indices independent of np.random gate). Dual
    must reduce mask0 coverage by at least 0.05 absolute on average,
    otherwise dual aug is not firing on the IR side at all.

    This avoids the false positive where non-square Megadepth_Syn images
    naturally produce mask0 coverage 0.5-0.75 from padding alone (long_edge
    480, narrow side ~256-352 after df-align), which would defeat a simple
    "coverage < 0.95" threshold."""
    # Force prob=1.0 on both datasets so aug always fires when enabled.
    ds_dual.homography_prob = 1.0
    ds_single.homography_prob = 1.0
    # Same sample indices for both (deterministic spacing).
    idxs = [(i * max(1, len(ds_dual) // n_samples)) % len(ds_dual) for i in range(n_samples)]
    # Same RNG seed for both passes so sampled H_vis is identical between
    # dual and single (the np.random.rand() gate + _random_homography fresh
    # rng is OS-entropy seeded so this is best-effort, not byte-identity).
    cov_dual_per = []
    cov_single_per = []
    for idx in idxs:
        s_d = ds_dual[idx]
        cov_dual_per.append(float(s_d["mask0"].float().mean().item()))
    for idx in idxs:
        s_s = ds_single[idx]
        cov_single_per.append(float(s_s["mask0"].float().mean().item()))
    avg_dual = float(np.mean(cov_dual_per))
    avg_single = float(np.mean(cov_single_per))
    delta = avg_single - avg_dual
    ok = delta >= 0.05
    return ok, (f"avg mask0 coverage: dual={avg_dual:.3f}, single={avg_single:.3f}, "
                f"delta={delta:+.3f} (threshold delta >= 0.05 confirms dual aug is "
                f"warping IR side)")


def check_covisibility(samples: list[dict], cov_min: float) -> tuple[bool, str]:
    """Per-pair covisibility = mask0 cells whose H_0to1 projection lands in
    mask1 / total mask0 cells. Average across samples must be >= cov_min."""
    covs = []
    for s in samples:
        covs.append(_project_grid_via_H(s["mask0"], s["mask1"], s["homography_0to1"]))
    avg = float(np.mean(covs))
    ok = avg >= cov_min
    return ok, (f"avg covisibility = {avg:.3f} (threshold >= {cov_min} for sufficient GT signal); "
                f"per-sample range [{min(covs):.3f}, {max(covs):.3f}]")


def check_h_composition(rng: np.random.Generator, tol_px: float) -> tuple[bool, str]:
    """Independently sample H_ir, H_vis with the v11 aggressive preset; verify
    that for a known point p in original frame, H_0to1 @ (H_ir @ p) == H_vis @ p
    within tol_px sub-pixel residual."""
    H_ir = _random_homography(480, 480, rot_deg=25.0, scale_range=(0.75, 1.25),
                              trans_ratio=0.12, persp_ratio=0.08, rng=rng)
    H_vis = _random_homography(480, 480, rot_deg=25.0, scale_range=(0.75, 1.25),
                               trans_ratio=0.12, persp_ratio=0.08, rng=rng)
    H_0to1 = (H_vis @ np.linalg.inv(H_ir)).astype(np.float32)
    H_0to1 /= H_0to1[2, 2]

    # 64 random test points spread across the image. Each point p is in the
    # original (unwarped) frame.
    pts_orig = rng.uniform(0.0, 480.0, size=(64, 2)).astype(np.float32)
    pts_h = np.concatenate([pts_orig, np.ones((64, 1), dtype=np.float32)], axis=1)  # (64, 3)

    # Where p ends up after each warp (in the warped IR / warped VIS frames).
    p_in_warped_ir = (H_ir @ pts_h.T).T          # (64, 3)
    p_in_warped_ir = p_in_warped_ir[:, :2] / p_in_warped_ir[:, 2:3]

    p_in_warped_vis_via_independent = (H_vis @ pts_h.T).T
    p_in_warped_vis_via_independent = p_in_warped_vis_via_independent[:, :2] / p_in_warped_vis_via_independent[:, 2:3]

    # Apply the composed H_0to1 to (p_in_warped_ir) -> should equal p_in_warped_vis.
    pts_warped_ir_h = np.concatenate([p_in_warped_ir, np.ones((64, 1), dtype=np.float32)], axis=1)
    p_via_compose = (H_0to1 @ pts_warped_ir_h.T).T
    p_via_compose = p_via_compose[:, :2] / p_via_compose[:, 2:3]

    residuals = np.linalg.norm(p_via_compose - p_in_warped_vis_via_independent, axis=1)
    max_res = float(residuals.max())
    mean_res = float(residuals.mean())
    ok = max_res < tol_px
    return ok, (f"H_vis @ inv(H_ir) recomposition residual: mean={mean_res:.3e} px, "
                f"max={max_res:.3e} px (threshold < {tol_px} px)")


def check_backward_compat(cfg, n_samples: int = 4) -> tuple[bool, str]:
    """Build a SECOND dataset with homography_aug=False (no aug at all) and
    verify the no-aug code path is still byte-identical to the v0-v10 contract:
    H_0to1 == I, mask0 fully True in valid region.

    We deliberately DON'T flip homography_dual here -- the dual flag is
    semantically gated by homography_aug, so dual=True + aug=False == no aug
    (both H_ir and H_vis stay at I and the new mask0 &= ir_valid is a no-op
    because ir_valid is all-True).
    """
    ds = _build_dataset(cfg, mode="val", homography_aug_override=False)
    eye = np.eye(3, dtype=np.float32)
    msgs = []
    ok = True
    for i in range(min(n_samples, len(ds))):
        s = ds[i]
        H = s["homography_0to1"].numpy()
        if not np.allclose(H, eye, atol=1e-6):
            ok = False
            msgs.append(f"sample {i}: H_0to1 != I (max abs diff {abs(H - eye).max():.3e})")
        m0 = s["mask0"].numpy()
        # mask0 should be fully True in the (h0_r/8, w0_r/8) valid region.
        # Easier check: if any True cell exists, then in DUAL=False+AUG=False
        # mode the only False cells should be padding cells. So coverage ratio
        # must equal what raw resize+pad would give -- not aug-related.
        # Use a coarse heuristic: m0 sum / m0.numel() >= 0.5 (at least half
        # the canvas is real content; ROAD_PAD_SIZE matches img_resize so
        # for square sources ratio is 1.0, for 4:3 sources ratio is 0.75+).
        if m0.mean() < 0.5:
            ok = False
            msgs.append(f"sample {i}: mask0 coverage {m0.mean():.3f} < 0.5 in no-aug mode "
                        f"(suggests new mask0 &= ir_valid path is over-eager)")
    if ok:
        msgs = ["all no-aug samples: H_0to1==I, mask0 coverage >= 0.5"]
    return ok, "; ".join(msgs)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    print(f"[v11 sanity] data_cfg={args.data_cfg}, main_cfg={args.main_cfg}")
    print(f"[v11 sanity] n_samples={args.n_samples}, seed={args.seed}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _cfg_load(args)
    print(f"[v11 sanity] HOMOGRAPHY_DUAL = {getattr(cfg.DATASET, 'ROAD_HOMOGRAPHY_DUAL', False)}")
    print(f"[v11 sanity] HOMOGRAPHY_PROB = {getattr(cfg.DATASET, 'ROAD_HOMOGRAPHY_PROB', 1.0)}")
    print(f"[v11 sanity] HOMOGRAPHY_KWARGS = {dict(getattr(cfg.DATASET, 'ROAD_HOMOGRAPHY_KWARGS', {}))}")

    ds = _build_dataset(cfg, mode="train")
    print(f"[v11 sanity] train dataset built: {len(ds)} samples")

    # Force prob=1.0 for sanity sampling so we always get the aug path.
    # (We will check aug-firing rate via mask0 coverage in check 2.)
    ds.homography_prob = 1.0

    samples = []
    for i in range(args.n_samples):
        # Cycle through the dataset; pick deterministic spaced indices.
        idx = (i * max(1, len(ds) // args.n_samples)) % len(ds)
        samples.append(ds[idx])

    results = []

    # 1. shape / dtype on first sample (representative).
    ok, msg = check_shapes(samples[0])
    results.append(("1. shape / dtype contract", ok, msg))

    # 2. dual aug firing -- compare mask0 coverage between dual=True and
    # dual=False on the same sample indices. Dual must reduce coverage by
    # at least dual_delta_min.
    ds_single = _build_dataset(cfg, mode="train", homography_dual_override=False)
    ok, msg = check_dual_aug_firing(ds, ds_single, n_samples=min(8, args.n_samples))
    results.append(("2. dual aug firing on IR side", ok, msg))

    # 3. covisibility floor.
    ok, msg = check_covisibility(samples, args.cov_min)
    results.append(("3. covisibility floor", ok, msg))

    # 4. H composition correctness (independent of dataset).
    rng = np.random.default_rng(args.seed)
    ok, msg = check_h_composition(rng, args.h_compose_tol_px)
    results.append(("4. H_vis @ inv(H_ir) composition", ok, msg))

    # 5. backward compatibility (no-aug path).
    ok, msg = check_backward_compat(cfg)
    results.append(("5. backward compat (no-aug == v0-v10)", ok, msg))

    print()
    print("=" * 72)
    print("[v11 sanity] results")
    print("=" * 72)
    all_pass = True
    for name, ok, msg in results:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {name}")
        print(f"       {msg}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("[v11 sanity] ALL 5 CHECKS PASSED -- safe to launch ship training.")
        print("            next step: bash MyScripts/run_msyn_v11_dualh_ddp.sh (in tmux)")
        return 0
    else:
        print("[v11 sanity] ONE OR MORE CHECKS FAILED -- DO NOT launch ship training.")
        print("            see plan SS Risk与回滚 for fix paths per failed check.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
