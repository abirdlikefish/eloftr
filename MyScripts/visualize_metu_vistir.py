"""METU_VISTIR independent evaluation + visualisation script.

Sister of MyScripts/eval_roadscene.py but for METU_VISTIR's pose-based
protocol (auc@5/10/20 deg + prec@5e-4 + per-pair RANSAC R/t error).

Design notes
============

Why a separate .py and not test.py?
  test.py drives PL trainer.test() which:
    - has no figure-saving logic in test_step
    - redirects all output through PL's logging which doesn't tee to stdout
      cleanly when the .bat redirects with `> file 2>&1`
  This script bypasses PL Trainer:
    - hand-rolled DataLoader loop (mirrors eval_roadscene.py)
    - direct print(..., flush=True) so progress bar shows in cmd in real time
    - figures saved at <out_dir>/figures/ capped to args.max_figs
    - overall.txt + summary.csv written in-process (no shell redirect needed)

Pipeline (per pair)
-------------------
1. matcher(batch)                                # gives mkpts0_f, mkpts1_f, mconf
2. compute_symmetrical_epipolar_errors(batch)    # writes batch['epi_errs']
3. 5-restart RANSAC pose:
     for k in range(args.ransac_times):
       shuf -> estimate_pose / estimate_lo_pose -> R, t, inliers
       relative_pose_error(T_0to1, R, t) -> R_err, t_err
     keep best (min max(R, t)); also accumulate ALL 5 R/t for aggregate.
4. record per-pair row + (optional) save figure (capped to max_figs)
5. print one progress line to stdout

Aggregation
-----------
Uses src.utils.metrics.aggregate_metrics with EVAL_TIMES = args.ransac_times
so the reshape(-1, EVAL_TIMES) inside aggregate_metrics matches our flat
R_errs / t_errs accumulator.
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.loftr import LoFTR
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    error_auc,
    estimate_pose,
    relative_pose_error,
)
from src.utils.misc import lower_config
from src.utils.plotting import dynamic_alpha, error_colormap, make_matching_figure


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint (.ckpt). Official or PL-trained.")
    p.add_argument("--main_cfg", required=True,
                   help="LoFTR yacs config used at training (architecture-defining).")
    p.add_argument("--data_cfg", required=True,
                   help="One of configs/data/metu_vistir_test_{all,cloudy_cloudy,cloudy_sunny}.py")
    p.add_argument("--out_dir", required=True,
                   help="Directory for figures/, summary.csv, overall.txt.")
    p.add_argument("--max_pairs", type=int, default=0,
                   help="0 = full set; otherwise stop after N pairs.")
    p.add_argument("--stride", type=int, default=1,
                   help="Sub-sample dataset by stride. Default 1 = full. >1 takes every "
                        "Nth pair so all npz scenes are touched but the total shrinks "
                        "to ~1/stride. e.g. stride=10 on all subset: 2590 -> ~259 pair "
                        "(~10x faster). NOTE: auc@5/10/20 under stride > 1 is only a "
                        "directional preview; cross-version results entries require stride=1.")
    p.add_argument("--max_figs", type=int, default=10,
                   help="Cap saved figure count (0 = none even if save_figures).")
    p.add_argument("--no_save_figures", dest="save_figures",
                   action="store_false", default=True,
                   help="Disable all figure saving regardless of max_figs.")
    p.add_argument("--ransac", default="RANSAC",
                   choices=["RANSAC", "LO-RANSAC"],
                   help="Pose estimation backend; LO-RANSAC needs `pip install poselib`.")
    p.add_argument("--ransac_thr", type=float, default=2.0,
                   help="RANSAC pixel threshold (in normalised image coords / mean K).")
    p.add_argument("--ransac_times", type=int, default=5,
                   help="Per-pair RANSAC restarts; aggregate_metrics' max(R, t) is "
                        "computed over all restarts. Higher = more robust auc.")
    p.add_argument("--thr", type=float, default=0.1,
                   help="LOFTR.MATCH_COARSE.THR override (default 0.1 matches v0..v11 eval; "
                        "official outdoor.ckpt AUC reproduction also uses 0.1, NOT cfg's "
                        "comment-recommended 0.2 -- see scripts/reproduce_test/outdoor_full_auc.sh).")
    p.add_argument("--megasize", type=int, default=None,
                   help="Override cfg.DATASET.MGDPT_IMG_RESIZE (long-edge resize). "
                        "Default None = keep cfg value (832 for METU). Official outdoor "
                        "AUC reproduction uses 1152 (--megasize 1152). For METU's 4K VIS "
                        "this gives ~3.3x downsample instead of ~4.6x, preserving more "
                        "texture for cross-modal matching. Mirrors test.py --megasize.")
    p.add_argument("--npe", action="store_true", default=False,
                   help="Enable NPE (Neural Position Encoding) extrapolation for ROPE. "
                        "When set: NPE = [832, 832, MGDPT_IMG_RESIZE, MGDPT_IMG_RESIZE], "
                        "telling RoPE that training was at 832 long-edge but testing is "
                        "at MGDPT_IMG_RESIZE. MUST be paired with --megasize when "
                        "MGDPT_IMG_RESIZE != 832, otherwise RoPE frequencies misalign. "
                        "Mirrors test.py --npe.")
    p.add_argument("--metu_side0", default=None, choices=[None, "vis", "thermal"],
                   help="Override cfg.DATASET.METU_SIDE0 (which modality goes into image0 "
                        "slot). Default None = keep cfg value. v10/v11/v12 finetuned ckpts "
                        "with modemb_ir bound to image0 require 'thermal'; official "
                        "outdoor.ckpt (no modemb) empirically prefers 'vis' (~4x higher AUC).")
    p.add_argument("--metu_side1", default=None, choices=[None, "vis", "thermal"],
                   help="Override cfg.DATASET.METU_SIDE1. Must differ from --metu_side0 "
                        "(cross-modal eval requires both sides used).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Config + model (mirrors eval_roadscene.py:104-155 byte-for-byte)
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg)
    cfg.merge_from_file(args.data_cfg)

    # --megasize override (mirrors test.py:94-95). MUST happen BEFORE the
    # --npe block below because NPE consumes MGDPT_IMG_RESIZE.
    if args.megasize is not None:
        cfg.DATASET.MGDPT_IMG_RESIZE = int(args.megasize)

    # --npe handling (mirrors test.py:97-106). When set, NPE is computed
    # from the (now possibly overridden) MGDPT_IMG_RESIZE so RoPE knows
    # it was trained at 832 long-edge but is being tested at a different
    # resolution. When NOT set, fall back to the legacy [832,832,832,832]
    # default that matches v0..v11 eval bytes-identically.
    if args.npe:
        if cfg.LOFTR.COARSE.ROPE:
            assert cfg.DATASET.NPE_NAME is not None, \
                "--npe requires cfg.DATASET.NPE_NAME (set in metu_vistir_test_*.py)"
        if cfg.DATASET.NPE_NAME == 'megadepth':
            cfg.LOFTR.COARSE.NPE = [832, 832,
                                    cfg.DATASET.MGDPT_IMG_RESIZE,
                                    cfg.DATASET.MGDPT_IMG_RESIZE]
        elif cfg.DATASET.NPE_NAME == 'scannet':
            cfg.LOFTR.COARSE.NPE = [832, 832,
                                    cfg.DATASET.SCAN_IMG_RESIZEX,
                                    cfg.DATASET.SCAN_IMG_RESIZEX]
    elif cfg.LOFTR.COARSE.NPE is None:
        cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    # --metu_side0 / --metu_side1 override (cross-modal eval side selection).
    # v10/v11/v12 finetuned ckpts: keep cfg default 'thermal'/'vis' (modemb_ir
    # bound to image0). Official outdoor.ckpt: pass 'vis'/'thermal' for ~4x
    # higher AUC (no modemb so the empirically better side wins).
    if args.metu_side0 is not None:
        cfg.DATASET.METU_SIDE0 = str(args.metu_side0)
    if args.metu_side1 is not None:
        cfg.DATASET.METU_SIDE1 = str(args.metu_side1)
    if cfg.DATASET.METU_SIDE0 == cfg.DATASET.METU_SIDE1:
        raise ValueError(f"METU_SIDE0 == METU_SIDE1 == {cfg.DATASET.METU_SIDE0!r}; "
                         f"cross-modal eval requires different sides.")

    # Force fp32 path so test-time numerics match training validation.
    cfg.LOFTR.MP = False
    cfg.LOFTR.HALF = False

    if args.thr is not None:
        cfg.LOFTR.MATCH_COARSE.THR = float(args.thr)

    # aggregate_metrics reshapes pose_errors to (N_pair, EVAL_TIMES); set
    # EVAL_TIMES so it matches our per-pair accumulator length.
    cfg.LOFTR.EVAL_TIMES = int(args.ransac_times)

    # RANSAC parameters consumed by compute_pose_errors (we don't call it,
    # but estimate_pose/estimate_lo_pose use them indirectly via cfg in
    # a few code paths). Setting them avoids surprise defaults if anyone
    # extends this script later.
    cfg.TRAINER.POSE_ESTIMATION_METHOD = args.ransac
    cfg.TRAINER.RANSAC_PIXEL_THR = float(args.ransac_thr)
    return cfg


def _strip_matcher_prefix(state_dict: Dict[str, torch.Tensor]):
    """PL ckpts prefix every LoFTR key with `matcher.`; official outdoor.ckpt
    has no prefix. Handle both (mirror of eval_roadscene.py:124-136)."""
    if not any(k.startswith("matcher.") for k in state_dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("matcher."):
            cleaned[k[len("matcher."):]] = v
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
              f"{msg.missing_keys[:6]}{' ...' if len(msg.missing_keys) > 6 else ''}",
              flush=True)
    if msg.unexpected_keys:
        print(f"  [load] unexpected_keys ({len(msg.unexpected_keys)}): "
              f"{msg.unexpected_keys[:6]}{' ...' if len(msg.unexpected_keys) > 6 else ''}",
              flush=True)

    matcher = matcher.eval().to(device)
    return matcher


# --------------------------------------------------------------------------- #
# Dataset (delegated to MultiSceneDataModule so the elif metu_vistir
# dispatch + METU_* cfg fields are reused as-is)
# --------------------------------------------------------------------------- #
class _DataModuleArgs:
    """Minimal stand-in for the train.py argparse Namespace; only the fields
    MultiSceneDataModule.__init__ touches in test mode are populated."""
    def __init__(self, num_workers: int):
        self.batch_size = 1
        self.num_workers = num_workers
        self.pin_memory = False
        self.parallel_load_data = False


def build_dataset(cfg, args: argparse.Namespace):
    dm = MultiSceneDataModule(_DataModuleArgs(args.num_workers), cfg)
    dm.setup(stage='test')
    return dm.test_dataset    # ConcatDataset of N METUVisTIRDataset


# --------------------------------------------------------------------------- #
# Per-pair RANSAC (mirrors src/utils/metrics.compute_pose_errors but keeps
# all restarts for aggregate_metrics + tracks the BEST one for csv display)
# --------------------------------------------------------------------------- #
def _ransac_pose_5(pts0, pts1, K0, K1, T_0to1,
                   ransac_thr: float, ransac_times: int,
                   backend: str = "RANSAC"):
    """Returns
    (R_errs_flat:list[float], t_errs_flat:list[float],  # length == ransac_times
     R_err_best:float, t_err_best:float,
     inliers_best:np.ndarray)
    """
    use_lo = (backend == "LO-RANSAC")
    if use_lo:
        from src.utils.metrics import estimate_lo_pose

    R_errs, t_errs = [], []
    R_best = t_best = float('inf')
    inl_best = np.array([], dtype=bool)

    n = len(pts0)
    for trial in range(ransac_times):
        if n < 5:
            R_errs.append(np.inf); t_errs.append(np.inf)
            continue
        rng = np.random.permutation(n)
        shuf0 = pts0[rng]
        shuf1 = pts1[rng]
        if use_lo:
            est = estimate_lo_pose(shuf0, shuf1, K0, K1, ransac_thr,
                                   conf=0.99999)
            if not est["success"]:
                R_errs.append(90.0); t_errs.append(90.0)
                continue
            M = est["M_0to1"]
            R = M.R.numpy() if hasattr(M.R, 'numpy') else np.asarray(M.R)
            t = M.t.numpy() if hasattr(M.t, 'numpy') else np.asarray(M.t)
            inl = est["inliers"].numpy().astype(bool)
        else:
            ret = estimate_pose(shuf0, shuf1, K0, K1, ransac_thr,
                                conf=0.99999)
            if ret is None:
                R_errs.append(np.inf); t_errs.append(np.inf)
                continue
            R, t, inl = ret
        t_e, R_e = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
        R_errs.append(float(R_e))
        t_errs.append(float(t_e))
        if max(R_e, t_e) < max(R_best, t_best):
            R_best, t_best = float(R_e), float(t_e)
            inl_best = inl.astype(bool)

    return R_errs, t_errs, R_best, t_best, inl_best


# --------------------------------------------------------------------------- #
# Figure (METU pose-based variant of _make_evaluation_figure; conf_matrix_gt
# field absent at test time so the recall-line is dropped)
# --------------------------------------------------------------------------- #
def _save_metu_pair_figure(out_path: Path, batch: dict, epi_errs: np.ndarray,
                           R_err: float, t_err: float, n_inl: int,
                           ckpt_name: str, conf_thr: float = 5e-4):
    img0 = (batch['image0'][0][0].detach().cpu().numpy() * 255).round().astype(np.int32)
    img1 = (batch['image1'][0][0].detach().cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = batch['mkpts0_f'].detach().cpu().numpy()
    kpts1 = batch['mkpts1_f'].detach().cpu().numpy()

    # raw-coord -> resize-coord. NOTE: dataset emits scale = [w_raw/w_resized,
    # h_raw/h_resized], so to undo we divide kpts (col 0 = x, col 1 = y) by
    # [s_w, s_h] directly with column-wise broadcast. The legacy
    # _make_evaluation_figure in src/utils/plotting.py uses scale[[1, 0]]
    # (= [s_h, s_w]) which swaps x/y axes -- a long-standing LoFTR-repo bug
    # that does not affect numeric metrics, only visual figure aspect.
    scale0 = batch['scale0'][0].detach().cpu().numpy()   # [s_w, s_h]
    scale1 = batch['scale1'][0].detach().cpu().numpy()
    kpts0_vis = kpts0 / scale0
    kpts1_vis = kpts1 / scale1

    correct_mask = epi_errs < conf_thr
    n_correct = int(correct_mask.sum())
    precision = float(correct_mask.mean()) if len(epi_errs) else 0.0

    color = error_colormap(epi_errs, conf_thr,
                           alpha=dynamic_alpha(len(epi_errs))) \
            if len(epi_errs) else np.zeros((0, 4))

    rel0 = batch['pair_names'][0][0]
    rel1 = batch['pair_names'][1][0]
    # Per-pair AUC@5/10/20 deg hit indicator. AUC itself is dataset-aggregate
    # (overall.txt) - on a single figure we can only show whether THIS pair's
    # max(R, t) falls under each threshold (= contributes to AUC@N). ASCII
    # 'OK' / '--' instead of unicode check / cross to avoid matplotlib font
    # box-out on Windows default fonts.
    pose_err = max(R_err, t_err)
    tag5  = 'OK' if pose_err <= 5.0  else '--'
    tag10 = 'OK' if pose_err <= 10.0 else '--'
    tag20 = 'OK' if pose_err <= 20.0 else '--'
    text = [
        f'ckpt: {ckpt_name}',
        f'#Matches {len(kpts0)}',
        f'Prec@{conf_thr:.0e}: {100 * precision:.1f}% ({n_correct}/{len(kpts0)})',
        f'R_err={R_err:6.2f} deg  t_err={t_err:6.2f} deg  inliers={n_inl}',
        f'pose_err=max(R,t)={pose_err:6.2f} deg    AUC5: {tag5}    AUC10: {tag10}    AUC20: {tag20}',
        f'{Path(rel0).name}  vs  {Path(rel1).name}',
    ]
    fig = make_matching_figure(img0, img1, kpts0_vis, kpts1_vis, color, text=text)
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
    fig_dir = out_dir / "figures"
    save_fig = args.save_figures and args.max_figs > 0
    if save_fig:
        fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Cannot find checkpoint: {ckpt_path}")

    cfg = build_config(args)
    matcher = build_matcher(cfg, ckpt_path, args.device)
    dataset = build_dataset(cfg, args)
    full_len = len(dataset)

    if args.stride > 1:
        from torch.utils.data import Subset
        selected = list(range(0, full_len, args.stride))
        dataset = Subset(dataset, selected)
        print(f"  stride={args.stride}: subsampled {len(dataset)}/{full_len} pairs "
              f"(touches every npz scene; auc/prec are PREVIEW-only)",
              flush=True)

    n_total = len(dataset)
    if args.max_pairs > 0:
        n_total = min(n_total, args.max_pairs)
    print(f"Evaluating {n_total} pairs (full dataset = {full_len}) with ckpt={ckpt_path}",
          flush=True)
    print(f"  main_cfg : {args.main_cfg}", flush=True)
    print(f"  data_cfg : {args.data_cfg}", flush=True)
    print(f"  out_dir  : {out_dir}", flush=True)
    print(f"  ransac   : {args.ransac}  thr={args.ransac_thr}  times={args.ransac_times}",
          flush=True)
    print(f"  thr      : LOFTR.MATCH_COARSE.THR = {cfg.LOFTR.MATCH_COARSE.THR}",
          flush=True)
    print(f"  figures  : save={save_fig}, max={args.max_figs}", flush=True)
    print(f"  side0/1  : {cfg.DATASET.METU_SIDE0} / {cfg.DATASET.METU_SIDE1}",
          flush=True)
    print(f"  undistort: {cfg.DATASET.METU_UNDISTORT}", flush=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=int(args.num_workers))

    rows: List[dict] = []
    identifiers: List[str] = []
    all_epi: List[np.ndarray] = []
    all_R: List[float] = []
    all_t: List[float] = []
    all_n_matches: List[int] = []
    all_inliers: List[np.ndarray] = []

    for idx, batch in enumerate(loader, start=1):
        if args.max_pairs and idx > args.max_pairs:
            break

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device, non_blocking=True)

        with torch.no_grad():
            matcher(batch)
        compute_symmetrical_epipolar_errors(batch)

        epi = batch['epi_errs'].detach().cpu().numpy()
        pts0 = batch['mkpts0_f'].detach().cpu().numpy()
        pts1 = batch['mkpts1_f'].detach().cpu().numpy()
        K0 = batch['K0'][0].cpu().numpy()
        K1 = batch['K1'][0].cpu().numpy()
        T_0to1 = batch['T_0to1'][0].cpu().numpy()

        R_errs_5, t_errs_5, R_best, t_best, inl_best = _ransac_pose_5(
            pts0, pts1, K0, K1, T_0to1,
            ransac_thr=args.ransac_thr,
            ransac_times=args.ransac_times,
            backend=args.ransac,
        )

        # batch field unwrap (default_collate at bs=1):
        #   scene_id : list[str], pair_names: tuple(list, list), pair_id: tensor([id])
        scene_id = batch['scene_id'][0] if isinstance(batch['scene_id'], (list, tuple)) \
                   else str(batch['scene_id'])
        pair_id_v = batch['pair_id']
        if torch.is_tensor(pair_id_v):
            pair_id = int(pair_id_v[0].item())
        elif isinstance(pair_id_v, (list, tuple)):
            pair_id = int(pair_id_v[0])
        else:
            pair_id = int(pair_id_v)
        rel0 = batch['pair_names'][0][0]
        rel1 = batch['pair_names'][1][0]

        identifiers.append(f"{scene_id}#{pair_id}")
        all_epi.append(epi)
        all_R.extend(R_errs_5)
        all_t.extend(t_errs_5)
        all_n_matches.append(int(len(epi)))
        all_inliers.append(inl_best.astype(bool))

        mconf = batch['mconf'].detach().cpu().numpy() if 'mconf' in batch \
                else np.zeros(0)
        pose_err_best = float(max(R_best, t_best))
        per = {
            'scene_id': scene_id,
            'pair_id': pair_id,
            'name0': rel0,
            'name1': rel1,
            'num_matches': int(len(epi)),
            'mean_conf': float(mconf.mean()) if len(mconf) else 0.0,
            'median_conf': float(np.median(mconf)) if len(mconf) else 0.0,
            'epi_err_median': float(np.median(epi)) if len(epi) else 0.0,
            'epi_err_p10': float(np.percentile(epi, 10)) if len(epi) else 0.0,
            'prec@5e-04': float((epi < 5e-4).mean()) if len(epi) else 0.0,
            'R_err_best': float(R_best),
            't_err_best': float(t_best),
            'inliers_best': int(inl_best.sum()),
            # Per-pair AUC@5/10/20 deg hit indicators. AUC itself is a
            # dataset-level aggregate (computed in overall.txt via
            # aggregate_metrics); the per-pair correct@Ndeg = 1 iff this
            # pair's max(R_err, t_err) <= N deg, i.e. the pair contributes
            # to AUC@N. inf <= N is naturally False (counts as miss).
            'pose_err_best': pose_err_best,
            'correct@5deg':  int(pose_err_best <= 5.0),
            'correct@10deg': int(pose_err_best <= 10.0),
            'correct@20deg': int(pose_err_best <= 20.0),
        }
        rows.append(per)

        if save_fig and idx <= args.max_figs:
            fig_path = fig_dir / f"{idx:04d}_{scene_id}_pair{pair_id:04d}.png"
            try:
                _save_metu_pair_figure(fig_path, batch, epi, R_best, t_best,
                                       int(inl_best.sum()), ckpt_path.name)
            except Exception as e:
                # Don't let figure failures kill the whole eval; just warn.
                print(f"  [WARN] figure save failed for {fig_path.name}: {e}",
                      flush=True)

        print(f"[{idx:4d}/{n_total}] {scene_id}#{pair_id:04d}  "
              f"matches={per['num_matches']:4d}  "
              f"epi_med={per['epi_err_median']:.2e}  "
              f"R={per['R_err_best']:6.2f}  t={per['t_err_best']:6.2f}  "
              f"pose={per['pose_err_best']:6.2f}  "
              f"hit5={per['correct@5deg']} hit10={per['correct@10deg']} hit20={per['correct@20deg']}  "
              f"prec@5e-4={100 * per['prec@5e-04']:5.1f}%",
              flush=True)

    # ---- per-pair csv ------------------------------------------------------
    csv_path = out_dir / "summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ---- overall.txt via MINIMA-style aggregation --------------------------
    # MINIMA / XoFTR test_relative_pose_infrared.py: per-npz error_auc -> split
    # ('_scene')[0] class-mean -> two-class final mean. NOT the legacy pooled
    # `aggregate_metrics` which flattens all 2590 pair errors into a single AUC
    # (pair-weighted, biased toward npz with more pairs).
    ransac_times = max(1, int(args.ransac_times))
    n_pair = len(identifiers)
    if len(all_R) != n_pair * ransac_times or len(all_t) != n_pair * ransac_times:
        raise RuntimeError(
            f"flat R/t length mismatch: len(all_R)={len(all_R)}, "
            f"len(all_t)={len(all_t)}, expected n_pair*ransac_times="
            f"{n_pair}*{ransac_times}={n_pair * ransac_times}")
    R_arr = np.array(all_R, dtype=np.float64).reshape(n_pair, ransac_times)
    t_arr = np.array(all_t, dtype=np.float64).reshape(n_pair, ransac_times)
    # MINIMA pose error = max(R, t) per single ransac call; we keep
    # per-pair "best across restarts" = min over restarts of max(R, t).
    # ransac_times=1 (MINIMA-exact) reduces to single-shot max(R, t).
    pose_err_per_pair = np.min(np.maximum(R_arr, t_arr), axis=1)

    # per-scene buckets (scene_id is the npz name without .npz, e.g.
    # 'cloudy_cloudy_scene_1'). identifier format = '{scene_id}#{pair_id}'.
    scene_to_pose_errs: Dict[str, List[float]] = defaultdict(list)
    scene_to_n_pairs: Dict[str, int] = defaultdict(int)
    for ident, pe in zip(identifiers, pose_err_per_pair):
        scene_name = ident.split('#')[0]
        scene_to_pose_errs[scene_name].append(float(pe))
        scene_to_n_pairs[scene_name] += 1

    # per-scene AUC. error_auc returns 0..1; multiply by 100 to align with
    # MINIMA paper Table 3 percentage units.
    scene_aucs: "OrderedDict[str, dict]" = OrderedDict()
    for scene_name in sorted(scene_to_pose_errs.keys()):
        aucs_dict = error_auc(scene_to_pose_errs[scene_name], [5, 10, 20])
        scene_aucs[scene_name] = {
            'auc@5':  100.0 * float(aucs_dict['auc@5']),
            'auc@10': 100.0 * float(aucs_dict['auc@10']),
            'auc@20': 100.0 * float(aucs_dict['auc@20']),
            'pairs':  scene_to_n_pairs[scene_name],
        }

    # per-class AUC: split('_scene')[0] yields 'cloudy_cloudy' / 'cloudy_sunny'
    # (mirrors XoFTR aggregiate_scenes). Simple arithmetic mean within class.
    class_to_scenes: Dict[str, List[str]] = defaultdict(list)
    for scene_name in scene_aucs.keys():
        cls = scene_name.split('_scene')[0]
        class_to_scenes[cls].append(scene_name)
    class_aucs: "OrderedDict[str, dict]" = OrderedDict()
    for cls in sorted(class_to_scenes.keys()):
        class_aucs[cls] = {
            f'auc@{t}': float(np.mean([scene_aucs[s][f'auc@{t}']
                                       for s in class_to_scenes[cls]]))
            for t in (5, 10, 20)
        }

    # overall: 2-class arithmetic mean ↔ MINIMA paper Table 3 row.
    all_class_mean = {
        f'auc@{t}': float(np.mean([cls_d[f'auc@{t}']
                                   for cls_d in class_aucs.values()]))
        for t in (5, 10, 20)
    }
    mean_num_matches = float(np.mean(all_n_matches)) if all_n_matches else 0.0

    overall_path = out_dir / "overall.txt"
    with overall_path.open("w", encoding="utf-8") as f:
        f.write("protocol: MINIMA / XoFTR (test_relative_pose_infrared.py)\n")
        f.write(f"ckpt: {ckpt_path}\n")
        f.write(f"main_cfg: {args.main_cfg}\n")
        f.write(f"data_cfg: {args.data_cfg}\n")
        f.write(f"pairs: {len(rows)}  scenes: {len(scene_aucs)}\n")
        f.write(f"side0: {cfg.DATASET.METU_SIDE0}  "
                f"side1: {cfg.DATASET.METU_SIDE1}\n")
        f.write(f"undistort: {cfg.DATASET.METU_UNDISTORT} "
                f"(getOptimalNewCameraMatrix alpha=0)\n")
        f.write(f"pad_to_square: "
                f"{getattr(cfg.DATASET, 'METU_PAD_TO_SQUARE', False)}\n")
        f.write(f"ransac: {args.ransac}  thr: {args.ransac_thr}  "
                f"times: {args.ransac_times}\n")
        f.write(f"loftr_thr: {cfg.LOFTR.MATCH_COARSE.THR}  "
                f"megasize: {cfg.DATASET.MGDPT_IMG_RESIZE}\n")
        f.write("---\n")
        f.write("[per-scene] (units: %)\n")
        for scene_name, d in scene_aucs.items():
            f.write(f"  {scene_name:34s}  auc@5: {d['auc@5']:6.3f}  "
                    f"auc@10: {d['auc@10']:6.3f}  "
                    f"auc@20: {d['auc@20']:6.3f}  "
                    f"pairs: {d['pairs']}\n")
        f.write("---\n")
        f.write("[per-class] (XoFTR aggregiate_scenes equivalent, units: %)\n")
        for cls, d in class_aucs.items():
            f.write(f"  {cls:34s}  auc@5: {d['auc@5']:6.3f}  "
                    f"auc@10: {d['auc@10']:6.3f}  "
                    f"auc@20: {d['auc@20']:6.3f}\n")
        f.write("---\n")
        f.write("[overall] (units: %)\n")
        f.write(f"  all_class_mean                      "
                f"auc@5: {all_class_mean['auc@5']:6.3f}  "
                f"auc@10: {all_class_mean['auc@10']:6.3f}  "
                f"auc@20: {all_class_mean['auc@20']:6.3f}"
                f"   # compare with MINIMA paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72\n")
        f.write(f"num_matches: {mean_num_matches:.2f}\n")

    print("", flush=True)
    print("============================================================", flush=True)
    print(f"  pairs        : {len(rows)}   scenes: {len(scene_aucs)}", flush=True)
    print(f"  per-class auc@5 / @10 / @20 (units: %):", flush=True)
    for cls, d in class_aucs.items():
        print(f"    {cls:18s}  {d['auc@5']:6.3f}  {d['auc@10']:6.3f}  "
              f"{d['auc@20']:6.3f}", flush=True)
    print(f"  all_class_mean      "
          f"{all_class_mean['auc@5']:6.3f}  "
          f"{all_class_mean['auc@10']:6.3f}  "
          f"{all_class_mean['auc@20']:6.3f}   "
          f"(paper Table 3 ELoFTR: 2.88 / 7.88 / 17.72)", flush=True)
    print(f"  num_matches  : {mean_num_matches:.2f}", flush=True)
    print(f"  summary.csv  : {csv_path}", flush=True)
    print(f"  overall.txt  : {overall_path}", flush=True)
    if save_fig:
        n_saved = min(len(rows), args.max_figs)
        print(f"  figures      : {fig_dir}  ({n_saved} png saved, capped at "
              f"--max_figs {args.max_figs})", flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
