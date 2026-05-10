"""Read a PyTorch-Lightning 1.3.5 TensorBoard events file and emit AI-friendly
JSON / CSV / table. Stable schema, designed to be the single entry point for AI
assistants that need to inspect training results without opening TensorBoard UI.

Why this exists
---------------
Each EfficientLoFTR experiment writes one ``events.out.tfevents.*`` file under
``logs/tb_logs/<exp>/version_<N>/``. The files can be huge (v9 = 784 MB) because
``add_figure`` encodes matplotlib figures as PNG into the event stream. Direct
``EventAccumulator(...).Reload()`` will happily decode every PNG and OOM on a
laptop. This script uses ``size_guidance={'images': 1, 'tensors': 1}`` so only
the scalar timeseries are decoded; figures stay as opaque pointers we never
materialise.

Sub-commands
------------
* ``tags``     -- table of every scalar tag, sample count, first/last value
* ``summary``  -- AI-optimised JSON: dataset type, training stats, val per-epoch,
                  best epoch + monitor value, and (for v9) auto-graded acceptance
* ``export <tag>`` -- dump (step, wall_time, value) of one scalar to CSV
* ``best <tag>``   -- find the max/min of one scalar with its step
* ``aggregate``    -- scan ``logs/tb_logs/*`` and emit cross-experiment Markdown
                      tables (training cost / best val epoch / modemb-MSBN diag);
                      backs ``results/tb_summary.md`` workflow

Path resolution (--logdir accepts any of)
-----------------------------------------
1. ``logs/tb_logs/m3fd_v9_e2e_outdoor/version_0/events.out.tfevents.1778078682.3090.382918.0``
2. ``logs/tb_logs/m3fd_v9_e2e_outdoor/version_0``
3. ``logs/tb_logs/m3fd_v9_e2e_outdoor``  (auto picks the highest-numbered version)

When ``--logdir`` is omitted the script ``input()``-prompts for a path so the
user can paste it interactively (no need to remember the long tfevents filename).

Examples
--------
    python MyScripts/read_tb_metrics.py summary --logdir logs/tb_logs/m3fd_v9_e2e_outdoor
    python MyScripts/read_tb_metrics.py summary  # interactive prompt
    python MyScripts/read_tb_metrics.py tags    --logdir logs/tb_logs/m3fd_v9_e2e_outdoor
    python MyScripts/read_tb_metrics.py export "metrics_0/precision@1px" \
        --logdir logs/tb_logs/m3fd_v9_e2e_outdoor --out p1.csv
    python MyScripts/read_tb_metrics.py best "metrics_0/precision@1px" \
        --logdir logs/tb_logs/m3fd_v9_e2e_outdoor

Dependencies
------------
``tensorboard`` (already pulled in by pytorch-lightning 1.3.5). Pure stdlib
otherwise. No pandas / tbparse required, so this script also runs inside the
slim ``eloftr_yurupeng`` env on the vlrlab server.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_events(logdir: Optional[str]) -> Path:
    """Normalise the user-provided path into a concrete events file."""
    if logdir is None:
        try:
            logdir = input(
                "请粘贴 TensorBoard 路径 "
                "(events 文件 / version_x 目录 / 实验目录): "
            ).strip().strip('"').strip("'")
        except EOFError:
            sys.exit("[err] no --logdir given and no stdin available")
    if not logdir:
        sys.exit("[err] empty path")

    p = Path(logdir).expanduser().resolve()
    if not p.exists():
        sys.exit(f"[err] path not found: {p}")

    if p.is_file():
        if not p.name.startswith("events.out.tfevents."):
            sys.exit(f"[err] not a tfevents file: {p}")
        return p

    # directory case: try as version_x first, then as experiment dir
    direct_events = sorted(p.glob("events.out.tfevents.*"))
    if direct_events:
        return _pick_largest(direct_events)

    versions = sorted(
        [d for d in p.glob("version_*") if d.is_dir()],
        key=lambda d: _safe_int(d.name.split("_")[-1]),
    )
    if versions:
        chosen = versions[-1]
        events = sorted(chosen.glob("events.out.tfevents.*"))
        if events:
            print(
                f"[info] auto-selected newest version: {chosen.name}",
                file=sys.stderr,
            )
            return _pick_largest(events)

    sys.exit(f"[err] no events.out.tfevents.* found under {p}")


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -1


def _pick_largest(events: List[Path]) -> Path:
    """When a directory contains multiple events files (rare, after a crash +
    resume), prefer the largest one — it's almost always the longest run."""
    if len(events) == 1:
        return events[0]
    chosen = max(events, key=lambda p: p.stat().st_size)
    print(
        f"[info] {len(events)} tfevents files, picked largest: {chosen.name} "
        f"({chosen.stat().st_size / 1024**2:.1f} MB)",
        file=sys.stderr,
    )
    return chosen


def load_ea(events_path: Path) -> EventAccumulator:
    """Construct EventAccumulator with figure-skipping size_guidance.

    images=1 / tensors=1 means: keep the index entries (so we know they exist)
    but never decode the PNG / serialised tensor. This is the difference
    between 'opens in 3 seconds' and 'OOMs on a 16 GB laptop'.
    """
    ea = EventAccumulator(
        str(events_path),
        size_guidance={
            "scalars": 0,
            "images": 1,
            "tensors": 1,
            "histograms": 1,
            "audio": 1,
            "graph": 0,
            "compressedHistograms": 1,
        },
    )
    ea.Reload()
    return ea


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_tags(ea: EventAccumulator) -> None:
    scalar_tags = sorted(ea.Tags().get("scalars", []))
    if not scalar_tags:
        print("[warn] no scalar tags found.")
    else:
        print(
            f"{'tag':<55} {'n':>6} {'first':>12} {'last':>12} "
            f"{'s0':>10} {'s_last':>12}"
        )
        print("-" * 110)
        for t in scalar_tags:
            ev = ea.Scalars(t)
            n = len(ev)
            v0 = ev[0].value
            vL = ev[-1].value
            s0 = ev[0].step
            sL = ev[-1].step
            print(
                f"{t:<55} {n:>6} {v0:>12.4f} {vL:>12.4f} {s0:>10} {sL:>12}"
            )

    images = ea.Tags().get("images", [])
    if images:
        print(
            f"\n[note] {len(images)} image tag(s) skipped "
            f"(size_guidance=1, no decode):"
        )
        for t in images[:8]:
            print(f"  {t}")
        if len(images) > 8:
            print(f"  ... ({len(images) - 8} more)")

    other_kinds = ["histograms", "tensors", "audio", "graph"]
    for k in other_kinds:
        ts = ea.Tags().get(k, [])
        if ts:
            print(f"\n[note] {len(ts)} {k} tag(s):")
            for t in ts[:5]:
                print(f"  {t}")
            if len(ts) > 5:
                print(f"  ... ({len(ts) - 5} more)")


def cmd_summary(ea: EventAccumulator, events_path: Path) -> Dict:
    scalar_tags = set(ea.Tags().get("scalars", []))

    is_aligned = any(t.startswith("metrics_0/precision@") for t in scalar_tags)
    is_megadepth = any(t.startswith("metrics_0/auc@") for t in scalar_tags)

    version_dir = events_path.parent
    exp_dir = version_dir.parent
    exp_name = exp_dir.name
    version_match = re.match(r"version_(\d+)", version_dir.name)
    version = int(version_match.group(1)) if version_match else None

    # wall-clock spans: scan a few short tags to bound min/max wall_time
    wall_times: List[float] = []
    sample_tags = []
    for t in ("train/avg_loss_on_epoch", "train/loss"):
        if t in scalar_tags:
            sample_tags.append(t)
    if not sample_tags and scalar_tags:
        sample_tags = list(scalar_tags)[:3]
    for t in sample_tags:
        for e in ea.Scalars(t):
            wall_times.append(e.wall_time)
    wall = None
    if wall_times:
        t0, t1 = min(wall_times), max(wall_times)
        wall = {
            "start_iso": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "end_iso": dt.datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
            "elapsed_hours": round((t1 - t0) / 3600.0, 3),
        }

    out: Dict = {
        "schema_version": 1,
        "logdir": str(version_dir),
        "events_file": events_path.name,
        "events_size_mb": round(events_path.stat().st_size / 1024**2, 2),
        "exp_name": exp_name,
        "version": version,
        "dataset_type": (
            "aligned_irvis"
            if is_aligned
            else ("megadepth" if is_megadepth else "unknown")
        ),
        "wall_clock": wall,
    }

    # ------------------- training section -------------------
    train: Dict = {}
    if "train/loss" in scalar_tags:
        ev = ea.Scalars("train/loss")
        train["total_global_steps"] = ev[-1].step
        train["final_train_loss"] = round(ev[-1].value, 6)
        train["min_train_loss"] = {
            "value": round(min(e.value for e in ev), 6),
            "step": min(ev, key=lambda x: x.value).step,
        }
    if "train/avg_loss_on_epoch" in scalar_tags:
        ev = ea.Scalars("train/avg_loss_on_epoch")
        train["total_epochs"] = ev[-1].step + 1
        train["final_avg_loss_on_epoch"] = round(ev[-1].value, 6)

    if "train/mod_emb_ir_norm" in scalar_tags:
        ir = ea.Scalars("train/mod_emb_ir_norm")
        vis = ea.Scalars("train/mod_emb_vis_norm")
        train["modality_emb"] = {
            "ir_norm_first": round(ir[0].value, 4),
            "ir_norm_last": round(ir[-1].value, 4),
            "vis_norm_first": round(vis[0].value, 4),
            "vis_norm_last": round(vis[-1].value, 4),
            "verdict": (
                "modemb learning (norm grew >0.05)"
                if max(ir[-1].value, vis[-1].value) > 0.05
                else "modemb dead (norm <=0.05; check MODALITY_EMB_INIT)"
            ),
        }

    if "train/bn_drift_ratio_layer1" in scalar_tags:
        msbn: Dict = {}
        for ln in ("layer1", "layer2"):
            ev = ea.Scalars(f"train/bn_drift_ratio_{ln}")
            msbn[f"{ln}_drift_first"] = round(ev[0].value, 4)
            msbn[f"{ln}_drift_last"] = round(ev[-1].value, 4)
            msbn[f"{ln}_drift_max"] = round(max(e.value for e in ev), 4)
        max_drift_last = max(
            msbn["layer1_drift_last"], msbn["layer2_drift_last"]
        )
        msbn["verdict"] = (
            "MSBN active (drift_last > 0.05)"
            if max_drift_last > 0.05
            else "MSBN dead (drift_last <= 0.05; PC+CLAHE may have aligned "
            "distributions in earlier layers)"
        )
        train["msbn"] = msbn

    out["training"] = train

    # ------------------- validation section -------------------
    val: Dict = {}
    if is_aligned:
        monitor = "metrics_0/precision@1px"
        nice_keys: Dict[str, str] = {
            "precision@1px": "metrics_0/precision@1px",
            "precision@3px": "metrics_0/precision@3px",
            "precision@5px": "metrics_0/precision@5px",
            "mean_pixel_error": "metrics_0/mean_pixel_error",
            "num_matches": "metrics_0/num_matches",
            "mean_conf": "metrics_0/mean_conf",
            "avg_loss": "val_0/avg_loss",
        }
        best_mode = "max"
    elif is_megadepth:
        monitor = "metrics_0/auc@5"
        nice_keys = {
            "auc@5": "metrics_0/auc@5",
            "auc@10": "metrics_0/auc@10",
            "auc@20": "metrics_0/auc@20",
            "avg_loss": "val_0/avg_loss",
        }
        best_mode = "max"
    else:
        monitor = None
        nice_keys = {}
        best_mode = "max"

    if monitor and monitor in scalar_tags:
        ev_monitor = ea.Scalars(monitor)

        per_ep_data: Dict[int, Dict[str, float]] = {}
        for nice, tag in nice_keys.items():
            if tag not in scalar_tags:
                continue
            for e in ea.Scalars(tag):
                per_ep_data.setdefault(e.step, {})[nice] = round(e.value, 6)

        per_epoch_list = [
            {"epoch": ep, **vals} for ep, vals in sorted(per_ep_data.items())
        ]

        cmp_fn = max if best_mode == "max" else min
        best_e = cmp_fn(ev_monitor, key=lambda x: x.value)
        best_metrics = per_ep_data.get(best_e.step, {})

        val["monitor_key"] = monitor
        val["monitor_mode"] = best_mode
        val["num_validations"] = len(ev_monitor)
        val["best"] = {
            "epoch": best_e.step,
            "value": round(best_e.value, 6),
            "all_metrics_at_best": best_metrics,
        }
        val["per_epoch"] = per_epoch_list
    elif monitor:
        val["note"] = f"monitor '{monitor}' not present in scalar tags"
    out["validation"] = val

    # ------------------- v9 acceptance auto-grade -------------------
    # Source of truth: MyScripts/run_m3fd_v9_e2e.sh L26-L30
    if "v9" in exp_name.lower() and val.get("best"):
        ep = val["best"]["epoch"]
        p1 = val["best"]["all_metrics_at_best"].get(
            "precision@1px", val["best"]["value"]
        )
        if p1 >= 0.51 and 35 <= ep <= 55:
            grade = "strong"
        elif 0.45 <= p1 < 0.51 and 35 <= ep <= 65:
            grade = "medium"
        elif 0.40 <= p1 < 0.45 and 35 <= ep <= 78:
            grade = "weak"
        else:
            grade = "fail"
        out["v9_acceptance"] = {
            "grade": grade,
            "p@1px": round(p1, 4),
            "best_epoch": ep,
            "rule_source": "MyScripts/run_m3fd_v9_e2e.sh L26-L30",
        }

    out["all_scalar_tags"] = sorted(scalar_tags)
    out["image_tags_count"] = len(ea.Tags().get("images", []))
    return out


def cmd_export(ea: EventAccumulator, tag: str, out_path: Path) -> None:
    if tag not in ea.Tags().get("scalars", []):
        sys.exit(
            f"[err] tag not found: {tag}\n"
            f"available: {sorted(ea.Tags().get('scalars', []))}"
        )
    events = ea.Scalars(tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_time", "value"])
        for e in events:
            w.writerow([e.step, e.wall_time, e.value])
    print(f"[ok] {len(events)} rows -> {out_path}")


def _exp_to_version_label(name: str) -> str:
    """Map exp dir name to short version label (v1, v6.1, v9, ...)."""
    m = re.search(r"v(\d+)(?:_(\d+))?", name)
    if not m:
        return name
    main = m.group(1)
    sub = m.group(2)
    return f"v{main}.{sub}" if sub else f"v{main}"


def _version_sort_key(label: str) -> tuple:
    """Sort version labels naturally: v1, v2, ..., v6, v6.1, v7, ..., v9."""
    m = re.match(r"v(\d+)(?:\.(\d+))?", label)
    if not m:
        return (999, 999, label)
    return (int(m.group(1)), int(m.group(2) or 0), label)


def _fmt_num(x, fmt: str = "{:.4f}") -> str:
    if x is None:
        return "—"
    try:
        return fmt.format(float(x))
    except (ValueError, TypeError):
        return str(x)


def cmd_aggregate(
    root_dir: Path,
    out_path: Optional[Path],
    include_pat: str,
    exclude_pat: str,
) -> None:
    """Scan logs/tb_logs/* and write cross-experiment Markdown tables.

    Three tables are emitted:
      §1 训练成本与收敛
      §2 训练 val 最佳 epoch + 主指标
      §3 modemb / MSBN 诊断

    The chosen version_<N> per experiment is the highest-numbered one (same
    rule as `summary` resolves), so call sites that have multiple versions
    auto-pick the latest. Override per-row by running `summary` directly with
    `--logdir logs/tb_logs/<exp>/version_<N>` if needed.
    """
    if not root_dir.exists():
        sys.exit(f"[err] root not found: {root_dir}")

    inc_re = re.compile(include_pat) if include_pat else None
    exc_re = re.compile(exclude_pat) if exclude_pat else None

    rows: List[Dict] = []
    candidates = sorted(d for d in root_dir.iterdir() if d.is_dir())
    for exp_dir in candidates:
        name = exp_dir.name
        if inc_re and not inc_re.search(name):
            continue
        if exc_re and exc_re.search(name):
            continue
        try:
            events = resolve_events(str(exp_dir))
        except SystemExit as exc:
            print(f"[skip] {name}: {exc}", file=sys.stderr)
            continue
        try:
            ea = load_ea(events)
            s = cmd_summary(ea, events)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[skip] {name}: load failed ({exc})", file=sys.stderr)
            continue
        s["version_label"] = _exp_to_version_label(name)
        rows.append(s)

    if not rows:
        sys.exit(f"[err] no experiments matched include={include_pat!r} "
                 f"exclude={exclude_pat!r} under {root_dir}")

    rows.sort(key=lambda r: _version_sort_key(r["version_label"]))

    lines: List[str] = []
    lines.append(f"<!-- auto-generated by MyScripts/read_tb_metrics.py aggregate -->")
    lines.append(f"<!-- root: {root_dir} | include: {include_pat} | exclude: {exclude_pat} -->")
    lines.append(f"<!-- experiments: {len(rows)} | timestamp: {dt.datetime.now().isoformat(timespec='seconds')} -->")
    lines.append("")

    # ---- Table 1: 训练成本与收敛
    lines.append("## §1 训练成本与收敛")
    lines.append("")
    lines.append("| 版本 | exp_name | version | dataset | total_epochs | total_steps | wall (h) | final loss | min loss | start | end |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        tr = r.get("training", {})
        wc = r.get("wall_clock") or {}
        min_loss = (tr.get("min_train_loss") or {}).get("value")
        lines.append(
            f"| {r['version_label']} | `{r['exp_name']}` | {r['version']} "
            f"| {r['dataset_type']} | {tr.get('total_epochs', '—')} "
            f"| {tr.get('total_global_steps', '—')} "
            f"| {_fmt_num(wc.get('elapsed_hours'), '{:.2f}')} "
            f"| {_fmt_num(tr.get('final_avg_loss_on_epoch'), '{:.4f}')} "
            f"| {_fmt_num(min_loss, '{:.4f}')} "
            f"| {wc.get('start_iso', '—')} | {wc.get('end_iso', '—')} |"
        )
    lines.append("")

    # ---- Table 2: 训练 val 最佳 epoch + 主指标
    lines.append("## §2 训练 val 最佳 epoch + 主指标（PL `metrics_0/*`）")
    lines.append("")
    lines.append("| 版本 | monitor (best mode) | best_epoch | val P@1px | val P@3px | val P@5px | val mpe | val num_matches | val avg_loss | num_validations |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        v = r.get("validation", {})
        best = v.get("best") or {}
        m = best.get("all_metrics_at_best") or {}
        monitor = (v.get("monitor_key") or "—").replace("metrics_0/", "")
        mode = v.get("monitor_mode", "")
        lines.append(
            f"| {r['version_label']} | `{monitor}` ({mode}) | {best.get('epoch', '—')} "
            f"| {_fmt_num(m.get('precision@1px'))} "
            f"| {_fmt_num(m.get('precision@3px'))} "
            f"| {_fmt_num(m.get('precision@5px'))} "
            f"| {_fmt_num(m.get('mean_pixel_error'))} "
            f"| {_fmt_num(m.get('num_matches'), '{:.0f}')} "
            f"| {_fmt_num(m.get('avg_loss'))} "
            f"| {v.get('num_validations', '—')} |"
        )
    lines.append("")

    # ---- Table 3: modemb / MSBN 诊断
    lines.append("## §3 modemb / MSBN 诊断（仅相应版本写出）")
    lines.append("")
    lines.append("| 版本 | modemb ir_norm | modemb vis_norm | modemb verdict | MSBN drift L1 (last/max) | MSBN drift L2 (last/max) | MSBN verdict |")
    lines.append("|---|---:|---:|---|---|---|---|")
    for r in rows:
        tr = r.get("training", {})
        em = tr.get("modality_emb") or {}
        ms = tr.get("msbn") or {}
        em_first_last = (
            f"{_fmt_num(em.get('ir_norm_first'))}→{_fmt_num(em.get('ir_norm_last'))}"
            if em else "—"
        )
        em_vis = (
            f"{_fmt_num(em.get('vis_norm_first'))}→{_fmt_num(em.get('vis_norm_last'))}"
            if em else "—"
        )
        ms_l1 = (
            f"{_fmt_num(ms.get('layer1_drift_last'))} / {_fmt_num(ms.get('layer1_drift_max'))}"
            if ms else "—"
        )
        ms_l2 = (
            f"{_fmt_num(ms.get('layer2_drift_last'))} / {_fmt_num(ms.get('layer2_drift_max'))}"
            if ms else "—"
        )
        lines.append(
            f"| {r['version_label']} | {em_first_last} | {em_vis} "
            f"| {em.get('verdict', '—')} | {ms_l1} | {ms_l2} | {ms.get('verdict', '—')} |"
        )
    lines.append("")

    text = "\n".join(lines)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[ok] aggregated {len(rows)} experiments -> {out_path}", file=sys.stderr)
    else:
        print(text)


def cmd_best(ea: EventAccumulator, tag: str, mode: str) -> Dict:
    if tag not in ea.Tags().get("scalars", []):
        sys.exit(f"[err] tag not found: {tag}")
    events = ea.Scalars(tag)
    cmp_fn = max if mode == "max" else min
    best = cmp_fn(events, key=lambda x: x.value)
    return {
        "tag": tag,
        "mode": mode,
        "best_step": best.step,
        "best_value": best.value,
        "wall_time": best.wall_time,
        "wall_time_iso": dt.datetime.fromtimestamp(best.wall_time).isoformat(
            timespec="seconds"
        ),
        "n_samples": len(events),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--logdir",
        default=None,
        help=(
            "events file, version_x dir, or experiment dir. "
            "Omit to be prompted interactively."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_tags = sub.add_parser("tags", help="list every scalar tag with stats")
    _add_common(sp_tags)

    sp_sum = sub.add_parser("summary", help="AI-friendly JSON summary")
    _add_common(sp_sum)
    sp_sum.add_argument(
        "--out",
        default=None,
        help="write JSON to this file instead of stdout",
    )

    sp_exp = sub.add_parser("export", help="export one scalar tag to CSV")
    sp_exp.add_argument("tag")
    _add_common(sp_exp)
    sp_exp.add_argument("--out", required=True)

    sp_best = sub.add_parser("best", help="find best step for one scalar")
    sp_best.add_argument("tag")
    _add_common(sp_best)
    sp_best.add_argument(
        "--mode", choices=["max", "min"], default="max"
    )

    sp_agg = sub.add_parser(
        "aggregate",
        help="scan logs/tb_logs/* and emit cross-experiment Markdown table",
    )
    sp_agg.add_argument(
        "--root", default="logs/tb_logs",
        help="root containing per-experiment subdirs (default: logs/tb_logs)",
    )
    sp_agg.add_argument(
        "--out", default=None,
        help="write Markdown to this file (default: stdout)",
    )
    sp_agg.add_argument(
        "--include", default=r"^(roadscene|m3fd|msyn)_v\d",
        help=r"regex; only exp dirs matching it are kept "
             r"(default: ^(roadscene|m3fd|msyn)_v\d)",
    )
    sp_agg.add_argument(
        "--exclude", default=r"(_debug|_small|_compat_test)",
        help=r"regex; exp dirs matching it are dropped "
             r"(default: (_debug|_small|_compat_test))",
    )

    args = ap.parse_args()

    if args.cmd == "aggregate":
        cmd_aggregate(
            Path(args.root).expanduser().resolve(),
            Path(args.out).expanduser().resolve() if args.out else None,
            args.include,
            args.exclude,
        )
        return

    events_path = resolve_events(getattr(args, "logdir", None))
    size_mb = events_path.stat().st_size / 1024**2
    print(
        f"[info] loading {events_path} ({size_mb:.1f} MB) ...",
        file=sys.stderr,
    )
    ea = load_ea(events_path)

    if args.cmd == "tags":
        cmd_tags(ea)
    elif args.cmd == "summary":
        result = cmd_summary(ea, events_path)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.out:
            out_p = Path(args.out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(text, encoding="utf-8")
            print(f"[ok] summary -> {out_p}", file=sys.stderr)
            # also print a short human-readable verdict line so users see
            # the take-away without having to open the JSON.
            v9 = result.get("v9_acceptance")
            if v9:
                print(
                    f"[v9] grade={v9['grade']}  p@1px={v9['p@1px']}  "
                    f"best_epoch={v9['best_epoch']}",
                    file=sys.stderr,
                )
            elif result.get("validation", {}).get("best"):
                b = result["validation"]["best"]
                print(
                    f"[best] epoch={b['epoch']}  "
                    f"{result['validation']['monitor_key']}={b['value']}",
                    file=sys.stderr,
                )
        else:
            print(text)
    elif args.cmd == "export":
        cmd_export(ea, args.tag, Path(args.out))
    elif args.cmd == "best":
        print(json.dumps(cmd_best(ea, args.tag, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
