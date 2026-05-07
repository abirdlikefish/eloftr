#!/usr/bin/env python3
"""
stop / sessionEnd hook: mirror Cursor agent transcripts into the project tree.

Source : ~/.cursor/projects/<encoded-repo-root>/agent-transcripts/<uuid>/<uuid>.jsonl
Target : <repo-root>/.cursor/chat-logs/<uuid>.jsonl

Cross-platform (Windows / Linux / macOS); no bash / jq / realpath dependency.
Hook entry registers failClosed=false, so any failure here only loses one backup
and never blocks the agent session.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """Locate the repo root via git, fall back to relative path inference."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return Path(out.decode().strip()).resolve()
    except Exception:
        return Path(__file__).resolve().parents[2]


def strict_encoded(repo_root: Path) -> str:
    """First-try encoding: drive-letter lowercased, separators -> '-'.

    Matches Cursor's encoding for plain-ASCII paths
    (e.g. /home/xyjiang/Desktop/yurupeng/eloftr -> home-xyjiang-Desktop-yurupeng-eloftr).
    Will MISS paths containing non-ASCII (Cursor strips Chinese segments) or
    underscores (Cursor replaces _ with -); strategy 2 covers those cases.
    """
    abs_path = str(repo_root)
    if os.name == "nt":
        abs_path = abs_path.replace(":", "")
    abs_path = abs_path.replace("\\", "-").replace("/", "-").lstrip("-")
    if os.name == "nt" and abs_path:
        abs_path = abs_path[0].lower() + abs_path[1:]
    return abs_path


def slug(name: str) -> str:
    """Cursor-like slug: non-alnum -> '-', collapse, lowercase."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s


def find_src_dir(repo_root: Path) -> Optional[Path]:
    """Resolve ~/.cursor/projects/<encoded>/agent-transcripts/ via 3 strategies."""
    projects_root = Path.home() / ".cursor" / "projects"
    if not projects_root.is_dir():
        return None

    p1 = projects_root / strict_encoded(repo_root) / "agent-transcripts"
    if p1.is_dir():
        return p1

    name_slug = slug(repo_root.name)
    if name_slug:
        matches = sorted(
            (
                p
                for p in projects_root.glob(f"*{name_slug}*/agent-transcripts")
                if p.is_dir()
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    all_dirs = [p for p in projects_root.glob("*/agent-transcripts") if p.is_dir()]
    if all_dirs:
        return max(all_dirs, key=lambda p: p.stat().st_mtime)

    return None


def main() -> int:
    try:
        sys.stdin.read()
    except Exception:
        pass

    repo_root = get_repo_root()
    src_dir = find_src_dir(repo_root)
    dst_dir = repo_root / ".cursor" / "chat-logs"
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    total = 0
    if src_dir is not None:
        for jsonl in src_dir.glob("*/*.jsonl"):
            total += 1
            target = dst_dir / jsonl.name
            try:
                if (
                    not target.exists()
                    or jsonl.stat().st_mtime > target.stat().st_mtime
                ):
                    shutil.copy2(jsonl, target)
                    copied += 1
            except OSError:
                continue

    print(
        json.dumps(
            {
                "_archived": copied,
                "_total": total,
                "_src": str(src_dir) if src_dir else None,
                "_strict_encoded": strict_encoded(repo_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
