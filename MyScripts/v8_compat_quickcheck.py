"""v8 R8 compatibility quick-check: validate R8a/R8b/R8c without running a
full training epoch. R8d (forward math byte-identical p@3=0.855) requires
real data + GPU, so it stays as a separate .bat invocation; this script only
checks model construction + ckpt load + param count.

R8a: no "MSBN inflated" log line printed during ckpt load
R8b: load_state_dict produces empty missing_keys + unexpected_keys
R8c: Trainable params == 16.00M, Total == 16.03M (matches v7 SKILL S6 14d)

Usage:
    python MyScripts/v8_compat_quickcheck.py
"""
import io
import logging
import sys
from pathlib import Path

import torch
from loguru import logger

# Capture all log lines to a string buffer so we can grep R8a after the fact.
log_buffer = io.StringIO()
logger.remove()
logger.add(log_buffer, level="DEBUG")
logger.add(sys.stderr, level="INFO")

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load v7 config (USE_MSBN defaults to False from default.py)
from src.config.default import get_cfg_defaults
import importlib.util


def load_cfg(cfg_path):
    spec = importlib.util.spec_from_file_location("v7_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cfg


cfg = load_cfg("configs/loftr/eloftr_full_v7_pcclahe.py")
print(f"\n[1/5] Loaded v7 cfg (USE_MSBN={cfg.LOFTR.USE_MSBN}, "
      f"BACKBONE_IN_CHANNELS={cfg.LOFTR.BACKBONE_IN_CHANNELS})")
assert cfg.LOFTR.USE_MSBN is False, "v7 cfg must default to USE_MSBN=False"

# Wire up the dummy trainer-side fields PL_LoFTR.__init__ reads
cfg.TRAINER.WORLD_SIZE = 1
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 1
# train.py sets this at runtime if None; for quickcheck we set it manually.
if cfg.LOFTR.COARSE.NPE is None:
    cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

# Build model
from src.lightning.lightning_loftr import PL_LoFTR

ckpt = "logs/tb_logs/m3fd_v7_pcclahe/version_0/checkpoints/" \
       "epoch=6-precision@1px=0.475-precision@3px=0.855-precision@5px=0.913.ckpt"
print(f"[2/5] Loading model with v7 ckpt: {ckpt}")
if not Path(ckpt).exists():
    print(f"  ERROR: ckpt not found at {ckpt}")
    sys.exit(2)

model = PL_LoFTR(cfg, pretrained_ckpt=ckpt)

# R8a: grep log for MSBN inflated
log_text = log_buffer.getvalue()
r8a_pass = "MSBN inflated" not in log_text
print(f"\n[3/5] R8a (no 'MSBN inflated' log line):     {'PASS' if r8a_pass else 'FAIL'}")
if not r8a_pass:
    for line in log_text.splitlines():
        if "MSBN" in line:
            print(f"      offending: {line}")

# R8b: missing_keys / unexpected_keys
# We need to manually re-run load_state_dict to capture msg here
# (PL_LoFTR.__init__ already did so but discarded the IncompatibleKeys obj)
state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
from src.lightning.lightning_loftr import _maybe_inflate_stage0, _maybe_inflate_msbn
state_dict = _maybe_inflate_stage0(state_dict, model.matcher.backbone, alpha=0.0)
state_dict = _maybe_inflate_msbn(state_dict, model.matcher)
msg = model.matcher.load_state_dict(state_dict, strict=False)
r8b_pass = (len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0)
print(f"[4/5] R8b (no missing_keys/unexpected_keys): {'PASS' if r8b_pass else 'FAIL'}")
if not r8b_pass:
    print(f"      missing_keys ({len(msg.missing_keys)}): {msg.missing_keys[:5]}")
    print(f"      unexpected_keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:5]}")

# R8c: param counts
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
trainable_m = n_trainable / 1e6
total_m = n_total / 1e6
# v7 SKILL gate 14d: 16.00M / Total: 16.03M
r8c_pass = (round(trainable_m, 2) == 16.00 and round(total_m, 2) == 16.03)
print(f"[5/5] R8c (params 16.00M / 16.03M):          "
      f"{'PASS' if r8c_pass else 'FAIL'}  "
      f"(actual: {trainable_m:.4f}M / {total_m:.4f}M)")

# Verdict
all_pass = r8a_pass and r8b_pass and r8c_pass
print(f"\n{'=' * 60}")
print(f"R8 quick-check verdict: {'ALL PASS' if all_pass else 'FAIL'} "
      f"(R8a={r8a_pass}, R8b={r8b_pass}, R8c={r8c_pass})")
print(f"R8d (p@3=0.855 byte-identical) requires running .bat with real data.")
print(f"{'=' * 60}")
sys.exit(0 if all_pass else 1)
