"""v13 = real K + W2C pose + depth supervision on cross-view + cross-modal
Megadepth_Syn (4-GPU DDP, B-pure stack).

Short-circuit inheritance from eloftr_full.py (NOT the v9_e2e -> v10 -> v11
chain that v9..v11 use). All v1-v8 USE_* flags
(USE_CONTRASTIVE / USE_MODALITY_EMB / USE_EDGE_INPUT / USE_CLAHE_IR /
USE_MSBN / BACKBONE_IN_CHANNELS=2) stay at their default ``False`` /
``1`` values; the model is byte-compatible with outdoor.ckpt's first-layer
conv so cold start needs no inflate hook.

Rationale (see SKILL eloftr-v13-pose docstring for the full story):
- v10/v11/v12 all train with 2D Homography supervision on same-viewpoint
  IR-VIS pairs. v12 isolated the H aug knob by short-circuiting v1-v8
  stack; result on METU was prec UP / AUC DOWN (modality + geometry mismatch
  vs essential matrix benchmark).
- v13 changes the supervision contract entirely: real camera geometry
  (K, W2C pose, depth) from LoFTR scene_info_0.1_0.7, plus pair_infos that
  pairs DIFFERENT viewpoints (idx0 != idx1). Hence "cross-view + cross-modal
  pose supervision".
- B-pure (this cfg) keeps v1-v8 stack OFF. Reason: outdoor.ckpt is trained
  exactly on K+pose+depth MegaDepth in 1-channel raw image space. v13 fine-
  tunes that exact pretext on cross-modal data, so any v1-v8 stack added
  here would obscure whether benefit comes from "pose supervision" vs
  "modality-invariance bias from MSBN/modemb/PC". B-fused (v13b, NOT this
  cfg) layers the stack back on top after v13a results land.

DDP LR / WARMUP timeline (4-GPU bs=2, effective_batch=8; reverse-scaled
to keep TRUE_LR == v10/v9 1.25e-4 despite halved bs).

Why bs=2 instead of bs=4: 832 path coarse_matching's sim_matrix is
(bs, 10816, 10816) fp32 = 1.87 GB / batch -- AGG_SIZE=4 only helps
attention, NOT coarse_matching's dual-softmax. With bs=4 + DDP the peak
hits ~22 GB of 24 GB on RTX 3090, OOM-prone (hit on the LoFTR-split
ship 2026-05-13). Halving to bs=2 cuts sim_matrix peak to 0.93 GB and
brings stable headroom to ~8 GB / card.

  _scaling = (2 * 4) / 64 = 0.125
  TRUE_LR  = CANONICAL_LR * _scaling = 1e-3 * 0.125 = 1.25e-4 (== v10 == v9)
  actual WARMUP step = WARMUP_STEP / _scaling = 225 / 0.125 = 1800 step

Schedule: max_ep=12, MSLR=[3,5,7] (LR x= 0.5 each), ES patience=3.
v13 wall-clock estimate: train ~152 min/ep (bs halved -> step doubled
to 9200/rank/ep at 1.0 it/s) + val ~23 min/ep (4493 pair * 0.5 limit /
4 ranks / bs=1 = 562 step at 2.45 s/step) = 175 min/ep, 12 ep ~ 35h.
Roughly 2x v10 ddp's 17h due to bs halving.

Sampler: scene-balance N_SAMPLES_PER_SUBSET=200 (LoFTR default; v13 train list
is LoFTR official train_list.txt = 153 scene / 368 npz / 8.86M pair pool).
4-GPU get_local_split shards npz at scene level (data.py L320), so each rank
gets ~92 npz * 200 = 18.4K pair pre-shuffle per ep, /bs=2 = 9200 step / rank
/ ep. Acceptable density for fine-tuning from outdoor.ckpt; no need to push
higher (LoFTR original recipe).

Val: LoFTR official val_list.txt is too large for 12-ep DDP (147K-pair
npz `0022_0.1_0.3` alone), so v13 uses a 2-line truncation (Trevi
``0015_0.3_0.5`` + Pantheon ``0022_0.5_0.7`` = 4493 pair) plus
``--limit_val_batches=0.5`` from the run script -> 2247 pair effective val
each epoch. Still > LoFTR megadepth_val_1500 baseline (1500 pair) so
best-ckpt selection signal is unaffected.

Acceptance (set in §6 of plan; reproduced here for grep-ability):
  Strong : METU all auc@20 >= 0.05 (5x of v10's 0.87%)
  Medium : METU all auc@20 in [0.02, 0.05]  -> reconsider B-fused next
  Weak   : METU all auc@20 < 0.02           -> revisit WARMUP / LR / N
  Crash  : NaN loss or METU auc < outdoor.ckpt -> dispatch / unit bug
"""
from configs.loftr.eloftr_full import cfg

# --- model stack: B-pure, byte-compatible with outdoor.ckpt ---
# (default values from src/config/default.py; listed here for auditability)
cfg.LOFTR.BACKBONE_IN_CHANNELS = 1
cfg.LOFTR.USE_EDGE_INPUT = False
cfg.LOFTR.USE_CLAHE_IR = False
cfg.LOFTR.USE_CLAHE_VIS = False
cfg.LOFTR.USE_MSBN = False
cfg.LOFTR.USE_MODALITY_EMB = False
cfg.LOFTR.LOSS.USE_CONTRASTIVE = False

# --- freeze: fully unfreeze, like v10 mode B (Megadepth_Syn ample data) ---
cfg.LOFTR.FREEZE_BACKBONE = False
cfg.LOFTR.FREEZE_BN = False
cfg.LOFTR.FREEZE_BACKBONE_BN = False

# --- DDP schedule (bs=2 reverse-scaled to TRUE_LR=1.25e-4 same as v10 ddp) ---
# bs=4 was OOM-prone on 3090 24GB at IMG_RESIZE=832 because coarse_matching's
# sim_matrix is (bs, 10816, 10816) fp32 = 1.87GB per batch (AGG_SIZE=4 only
# helps attention, not coarse_matching). bs=2 halves sim_matrix peak.
cfg.TRAINER.CANONICAL_LR = 1e-3              # TRUE_LR = 1e-3 * 0.125 = 1.25e-4
cfg.TRAINER.WARMUP_STEP = 225                # actual = 225 / 0.125 = 1800 step
cfg.TRAINER.MSLR_MILESTONES = [3, 5, 7]
cfg.TRAINER.MSLR_GAMMA = 0.5                 # default; listed for audit
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 3

# --- sampler: LoFTR default N=200, no replacement (use 200 distinct pairs/scene) ---
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 200
cfg.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = False

# --- plotting: enable (pose-based supervision can use the default epi-error path) ---
# eloftr_full.py default already True; v13 needs no override here.

# bat / sh: --max_epochs=12, --batch_size=2, --gpus=4, --disable_mp, --thr 0.1
