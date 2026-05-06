"""v8_msbn: Modality-Specific BatchNorm on fine_preprocess (cross-modal path E1).

Continues from v7's PC+CLAHE input-side optimisation by adding the orthogonal
"fine de-modality-blind" architectural change: replace the 2 single BatchNorm2d
layers in fine_preprocess.layer{1,2}_outconv2 with a pair of (bn_ir / bn_vis)
each. fine_preprocess.forward routes IR features through bn_ir and VIS through
bn_vis, allowing the two modalities to maintain their own running statistics
and affine parameters.

==============================================================================
Design rationale (cross-modal §4 path E1, two paths combined)
------------------------------------------------------------------------------
v7 (path F) attacks the cross-modal gap at the INPUT side: PC edges + IR CLAHE
align IR/VIS distributions BEFORE the network sees them. v8 (path E1) attacks
it at the FINE-PROCESSING side: the BNs that were previously seeing IR+VIS as
a single mixed batch (the "BN eats modemb bias" mechanism documented in
eloftr-v2-modemb SKILL §"BN ate the offset") now see only same-modality
samples per branch. The two paths are ORTHOGONAL: PC+CLAHE harmonises the
input distribution, MSBN lets fine BNs adapt per-modality even when residual
distribution differences leak through.

Expected behaviour (from §6.2):
  - Strong success: M3FD val p@1 >= 0.485 AND OOD p@1 not regress past 0.21
                    -> Ship v8 final, write "PC+CLAHE x MSBN orthogonal"
  - Weak success:   M3FD val p@1 in [0.475, 0.485]
                    -> Ship v7 final, v8 as marginal validation
  - Failure:        M3FD val p@1 < 0.475
                    -> Ship v7 final, v8 as negative result demonstrating
                       PC+CLAHE already absorbed the cross-modal BN noise

==============================================================================
Inherits everything else from v7 (which inherits from v6.1, v6, v5, v4 REV2,
v3, v2, v1):
  - schedule (CHANGED below): TRUE_LR=2.5e-5 inherited; WARMUP/MSLR/ES tuned
  - architecture: contrastive (v1) + modemb (v2) + FREEZE_BACKBONE_BN=True (v4)
                  + USE_EDGE_INPUT (v7) + BACKBONE_IN_CHANNELS=2 (v7)
                  + USE_CLAHE_IR (v7) + NEW: USE_MSBN=True
  - data:        M3FD 3780 train + 210 val (v5)
  - resume:      v7 ep6 ckpt via --ckpt_path in run_m3fd_v8_msbn.bat
                 (the v7 SOTA: M3FD val p@1=0.475 / p@3=0.855 / p@5=0.913)
  - operator:    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (v6.1)

==============================================================================
v8 cfg overrides (1 architectural flag + 3 schedule fixes)
------------------------------------------------------------------------------
USE_MSBN = True
  Enables fine_preprocess MSBN dual-branch (see fine_preprocess.py docstring).
  fine_preprocess.layer{1,2}_outconv2.1 becomes nn.Identity(); 4 new BN
  branches (layer{1,2}_outconv2_bn_{ir,vis}) are added. lightning_loftr.py's
  _maybe_inflate_msbn() copies the v7 ckpt's single BN weights into both
  branches at load time, so v8 epoch 0 is byte-identical to v7 ep6.

==============================================================================
Schedule fixes for v7 dead code (v7 ran with --max_epochs=10)
------------------------------------------------------------------------------
v7 inherited v6.1's schedule designed for max_epochs=50, which left three
flags as dead code under v7's max_epochs=10:
  - v7 MSLR=[15,25,35]  never fired (max_ep=10 < 15) -> full LR throughout
  - v7 ES patience=12   mathematically impossible to trigger (10 < 12)
  - v7 WARMUP=50        wasted ~1 epoch on a converged-resume start

Plan B chosen (max_epochs=20, see plan §6.1 for ES trigger preview):
  WARMUP=30           ~300 abs step ~= 0.32 epoch. Buffers AdamW second-
                      moment reset + the new MSBN branches' adaptation.
  MSLR=[6, 11, 15]    All three milestones fall within max_epochs=20:
                       ep 6 first decay -- AFTER expected peak (~ep 6 like v7)
                       ep 11 second decay -- low-LR refinement
                       ep 15 third decay -- end-of-run stabilisation
  ES patience=8       Will actually trigger if the run plateaus (max_ep=20 > 8)
  --max_epochs=20     (in .bat, not in cfg)

LR timeline (TRUE_LR=2.5e-5, bs=4, M3FD ~945 step/epoch):
  warmup    ep 0     - 0.32   ->  0       -> 2.5e-5
  full      ep 0.32  - 6      ->  2.5e-5
  MSLR #1   ep 6     - 11     ->  1.25e-5
  MSLR #2   ep 11    - 15     ->  6.25e-6
  MSLR #3   ep 15    - 20     ->  3.13e-6
Total ~19k steps (v7 had ~9.5k). Three decays now actually fire.

==============================================================================
v0-v7 backward compatibility
------------------------------------------------------------------------------
This config only opts in to USE_MSBN. The lightning_loftr.py changes
(freeze_logic refactor + _maybe_inflate_msbn hook + monitor block) are all
guarded by USE_MSBN/use_msbn defaults to False, so any v0-v7 cfg merge
produces byte-identical behaviour to before v8 landed. See plan §4.5
"v0-v7 reproduction compatibility checklist" (10 items).
"""
from configs.loftr.eloftr_full_v7_pcclahe import cfg

# v8: MSBN on fine_preprocess (cross-modal §4 E1)
cfg.LOFTR.USE_MSBN = True

# Schedule fixes for v7 dead code under max_epochs=10
cfg.TRAINER.WARMUP_STEP             = 30           # v7=50 (too long for resume)
cfg.TRAINER.MSLR_MILESTONES         = [6, 11, 15]  # v7=[15,25,35] all dead under max_ep=10
cfg.TRAINER.EARLY_STOPPING_PATIENCE = 8            # v7=12 unreachable under max_ep=10
# CANONICAL_LR=4e-4 (TRUE_LR=2.5e-5) inherited from v7 unchanged -- already at v2 effective peak.
# bat: --max_epochs=20  (v7 was 10)
