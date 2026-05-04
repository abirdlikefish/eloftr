"""v4: a softer v3 — keep selective BN freeze + EarlyStopping, undo over-aggressive choices.

==============================================================================
REVISION 2 (BN-only fix): roll back LR-only revision; switch from FREEZE_BN=True
to FREEZE_BACKBONE_BN=True (freeze backbone BN only, free fine_preprocess BN)
------------------------------------------------------------------------------
Previous revision (CANONICAL_LR 2e-3 -> 4e-4, max_epochs 30 -> 80) was
empirically refuted: lowering LR by ~5x to ~2.4e-5 made precision@3px DROP
to ~0.684, even worse than original v4's 0.711. This rules out "LR + step
count alone" as the cause of v4 < v2.

Revised hypothesis: original v4's gap to v2 is dominated by FREEZE_BN=True
freezing BOTH backbone BN AND fine_preprocess BN (the 2 BN layers inside
fine_preprocess.layer{1,2}_outconv2). v2 had FREEZE_BN=False so both BN
groups stayed trainable; that let fine_preprocess BN slowly adapt to the
IR-VIS cross-modal distribution over its very long (~6400-step) run. v3/v4
froze fine BN and lost that adaptation entirely.

This revision tests the minimal targeted fix:
  * roll CANONICAL_LR back to 2e-3 (original v4 value, TRUE_LR=1.25e-4)
  * roll max_epochs back to 30 (original v4 value)
  * set FREEZE_BN = False (was True)
  * set FREEZE_BACKBONE_BN = True (NEW flag)
        -> backbone BN: eval-mode + no-grad (pin MegaDepth running stats)
        -> fine_preprocess BN (2 layers, ~768 affine params): trainable + train-mode

If precision@3px climbs above original v4's 0.711 toward v2's 0.80, the
diagnosis is confirmed: the bottleneck was fine BN being frozen, not LR /
step count / backbone capacity. If precision@3px stays around 0.71 or drops,
fine BN is NOT the bottleneck and we'll move on to MSLR/ES patience tuning
(see v5 open question at bottom).

Original v4 (LR=2e-3 / max_epochs=30 / FREEZE_BN=True) trained result is
preserved at logs/tb_logs/roadscene_v4_combined/version_* and remains
comparable in TensorBoard alongside this revision's run.
==============================================================================

Why v3 underperformed v2 (best 0.678 < 0.806):
  - FREEZE_BACKBONE=True killed 9.5M conv params (out of 16M total), capping
    IR-domain adaptation. v2 kept them trainable and that gave it the lead;
    v3 still has step1 (InfoNCE) + step2 (modemb) but those alone cannot
    compensate for the lost backbone capacity on IR.
  - WARMUP_STEP=1 + CANONICAL_LR=8e-3 (TRUE_LR=5e-4) hits full LR by step 16,
    too hot for the small RoadScene set; combined with MSLR_MILESTONES=[5,7]
    and EarlyStopping patience=4, training peaked early then crashed before
    it could recover from the high-LR overshoot.

Why we don't just retry v2:
  - v2 best (0.806) came from CANONICAL_LR=8e-3 + WARMUP_STEP=1875 — at bs=4
    that warmup auto-scales to ~30000 steps, so the entire 50-epoch run was
    stuck inside a slow ramp from ~1.25e-5 up to ~5e-5. The peak LR v2 ever
    actually saw was ~4e-5. Resuming v2 from its 0.7996 ckpt yielded 0.7932
    at the next val, fluctuating in 0.79-0.80 — v2 has *plateaued* under
    that schedule and just adding epochs won't help. We need a different
    LR regime.

v4 keeps what worked in v3 and softens what didn't:
  1) FREEZE_BACKBONE=False (override v3=True)
       Restore the 9.5M backbone — this is the biggest single reason v2
       beat v3 and we want that capacity back for IR adaptation.
  2) FREEZE_BN=False + FREEZE_BACKBONE_BN=True (REVISION 2: was FREEZE_BN=True)
       Pin backbone BN running_mean/running_var to MegaDepth pretrained
       values (with bs=4 the batch BN stats would be too noisy and would
       drift over many epochs of small-batch training), but ALLOW the 2
       fine_preprocess BN layers to keep updating so they can adapt to the
       IR-VIS cross-modal distribution. This is the central change tested
       by REVISION 2.
  3) CANONICAL_LR=2e-3 (override v3=8e-3) -> TRUE_LR = 2e-3 * 4/64 = 1.25e-4
       4x lower than v3's 5e-4 (calmer for full-capacity training);
       ~3x v2's effective peak (~4e-5, escapes the ramp trap).
       Sits in the sweet spot between v2's "too cold" and v3's "too hot".
  4) WARMUP_STEP=2 (override v3=1) -> auto-scaled to 32 steps (~1 epoch)
       v3's 1 was effectively no warmup. With 9.5M backbone now trainable
       we want a brief ramp so AdamW second-moment estimates settle before
       full LR; 1 epoch is enough given the pretrained init.
  5) MSLR_MILESTONES=[10, 15, 20] (override v3=[5, 7])
       Three decays (0.5x / 0.25x / 0.125x) staged after at least 10 epochs
       at full LR. v3's two decays at [5, 7] gave the model only 5 high-LR
       epochs, way too short for backbone-trainable runs.
  6) EARLY_STOPPING_PATIENCE=8 (override v3=4)
       v3's 4 stopped runs that hadn't really converged. Mid-range LR with
       full capacity will oscillate short-term; 8 lets us tolerate that.
  EARLY_STOPPING=True, USE_CONTRASTIVE=True, USE_MODALITY_EMB=True all
  inherited unchanged from v3 -> v2 -> v1.

Resulting effective TRUE LR trajectory at bs=4 (~40 train batches/epoch),
post-REVISION 2 (back to CANONICAL_LR=2e-3, max_epochs=30):
    epoch 0  step  0     LR ramps from 0 over ~32 steps
    epoch  1 onward      LR = 1.25e-4 (full)
    epoch 10 end         LR = 6.25e-5 (MSLR step #1, gamma=0.5)
    epoch 15 end         LR = 3.13e-5 (MSLR step #2)
    epoch 20 end         LR = 1.56e-5 (MSLR step #3)
    epoch 20-30          LR stays 1.56e-5
    EarlyStopping (patience=8) most often fires before epoch 30.
  (REVISION 1's TRUE_LR ~2.5e-5 / 80-epoch trajectory deleted because that
   revision is being rolled back.)

Open question for v5 (kept out of this config for clean ablation):
  If REVISION 2 approaches v2 (~0.80), the modemb growth issue was a
  symptom not a cause and no v5 modemb L2 reg is needed. If it still
  underperforms v2, the next steps in priority order are:
    1) Push MSLR_MILESTONES to [40, 60] + max_epochs to 80 so most of the
       run sees full LR (currently MSLR kills LR by epoch 20).
    2) Disable EarlyStopping or raise patience to 16 to let the late-epoch
       co-adaptation finish (v2's best was at epoch 49).
    3) Only after 1+2 are exhausted, consider modemb L2 reg
       (USE_MODEMB_REG / MODEMB_REG_WEIGHT) on top of v4.
"""
from configs.loftr.eloftr_full_v3_combined import cfg

cfg.LOFTR.FREEZE_BACKBONE              = False         # override v3=True; restore 9.5M backbone
cfg.LOFTR.FREEZE_BN                    = False         # REVISION 2: was True; release fine BN
cfg.LOFTR.FREEZE_BACKBONE_BN           = True          # REVISION 2 NEW: pin backbone BN only

cfg.TRAINER.EARLY_STOPPING_PATIENCE    = 8             # override v3=4; tolerate mid-LR oscillation

cfg.TRAINER.CANONICAL_LR               = 2e-3          # REVISION 2: rolled back from 4e-4 to original 2e-3; TRUE_LR = 2e-3 * 4/64 = 1.25e-4 at bs=4
cfg.TRAINER.WARMUP_STEP                = 2             # override v3=1; auto-scaled to ~32 steps
cfg.TRAINER.MSLR_MILESTONES            = [10, 15, 20]  # override v3=[5,7]; three decays after full LR
